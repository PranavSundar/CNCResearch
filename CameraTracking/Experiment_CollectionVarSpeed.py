"""
Experiment_CollectionVarSpeed.py

Unified experiment controller: AprilTag Kalman tracking, NI cDAQ sensors,
CNC motion, and configurable attack modes for security research.

Trajectory modes: "square" | "circle" | "varspeed"
Attack modes:
  0 — Nominal (no attack)
  1 — G-code injection (extra G-code inserted mid-trajectory)
  2 — Feedrate tamper (override feed rates during motion)
  3 — Telemetry/vision spoofing (camera observations replaced with synthetic path)

Post-run outputs saved to OUTPUT_DIR:
  <stem>_cam.csv        14-col camera records
  <stem>_grbl.csv        8-col GRBL position records
  <stem>_events.csv      timestamped event log
  <stem>_commands.csv    timestamped command log
  <stem>_archive.npz     compressed archive (all streams + metadata JSON)
  <stem>_tracking.png    tracking figure
  <stem>_accel.png       accelerometer analysis
  <stem>_ae.png          AE sensor analysis
  <stem>_report.pdf      multi-page report

Dependencies:
  pip install pyserial numpy opencv-contrib-python matplotlib scipy nidaqmx
"""

import sys
import os
import json
import serial
import time
import re
import csv
import threading
import traceback
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pub_fig import pub_fig

if not hasattr(cv2, "aruco"):
    sys.exit(
        "cv2.aruco not found — install the contrib build:\n"
        "  pip uninstall opencv-python && pip install opencv-contrib-python"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RUN_LABEL       = "exp"         # short label prepended to output filenames
TRAJECTORY_MODE = "varspeed"    # "square" | "circle" | "varspeed"
ATTACK_MODE     = 0             # 0=nominal  1=gcode_inject  2=feed_tamper  3=spoof

# ── GRBL / CNC ────────────────────────────────────────────────────────────────
PORT   = "COM3"
BAUD   = 115200
Z_SAFE = 4.0      # mm — plunge depth (positive = downward in machine coords)
TARGET_FPS = 80   # GRBL status-poll rate (Hz)

# Variable-speed staircase trajectory
# Each entry: (feedrate_mm_min, n_segments)
# Segments alternate Y-negative / X-positive moves of VARSPEED_STEP_MM each.
VARSPEED_STEP_MM = 10
VARSPEED_PROFILE = [
    (200,  3),   # 3 segments at F200
    (600,  3),   # 3 segments at F600
    (1200, 4),   # 4 segments at F1200
]

# Derived — used for reference display and metadata; XY motion uses per-zone feeds
_vs_total_segs = sum(n for _, n in VARSPEED_PROFILE)
SIDE = VARSPEED_STEP_MM * (_vs_total_segs // 2)   # bounding box edge (= 50 mm)
FEED = VARSPEED_PROFILE[0][0]                      # plunge feedrate = first zone

# $13=0 forces GRBL to report positions in mm regardless of G20/G21 state.
# G21 = metric mode, G92 zeros the work coordinate at the home position.
GRBL_POSITION_UNITS = "mm"   # "mm" | "in" — must match machine $13/$20 setting

# Attack-mode parameters
ATTACK_INJECT_GCODE = "G4 P0.5"  # G-code inserted mid-move (attack mode 1)
ATTACK_FEED_SCALE   = 0.5         # feedrate multiplier applied to every zone (attack mode 2)
ATTACK_SPOOF_DELAY  = 5.0         # seconds after motion start to activate spoof (mode 3)

# ── ESP32 STEP/DIR + CURRENT SENSOR ──────────────────────────────────────────
ESP32_PORT   = "COM7"    # separate serial port from GRBL (COM3)
ESP32_BAUD   = 115200
STEPS_PER_MM = 320       # steps/mm from GRBL $100/$101; used to convert step count → mm

# ── CAMERA ────────────────────────────────────────────────────────────────────
CAM_INDEX   = 1
CAM_W       = 1920
CAM_H       = 1080
TAG_SIZE_MM = 30.0    # physical AprilTag side length (mm)

# ── KALMAN / RTS ──────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE     = 5.0    # mm/s²
KALMAN_MEASUREMENT_NOISE = 0.3    # mm
RTS_PROCESS_NOISE        = 50.0   # process noise for RTS smoother — higher = sharper corners
POSE_JUMP_MM             = 80.0   # outlier-rejection threshold
MIN_TAG_SIDE_PX          = 20.0   # discard tiny detections

# ── DAQ ───────────────────────────────────────────────────────────────────────
DAQ_DEVICE        = "cDAQ9185-22C6F90"

ACCEL_MODULE      = 1
ACCEL_CHANNEL     = "ai0"
ACCEL_SAMPLE_RATE = 8192.5243
ACCEL_RANGE       = 10.0      # ±V

AE_MODULE         = 3
AE_CHANNEL        = "ai1"
AE_SAMPLE_RATE    = 131147.541
AE_RANGE          = 10.0      # ±V

DAQ_BURST_DURATION = 0.1      # seconds per read burst

# ── LIVE DISPLAY ──────────────────────────────────────────────────────────────
LIVE_WINDOW_S         = 2.0
LIVE_UPDATE_HZ        = 10
LIVE_DOWNSAMPLE_ACCEL = 4
LIVE_DOWNSAMPLE_AE    = 64

# ── OUTPUT ────────────────────────────────────────────────────────────────────
PLOT_DPI        = 100
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
SAVE_SENSOR_CSV = False   # sensor CSVs are large; NPZ archive is the primary record


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CLOCK & SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentClock:
    """High-precision experiment-relative timer using perf_counter_ns."""

    def __init__(self):
        self._t0_ns = time.perf_counter_ns()

    def now(self):
        """Return seconds elapsed since clock creation."""
        return (time.perf_counter_ns() - self._t0_ns) * 1e-9

    def reset(self):
        self._t0_ns = time.perf_counter_ns()


class SharedState:
    """Thread-safe container for inter-thread attack coordination."""

    def __init__(self):
        self._lock            = threading.Lock()
        self.motion_start_t   = None   # set by grbl_worker when first move begins
        self.spoof_active     = False
        self.attack_triggered = False

    def set_motion_start(self, t):
        with self._lock:
            self.motion_start_t = t

    def get_motion_start(self):
        with self._lock:
            return self.motion_start_t

    def activate_spoof(self):
        with self._lock:
            self.spoof_active     = True
            self.attack_triggered = True

    def is_spoof_active(self):
        with self._lock:
            return self.spoof_active


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log_event(event_log, clock, tag, msg):
    """Append a timestamped event entry (thread-safe via list.append / GIL)."""
    t = clock.now()
    event_log.append((t, tag, msg))
    print(f"  [event {t:.3f}s] [{tag}] {msg}")


def log_command(command_log, clock, cmd, label):
    """Append a timestamped command entry."""
    command_log.append((clock.now(), cmd, label))


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_xy_waypoints(mode, side, n_circle_pts=72):
    """
    Return (waypoints_xy, mode_flag) where waypoints_xy is Nx2.
    Circle is approximated as a closed polygon with n_circle_pts segments.
    """
    if mode == "square":
        pts = np.array([
            [0,    0],
            [side, 0],
            [side, -side],
            [0,    -side],
            [0,    0],
        ], float)
        return pts, "square"
    elif mode == "circle":
        r      = side / 2.0
        angles = np.linspace(0, 2 * np.pi, n_circle_pts + 1)
        cx, cy = r, -r
        pts    = np.column_stack([cx + r * np.cos(angles),
                                  cy + r * np.sin(angles)])
        return pts, "circle"
    elif mode == "varspeed":
        # Staircase: alternating Y-negative (even) / X-positive (odd) steps
        total_segs = sum(n for _, n in VARSPEED_PROFILE)
        pts = [(0.0, 0.0)]
        x, y = 0.0, 0.0
        for i in range(total_segs):
            if i % 2 == 0:
                y -= VARSPEED_STEP_MM
            else:
                x += VARSPEED_STEP_MM
            pts.append((x, y))
        return np.array(pts, float), "varspeed"
    else:
        raise ValueError(f"Unknown TRAJECTORY_MODE: {mode!r}")


def _build_varspeed_program(z_safe, attack_mode):
    """
    Build the G-code sequence for the variable-speed staircase trajectory.
    Each zone in VARSPEED_PROFILE runs at its own feedrate; attack mode 2
    scales every zone's feed by ATTACK_FEED_SCALE.
    """
    waypoints, _ = _build_xy_waypoints("varspeed", SIDE)
    scale        = ATTACK_FEED_SCALE if attack_mode == 2 else 1.0

    # Per-segment feedrate list (matches waypoint segments 1..N)
    seg_feeds = []
    for f, n in VARSPEED_PROFILE:
        seg_feeds.extend([f] * n)

    sequence = [("G61", "Exact stop mode")]

    # Plunge at first-zone speed
    plunge_f = int(seg_feeds[0] * scale)
    sequence.append((f"G1 Z-{z_safe:.1f} F{plunge_f}", "Plunge Z"))

    inject_at = len(waypoints) // 2
    for i in range(1, len(waypoints)):
        xn, yn = waypoints[i]
        f_zone = seg_feeds[i - 1]
        f_cmd  = int(f_zone * scale)
        label  = f"-> ({xn:.0f},{yn:.0f}) F{f_zone}"
        sequence.append((f"G1 X{xn:.3f} Y{yn:.3f} F{f_cmd}", label))
        if attack_mode == 1 and i == inject_at:
            sequence.append((ATTACK_INJECT_GCODE, "[ATTACK] injected G-code"))

    # Return home and retract at last-zone speed
    last_f = int(seg_feeds[-1] * scale)
    sequence.append((f"G1 X0 Y0 F{last_f}", "<- Home"))
    sequence.append((f"G1 Z{z_safe + 5:.1f} F{last_f}", "Retract Z"))

    return sequence


def build_motion_program(mode, side, feed, z_safe, attack_mode=0):
    """
    Build the full G-code move sequence as [(cmd, label), ...].
    Handles attack mode 1 (G-code injection) and mode 2 (feedrate tamper).
    """
    if mode == "varspeed":
        return _build_varspeed_program(z_safe, attack_mode)

    actual_feed = feed * ATTACK_FEED_SCALE if attack_mode == 2 else feed
    waypoints, traj_flag = _build_xy_waypoints(mode, side)
    sequence = []

    # Mode setting — G61 exact stop for square corners, G64 continuous for circle
    if traj_flag in ("square", "diamond"):
        sequence.append(("G61", "Exact stop mode"))
    else:
        sequence.append(("G64", "Continuous mode"))

    # Plunge Z
    sequence.append((f"G1 Z-{z_safe:.1f} F{actual_feed:.0f}", "Plunge Z"))

    # Traverse each waypoint segment
    for i in range(1, len(waypoints)):
        xp, yp = waypoints[i - 1]
        xn, yn = waypoints[i]
        label  = f"-> ({xn:.0f},{yn:.0f})"
        sequence.append((f"G1 X{xn:.3f} Y{yn:.3f} F{actual_feed:.0f}", label))

        # Attack mode 1: inject extra G-code at midpoint of trajectory
        if attack_mode == 1 and i == len(waypoints) // 2:
            sequence.append((ATTACK_INJECT_GCODE, "[ATTACK] injected G-code"))

    # Retract
    sequence.append((f"G1 X0 Y0 F{actual_feed:.0f}", "<- Home"))
    sequence.append((f"G1 Z{z_safe + 5:.1f} F{actual_feed:.0f}", "Retract Z"))

    return sequence


def build_reference_arrays(mode, side, feed, dt=0.05):
    """
    Dense time-sampled ideal trajectory (t, x, y, z) for error comparison.
    Returns (t, x, y, z) as numpy arrays.
    For varspeed mode, each segment is timed using its own zone feedrate.
    """
    if mode == "varspeed":
        waypoints, _ = _build_xy_waypoints("varspeed", side)
        seg_feeds = []
        for f, n in VARSPEED_PROFILE:
            seg_feeds.extend([f] * n)

        # Cumulative time at each waypoint boundary
        t_wp = [0.0]
        for i in range(1, len(waypoints)):
            dist = float(np.linalg.norm(waypoints[i] - waypoints[i - 1]))
            t_wp.append(t_wp[-1] + dist / (seg_feeds[i - 1] / 60.0))
        t_wp  = np.array(t_wp)
        t_total = t_wp[-1]

        t_out, x_out, y_out = [], [], []
        for t in np.arange(0, t_total, dt):
            k     = int(np.clip(np.searchsorted(t_wp, t, side="right") - 1,
                                0, len(waypoints) - 2))
            alpha = (t - t_wp[k]) / max(t_wp[k + 1] - t_wp[k], 1e-9)
            xy    = waypoints[k] + alpha * (waypoints[k + 1] - waypoints[k])
            t_out.append(t); x_out.append(xy[0]); y_out.append(xy[1])

        return (np.array(t_out), np.array(x_out),
                np.array(y_out), np.zeros(len(t_out)))

    waypoints, _ = _build_xy_waypoints(mode, side)
    dists        = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total        = dists.sum()
    feed_mm_s    = feed / 60.0
    t_total      = total / feed_mm_s
    cum          = np.concatenate([[0], np.cumsum(dists)])

    t_out, x_out, y_out = [], [], []
    for t in np.arange(0, t_total, dt):
        s       = (t / t_total) * total
        seg_idx = int(np.clip(np.searchsorted(cum, s, side="right") - 1,
                              0, len(waypoints) - 2))
        alpha   = (s - cum[seg_idx]) / max(dists[seg_idx], 1e-9)
        p0, p1  = waypoints[seg_idx], waypoints[seg_idx + 1]
        xy      = p0 + alpha * (p1 - p0)
        t_out.append(t); x_out.append(xy[0]); y_out.append(xy[1])

    return (np.array(t_out), np.array(x_out),
            np.array(y_out), np.zeros(len(t_out)))


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK MODE 3: POSITION SPOOF
# ══════════════════════════════════════════════════════════════════════════════

# Cache reference arrays so apply_spoof_if_active doesn't rebuild each frame.
_spoof_ref_cache = None


def _get_spoof_ref():
    global _spoof_ref_cache
    if _spoof_ref_cache is None:
        _spoof_ref_cache = build_reference_arrays(TRAJECTORY_MODE, SIDE, FEED)
    return _spoof_ref_cache


def apply_spoof_if_active(state, clock, tx_real, ty_real, tz_real):
    """
    If attack mode 3 is active and the spoof delay has elapsed, return a
    synthetic position from the ideal trajectory instead of the real observation.
    Returns (x_obs, y_obs, z_obs, spoof_flag).
    """
    if ATTACK_MODE != 3:
        return tx_real, ty_real, tz_real, 0

    motion_t = state.get_motion_start()
    if motion_t is None:
        return tx_real, ty_real, tz_real, 0

    elapsed = clock.now() - motion_t
    if elapsed >= ATTACK_SPOOF_DELAY and not state.is_spoof_active():
        state.activate_spoof()

    if state.is_spoof_active():
        t_ref, x_ref, y_ref, _ = _get_spoof_ref()
        t_now = min(elapsed, t_ref[-1])
        x_spoof = float(np.interp(t_now, t_ref, x_ref))
        y_spoof = float(np.interp(t_now, t_ref, y_ref))
        return x_spoof, y_spoof, tz_real, 1

    return tx_real, ty_real, tz_real, 0


# ══════════════════════════════════════════════════════════════════════════════
# 3D KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

class KalmanFilter3D:
    """6-state Kalman filter: state = [x, y, z, vx, vy, vz]."""

    def __init__(self, dt, process_noise_sigma, measurement_noise_sigma,
                 measurement_noise_z=None):
        if measurement_noise_z is None:
            measurement_noise_z = measurement_noise_sigma * 1.5
        self.state = np.zeros(6)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], float)
        rxy = measurement_noise_sigma ** 2
        rz  = measurement_noise_z ** 2
        self.R = np.diag([rxy, rxy, rz])
        self.P = np.diag([0.1, 0.1, 0.1, 100., 100., 100.])
        self.F = self._make_F(dt)
        self.Q = self._make_Q(dt, process_noise_sigma ** 2)
        self.is_initialized = False

    @staticmethod
    def _make_F(dt):
        return np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ], float)

    @staticmethod
    def _make_Q(dt, q):
        return q * np.array([
            [dt**4/4,       0,       0, dt**3/2,       0,       0],
            [      0, dt**4/4,       0,       0, dt**3/2,       0],
            [      0,       0, dt**4/4,       0,       0, dt**3/2],
            [dt**3/2,       0,       0,   dt**2,       0,       0],
            [      0, dt**3/2,       0,       0,   dt**2,       0],
            [      0,       0, dt**3/2,       0,       0,   dt**2],
        ])

    def initialize(self, x, y, z):
        self.state = np.array([x, y, z, 0., 0., 0.])
        self.P = np.diag([0.1, 0.1, 0.1, 100., 100., 100.])
        self.is_initialized = True

    def _rebuild(self, dt):
        self.F = self._make_F(dt)
        self.Q = self._make_Q(dt, KALMAN_PROCESS_NOISE ** 2)

    def predict(self, dt):
        dt = np.clip(dt, 0.005, 0.2)
        self._rebuild(dt)
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z    = np.asarray(measurement, float)
        y    = z - self.H @ self.state
        S    = self.H @ self.P @ self.H.T + self.R
        K    = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_position(self):    return self.state[:3]
    def get_velocity(self):    return self.state[3:]
    def get_position_uncertainty(self):
        return np.sqrt([self.P[0, 0], self.P[1, 1], self.P[2, 2]])


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA THREAD
# 14-column records:
#   t, cx, cy, tx_f, ty_f, tz_f, tag_px,
#   tx_raw, ty_raw, tz_raw,
#   x_obs, y_obs, z_obs, spoof_active
# ══════════════════════════════════════════════════════════════════════════════

def camera_worker(cam_records, clock, stop_event, home_event,
                  motion_done_event, homing_done_event, shared_state, event_log):
    """Detect AprilTag, run Kalman filter, append 14-column records."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_det    = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx_origin = cy_origin = scale_origin = tag_size_origin = None
    prev_cx = prev_cy = None
    _origin_buf = []

    kf = KalmanFilter3D(
        dt=1.0 / 30.0,
        process_noise_sigma=KALMAN_PROCESS_NOISE,
        measurement_noise_sigma=KALMAN_MEASUREMENT_NOISE,
        measurement_noise_z=KALMAN_MEASUREMENT_NOISE * 4.0,
    )
    prev_frame_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        t_now      = clock.now()
        frame_time = time.time()
        dt         = frame_time - prev_frame_time
        prev_frame_time = frame_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Predict outside detection loop so filter runs even on missed frames
        if kf.is_initialized:
            kf.predict(dt)

        corners_list, ids, _ = aruco_det.detectMarkers(gray)
        n_tags     = 0 if ids is None else len(ids)
        frame_disp = frame.copy()

        if ids is not None:
            for corners, tag_id in zip(corners_list, ids.flatten()):
                pts = corners[0]
                tag_side_px = np.mean([
                    np.linalg.norm(pts[1] - pts[0]),
                    np.linalg.norm(pts[2] - pts[1]),
                    np.linalg.norm(pts[3] - pts[2]),
                    np.linalg.norm(pts[0] - pts[3]),
                ])
                if tag_side_px < MIN_TAG_SIDE_PX:
                    continue

                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()

                # Gate: wait for GRBL homing to finish before latching origin
                if not homing_done_event.is_set():
                    continue

                # Accumulate 20 frames at home; latch origin from median
                if cx_origin is None:
                    _origin_buf.append((cx, cy, tag_side_px))
                    if len(_origin_buf) < 20:
                        continue
                    cx_origin       = float(np.median([s[0] for s in _origin_buf]))
                    cy_origin       = float(np.median([s[1] for s in _origin_buf]))
                    tag_size_origin = float(np.median([s[2] for s in _origin_buf]))
                    scale_origin    = TAG_SIZE_MM / tag_size_origin
                    prev_cx, prev_cy = cx_origin, cy_origin
                    kf.initialize(0., 0., 0.)
                    log_event(event_log, clock, "camera",
                              f"Home origin latched ({len(_origin_buf)} frames)  "
                              f"({cx_origin:.1f},{cy_origin:.1f}) px  "
                              f"tag={tag_size_origin:.1f} px  "
                              f"scale={scale_origin:.4f} mm/px")
                    home_event.set()
                    continue

                if not home_event.is_set():
                    continue

                jump_px = POSE_JUMP_MM / scale_origin
                is_jump = (abs(cx - prev_cx) > jump_px or
                           abs(cy - prev_cy) > jump_px)

                tx_raw =  (cx - cx_origin) * scale_origin
                ty_raw = -(cy - cy_origin) * scale_origin
                tz_raw = TAG_SIZE_MM / tag_size_origin - TAG_SIZE_MM / tag_side_px

                if not is_jump:
                    kf.update([tx_raw, ty_raw, tz_raw])

                fp = kf.get_position()
                prev_cx, prev_cy = cx, cy

                # Attack mode 3: replace observed position with spoofed path
                x_obs, y_obs, z_obs, spoof_flag = apply_spoof_if_active(
                    shared_state, clock, fp[0], fp[1], fp[2]
                )

                if not motion_done_event.is_set():
                    cam_records.append([
                        t_now, cx, cy,
                        fp[0], fp[1], fp[2],
                        tag_side_px,
                        tx_raw, ty_raw, tz_raw,
                        x_obs, y_obs, z_obs, float(spoof_flag),
                    ])

                cv2.polylines(frame_disp, [pts.astype(int)], True, (0, 255, 0), 2)
                for pt in pts.astype(int):
                    cv2.circle(frame_disp, tuple(pt), 5, (0, 255, 0), -1)
                cv2.circle(frame_disp, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                vx, vy, vz = kf.get_velocity()
                speed      = np.sqrt(vx**2 + vy**2 + vz**2)
                spoof_str  = " [SPOOF]" if spoof_flag else ""
                cv2.putText(
                    frame_disp,
                    f"X:{fp[0]:.1f} Y:{fp[1]:.1f} Z:{fp[2]:.1f} mm"
                    f" | V:{speed:.1f} mm/s{spoof_str}",
                    (int(cx) - 200, int(cy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
                )

        status = "ZEROED (Kalman 3D)" if cx_origin else "awaiting home"
        cv2.putText(frame_disp,
                    f"t={t_now:.2f}s  tags={n_tags}  [{status}]  atk={ATTACK_MODE}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.imshow("Experiment Collection — VarSpeed", frame_disp)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# GRBL THREAD
# 8-column records:
#   t, x_true, y_true, z_true, x_reported, y_reported, z_reported, spoof_active
#   (x_true = EMA-smoothed; x_reported = raw value from status report)
# ══════════════════════════════════════════════════════════════════════════════

def grbl_worker(grbl_records, clock, stop_event, home_event,
                motion_done_event, homing_done_event, shared_state,
                event_log, command_log):
    """Home machine, run trajectory, poll and record positions."""
    ema_pos  = [None, None, None]
    interval = 1.0 / TARGET_FPS

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1, write_timeout=None)
    except Exception as e:
        log_event(event_log, clock, "grbl", f"Cannot open {PORT}: {e}")
        return

    try:
        time.sleep(2)
        ser.write(b"\r\n\r\n"); time.sleep(2); ser.reset_input_buffer()
        ser.write(b"$X\n");     time.sleep(1)
        ser.write(b"$10=0\n"); time.sleep(0.5)
        ser.write(b"$13=0\n"); time.sleep(0.5)   # Force mm in status reports
        ser.write(b"$H\n");     time.sleep(5)
        ser.write(b"G21 G90 G92 X0 Y0 Z0\n"); time.sleep(1)
        homing_done_event.set()
        log_event(event_log, clock, "grbl", "Homing complete — G92 origin set")

        log_event(event_log, clock, "grbl", "Waiting for camera home latch...")
        if not home_event.wait(timeout=30.0):
            log_event(event_log, clock, "grbl", "WARNING: tag not detected within 30 s")

        move_sequence = build_motion_program(
            TRAJECTORY_MODE, SIDE, FEED, Z_SAFE, attack_mode=ATTACK_MODE
        )
        log_event(event_log, clock, "grbl",
                  f"Starting {TRAJECTORY_MODE} trajectory  "
                  f"SIDE={SIDE} mm  FEED={FEED} mm/min  attack={ATTACK_MODE}")

        motion_start_logged = False
        MODE_CMDS = {"G61", "G64"}   # mode-setting commands don't trigger motion

        for cmd, label in move_sequence:
            log_command(command_log, clock, cmd, label)
            ser.write((cmd + "\n").encode())

            ser.reset_input_buffer()
            time.sleep(0.15)  # let GRBL begin executing before first poll

            deadline   = time.time() + 90.0
            idle_count = 0

            while time.time() < deadline and not stop_event.is_set():
                poll_start = time.time()
                ser.write(b"?\n"); time.sleep(0.02)

                raw = ""
                while ser.in_waiting:
                    try:
                        raw += ser.readline().decode(errors="ignore").strip()
                    except Exception:
                        pass

                if raw:
                    m = re.search(r"WPos:([\d.\-]+),([\d.\-]+),([\d.\-]+)", raw) or \
                        re.search(r"MPos:([\d.\-]+),([\d.\-]+),([\d.\-]+)", raw)
                    if m:
                        rp    = [float(m.group(i)) for i in (1, 2, 3)]
                        t_now = clock.now()

                        if GRBL_POSITION_UNITS == "in":
                            rp = [v * 25.4 for v in rp]

                        for ax in range(3):
                            ema_pos[ax] = (rp[ax] if ema_pos[ax] is None
                                           else 0.8 * rp[ax] + 0.2 * ema_pos[ax])

                        spoof_flag = 1.0 if shared_state.is_spoof_active() else 0.0
                        grbl_records.append([
                            t_now,
                            ema_pos[0], ema_pos[1], ema_pos[2],
                            rp[0],      rp[1],      rp[2],
                            spoof_flag,
                        ])

                        if not motion_start_logged and cmd not in MODE_CMDS:
                            shared_state.set_motion_start(t_now)
                            motion_start_logged = True
                            log_event(event_log, clock, "grbl",
                                      f"Motion started: {label}")

                    if "Run" in raw or "Jog" in raw:
                        idle_count = 0
                    elif "Idle" in raw:
                        idle_count += 1
                        if idle_count >= 3:
                            break

                sleep_for = interval - (time.time() - poll_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        log_event(event_log, clock, "grbl", "Trajectory complete")

    except Exception as e:
        log_event(event_log, clock, "grbl", f"Error: {e}")
        traceback.print_exc()
    finally:
        motion_done_event.set()
        stop_event.set()
        try:
            ser.write(b"$H\n"); time.sleep(5)
            ser.close()
        except Exception:
            pass
        log_event(event_log, clock, "grbl", "Done.")


# ══════════════════════════════════════════════════════════════════════════════
# DAQ THREAD
# ══════════════════════════════════════════════════════════════════════════════

def daq_worker(daq_bursts, accel_live, ae_live, clock, stop_event, home_event):
    """
    Acquires sensor data in DAQ_BURST_DURATION-second chunks using two
    continuous NI-DAQmx tasks (one per sensor module).

    Waits for home_event before starting hardware tasks to prevent ring-buffer
    overflow (only ~0.8 s deep) during the homing phase.
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    except ImportError:
        print("[daq] nidaqmx not installed — skipping sensor acquisition")
        return

    accel_task = ae_task = None
    try:
        burst_accel = int(DAQ_BURST_DURATION * ACCEL_SAMPLE_RATE)
        burst_ae    = int(DAQ_BURST_DURATION * AE_SAMPLE_RATE)

        accel_task = nidaqmx.Task()
        accel_task.ai_channels.add_ai_voltage_chan(
            f"{DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}",
            min_val=-ACCEL_RANGE, max_val=ACCEL_RANGE,
            terminal_config=TerminalConfiguration.DIFF,
        )
        accel_task.timing.cfg_samp_clk_timing(
            rate=ACCEL_SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=burst_accel * 8,
        )

        ae_task = nidaqmx.Task()
        ae_task.ai_channels.add_ai_voltage_chan(
            f"{DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}",
            min_val=-AE_RANGE, max_val=AE_RANGE,
            terminal_config=TerminalConfiguration.DIFF,
        )
        ae_task.timing.cfg_samp_clk_timing(
            rate=AE_SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=burst_ae * 8,
        )

        if not home_event.wait(timeout=30.0):
            print("[daq] WARNING: home latch timed out after 30 s — starting anyway")
        if stop_event.is_set():
            print("[daq] stop_event set before homing completed — exiting")
            return

        accel_task.start()
        ae_task.start()
        print("[daq] Sensors started (continuous mode)")

        while not stop_event.is_set():
            t_burst = clock.now()

            accel_chunk = np.array(
                accel_task.read(
                    number_of_samples_per_channel=burst_accel,
                    timeout=DAQ_BURST_DURATION * 5,
                )
            )
            ae_chunk = np.array(
                ae_task.read(
                    number_of_samples_per_channel=burst_ae,
                    timeout=DAQ_BURST_DURATION * 5,
                )
            )

            daq_bursts.append((t_burst, accel_chunk, ae_chunk))

            for i in range(0, len(accel_chunk), LIVE_DOWNSAMPLE_ACCEL):
                accel_live.append((t_burst + i / ACCEL_SAMPLE_RATE, accel_chunk[i]))
            for i in range(0, len(ae_chunk), LIVE_DOWNSAMPLE_AE):
                ae_live.append((t_burst + i / AE_SAMPLE_RATE, ae_chunk[i]))

    except Exception as e:
        print(f"[daq] Error: {e}")
        traceback.print_exc()
    finally:
        for t in (accel_task, ae_task):
            if t is not None:
                try: t.stop(); t.close()
                except Exception: pass
        print("[daq] Tasks closed")


# ══════════════════════════════════════════════════════════════════════════════
# ESP32 THREAD  (STEP/DIR + current sensor)
# 10-column records:
#   exp_time_s, esp32_time_s, step_count, pos_mm,
#   pulse_rate, dir_state, voltage_A, voltage_B, current_A, current_B
# ══════════════════════════════════════════════════════════════════════════════

def esp32_worker(esp32_records, clock, stop_event):
    """
    Read 8-field CSV lines from the ESP32 motor monitor.
    Expected format (one line per sample):
      time_ms, stepCount, pulseRate, dirState, voltageA, voltageB, currentA, currentB

    Records are stored with an experiment-clock timestamp prepended so they can
    be aligned with the GRBL and camera streams in post-processing.
    """
    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
    except Exception as e:
        print(f"[esp32] Cannot open {ESP32_PORT}: {e} — skipping ESP32 acquisition")
        return

    print(f"[esp32] Connected on {ESP32_PORT} @ {ESP32_BAUD}")
    try:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 8:
                continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            t_ms, step_count, pulse_rate, dir_state, va, vb, ia, ib = vals
            pos_mm = step_count / STEPS_PER_MM
            esp32_records.append([
                clock.now(),        # experiment-relative time (s)
                t_ms / 1000.0,      # ESP32 millis() converted to seconds
                step_count,
                pos_mm,
                pulse_rate,
                dir_state,
                va, vb, ia, ib,
            ])
    except Exception as e:
        print(f"[esp32] Error: {e}")
        traceback.print_exc()
    finally:
        ser.close()
        print(f"[esp32] Done. {len(esp32_records)} records captured.")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DISPLAY  (main thread)
# ══════════════════════════════════════════════════════════════════════════════

def run_live_display(accel_live, ae_live, daq_bursts, clock, stop_event):
    """Scrolling live sensor plot — must run on the main thread."""
    plt.ion()
    fig_live, (ax_a, ax_ae) = plt.subplots(
        2, 1, figsize=(13, 6), dpi=PLOT_DPI,
        constrained_layout=True,
    )
    fig_live.suptitle(
        "Live Sensor Data — experiment running  (close window or Ctrl+C to stop)",
        fontsize=11,
    )

    (line_a,)  = ax_a.plot([], [], color="steelblue", lw=0.8)
    (line_ae,) = ax_ae.plot([], [], color="crimson",  lw=0.8)

    for ax, ylim, title in [
        (ax_a,  ACCEL_RANGE, f"Accelerometer  {DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}"),
        (ax_ae, AE_RANGE,    f"AE Sensor  {DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}"),
    ]:
        ax.set_ylim(-ylim * 1.05, ylim * 1.05)
        ax.set_ylabel("Amplitude (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    ax_ae.set_xlabel("Time (s)")

    fig_live.canvas.draw()
    fig_live.canvas.flush_events()

    plot_interval = 1.0 / LIVE_UPDATE_HZ

    while not stop_event.is_set():
        t0    = time.time()
        t_now = clock.now()
        t_min = t_now - LIVE_WINDOW_S

        if accel_live:
            try:
                arr  = np.array(list(accel_live))
                mask = arr[:, 0] >= t_min
                if mask.sum() > 1:
                    line_a.set_data(arr[mask, 0], arr[mask, 1])
                    ax_a.set_xlim(t_min, t_now)
                    rms = np.sqrt(np.mean(arr[mask, 1] ** 2))
                    ax_a.set_title(
                        f"Accelerometer  {DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}"
                        f"   RMS = {rms:.4f} V"
                    )
            except Exception:
                pass

        if ae_live:
            try:
                arr  = np.array(list(ae_live))
                mask = arr[:, 0] >= t_min
                if mask.sum() > 1:
                    line_ae.set_data(arr[mask, 0], arr[mask, 1])
                    ax_ae.set_xlim(t_min, t_now)
                    rms = np.sqrt(np.mean(arr[mask, 1] ** 2))
                    ax_ae.set_title(
                        f"AE Sensor  {DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}"
                        f"   RMS = {rms:.4f} V"
                    )
            except Exception:
                pass

        fig_live.suptitle(
            f"Live Sensor Data   t = {t_now:.1f} s   "
            f"bursts = {len(daq_bursts)}   "
            f"(close window or Ctrl+C to stop)",
            fontsize=11,
        )

        if not plt.fignum_exists(fig_live.number):
            stop_event.set()
            break

        elapsed   = time.time() - t0
        sleep_for = plot_interval - elapsed
        plt.pause(max(0.001, sleep_for))

    plt.ioff()
    if plt.fignum_exists(fig_live.number):
        plt.close(fig_live)


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_sensor_arrays(daq_bursts):
    """Concatenate burst list into full waveforms with experiment-relative time axes."""
    a_t, a_d, e_t, e_d = [], [], [], []
    for t_start, accel_chunk, ae_chunk in daq_bursts:
        n_a = len(accel_chunk)
        n_e = len(ae_chunk)
        a_t.append(t_start + np.arange(n_a) / ACCEL_SAMPLE_RATE)
        e_t.append(t_start + np.arange(n_e) / AE_SAMPLE_RATE)
        a_d.append(accel_chunk)
        e_d.append(ae_chunk)
    return (np.concatenate(a_t), np.concatenate(a_d),
            np.concatenate(e_t), np.concatenate(e_d))


# ══════════════════════════════════════════════════════════════════════════════
# RTS SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

def _build_FQ(dt, process_noise=None):
    """State transition matrix F and process noise Q for the constant-velocity model."""
    if process_noise is None:
        process_noise = KALMAN_PROCESS_NOISE
    F = np.array([
        [1, 0, 0, dt, 0,  0 ],
        [0, 1, 0, 0,  dt, 0 ],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0 ],
        [0, 0, 0, 0,  1,  0 ],
        [0, 0, 0, 0,  0,  1 ],
    ])
    q = process_noise ** 2
    Q = q * np.array([
        [dt**4/4, 0,       0,       dt**3/2, 0,       0      ],
        [0,       dt**4/4, 0,       0,       dt**3/2, 0      ],
        [0,       0,       dt**4/4, 0,       0,       dt**3/2],
        [dt**3/2, 0,       0,       dt**2,   0,       0      ],
        [0,       dt**3/2, 0,       0,       dt**2,   0      ],
        [0,       0,       dt**3/2, 0,       0,       dt**2  ],
    ])
    return F, Q


def rts_smooth(cam_t, tx_raw, ty_raw, tz_raw):
    """
    Forward Kalman + Rauch-Tung-Striebel backward smoother on saved raw
    measurements.  Removes causality lag.  Returns smoothed (tx, ty, tz).
    """
    n = len(cam_t)
    if n < 3:
        return tx_raw.copy(), ty_raw.copy(), tz_raw.copy()

    H   = np.eye(3, 6)
    rxy = KALMAN_MEASUREMENT_NOISE ** 2
    rz  = (KALMAN_MEASUREMENT_NOISE * 4.0) ** 2
    R   = np.diag([rxy, rxy, rz])
    I6  = np.eye(6)

    xs  = np.zeros((n, 6));   Ps  = np.zeros((n, 6, 6))
    xpr = np.zeros((n, 6));   Ppr = np.zeros((n, 6, 6))

    x = np.array([tx_raw[0], ty_raw[0], tz_raw[0], 0., 0., 0.])
    P = np.diag([0.1, 0.1, 0.1, 100., 100., 100.])

    for k in range(n):
        dt        = float(np.clip(cam_t[k] - cam_t[k-1], 0.005, 0.2)) if k > 0 else 1.0/80.0
        F, Q      = _build_FQ(dt, RTS_PROCESS_NOISE)
        xpr[k]    = F @ x
        Ppr[k]    = F @ P @ F.T + Q
        z         = np.array([tx_raw[k], ty_raw[k], tz_raw[k]])
        K         = Ppr[k] @ H.T @ np.linalg.inv(H @ Ppr[k] @ H.T + R)
        IKH       = I6 - K @ H
        x         = xpr[k] + K @ (z - H @ xpr[k])
        P         = IKH @ Ppr[k] @ IKH.T + K @ R @ K.T
        xs[k]     = x;   Ps[k] = P

    xs_s = xs.copy();   Ps_s = Ps.copy()
    for k in range(n - 2, -1, -1):
        dt        = float(np.clip(cam_t[k+1] - cam_t[k], 0.005, 0.2))
        F, _      = _build_FQ(dt, RTS_PROCESS_NOISE)
        G         = Ps[k] @ F.T @ np.linalg.inv(Ppr[k+1])
        xs_s[k]   = xs[k]  + G @ (xs_s[k+1]  - xpr[k+1])
        Ps_s[k]   = Ps[k]  + G @ (Ps_s[k+1]  - Ppr[k+1]) @ G.T

    return xs_s[:, 0], xs_s[:, 1], xs_s[:, 2]


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def align_camera_to_grbl(cam_t, cam_tx, cam_ty, cam_tz, cam_tag_px,
                          grbl_t, grbl_x, grbl_y, grbl_z):
    """Quadratic least-squares regression: camera measurements → GRBL mm."""
    vp = (np.isfinite(cam_tx) & np.isfinite(cam_ty) &
          np.isfinite(cam_tz) & np.isfinite(cam_tag_px) & (cam_tag_px > 0))
    if vp.sum() < 20 or len(grbl_t) < 2:
        return cam_tx, cam_ty, cam_tz, np.full_like(cam_tx, np.nan)

    cam_tx_v = cam_tx[vp]
    cam_ty_v = cam_ty[vp]
    cam_tz_v = cam_tz[vp]
    cam_t_v  = cam_t[vp]

    grbl_x_v = np.interp(cam_t_v, grbl_t, grbl_x)
    grbl_y_v = np.interp(cam_t_v, grbl_t, grbl_y)
    grbl_z_v = np.interp(cam_t_v, grbl_t, grbl_z)

    valid = (cam_t_v >= grbl_t[0]) & (cam_t_v <= grbl_t[-1])
    if valid.sum() < 10:
        return cam_tx, cam_ty, cam_tz, np.full_like(cam_tx, np.nan)

    grbl_x_v = grbl_x_v[valid]
    grbl_y_v = grbl_y_v[valid]
    grbl_z_v = grbl_z_v[valid]
    cam_tx_v = cam_tx_v[valid]
    cam_ty_v = cam_ty_v[valid]
    cam_tz_v = cam_tz_v[valid]

    # 9-feature quadratic matrix: linear + squared + cross-axis coupling
    A = np.column_stack([
        cam_tx_v,
        cam_ty_v,
        cam_tz_v,
        cam_tx_v ** 2,
        cam_ty_v ** 2,
        cam_tx_v * cam_ty_v,
        cam_tx_v * cam_tz_v,
        cam_ty_v * cam_tz_v,
        np.ones_like(cam_tx_v),
    ])

    c_x, _, _, _ = np.linalg.lstsq(A, grbl_x_v, rcond=None)
    tx_corrected = A @ c_x

    c_y, _, _, _ = np.linalg.lstsq(A, grbl_y_v, rcond=None)
    ty_corrected = A @ c_y

    c_z, _, _, _ = np.linalg.lstsq(A, grbl_z_v, rcond=None)
    tz_corrected = A @ c_z

    vp_idx        = np.where(vp)[0]
    tx_out        = np.full_like(cam_tx, np.nan)
    ty_out        = np.full_like(cam_ty, np.nan)
    tz_out        = np.full_like(cam_tz, np.nan)
    residuals_out = np.full_like(cam_tx, np.nan)
    tx_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tx_corrected)
    ty_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], ty_corrected)
    tz_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tz_corrected)
    residuals_out[vp_idx[valid]] = np.sqrt(
        (tx_corrected - grbl_x_v)**2 +
        (ty_corrected - grbl_y_v)**2 +
        (tz_corrected - grbl_z_v)**2
    )

    return tx_out, ty_out, tz_out, residuals_out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: TRACKING (3×3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tracking(cam_records, grbl_records, stem):
    if not cam_records or not grbl_records:
        print("[plot] Insufficient tracking data"); return None

    cam  = np.array(cam_records)
    grbl = np.array(grbl_records)

    # 14-col cam unpack
    (cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px,
     cam_tx_raw, cam_ty_raw, cam_tz_raw,
     cam_x_obs, cam_y_obs, cam_z_obs, cam_spoof) = cam.T

    # 8-col grbl: columns 0-3 = t, x_true, y_true, z_true
    grbl_t = grbl[:, 0]
    grbl_x = grbl[:, 1]
    grbl_y = grbl[:, 2]
    grbl_z = grbl[:, 3]

    # Discard camera frames recorded after CNC motion ended
    cam_mask = cam_t <= grbl_t[-1]
    (cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px,
     cam_tx_raw, cam_ty_raw, cam_tz_raw,
     cam_x_obs, cam_y_obs, cam_z_obs, cam_spoof) = (
        v[cam_mask] for v in (
            cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px,
            cam_tx_raw, cam_ty_raw, cam_tz_raw,
            cam_x_obs, cam_y_obs, cam_z_obs, cam_spoof,
        )
    )

    # RTS smoother on raw measurements
    cam_tx, cam_ty, cam_tz = rts_smooth(cam_t, cam_tx_raw, cam_ty_raw, cam_tz_raw)

    tx_a, ty_a, tz_a, residuals = align_camera_to_grbl(
        cam_t, cam_tx, cam_ty, cam_tz, cam_px,
        grbl_t, grbl_x, grbl_y, grbl_z,
    )

    fig = plt.figure(figsize=(16, 10), dpi=PLOT_DPI)
    fig.suptitle(
        f"Tracking — {stem}  [traj={TRAJECTORY_MODE}  attack={ATTACK_MODE}]",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    ref_pts, _ = _build_xy_waypoints(TRAJECTORY_MODE, float(SIDE))
    ref_x, ref_y = ref_pts[:, 0], ref_pts[:, 1]

    # ── Row 0: paths ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ref_x, ref_y, "k--", lw=1.5, alpha=0.7, label="Planned")
    ax.plot(grbl_x, grbl_y, "b.-", lw=1.5, ms=2, label="Measured")
    ax.plot(grbl_x[0],  grbl_y[0],  "go", ms=8, label="Start")
    ax.plot(grbl_x[-1], grbl_y[-1], "ro", ms=8, label="End")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="GRBL Toolpath")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    pub_fig(ax, fig)

    ax = fig.add_subplot(gs[0, 1])
    valid = np.isfinite(tx_a) & np.isfinite(ty_a)
    ax.plot(ref_x, ref_y, "k--", lw=1.5, alpha=0.7, label="Planned")
    if valid.sum() > 10:
        ax.plot(tx_a[valid], ty_a[valid], "m.-", lw=1.5, ms=2, label="Camera")
        ax.set_aspect("equal")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="Camera Aligned Path")
    ax.legend(fontsize=8); pub_fig(ax, fig)

    ax = fig.add_subplot(gs[0, 2])
    if len(grbl_x) > 1:
        x_rng = grbl_x.max() - grbl_x.min() + 1e-9
        y_rng = grbl_y.max() - grbl_y.min() + 1e-9
        ax.plot([0, 1, 1, 0, 0], [1, 1, 0, 0, 1], "k--", lw=1.5, alpha=0.7, label="Planned")
        ax.plot((grbl_x - grbl_x.min()) / x_rng,
                (grbl_y - grbl_y.min()) / y_rng, "b-", lw=1.5, label="GRBL")
        if valid.sum() > 1:
            ax.plot((tx_a[valid] - grbl_x.min()) / x_rng,
                    (ty_a[valid] - grbl_y.min()) / y_rng,
                    "m-", lw=1.5, label="Camera")
    ax.set(xlabel="Norm X", ylabel="Norm Y", title="Overlay (normalised)")
    ax.legend(fontsize=8); pub_fig(ax, fig)

    # ── Row 1: X / Y / Z vs time ──────────────────────────────────────────────
    for col_idx, (col_a, col_g, ylabel, title) in enumerate([
        (tx_a, grbl_x, "X (mm)", "X Position"),
        (ty_a, grbl_y, "Y (mm)", "Y Position"),
        (tz_a, grbl_z, "Z (mm)", "Z Position"),
    ]):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.plot(grbl_t, col_g, "b-", lw=1.5, label="GRBL")
        vt = np.isfinite(col_a)
        if vt.sum():
            ax.plot(cam_t[vt], col_a[vt], "k--", lw=1, alpha=0.7, label="Camera")
        ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
        ax.legend(fontsize=8); pub_fig(ax, fig)

    # ── Row 2: residual / raw scatter / error bar chart ───────────────────────
    ax = fig.add_subplot(gs[2, 0])
    vt = np.isfinite(residuals)
    if vt.sum():
        ax.plot(cam_t[vt], residuals[vt], "r-", lw=1.5)
    ax.set(xlabel="Time (s)", ylabel="Error (mm)", title="3D Residual")
    pub_fig(ax, fig)

    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(cam_tx, cam_ty, s=5, alpha=0.6, c=cam_t, cmap="plasma",
               rasterized=True)
    ax.set_aspect("equal")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="Camera Raw XY (camera space)")
    pub_fig(ax, fig)

    ax = fig.add_subplot(gs[2, 2])
    WARMUP_S   = 10.0
    err_labels = ["X", "Y", "Z"]
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    avg_vals, max_vals, rms_vals = [], [], []
    for col_cam, col_grbl in zip([tx_a, ty_a, tz_a], [grbl_x, grbl_y, grbl_z]):
        vt = np.isfinite(col_cam) & (cam_t > WARMUP_S)
        if vt.sum() > 1:
            err = np.abs(col_cam[vt] - np.interp(cam_t[vt], grbl_t, col_grbl))
            avg_vals.append(err.mean())
            max_vals.append(err.max())
            rms_vals.append(np.sqrt((err**2).mean()))
        else:
            avg_vals.append(np.nan); max_vals.append(np.nan); rms_vals.append(np.nan)

    print("\nPosition Error (camera vs GRBL):")
    for i, lbl in enumerate(err_labels):
        print(f"  {lbl}  Avg={avg_vals[i]:.2f}  Max={max_vals[i]:.2f}  RMS={rms_vals[i]:.2f}  (mm)")

    bx = np.arange(3); bw = 0.25
    for j, (lbl, a, m, r) in enumerate(zip(err_labels, avg_vals, max_vals, rms_vals)):
        ax.bar(bx + j * bw, [a, m, r], bw, label=lbl, color=colors[j])
    ax.set_xticks(bx + bw); ax.set_xticklabels(["Avg", "Max", "RMS"])
    ax.set(ylabel="Error (mm)", title="Error Statistics")
    ax.legend(fontsize=8); pub_fig(ax, fig)

    save_path = os.path.join(OUTPUT_DIR, f"{stem}_tracking.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[plot] Saved: {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 2 & 3: SENSOR ANALYSIS (2×2)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensor(data, t_vec, sample_rate, label, color, stem, suffix):
    """2×2 sensor analysis figure with experiment-aligned time axis."""
    N = len(data)
    if N == 0:
        print(f"[plot] No {label} data"); return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PLOT_DPI, constrained_layout=True)
    fig.suptitle(
        f"{label}  —  {N:,} samples @ {sample_rate:.0f} Hz"
        f"    t = [{t_vec[0]:.1f} … {t_vec[-1]:.1f}] s",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0, 0]
    DISP_MAX = 5000
    if N > DISP_MAX:
        idx = np.linspace(0, N - 1, DISP_MAX, dtype=int)
        t_disp, d_disp = t_vec[idx], data[idx]
    else:
        t_disp, d_disp = t_vec, data
    rms = np.sqrt(np.mean(data**2))
    ax.plot(t_disp, d_disp, color=color, lw=0.6, rasterized=True)
    ax.axhline( rms, color="k", ls="--", lw=0.9, label=f"RMS = {rms:.4f} V")
    ax.axhline(-rms, color="k", ls="--", lw=0.9)
    ax.set(xlabel="Time (s)", ylabel="Amplitude (V)", title="Time Domain")
    ax.legend(fontsize=8); pub_fig(ax, fig)

    ax = axes[0, 1]
    nperseg = min(512, N // 8)
    f_s, t_s, Sxx = signal.spectrogram(
        data, fs=sample_rate,
        nperseg=nperseg, noverlap=nperseg // 2, window="hann",
    )
    t_s_exp = t_s + t_vec[0]
    img = ax.pcolormesh(t_s_exp, f_s, 10 * np.log10(Sxx + 1e-12),
                        shading="gouraud", cmap="viridis", rasterized=True)
    plt.colorbar(img, ax=ax, label="Power (dB)")
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Spectrogram")
    pub_fig(ax, fig)

    ax = axes[1, 0]
    data_w  = (data - data.mean()) * np.hanning(N)
    fft_mag = np.abs(np.fft.rfft(data_w)) * (2.0 / N)
    freqs   = np.fft.rfftfreq(N, 1.0 / sample_rate)
    FREQ_MAX_PTS = 8192
    if len(freqs) > FREQ_MAX_PTS:
        idx_f = np.linspace(0, len(freqs) - 1, FREQ_MAX_PTS, dtype=int)
        freqs_d, fft_d = freqs[idx_f], fft_mag[idx_f]
    else:
        freqs_d, fft_d = freqs, fft_mag
    ax.semilogy(freqs_d, np.maximum(fft_d, 1e-12), color=color, lw=0.7,
                rasterized=True)
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (V)", title="FFT Spectrum")
    ax.set_xlim(0, sample_rate / 2)
    pub_fig(ax, fig)

    ax = axes[1, 1]
    nperseg_w = min(1024, N // 4)
    f_w, psd  = signal.welch(data, fs=sample_rate,
                              nperseg=nperseg_w, window="hann")
    ax.semilogy(f_w, np.maximum(psd, 1e-20), color=color, lw=0.8,
                rasterized=True)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
           title="Power Spectral Density (Welch)")
    ax.set_xlim(0, sample_rate / 2)
    pub_fig(ax, fig)

    save_path = os.path.join(OUTPUT_DIR, f"{stem}_{suffix}.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[plot] Saved: {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT: NPZ ARCHIVE + CSVs
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(stem, cam_records, grbl_records, event_log, command_log,
                 daq_bursts, esp32_records):
    """
    Write all post-run data.
    Primary output: compressed NPZ archive with all streams and metadata JSON.
    Supplementary: cam/grbl/event/command CSVs; sensor CSVs only if SAVE_SENSOR_CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if cam_records:
        path = os.path.join(OUTPUT_DIR, f"{stem}_cam.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "cx_px", "cy_px",
                        "tx_mm", "ty_mm", "tz_mm", "tag_side_px",
                        "tx_raw_mm", "ty_raw_mm", "tz_raw_mm",
                        "x_obs_mm", "y_obs_mm", "z_obs_mm", "spoof_active"])
            w.writerows(cam_records)
        print(f"[data] {path}")

    if grbl_records:
        path = os.path.join(OUTPUT_DIR, f"{stem}_grbl.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s",
                        "x_true_mm", "y_true_mm", "z_true_mm",
                        "x_rep_mm",  "y_rep_mm",  "z_rep_mm",
                        "spoof_active"])
            w.writerows(grbl_records)
        print(f"[data] {path}")

    if event_log:
        path = os.path.join(OUTPUT_DIR, f"{stem}_events.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "tag", "message"])
            w.writerows(event_log)
        print(f"[data] {path}")

    if command_log:
        path = os.path.join(OUTPUT_DIR, f"{stem}_commands.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "command", "label"])
            w.writerows(command_log)
        print(f"[data] {path}")

    if esp32_records:
        path = os.path.join(OUTPUT_DIR, f"{stem}_esp32.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exp_time_s", "esp32_time_s",
                        "step_count", "pos_mm", "pulse_rate", "dir_state",
                        "voltage_A", "voltage_B", "current_A", "current_B"])
            w.writerows(esp32_records)
        print(f"[data] {path}")

    # ── NPZ archive ───────────────────────────────────────────────────────────
    npz_path = os.path.join(OUTPUT_DIR, f"{stem}_archive.npz")
    arrays   = {}

    if cam_records:
        arrays["cam"] = np.array(cam_records, dtype=np.float64)
    if grbl_records:
        arrays["grbl"] = np.array(grbl_records, dtype=np.float64)
    if event_log:
        arrays["event_t"]   = np.array([r[0] for r in event_log])
        arrays["event_tag"] = np.array([r[1] for r in event_log], dtype=object)
        arrays["event_msg"] = np.array([r[2] for r in event_log], dtype=object)
    if command_log:
        arrays["cmd_t"]     = np.array([r[0] for r in command_log])
        arrays["cmd_cmd"]   = np.array([r[1] for r in command_log], dtype=object)
        arrays["cmd_label"] = np.array([r[2] for r in command_log], dtype=object)

    if esp32_records:
        arrays["esp32"] = np.array(esp32_records, dtype=np.float64)

    if daq_bursts:
        accel_t, accel_data, ae_t, ae_data = build_sensor_arrays(daq_bursts)
        arrays["accel_t"]    = accel_t
        arrays["accel_data"] = accel_data
        arrays["ae_t"]       = ae_t
        arrays["ae_data"]    = ae_data

        if SAVE_SENSOR_CSV:
            path = os.path.join(OUTPUT_DIR, f"{stem}_accel.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "accel_V"])
                for t_s, v in zip(accel_t, accel_data):
                    w.writerow([f"{t_s:.6f}", f"{v:.6f}"])
            print(f"[data] {path}")

            path = os.path.join(OUTPUT_DIR, f"{stem}_ae.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "ae_V"])
                for t_s, v in zip(ae_t, ae_data):
                    w.writerow([f"{t_s:.6f}", f"{v:.6f}"])
            print(f"[data] {path}")

    metadata = {
        "run_label":          RUN_LABEL,
        "trajectory_mode":    TRAJECTORY_MODE,
        "attack_mode":        ATTACK_MODE,
        "varspeed_step_mm":   VARSPEED_STEP_MM,
        "varspeed_profile":   VARSPEED_PROFILE,
        "side_mm":            SIDE,
        "feed_mm_min":        FEED,
        "z_safe_mm":          Z_SAFE,
        "tag_size_mm":        TAG_SIZE_MM,
        "kalman_proc":        KALMAN_PROCESS_NOISE,
        "kalman_meas":        KALMAN_MEASUREMENT_NOISE,
        "rts_proc":           RTS_PROCESS_NOISE,
        "accel_sr":           ACCEL_SAMPLE_RATE,
        "ae_sr":              AE_SAMPLE_RATE,
        "stem":               stem,
    }
    arrays["metadata_json"] = np.array([json.dumps(metadata)])
    np.savez_compressed(npz_path, **arrays)
    print(f"[data] NPZ archive: {npz_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    profile_str = "  ".join(f"F{f}×{n}" for f, n in VARSPEED_PROFILE)
    print("\n" + "=" * 70)
    print(f"  Experiment_CollectionVarSpeed   traj={TRAJECTORY_MODE}  "
          f"attack={ATTACK_MODE}  label={RUN_LABEL}")
    print("=" * 70 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"{RUN_LABEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    clock  = ExperimentClock()
    shared = SharedState()

    stop_event        = threading.Event()
    home_event        = threading.Event()
    motion_done_event = threading.Event()
    homing_done_event = threading.Event()

    cam_records   = []   # 14-col
    grbl_records  = []   # 8-col
    esp32_records = []   # 10-col
    event_log     = []   # [(t, tag, msg), ...]
    command_log   = []   # [(t, cmd, label), ...]
    daq_bursts    = []   # [(t_start, accel_chunk, ae_chunk), ...]

    live_maxlen_a  = int(LIVE_WINDOW_S * ACCEL_SAMPLE_RATE / LIVE_DOWNSAMPLE_ACCEL * 2)
    live_maxlen_ae = int(LIVE_WINDOW_S * AE_SAMPLE_RATE    / LIVE_DOWNSAMPLE_AE    * 2)
    accel_live = deque(maxlen=live_maxlen_a)
    ae_live    = deque(maxlen=live_maxlen_ae)

    cam_thread = threading.Thread(
        target=camera_worker,
        args=(cam_records, clock, stop_event, home_event,
              motion_done_event, homing_done_event, shared, event_log),
        daemon=True,
    )
    grbl_thread = threading.Thread(
        target=grbl_worker,
        args=(grbl_records, clock, stop_event, home_event,
              motion_done_event, homing_done_event, shared,
              event_log, command_log),
        daemon=True,
    )
    daq_thread = threading.Thread(
        target=daq_worker,
        args=(daq_bursts, accel_live, ae_live, clock, stop_event, home_event),
        daemon=True,
    )
    esp32_thread = threading.Thread(
        target=esp32_worker,
        args=(esp32_records, clock, stop_event),
        daemon=True,
    )

    log_event(event_log, clock, "main", f"Experiment start  stem={stem}")
    print("  Config:")
    print(f"    Camera    : index={CAM_INDEX}  {CAM_W}×{CAM_H}")
    print(f"    GRBL      : {PORT} @ {BAUD}  varspeed  step={VARSPEED_STEP_MM} mm")
    print(f"    Profile   : {profile_str}  ({_vs_total_segs} segments total)")
    print(f"    G-code    : G61 + staircase  bounding box {SIDE}×{SIDE} mm")
    print(f"    Attack    : mode={ATTACK_MODE}")
    print(f"    Kalman    : proc={KALMAN_PROCESS_NOISE}  meas={KALMAN_MEASUREMENT_NOISE}  RTS={RTS_PROCESS_NOISE}")
    print(f"    Accel     : {DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}  {ACCEL_SAMPLE_RATE:.0f} Hz")
    print(f"    AE        : {DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}  {AE_SAMPLE_RATE:.0f} Hz")
    print(f"    Burst     : {DAQ_BURST_DURATION} s  Live: {LIVE_WINDOW_S} s")
    print(f"    ESP32     : {ESP32_PORT} @ {ESP32_BAUD}  steps/mm={STEPS_PER_MM}\n")

    cam_thread.start()
    grbl_thread.start()
    daq_thread.start()
    esp32_thread.start()

    try:
        run_live_display(accel_live, ae_live, daq_bursts, clock, stop_event)
    except KeyboardInterrupt:
        print("\n[main] Interrupt — shutting down...")
        stop_event.set()

    cam_thread.join(timeout=3)
    grbl_thread.join(timeout=3)
    daq_thread.join(timeout=DAQ_BURST_DURATION * 5 + 3)
    esp32_thread.join(timeout=3)

    log_event(event_log, clock, "main",
              f"Threads joined  cam={len(cam_records)}  "
              f"grbl={len(grbl_records)}  daq_bursts={len(daq_bursts)}  "
              f"esp32={len(esp32_records)}")
    print(f"\n[data] cam={len(cam_records)}  grbl={len(grbl_records)}  "
          f"daq_bursts={len(daq_bursts)}  esp32={len(esp32_records)}")

    save_outputs(stem, cam_records, grbl_records, event_log, command_log,
                 daq_bursts, esp32_records)

    print("\n[plot] Generating post-run figures...")
    figures = []

    fig1 = plot_tracking(cam_records, grbl_records, stem)
    if fig1 is not None:
        figures.append(fig1)

    if daq_bursts:
        accel_t, accel_data, ae_t, ae_data = build_sensor_arrays(daq_bursts)
        fig2 = plot_sensor(
            accel_data, accel_t, ACCEL_SAMPLE_RATE,
            "Accelerometer (NI 9215)", "steelblue", stem, "accel",
        )
        if fig2 is not None:
            figures.append(fig2)
        fig3 = plot_sensor(
            ae_data, ae_t, AE_SAMPLE_RATE,
            "AE Sensor (NI 9223)", "crimson", stem, "ae",
        )
        if fig3 is not None:
            figures.append(fig3)
    else:
        print("[plot] No DAQ data captured — sensor figures skipped")

    if figures:
        pdf_path = os.path.join(OUTPUT_DIR, f"{stem}_report.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches="tight", dpi=150)
        print(f"[plot] Saved multi-page PDF: {pdf_path}")

    log_event(event_log, clock, "main", "Done")
    plt.show(block=True)
    print("\n[done]\n")


if __name__ == "__main__":
    main()
