"""
AprilTagTrackKalman.py
AprilTag tracking with Kalman filter state estimation for robust position tracking.

Similar to AprilTagTrackWebcam.py but replaces EMA smoothing with a 2D Kalman filter
that tracks both position and velocity, providing better outlier rejection and prediction.

Usage:
  python AprilTagTrackKalman.py

Dependencies:
  pip install pyserial numpy opencv-contrib-python matplotlib
"""

import sys
import os
import serial
import time
import re
import csv
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pub_fig import pub_fig

if not hasattr(cv2, "aruco"):
    sys.exit(
        "cv2.aruco not found — install the contrib build:\n"
        "  pip uninstall opencv-python && pip install opencv-contrib-python"
    )


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
PORT        = "COM3"
BAUD        = 115200
SIDE        = 200       # mm  — square side length
FEED        = 1500      # mm/min
Z_SAFE      = 4.0       # mm  — plunge depth
TARGET_FPS  = 80        # GRBL poll rate (Hz)

CAM_INDEX   = 1         # 0 = first USB camera
CAM_W       = 1920
CAM_H       = 1080
TAG_FAMILY  = "tag36h11"
TAG_SIZE_MM = 30.0      # physical side length of printed AprilTag

# Kalman Filter Configuration
KALMAN_PROCESS_NOISE     = 5.0   # Process noise for live filter (mm/s²)
KALMAN_MEASUREMENT_NOISE = 0.3   # Measurement noise (mm)
RTS_PROCESS_NOISE        = 50.0  # Process noise for RTS smoother — higher = sharper corners

POSE_JUMP_MM    = 80.0      # Maximum allowed jump in pixel space (fallback)
MIN_TAG_SIDE_PX = 20.0      # reject detections where tag side < this many pixels


# ── 3D KALMAN FILTER ───────────────────────────────────────────────────────────
class KalmanFilter3D:
    """
    3D Kalman filter for position + velocity tracking.
    
    State vector: [x, y, z, vx, vy, vz]
    - Assumes constant velocity model with process noise
    - Measurement: [x, y, z] from camera detection
    - XY: from pixel displacement
    - Z: from AprilTag apparent size (TAG_SIZE_MM / tag_side_px)
    """
    
    def __init__(self, dt, process_noise_sigma, measurement_noise_sigma, measurement_noise_z=None):
        """
        Initialize Kalman filter.
        
        Args:
            dt: Time step (seconds)
            process_noise_sigma: Standard deviation of acceleration (mm/s²)
            measurement_noise_sigma: Standard deviation of XY position measurement (mm)
            measurement_noise_z: Standard deviation of Z measurement (mm). If None, uses measurement_noise_sigma.
        """
        self.dt = dt
        if measurement_noise_z is None:
            measurement_noise_z = measurement_noise_sigma * 1.5  # Z typically noisier
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        
        # Measurement matrix (measure position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance (models uncertainty in acceleration)
        q = process_noise_sigma ** 2
        self.Q = q * np.array([
            [dt**4/4, 0,        0,        dt**3/2, 0,        0],
            [0,       dt**4/4,  0,        0,       dt**3/2,  0],
            [0,       0,        dt**4/4,  0,       0,        dt**3/2],
            [dt**3/2, 0,        0,        dt**2,   0,        0],
            [0,       dt**3/2,  0,        0,       dt**2,    0],
            [0,       0,        dt**3/2,  0,       0,        dt**2]
        ])
        
        # Measurement noise covariance (XY and Z may have different noise)
        rxy = measurement_noise_sigma ** 2
        rz = measurement_noise_z ** 2
        self.R = np.array([
            [rxy, 0,   0],
            [0,   rxy, 0],
            [0,   0,   rz]
        ])
        
        # State covariance (uncertainty in state estimate)
        self.P = np.eye(6) * 10.0  # Initialize with high uncertainty
        
        self.is_initialized = False
    
    def initialize(self, x, y, z):
        """Initialize filter with first measurement."""
        self.state = np.array([x, y, z, 0.0, 0.0, 0.0])
        # Position is known exactly (home), velocity is completely unknown
        self.P = np.diag([0.1, 0.1, 0.1, 100.0, 100.0, 100.0])
        self.is_initialized = True

    def _rebuild(self, dt):
        """Rebuild F and Q matrices with actual dt."""
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        q = KALMAN_PROCESS_NOISE ** 2
        self.Q = q * np.array([
            [dt**4/4, 0,        0,        dt**3/2, 0,        0],
            [0,       dt**4/4,  0,        0,       dt**3/2,  0],
            [0,       0,        dt**4/4,  0,       0,        dt**3/2],
            [dt**3/2, 0,        0,        dt**2,   0,        0],
            [0,       dt**3/2,  0,        0,       dt**2,    0],
            [0,       0,        dt**3/2,  0,       0,        dt**2]
        ])

    def predict(self, dt):
        """Prediction step: advance state by one time step with actual dt."""
        dt = np.clip(dt, 0.005, 0.2)  # Sanity check: 5-200 ms
        self._rebuild(dt)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement, is_valid=True):
        """
        Update step: incorporate new measurement.
        
        Args:
            measurement: [x, y, z] position from camera
            is_valid: If False, use only prediction (measurement rejected)
        
        Returns:
            innovation_squared: Normalized squared innovation (for outlier detection)
        """
        if not is_valid:
            return np.inf  # Measurement rejected
        
        # Innovation (measurement residual)
        z = np.array([measurement[0], measurement[1], measurement[2]])
        y = z - (self.H @ self.state)  # Innovation
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y

        # Joseph form: numerically stable, keeps P symmetric positive-semidefinite
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return float(y @ np.linalg.solve(S, y))

    def get_position(self):
        """Return current position estimate [x, y, z]."""
        return self.state[:3]
    
    def get_velocity(self):
        """Return current velocity estimate [vx, vy, vz]."""
        return self.state[3:]
    
    def get_position_uncertainty(self):
        """Return position uncertainty (std dev) [σx, σy, σz]."""
        return np.sqrt([self.P[0, 0], self.P[1, 1], self.P[2, 2]])


# ── CAMERA TRACKING THREAD ─────────────────────────────────────────────────────
def camera_worker(cam_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event):
    """
    Background thread: captures frames, detects AprilTag, appends to cam_records.
    Each record: [elapsed_t, cx_px, cy_px, tx_mm, ty_mm, tag_side_px]

    Uses Kalman filter for robust position estimation.
    """
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx_origin    = None
    cy_origin    = None
    scale_origin = None
    tag_size_origin = None
    prev_cx      = None
    prev_cy      = None
    _origin_buf  = []  # Accumulate frames at home for a stable origin estimate
    
    # Kalman filter for camera measurements (3D)
    kf = KalmanFilter3D(
        dt=1.0/30.0,  # Assume ~30Hz camera update
        process_noise_sigma=KALMAN_PROCESS_NOISE,
        measurement_noise_sigma=KALMAN_MEASUREMENT_NOISE,
        measurement_noise_z=KALMAN_MEASUREMENT_NOISE * 4.0  # Z from tag-size is noisier than pixel XY
    )
    
    prev_frame_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        t_now = time.time() - start_ref[0]
        frame_time = time.time()
        dt = frame_time - prev_frame_time
        prev_frame_time = frame_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if kf.is_initialized:
            kf.predict(dt)

        corners_list, ids, _ = aruco_detector.detectMarkers(gray)
        n_tags = 0 if ids is None else len(ids)

        frame_disp = frame.copy()
        measurement_valid = False

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

                # Only latch origin after GRBL homing cycle has finished
                if not homing_done_event.is_set():
                    continue

                # Accumulate frames at home; latch origin from 20-frame median
                if cx_origin is None:
                    _origin_buf.append((cx, cy, tag_side_px))
                    if len(_origin_buf) < 20:
                        continue
                    cx_origin       = float(np.median([s[0] for s in _origin_buf]))
                    cy_origin       = float(np.median([s[1] for s in _origin_buf]))
                    tag_size_origin = float(np.median([s[2] for s in _origin_buf]))
                    scale_origin    = TAG_SIZE_MM / tag_size_origin
                    prev_cx, prev_cy = cx_origin, cy_origin
                    kf.initialize(0.0, 0.0, 0.0)
                    print(f"  [camera] Home origin latched ({len(_origin_buf)} frames)  "
                          f"({cx_origin:.1f}, {cy_origin:.1f}) px  "
                          f"tag={tag_size_origin:.1f} px  "
                          f"scale={scale_origin:.4f} mm/px")
                    home_event.set()
                    continue

                # Only record measurements after homing is confirmed
                if not home_event.is_set():
                    continue

                # Outlier rejection in pixel space
                jump_px = POSE_JUMP_MM / scale_origin
                is_jump = abs(cx - prev_cx) > jump_px or abs(cy - prev_cy) > jump_px
                
                # Convert to mm (X from horizontal pixel shift, Y from vertical pixel shift)
                tx =  (cx - cx_origin) * scale_origin
                ty = -(cy - cy_origin) * scale_origin
                # Z from tag size change: larger apparent size = closer (smaller Z)
                tz = (TAG_SIZE_MM / tag_size_origin - TAG_SIZE_MM / tag_side_px)
                
                # Only reject pixel-space jumps (false detections); update on everything else
                if not is_jump:
                    kf.update([tx, ty, tz])
                
                # Get filtered position
                filtered_pos = kf.get_position()
                tx_filtered, ty_filtered, tz_filtered = filtered_pos[0], filtered_pos[1], filtered_pos[2]
                
                prev_cx, prev_cy = cx, cy
                if not motion_done_event.is_set():
                    cam_records.append([t_now, cx, cy, tx_filtered, ty_filtered, tz_filtered, tag_side_px,
                                        tx, ty, tz])

                # Draw detection overlay
                cv2.polylines(frame_disp, [pts.astype(int)], True, (0, 255, 0), 2)
                for pt in pts.astype(int):
                    cv2.circle(frame_disp, tuple(pt), 5, (0, 255, 0), -1)
                cv2.circle(frame_disp, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                
                # Display position and velocity
                vx, vy, vz = kf.get_velocity()
                ux, uy, uz = kf.get_position_uncertainty()
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                label = f"X:{tx_filtered:.1f} Y:{ty_filtered:.1f} Z:{tz_filtered:.1f} mm | V:{speed:.1f} mm/s"
                cv2.putText(frame_disp, label,
                            (int(cx) - 200, int(cy) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw status
        status = "ZEROED (Kalman 3D)" if cx_origin is not None else "awaiting home"
        cv2.putText(
            frame_disp,
            f"t={t_now:.2f}s   tags={n_tags}   [{status}]",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
        )
        cv2.imshow("AprilTag Tracking (3D Kalman Filter)", frame_disp)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# ── GRBL SERIAL THREAD ─────────────────────────────────────────────────────────
def grbl_worker(grbl_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event):
    """Background thread: sets up GRBL, runs motion sequence, polls position."""
    ema_pos = [None, None, None]
    interval = 1.0 / TARGET_FPS

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1, write_timeout=None)
    except Exception as e:
        print(f"  [grbl] Failed to open port: {e}")
        return

    time.sleep(2)
    ser.write(b"\r\n\r\n")
    time.sleep(2)
    ser.reset_input_buffer()

    print("  [grbl] Setting up machine...")
    ser.write(b"$X\n");    time.sleep(1)
    ser.write(b"$10=0\n"); time.sleep(0.5)
    ser.write(b"$13=0\n"); time.sleep(0.5)  # Force mm in status reports
    ser.write(b"$H\n");    time.sleep(5)
    ser.write(b"G21 G90 G92 X0 Y0 Z0\n"); time.sleep(1)
    homing_done_event.set()

    print("  [grbl] Waiting for camera home latch...")
    if not home_event.wait(timeout=30.0):
        print("  [grbl] WARNING: tag not detected within 30 s")

    print(f"  [grbl] Starting square pattern ({SIDE} mm @ {FEED} mm/min)")

    move_sequence = [
        ("G61",                            "Exact stop mode"),
        (f"G1 Z-{Z_SAFE:.1f} F{FEED}",   "Plunge Z"),
        (f"G1 X{SIDE} Y0 F{FEED}",        "-> X+"),
        (f"G1 X{SIDE} Y-{SIDE} F{FEED}", "-> Corner"),
        (f"G1 X0 Y-{SIDE} F{FEED}",      "<- X-"),
        (f"G1 X0 Y0 F{FEED}",            "<- Home"),
        (f"G1 Z{Z_SAFE + 5:.1f} F{FEED}","Retract Z"),
    ]

    start_time = start_ref[0]

    for cmd, label in move_sequence:
        print(f"  [grbl] {label}")
        ser.write((cmd + "\n").encode())

        ser.reset_input_buffer()
        time.sleep(0.15)  # Let GRBL start moving before we poll

        deadline   = time.time() + 90.0
        idle_count = 0

        while time.time() < deadline and not stop_event.is_set():
            poll_start = time.time()

            ser.write(b"?\n")
            time.sleep(0.02)

            raw = ""
            while ser.in_waiting:
                try:
                    raw += ser.readline().decode(errors="ignore").strip()
                except Exception:
                    pass

            if raw:
                match = re.search(
                    r"WPos:([\d\.\-]+),([\d\.\-]+),([\d\.\-]+)", raw
                ) or re.search(
                    r"MPos:([\d\.\-]+),([\d\.\-]+),([\d\.\-]+)", raw
                )
                if match:
                    raw_pos = [float(match.group(i)) for i in (1, 2, 3)]
                    t_now = time.time() - start_time

                    for ax in range(3):
                        if ema_pos[ax] is None:
                            ema_pos[ax] = raw_pos[ax]
                        else:
                            ema_pos[ax] = (
                                0.8 * raw_pos[ax] + 0.2 * ema_pos[ax]
                            )

                    grbl_records.append([t_now, ema_pos[0], ema_pos[1], ema_pos[2]])

                if "Run" in raw or "Jog" in raw:
                    idle_count = 0
                elif "Idle" in raw:
                    idle_count += 1
                    if idle_count >= 3:
                        break

            elapsed = time.time() - poll_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    motion_done_event.set()
    ser.write(b"$H\n"); time.sleep(5)
    stop_event.set()
    ser.close()
    print("  [grbl] Done.")


# ── RTS SMOOTHER ───────────────────────────────────────────────────────────────
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
    Forward Kalman filter + Rauch-Tung-Striebel backward smoother on saved raw
    measurements.  Removes causality lag with no noise/smoothness tradeoff.
    Returns smoothed (tx, ty, tz).
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


# ── GEOMETRY & ALIGNMENT ───────────────────────────────────────────────────────
def safe_range(arr):
    """Returns max-min, avoiding zero division."""
    r = float(arr.max() - arr.min())
    return r if r > 1e-9 else 1.0


def nearest_value(query_t, data_t, data_v):
    """Map data_v onto query_t by nearest-neighbour lookup."""
    out = np.full(len(query_t), np.nan)
    if len(data_t) == 0:
        return out
    idx = np.clip(np.searchsorted(data_t, query_t), 0, len(data_t) - 1)
    out[:] = data_v[idx]
    return out


def align_camera_to_grbl(cam_t, cam_tx, cam_ty, cam_tz, cam_tag_px, grbl_t, grbl_x, grbl_y, grbl_z):
    """
    Recover CNC XYZ from filtered camera data via least-squares regression.
    Now includes Z from tag apparent size estimation.
    """
    vp = np.isfinite(cam_tx) & np.isfinite(cam_ty) & np.isfinite(cam_tz) & np.isfinite(cam_tag_px) & (cam_tag_px > 0)
    if vp.sum() < 20 or len(grbl_t) < 2:
        return cam_tx, cam_ty, cam_tz, np.full_like(cam_tx, np.nan)

    cam_tx_v = cam_tx[vp]
    cam_ty_v = cam_ty[vp]
    cam_tz_v = cam_tz[vp]
    cam_t_v = cam_t[vp]

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

    # Quadratic feature matrix: captures linear + lens distortion + cross-axis coupling
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

    # Fit X
    c_x, _, _, _ = np.linalg.lstsq(A, grbl_x_v, rcond=None)
    tx_corrected = A @ c_x

    # Fit Y
    c_y, _, _, _ = np.linalg.lstsq(A, grbl_y_v, rcond=None)
    ty_corrected = A @ c_y

    # Fit Z
    c_z, _, _, _ = np.linalg.lstsq(A, grbl_z_v, rcond=None)
    tz_corrected = A @ c_z

    # Expand back to full length
    vp_idx = np.where(vp)[0]
    tx_out = np.full_like(cam_tx, np.nan)
    ty_out = np.full_like(cam_ty, np.nan)
    tz_out = np.full_like(cam_tz, np.nan)
    residuals_out = np.full_like(cam_tx, np.nan)
    tx_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tx_corrected)
    ty_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], ty_corrected)
    tz_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tz_corrected)
    residuals_out[vp_idx[valid]] = np.sqrt((tx_corrected - grbl_x_v)**2 + (ty_corrected - grbl_y_v)**2 + (tz_corrected - grbl_z_v)**2)

    return tx_out, ty_out, tz_out, residuals_out


# ── PLOTTING ───────────────────────────────────────────────────────────────────
def plot_results(cam_records, grbl_records, csv_stem):
    """Plot and save camera & GRBL trajectories in 3D with pub_fig styling."""
    if not cam_records or not grbl_records:
        print("[plot] Insufficient data to plot")
        return

    cam = np.array(cam_records)
    grbl = np.array(grbl_records)

    cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px, cam_tx_raw, cam_ty_raw, cam_tz_raw = cam.T
    grbl_t, grbl_x, grbl_y, grbl_z = grbl.T

    # Discard camera frames recorded after CNC motion ended
    cam_mask = cam_t <= grbl_t[-1]
    (cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px,
     cam_tx_raw, cam_ty_raw, cam_tz_raw) = (
        cam_t[cam_mask], cam_cx[cam_mask], cam_cy[cam_mask],
        cam_tx[cam_mask], cam_ty[cam_mask], cam_tz[cam_mask], cam_px[cam_mask],
        cam_tx_raw[cam_mask], cam_ty_raw[cam_mask], cam_tz_raw[cam_mask]
    )

    # RTS smoother: re-run Kalman forward+backward on raw measurements
    cam_tx, cam_ty, cam_tz = rts_smooth(cam_t, cam_tx_raw, cam_ty_raw, cam_tz_raw)

    tx_aligned, ty_aligned, tz_aligned, residuals = align_camera_to_grbl(
        cam_t, cam_tx, cam_ty, cam_tz, cam_px, grbl_t, grbl_x, grbl_y, grbl_z
    )

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"AprilTag Kalman Tracking (3D)  —  {len(cam_t)} cam frames / {len(grbl_t)} GRBL pts  —  {csv_stem}", fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    # Row 0 — GRBL path
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(grbl_x, grbl_y, "b.-", lw=1.5, ms=2)
    ax.plot(grbl_x[0],  grbl_y[0],  "go", ms=8)
    ax.plot(grbl_x[-1], grbl_y[-1], "ro", ms=8)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="GRBL Toolpath")
    ax.set_aspect("equal"); ax.grid(True)
    ax.legend(["Path", "Start", "End"], fontsize=8)
    pub_fig(ax, fig)

    # Row 0 — Camera aligned path
    ax = fig.add_subplot(gs[0, 1])
    valid = np.isfinite(tx_aligned) & np.isfinite(ty_aligned)
    if valid.sum() > 10:
        ax.plot(tx_aligned[valid], ty_aligned[valid], "m.-", lw=1.5, ms=2)
        ax.set_aspect("equal")
        ax.set(xlabel="X cam (mm)", ylabel="Y cam (mm)", title="Camera Aligned Path")
    ax.grid(True)
    pub_fig(ax, fig)

    # Row 0 — Normalized overlay
    ax = fig.add_subplot(gs[0, 2])
    if len(grbl_x) > 1:
        x_min, x_max = grbl_x.min(), grbl_x.max()
        y_min, y_max = grbl_y.min(), grbl_y.max()
        gx = (grbl_x - x_min) / (x_max - x_min + 1e-6)
        gy = (grbl_y - y_min) / (y_max - y_min + 1e-6)
        ax.plot(gx, gy, "b-", lw=1.5, alpha=0.8, label="GRBL")
        if valid.sum() > 1:
            # Use GRBL scale so amplitude errors are visible, not hidden by independent normalisation
            nx = (tx_aligned[valid] - x_min) / (x_max - x_min + 1e-6)
            ny = (ty_aligned[valid] - y_min) / (y_max - y_min + 1e-6)
            ax.plot(nx, ny, "m-", lw=1.5, alpha=0.8, label="Camera")
    ax.set(xlabel="Norm X", ylabel="Norm Y", title="Path Overlay (normalised)")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    # Row 1 — X vs time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(grbl_t, grbl_x, "b-", lw=1.5, label="GRBL")
    valid_t = np.isfinite(tx_aligned)
    if valid_t.sum() > 0:
        ax.plot(cam_t[valid_t], tx_aligned[valid_t], "k--", lw=1, alpha=0.7, label="Camera")
    ax.set(xlabel="Time (s)", ylabel="X (mm)", title="X Position")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    # Row 1 — Y vs time
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(grbl_t, grbl_y, "b-", lw=1.5, label="GRBL")
    valid_t = np.isfinite(ty_aligned)
    if valid_t.sum() > 0:
        ax.plot(cam_t[valid_t], ty_aligned[valid_t], "k--", lw=1, alpha=0.7, label="Camera")
    ax.set(xlabel="Time (s)", ylabel="Y (mm)", title="Y Position")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    # Row 1 — Z vs time
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(grbl_t, grbl_z, "b-", lw=1.5, label="GRBL")
    valid_t = np.isfinite(tz_aligned)
    if valid_t.sum() > 0:
        ax.plot(cam_t[valid_t], tz_aligned[valid_t], "k--", lw=1, alpha=0.7, label="Camera")
    ax.set(xlabel="Time (s)", ylabel="Z (mm)", title="Z Position")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    # Row 2 — Error stats
    ax = fig.add_subplot(gs[2, 0])
    err_labels  = ["X", "Y", "Z"]
    err_avg     = []
    err_max     = []
    err_rms     = []
    WARMUP_S = 10.0  # skip initial filter warm-up / machine acceleration phase
    cam_aligned = [tx_aligned, ty_aligned, tz_aligned]
    for ci, grbl_col in enumerate([grbl_x, grbl_y, grbl_z]):
        cc = cam_aligned[ci]
        valid_pts = np.isfinite(cc) & (cam_t > WARMUP_S)
        if valid_pts.sum() > 1:
            ct_valid  = cam_t[valid_pts]
            grbl_interp = np.interp(ct_valid, grbl_t, grbl_col)
            err = np.abs(cc[valid_pts] - grbl_interp)
            err_avg.append(float(err.mean()))
            err_max.append(float(err.max()))
            err_rms.append(float(np.sqrt((err**2).mean())))
        else:
            err_avg.append(np.nan); err_max.append(np.nan); err_rms.append(np.nan)

    print("\nPosition Error (camera vs GRBL):")
    for i, lbl in enumerate(err_labels):
        print(f"  {lbl}  Avg={err_avg[i]:.2f}  Max={err_max[i]:.2f}  RMS={err_rms[i]:.2f}  (mm)")

    if len(residuals) > 0:
        ax.plot(cam_t, residuals, "r-", lw=1.5)
        ax.set(xlabel="Time (s)", ylabel="Error (mm)", title="Position Error (3D)")
    ax.grid(True)
    pub_fig(ax, fig)

    # Row 2 — Camera raw XY
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(cam_tx, cam_ty, s=5, alpha=0.6, c=cam_t, cmap='plasma')
    ax.set_aspect("equal")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="Camera Raw XY")
    ax.grid(True)
    pub_fig(ax, fig)

    # Row 2 — Error bar chart
    ax = fig.add_subplot(gs[2, 2])
    bar_x   = np.arange(3)           # Avg, Max, RMS
    bar_w   = 0.25
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # X=blue, Y=orange, Z=green
    metrics = [err_avg, err_max, err_rms]
    metric_labels = ["Avg", "Max", "RMS"]
    for j, (lbl, vals) in enumerate(zip(err_labels, zip(*metrics))):
        ax.bar(bar_x + j * bar_w, vals, bar_w, label=lbl, color=colors[j])
    ax.set_xticks(bar_x + bar_w)
    ax.set_xticklabels(metric_labels)
    ax.set(ylabel="Error (mm)", title="Position Error Statistics")
    ax.legend(fontsize=8); ax.grid(True, axis="y")
    pub_fig(ax, fig)

    plt.savefig("apriltag_run.png", dpi=150)
    plt.show()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Data")
    plot_path = os.path.join(data_dir, f"{csv_stem}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[plot] Saved: {plot_path}")
    plt.close()


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Get script directory and create Data folder if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    now = datetime.now()
    csv_stem = f"apriltag_kalman_{now.strftime('%Y%m%d_%H%M%S')}"

    start_ref = [time.time()]
    stop_event = threading.Event()
    home_event = threading.Event()
    motion_done_event = threading.Event()
    homing_done_event = threading.Event()

    cam_records = []
    grbl_records = []

    cam_thread = threading.Thread(target=camera_worker, args=(cam_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event), daemon=True)
    grbl_thread = threading.Thread(target=grbl_worker, args=(grbl_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event), daemon=True)

    cam_thread.start()
    grbl_thread.start()

    print("\n=================================================================")
    print("     AprilTag Tracking with 3D Kalman Filter (Position + Velocity)")
    print("=================================================================\n")
    print(f"  Config:")
    print(f"    CAM_INDEX = {CAM_INDEX}  |  RES = {CAM_W}x{CAM_H}")
    print(f"    GRBL: {PORT} @ {BAUD}")
    print(f"    Process Noise: {KALMAN_PROCESS_NOISE} mm/s²")
    print(f"    Measurement Noise XY: {KALMAN_MEASUREMENT_NOISE} mm")
    print(f"    Measurement Noise Z: {KALMAN_MEASUREMENT_NOISE*2.0} mm\n")

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        print("\n[main] Interrupt received, shutting down...")
        stop_event.set()

    cam_thread.join(timeout=2)
    grbl_thread.join(timeout=2)

    print(f"\n[data] Captured {len(cam_records)} camera records, {len(grbl_records)} GRBL records")

    if cam_records:
        cam_file = os.path.join(data_dir, f"{csv_stem}_cam.csv")
        with open(cam_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "cx_px", "cy_px", "tx_mm", "ty_mm", "tz_mm", "tag_side_px",
                             "tx_raw_mm", "ty_raw_mm", "tz_raw_mm"])
            writer.writerows(cam_records)
        print(f"[data] Saved: {cam_file}")

    if grbl_records:
        grbl_file = os.path.join(data_dir, f"{csv_stem}_grbl.csv")
        with open(grbl_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "x_mm", "y_mm", "z_mm"])
            writer.writerows(grbl_records)
        print(f"[data] Saved: {grbl_file}")

    plot_results(cam_records, grbl_records, csv_stem)
    print("\n[main] Done.")