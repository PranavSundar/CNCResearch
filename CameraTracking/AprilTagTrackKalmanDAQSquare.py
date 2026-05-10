"""
AprilTagTrackKalmanDAQ.py

AprilTag Kalman tracking + NI cDAQ (accelerometer & AE sensor).
All three threads share one time reference so live and post-run plots align.

Runtime:
  - OpenCV window  : live camera / AprilTag overlay
  - Live figure    : two scrolling sensor traces (accelerometer top, AE bottom)

Three post-run figures (saved as PNG + shown interactively):
  Figure 1 — Tracking     : 3×3 (GRBL path, camera path, overlay, X/Y/Z vs time,
                              residual, raw XY scatter, error bar chart)
  Figure 2 — Accelerometer: 2×2 (time domain, spectrogram, FFT, PSD Welch)
  Figure 3 — AE Sensor    : 2×2 (same layout)

Time axes on all panels share the same origin (start_ref) so events at e.g.
t = 15 s are directly comparable across every figure.

Usage:
  python AprilTagTrackKalmanDAQ.py

Dependencies:
  pip install pyserial numpy opencv-contrib-python matplotlib scipy nidaqmx
"""

import sys
import os
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


# ── GRBL / CAMERA CONFIGURATION ────────────────────────────────────────────────
PORT       = "COM3"
BAUD       = 115200
SIDE       = 200        # mm — square side length
FEED       = 1500       # mm/min
Z_SAFE     = 4.0        # mm — plunge depth
TARGET_FPS = 80         # GRBL poll rate (Hz)

CAM_INDEX   = 1
CAM_W       = 1920
CAM_H       = 1080
TAG_SIZE_MM = 30.0      # physical tag side length (mm)

KALMAN_PROCESS_NOISE     = 5.0    # mm/s²
KALMAN_MEASUREMENT_NOISE = 0.3    # mm
RTS_PROCESS_NOISE        = 50.0   # process noise for RTS smoother — higher = sharper corners
POSE_JUMP_MM             = 80.0
MIN_TAG_SIDE_PX          = 20.0

# ── DAQ CONFIGURATION ──────────────────────────────────────────────────────────
DAQ_DEVICE        = "cDAQ9185-22C6F90"

ACCEL_MODULE      = 1
ACCEL_CHANNEL     = "ai0"
ACCEL_SAMPLE_RATE = 8192.5243       # Hz
ACCEL_RANGE       = 10.0            # ±V

AE_MODULE         = 3
AE_CHANNEL        = "ai1"
AE_SAMPLE_RATE    = 131147.541      # Hz
AE_RANGE          = 10.0            # ±V

DAQ_BURST_DURATION    = 0.1   # seconds per read — small for responsive live display

# ── LIVE DISPLAY ───────────────────────────────────────────────────────────────
LIVE_WINDOW_S         = 2.0   # seconds of data shown in scrolling live plot
LIVE_UPDATE_HZ        = 10    # live plot refresh rate
LIVE_DOWNSAMPLE_ACCEL = 4     # keep 1-in-N accel samples for live deque (~2048 pts/s)
LIVE_DOWNSAMPLE_AE    = 64    # keep 1-in-N AE samples for live deque (~2049 pts/s)

# ── OUTPUT ─────────────────────────────────────────────────────────────────────
PLOT_DPI   = 100
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")


# ── 3D KALMAN FILTER ───────────────────────────────────────────────────────────
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

    def get_position(self):           return self.state[:3]
    def get_velocity(self):           return self.state[3:]
    def get_position_uncertainty(self):
        return np.sqrt([self.P[0, 0], self.P[1, 1], self.P[2, 2]])


# ── CAMERA THREAD ──────────────────────────────────────────────────────────────
def camera_worker(cam_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event):
    """Detect AprilTag, run Kalman filter, append 10-column records."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params   = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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

        t_now      = time.time() - start_ref[0]
        frame_time = time.time()
        dt         = frame_time - prev_frame_time
        prev_frame_time = frame_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if kf.is_initialized:
            kf.predict(dt)

        corners_list, ids, _ = aruco_detector.detectMarkers(gray)
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

                # Skip until GRBL homing cycle has finished
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
                    kf.initialize(0., 0., 0.)
                    print(f"  [camera] Home origin latched ({len(_origin_buf)} frames)  "
                          f"({cx_origin:.1f}, {cy_origin:.1f}) px  "
                          f"tag={tag_size_origin:.1f} px  "
                          f"scale={scale_origin:.4f} mm/px")
                    home_event.set()
                    continue

                if not home_event.is_set():
                    continue

                jump_px = POSE_JUMP_MM / scale_origin
                is_jump = (abs(cx - prev_cx) > jump_px or
                           abs(cy - prev_cy) > jump_px)

                tx =  (cx - cx_origin) * scale_origin
                ty = -(cy - cy_origin) * scale_origin
                tz = TAG_SIZE_MM / tag_size_origin - TAG_SIZE_MM / tag_side_px

                if not is_jump:
                    kf.update([tx, ty, tz])

                fp = kf.get_position()
                prev_cx, prev_cy = cx, cy
                if not motion_done_event.is_set():
                    cam_records.append([t_now, cx, cy, fp[0], fp[1], fp[2], tag_side_px,
                                        tx, ty, tz])

                cv2.polylines(frame_disp, [pts.astype(int)], True, (0, 255, 0), 2)
                for pt in pts.astype(int):
                    cv2.circle(frame_disp, tuple(pt), 5, (0, 255, 0), -1)
                cv2.circle(frame_disp, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                vx, vy, vz = kf.get_velocity()
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                cv2.putText(
                    frame_disp,
                    f"X:{fp[0]:.1f} Y:{fp[1]:.1f} Z:{fp[2]:.1f} mm | V:{speed:.1f} mm/s",
                    (int(cx) - 200, int(cy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
                )

        status = "ZEROED (Kalman 3D)" if cx_origin else "awaiting home"
        cv2.putText(frame_disp,
                    f"t={t_now:.2f}s  tags={n_tags}  [{status}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.imshow("AprilTag Kalman + DAQ", frame_disp)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# ── GRBL THREAD ────────────────────────────────────────────────────────────────
def grbl_worker(grbl_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event):
    """Home machine, wait for camera latch, run square move sequence, poll position."""
    ema_pos  = [None, None, None]
    interval = 1.0 / TARGET_FPS

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1, write_timeout=None)
    except Exception as e:
        print(f"  [grbl] Cannot open {PORT}: {e}")
        return

    time.sleep(2)
    ser.write(b"\r\n\r\n"); time.sleep(2); ser.reset_input_buffer()
    ser.write(b"$X\n");     time.sleep(1)
    ser.write(b"$10=0\n");  time.sleep(0.5)
    ser.write(b"$13=0\n");  time.sleep(0.5)  # Force mm in status reports
    ser.write(b"$H\n");     time.sleep(5)
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

    start_t = start_ref[0]
    for cmd, label in move_sequence:
        print(f"  [grbl] {label}")
        ser.write((cmd + "\n").encode())

        ser.reset_input_buffer()
        time.sleep(0.15)  # Let GRBL start moving before we poll

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
                    t_now = time.time() - start_t
                    for ax in range(3):
                        ema_pos[ax] = (rp[ax] if ema_pos[ax] is None
                                       else 0.8 * rp[ax] + 0.2 * ema_pos[ax])
                    grbl_records.append([t_now, ema_pos[0], ema_pos[1], ema_pos[2]])

                if "Run" in raw or "Jog" in raw:
                    idle_count = 0
                elif "Idle" in raw:
                    idle_count += 1
                    if idle_count >= 3:
                        break

            sleep_for = interval - (time.time() - poll_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    # Motion complete — stop camera recording, then stop all threads before homing rapid
    motion_done_event.set()
    print("  [grbl] Pattern complete — stopping recording before homing")
    stop_event.set()
    ser.write(b"$H\n"); time.sleep(5)
    ser.close()
    print("  [grbl] Done.")


# ── DAQ THREAD ─────────────────────────────────────────────────────────────────
def daq_worker(daq_bursts, accel_live, ae_live, start_ref, stop_event, home_event):
    """
    Acquires sensor data in DAQ_BURST_DURATION-second chunks using two
    continuous NI-DAQmx tasks (one per sensor module).

    Full-resolution chunks are appended to daq_bursts for post-run analysis.
    Downsampled data is pushed to accel_live / ae_live deques for the live plot.
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

        # Wait for the camera to latch home BEFORE starting the hardware tasks.
        # Starting first causes the ring buffer (only ~0.8 s deep) to overflow
        # during the homing phase; the subsequent read() raises a DaqError and
        # the entire acquisition loop exits silently with no data collected.
        if not home_event.wait(timeout=30.0):
            print("[daq] WARNING: home latch timed out after 30 s — starting anyway")
        if stop_event.is_set():
            print("[daq] stop_event set before homing completed — exiting")
            return

        accel_task.start()
        ae_task.start()
        print("[daq] Sensors started (continuous mode)")

        while not stop_event.is_set():
            t_burst = time.time() - start_ref[0]

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

            # Full-resolution storage for post-run analysis
            daq_bursts.append((t_burst, accel_chunk, ae_chunk))

            # Downsampled storage for live display
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


# ── LIVE DISPLAY (main thread) ─────────────────────────────────────────────────
def run_live_display(accel_live, ae_live, daq_bursts, start_ref, stop_event):
    """
    Scrolling live plot for both sensors — must run on the main thread.
    Blocks until stop_event is set, then closes the figure and returns.
    """
    plt.ion()
    fig_live, (ax_a, ax_ae) = plt.subplots(
        2, 1, figsize=(13, 6), dpi=PLOT_DPI,
        gridspec_kw={"hspace": 0.45},
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

    plt.tight_layout()
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()

    plot_interval = 1.0 / LIVE_UPDATE_HZ

    while not stop_event.is_set():
        t0    = time.time()
        t_now = t0 - start_ref[0]
        t_min = t_now - LIVE_WINDOW_S

        # Update accelerometer — list() gives a thread-safe snapshot of the deque
        if accel_live:
            try:
                arr = np.array(list(accel_live))
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

        # Update AE sensor
        if ae_live:
            try:
                arr = np.array(list(ae_live))
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

        # Close if the user manually closes the window
        if not plt.fignum_exists(fig_live.number):
            stop_event.set()
            break

        # plt.pause() drives the backend event loop for the sleep duration —
        # required on Windows for the window to update and stay responsive.
        elapsed   = time.time() - t0
        sleep_for = plot_interval - elapsed
        plt.pause(max(0.001, sleep_for))

    plt.ioff()
    if plt.fignum_exists(fig_live.number):
        plt.close(fig_live)


# ── SENSOR RECONSTRUCTION ──────────────────────────────────────────────────────
def build_sensor_arrays(daq_bursts):
    """
    Concatenate burst list into full waveforms with experiment-relative time axes.
    Returns (accel_t, accel_data, ae_t, ae_data) — all numpy arrays.
    """
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


# ── ALIGNMENT ──────────────────────────────────────────────────────────────────
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

    vp_idx = np.where(vp)[0]
    tx_out = np.full_like(cam_tx, np.nan)
    ty_out = np.full_like(cam_ty, np.nan)
    tz_out = np.full_like(cam_tz, np.nan)
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


# ── FIGURE 1: TRACKING (3×3) ───────────────────────────────────────────────────
def plot_tracking(cam_records, grbl_records, stem):
    if not cam_records or not grbl_records:
        print("[plot] Insufficient tracking data"); return None

    cam  = np.array(cam_records)
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

    tx_a, ty_a, tz_a, residuals = align_camera_to_grbl(
        cam_t, cam_tx, cam_ty, cam_tz, cam_px,
        grbl_t, grbl_x, grbl_y, grbl_z,
    )

    fig = plt.figure(figsize=(16, 10), dpi=PLOT_DPI)
    fig.suptitle(f"Tracking — {stem}", fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    # Planned square path in mm
    s = float(SIDE)
    ref_x = np.array([0,  s,  s, 0, 0])
    ref_y = np.array([0,  0, -s, -s, 0])

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
    for col, (col_a, col_g, ylabel, title) in enumerate([
        (tx_a, grbl_x, "X (mm)", "X Position"),
        (ty_a, grbl_y, "Y (mm)", "Y Position"),
        (tz_a, grbl_z, "Z (mm)", "Z Position"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(grbl_t, col_g, "b-", lw=1.5, label="GRBL")
        vt = np.isfinite(col_a)
        if vt.sum():
            ax.plot(cam_t[vt], col_a[vt], "k--", lw=1, alpha=0.7, label="Camera")
        ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
        ax.legend(fontsize=8); pub_fig(ax, fig)

    # ── Row 2: residual / raw scatter / error bars ─────────────────────────────
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
    WARMUP_S = 10.0
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


# ── FIGURES 2 & 3: SENSOR ANALYSIS (2×2) ──────────────────────────────────────
def plot_sensor(data, t_vec, sample_rate, label, color, stem, suffix):
    """
    2×2 sensor analysis figure.

    t_vec carries experiment-relative time (same origin as camera/GRBL) so the
    time-domain and spectrogram x-axes align directly with Figure 1.
    """
    N = len(data)
    if N == 0:
        print(f"[plot] No {label} data"); return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PLOT_DPI)
    fig.suptitle(
        f"{label}  —  {N:,} samples @ {sample_rate:.0f} Hz"
        f"    t = [{t_vec[0]:.1f} … {t_vec[-1]:.1f}] s",
        fontsize=13, fontweight="bold",
    )

    # ── Time domain (downsampled for display) ──────────────────────────────────
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

    # ── Spectrogram — x-axis shifted to experiment time ────────────────────────
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

    # ── FFT (demean + Hann window) ─────────────────────────────────────────────
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

    # ── PSD Welch ──────────────────────────────────────────────────────────────
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

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{stem}_{suffix}.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[plot] Saved: {save_path}")
    return fig


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  AprilTag Kalman Tracking + NI cDAQ (Accelerometer & AE Sensor)")
    print("=" * 65 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    start_ref         = [time.time()]
    stop_event        = threading.Event()
    home_event        = threading.Event()
    motion_done_event = threading.Event()
    homing_done_event = threading.Event()

    cam_records  = []      # [t, cx, cy, tx, ty, tz, tag_px, tx_raw, ty_raw, tz_raw]
    grbl_records = []      # [t, x, y, z]
    daq_bursts   = []      # [(t_start, accel_chunk, ae_chunk), ...]

    live_maxlen_a  = int(LIVE_WINDOW_S * ACCEL_SAMPLE_RATE / LIVE_DOWNSAMPLE_ACCEL * 2)
    live_maxlen_ae = int(LIVE_WINDOW_S * AE_SAMPLE_RATE    / LIVE_DOWNSAMPLE_AE    * 2)
    accel_live = deque(maxlen=live_maxlen_a)
    ae_live    = deque(maxlen=live_maxlen_ae)

    cam_thread  = threading.Thread(
        target=camera_worker,
        args=(cam_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event),
        daemon=True,
    )
    grbl_thread = threading.Thread(
        target=grbl_worker,
        args=(grbl_records, start_ref, stop_event, home_event, motion_done_event, homing_done_event),
        daemon=True,
    )
    daq_thread  = threading.Thread(
        target=daq_worker,
        args=(daq_bursts, accel_live, ae_live, start_ref, stop_event, home_event),
        daemon=True,
    )

    print("  Config:")
    print(f"    Camera  : index={CAM_INDEX}  {CAM_W}×{CAM_H}")
    print(f"    GRBL    : {PORT} @ {BAUD}  square={SIDE} mm @ {FEED} mm/min")
    print(f"    Kalman  : proc={KALMAN_PROCESS_NOISE} mm/s²  meas={KALMAN_MEASUREMENT_NOISE} mm  RTS={RTS_PROCESS_NOISE}")
    print(f"    Accel   : {DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}  {ACCEL_SAMPLE_RATE:.0f} Hz")
    print(f"    AE      : {DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}  {AE_SAMPLE_RATE:.0f} Hz")
    print(f"    Burst   : {DAQ_BURST_DURATION} s  Live window: {LIVE_WINDOW_S} s\n")

    cam_thread.start()
    grbl_thread.start()
    daq_thread.start()

    try:
        run_live_display(accel_live, ae_live, daq_bursts, start_ref, stop_event)
    except KeyboardInterrupt:
        print("\n[main] Interrupt — shutting down...")
        stop_event.set()

    cam_thread.join(timeout=3)
    grbl_thread.join(timeout=3)
    daq_thread.join(timeout=DAQ_BURST_DURATION * 5 + 3)

    print(f"\n[data] cam={len(cam_records)}  grbl={len(grbl_records)}  "
          f"daq_bursts={len(daq_bursts)}")

    # ── Save CSVs ───────────────────────────────────────────────────────────────
    if cam_records:
        path = os.path.join(OUTPUT_DIR, f"{stem}_cam.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "cx_px", "cy_px",
                        "tx_mm", "ty_mm", "tz_mm", "tag_side_px",
                        "tx_raw_mm", "ty_raw_mm", "tz_raw_mm"])
            w.writerows(cam_records)
        print(f"[data] {path}")

    if grbl_records:
        path = os.path.join(OUTPUT_DIR, f"{stem}_grbl.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "x_mm", "y_mm", "z_mm"])
            w.writerows(grbl_records)
        print(f"[data] {path}")

    if daq_bursts:
        accel_t, accel_data, ae_t, ae_data = build_sensor_arrays(daq_bursts)

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

    # ── Generate post-run figures ───────────────────────────────────────────────
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

    # ── Save multi-page PDF ─────────────────────────────────────────────────────
    if figures:
        pdf_path = os.path.join(OUTPUT_DIR, f"{stem}_report.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches="tight", dpi=150)
        print(f"[plot] Saved multi-page PDF: {pdf_path}")

    plt.show(block=True)
    print("\n[done]\n")


if __name__ == "__main__":
    main()
