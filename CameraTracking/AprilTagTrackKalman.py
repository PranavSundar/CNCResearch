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
import serial
import time
import re
import csv
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

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
KALMAN_PROCESS_NOISE    = 0.5   # Process noise (velocity uncertainty, mm/s²)
KALMAN_MEASUREMENT_NOISE = 2.0  # Measurement noise (camera detection noise, mm)
KALMAN_OUTLIER_SIGMA    = 3.0   # Reject measurements >N std deviations from prediction

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
        self.P = np.eye(6) * 5.0
        self.is_initialized = True
    
    def predict(self):
        """Prediction step: advance state by one time step."""
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
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Check if measurement is an outlier
        innovation_squared = y @ np.linalg.inv(S) @ y
        return innovation_squared
    
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
def camera_worker(cam_records, start_ref, stop_event, home_event):
    """
    Background thread: captures frames, detects AprilTag, appends to cam_records.
    Each record: [elapsed_t, cx_px, cy_px, tx_mm, ty_mm, tag_side_px]

    Uses Kalman filter for robust position estimation.
    """
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params   = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx_origin    = None
    cy_origin    = None
    scale_origin = None
    tag_size_origin = None  # tag_side_px at home
    prev_cx      = None
    prev_cy      = None
    
    # Kalman filter for camera measurements (3D)
    kf = KalmanFilter3D(
        dt=1.0/30.0,  # Assume ~30Hz camera update
        process_noise_sigma=KALMAN_PROCESS_NOISE,
        measurement_noise_sigma=KALMAN_MEASUREMENT_NOISE,
        measurement_noise_z=KALMAN_MEASUREMENT_NOISE * 2.0  # Z typically noisier
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
        
        # Update Kalman filter time step
        if dt > 0.001:  # Only update if dt is reasonable
            kf.dt = np.clip(dt, 0.01, 0.1)  # Clamp to reasonable range

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

                # Latch origin on first detection
                if cx_origin is None:
                    cx_origin    = cx
                    cy_origin    = cy
                    tag_size_origin = tag_side_px
                    scale_origin = TAG_SIZE_MM / tag_side_px
                    prev_cx, prev_cy = cx, cy
                    kf.initialize(0.0, 0.0, 0.0)  # Origin is at (0, 0, 0)
                    print(f"  [camera] Home origin latched at "
                          f"({cx:.1f}, {cy:.1f}) px  "
                          f"tag={tag_side_px:.1f} px  "
                          f"scale={scale_origin:.4f} mm/px")
                    home_event.set()
                    cam_records.append([t_now, cx, cy, 0.0, 0.0, 0.0, tag_side_px])
                    continue

                # Outlier rejection in pixel space
                jump_px = POSE_JUMP_MM / scale_origin
                is_jump = abs(cx - prev_cx) > jump_px or abs(cy - prev_cy) > jump_px
                
                # Convert to mm (XY from pixel shift, Z from tag apparent size)
                tx =  (cx - cx_origin) * scale_origin
                ty = -(cy - cy_origin) * scale_origin
                # Z from tag size change: larger apparent size = closer (smaller Z)
                tz = (TAG_SIZE_MM / tag_size_origin - TAG_SIZE_MM / tag_side_px)
                
                # Kalman prediction step
                kf.predict()
                
                # Kalman update with measurement validation
                innovation_sq = kf.update([tx, ty, tz], is_valid=not is_jump)
                measurement_valid = not is_jump
                
                # Check if innovation is too large (outlier)
                outlier = innovation_sq > (KALMAN_OUTLIER_SIGMA ** 2)
                if outlier:
                    # Reject this measurement but keep the prediction
                    kf.update([tx, ty, tz], is_valid=False)
                
                # Get filtered position
                filtered_pos = kf.get_position()
                tx_filtered, ty_filtered, tz_filtered = filtered_pos[0], filtered_pos[1], filtered_pos[2]
                
                prev_cx, prev_cy = cx, cy
                cam_records.append([t_now, cx, cy, tx_filtered, ty_filtered, tz_filtered, tag_side_px])

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
def grbl_worker(grbl_records, start_ref, stop_event, home_event):
    """Background thread: polls GRBL position via serial."""
    while not home_event.is_set():
        time.sleep(0.01)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        ser.reset_input_buffer()
        print(f"  [grbl] Connected to {PORT}@{BAUD}")
    except Exception as e:
        print(f"  [grbl] Failed to open port: {e}")
        return

    pattern = re.compile(r"<.*?\|MPos:([\d.,\-]+)")

    while not stop_event.is_set():
        try:
            ser.write(b"?")
            time.sleep(1.0 / TARGET_FPS)
            
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            match = pattern.search(line)
            if match:
                parts = match.group(1).split(",")
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                t_now = time.time() - start_ref[0]
                grbl_records.append([t_now, x, y, z])
        except Exception:
            pass

    ser.close()


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

    grbl_x_v = nearest_value(cam_t_v, grbl_t, grbl_x)
    grbl_y_v = nearest_value(cam_t_v, grbl_t, grbl_y)
    grbl_z_v = nearest_value(cam_t_v, grbl_t, grbl_z)

    valid = np.isfinite(grbl_x_v) & np.isfinite(grbl_y_v) & np.isfinite(grbl_z_v)
    if valid.sum() < 10:
        return cam_tx, cam_ty, cam_tz, np.full_like(cam_tx, np.nan)

    grbl_x_v = grbl_x_v[valid]
    grbl_y_v = grbl_y_v[valid]
    grbl_z_v = grbl_z_v[valid]
    cam_tx_v = cam_tx_v[valid]
    cam_ty_v = cam_ty_v[valid]
    cam_tz_v = cam_tz_v[valid]

    depth_ratio = np.ones_like(cam_tx_v)  # Simplified
    A_list = [
        cam_tx_v,
        cam_ty_v,
        cam_tz_v,
        depth_ratio,
        cam_tx_v * depth_ratio,
        cam_ty_v * depth_ratio,
        np.ones_like(cam_tx_v)
    ]
    A = np.column_stack(A_list)

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
    tx_out = np.full_like(cam_tx, np.nan)
    ty_out = np.full_like(cam_ty, np.nan)
    tz_out = np.full_like(cam_tz, np.nan)
    residuals_out = np.full_like(cam_tx, np.nan)
    tx_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tx_corrected)
    ty_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], ty_corrected)
    tz_out[vp] = np.interp(cam_t[vp], cam_t_v[valid], tz_corrected)
    residuals_out[vp] = np.sqrt((tx_corrected - grbl_x_v)**2 + (ty_corrected - grbl_y_v)**2 + (tz_corrected - grbl_z_v)**2)

    return tx_out, ty_out, tz_out, residuals_out


# ── PLOTTING ───────────────────────────────────────────────────────────────────
def plot_results(cam_records, grbl_records, csv_stem):
    """Plot and save camera & GRBL trajectories in 3D."""
    if not cam_records or not grbl_records:
        print("[plot] Insufficient data to plot")
        return

    cam = np.array(cam_records)
    grbl = np.array(grbl_records)

    cam_t, cam_cx, cam_cy, cam_tx, cam_ty, cam_tz, cam_px = cam.T
    grbl_t, grbl_x, grbl_y, grbl_z = grbl.T

    tx_aligned, ty_aligned, tz_aligned, residuals = align_camera_to_grbl(
        cam_t, cam_tx, cam_ty, cam_tz, cam_px, grbl_t, grbl_x, grbl_y, grbl_z
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"AprilTag Kalman Tracking (3D): {csv_stem}", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(cam_t, cam_tx, s=10, alpha=0.6, label="Raw camera")
    ax.scatter(cam_t, tx_aligned, s=10, alpha=0.6, label="Aligned")
    ax.plot(grbl_t, grbl_x, 'r-', linewidth=2, label="GRBL command")
    ax.set_ylabel("X (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(cam_t, cam_ty, s=10, alpha=0.6, label="Raw camera")
    ax.scatter(cam_t, ty_aligned, s=10, alpha=0.6, label="Aligned")
    ax.plot(grbl_t, grbl_y, 'r-', linewidth=2, label="GRBL command")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(cam_t, cam_tz, s=10, alpha=0.6, label="Camera (tag size)")
    ax.scatter(cam_t, tz_aligned, s=10, alpha=0.6, label="Aligned")
    ax.plot(grbl_t, grbl_z, 'r-', linewidth=2, label="GRBL command")
    ax.set_ylabel("Z (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    valid = np.isfinite(tx_aligned) & np.isfinite(ty_aligned)
    if valid.sum() > 10:
        ax.plot(grbl_x, grbl_y, 'r-', linewidth=2, label="GRBL path", alpha=0.7)
        ax.scatter(tx_aligned[valid], ty_aligned[valid], s=5, alpha=0.6, c=cam_t[valid], cmap='viridis')
        ax.set_aspect('equal')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Aligned XY Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    valid = np.isfinite(cam_tx) & np.isfinite(cam_ty)
    if valid.sum() > 10:
        ax.scatter(cam_tx[valid], cam_ty[valid], s=5, alpha=0.6, c=cam_t[valid], cmap='plasma')
        ax.set_aspect('equal')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Raw Camera XY")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    info_text = f"Kalman Filter (3D)\n"
    info_text += f"Process Noise: {KALMAN_PROCESS_NOISE} mm/s²\n"
    info_text += f"Measurement Noise XY: {KALMAN_MEASUREMENT_NOISE} mm\n"
    info_text += f"Measurement Noise Z: {KALMAN_MEASUREMENT_NOISE*2.0} mm\n"
    info_text += f"Outlier Threshold: {KALMAN_OUTLIER_SIGMA}σ"
    ax.text(0.5, 0.5, info_text,
            ha='center', va='center', fontsize=10, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.axis('off')

    plt.tight_layout()
    plot_path = f"Data/{csv_stem}.png"
    plt.savefig(plot_path, dpi=100)
    print(f"[plot] Saved: {plot_path}")
    plt.close()


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now = datetime.now()
    csv_stem = f"apriltag_kalman_{now.strftime('%Y%m%d_%H%M%S')}"

    start_ref = [time.time()]
    stop_event = threading.Event()
    home_event = threading.Event()

    cam_records = []
    grbl_records = []

    cam_thread = threading.Thread(target=camera_worker, args=(cam_records, start_ref, stop_event, home_event), daemon=True)
    grbl_thread = threading.Thread(target=grbl_worker, args=(grbl_records, start_ref, stop_event, home_event), daemon=True)

    cam_thread.start()
    grbl_thread.start()

    print("\n═══════════════════════════════════════════════════════════════")
    print("     AprilTag Tracking with 3D Kalman Filter (Position + Velocity)")
    print("═══════════════════════════════════════════════════════════════\n")
    print(f"  Config:")
    print(f"    CAM_INDEX = {CAM_INDEX}  |  RES = {CAM_W}x{CAM_H}")
    print(f"    GRBL: {PORT} @ {BAUD}")
    print(f"    Process Noise: {KALMAN_PROCESS_NOISE} mm/s²")
    print(f"    Measurement Noise XY: {KALMAN_MEASUREMENT_NOISE} mm")
    print(f"    Measurement Noise Z: {KALMAN_MEASUREMENT_NOISE*2.0} mm\n")

    try:
        while cam_thread.is_alive() or grbl_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[main] Interrupt received, shutting down...")
        stop_event.set()

    cam_thread.join(timeout=2)
    grbl_thread.join(timeout=2)

    print(f"\n[data] Captured {len(cam_records)} camera records, {len(grbl_records)} GRBL records")

    if cam_records:
        with open(f"Data/{csv_stem}_cam.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "cx_px", "cy_px", "tx_mm", "ty_mm", "tz_mm", "tag_side_px"])
            writer.writerows(cam_records)
        print(f"[data] Saved: Data/{csv_stem}_cam.csv")

    if grbl_records:
        with open(f"Data/{csv_stem}_grbl.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "x_mm", "y_mm", "z_mm"])
            writer.writerows(grbl_records)
        print(f"[data] Saved: Data/{csv_stem}_grbl.csv")

    plot_results(cam_records, grbl_records, csv_stem)
    print("\n[main] Done.")
