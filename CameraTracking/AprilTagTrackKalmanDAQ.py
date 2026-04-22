"""
AprilTagTrackKalmanDAQ.py
Integrated AprilTag tracking with Kalman filter and NI cDAQ sensor acquisition.

Combines robust position tracking (Kalman filter) with synchronized sensor data
acquisition from accelerometer and acoustic emission sensors.

Usage:
  python AprilTagTrackKalmanDAQ.py

Dependencies:
  pip install pyserial numpy opencv-contrib-python matplotlib nidaqmx scipy
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
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from scipy import signal, integrate
from scipy.fft import fft, fftfreq

if not hasattr(cv2, "aruco"):
    sys.exit(
        "cv2.aruco not found — install the contrib build:\n"
        "  pip uninstall opencv-python && pip install opencv-contrib-python"
    )


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# GRBL/CNC Configuration
PORT        = "COM3"
BAUD        = 115200
SIDE        = 200       # mm  — square side length
FEED        = 1500      # mm/min
Z_SAFE      = 4.0       # mm  — plunge depth
TARGET_FPS  = 80        # GRBL poll rate (Hz)

# Camera Configuration
CAM_INDEX   = 1         # 0 = first USB camera
CAM_W       = 1920
CAM_H       = 1080
TAG_FAMILY  = "tag36h11"
TAG_SIZE_MM = 30.0      # physical side length of printed AprilTag

# Kalman Filter Configuration
KALMAN_PROCESS_NOISE    = 0.5   # Process noise (velocity uncertainty, mm/s²)
KALMAN_MEASUREMENT_NOISE = 2.0  # Measurement noise (camera detection noise, mm)

# DAQ Configuration (from Measurement & Automation Explorer)
DAQ_DEVICE = "cDAQ9185_22C6F90"                   # Ethernet cDAQ-9185 chassis S/N
ACCEL_MODULE = 1                                   # Accelerometer module slot
ACCEL_CHANNEL = "ai0"                              # Accelerometer channel (port 0)
ACCEL_SAMPLE_RATE = 8192.5243                      # Sampling rate in scans/s
ACCEL_RANGE = 10.0                                 # Input range (±10V)
AE_MODULE = 2                                      # AE sensor module slot
AE_CHANNEL = "ai0"                                 # AE sensor channel (port 0)
AE_SAMPLE_RATE = 131147.541                        # Sampling rate in scans/s
AE_RANGE = 10.0                                    # Input range (±10V)
DAQ_DURATION = 0.1                                 # DAQ acquisition duration per tracking cycle (seconds)
TERMINAL_CONFIG = TerminalConfiguration.DIFFERENTIAL

# Data Output
OUTPUT_DIR = "Data"                                # Output directory for CSV files
SAVE_PLOTS = True                                  # Save tracking plots
SAVE_SENSOR_DATA = True                           # Save sensor data with timestamps


# ── KALMAN FILTER CLASS ────────────────────────────────────────────────────────
class KalmanFilter3D:
    """
    6-state Kalman filter for 3D position + velocity tracking.
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z] (position only)
    """

    def __init__(self, dt, process_noise_sigma, measurement_noise_sigma):
        self.dt = dt

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

        # Process noise covariance
        process_noise = process_noise_sigma ** 2
        self.Q = np.array([
            [dt**4/4, 0, 0, dt**3/2, 0, 0],
            [0, dt**4/4, 0, 0, dt**3/2, 0],
            [0, 0, dt**4/4, 0, 0, dt**3/2],
            [dt**3/2, 0, 0, dt**2, 0, 0],
            [0, dt**3/2, 0, 0, dt**2, 0],
            [0, 0, dt**3/2, 0, 0, dt**2]
        ]) * process_noise

        # Measurement noise covariance
        measurement_noise_xy = measurement_noise_sigma ** 2
        measurement_noise_z = (measurement_noise_sigma * 1.5) ** 2  # Z typically noisier
        self.R = np.array([
            [measurement_noise_xy, 0, 0],
            [0, measurement_noise_xy, 0],
            [0, 0, measurement_noise_z]
        ])

        # Initial state covariance (uncertainty)
        self.P = np.eye(6) * 100.0

        # Identity matrix
        self.I = np.eye(6)

    def predict(self):
        """Predict next state and covariance."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update state with new measurement."""
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_position(self):
        """Get current position estimate [x, y, z]."""
        return self.state[:3]

    def get_velocity(self):
        """Get current velocity estimate [vx, vy, vz]."""
        return self.state[3:]


# ── DAQ FUNCTIONS ─────────────────────────────────────────────────────────────
def initialize_daq():
    """
    Initialize dual DAQ channels for accelerometer and AE sensor.
    Returns configured task or None on error.
    """
    try:
        task = nidaqmx.Task()

        # Add accelerometer channel (NI 9215, Module 1)
        accel_phys_ch = f"{DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}"
        task.ai_channels.add_ai_voltage_chan(
            physical_channel=accel_phys_ch,
            min_val=-ACCEL_RANGE,
            max_val=ACCEL_RANGE,
            terminal_config=TERMINAL_CONFIG,
            name_to_assign_to_channel="Accelerometer"
        )

        # Add AE sensor channel (NI 9223, Module 2)
        ae_phys_ch = f"{DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}"
        task.ai_channels.add_ai_voltage_chan(
            physical_channel=ae_phys_ch,
            min_val=-AE_RANGE,
            max_val=AE_RANGE,
            terminal_config=TERMINAL_CONFIG,
            name_to_assign_to_channel="AE_Sensor"
        )

        # Calculate samples for each sensor
        accel_samples = int(DAQ_DURATION * ACCEL_SAMPLE_RATE)
        ae_samples = int(DAQ_DURATION * AE_SAMPLE_RATE)

        # Configure timing using the faster sample rate (AE sensor)
        task.timing.cfg_samp_clk_timing(
            rate=AE_SAMPLE_RATE,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=ae_samples
        )

        return task, accel_samples, ae_samples

    except Exception as e:
        print(f"[daq_error] Failed to initialize DAQ: {e}")
        return None, None, None


def acquire_sensor_data(task, accel_samples, ae_samples):
    """
    Acquire a short burst of sensor data.
    Returns (accel_data, ae_data) or (None, None) on error.
    """
    try:
        task.start()
        raw_data = task.read(number_of_samples_per_channel=ae_samples, timeout=DAQ_DURATION+1)
        task.stop()

        # Convert to numpy array and separate channels
        data_array = np.array(raw_data).T

        # Extract channels (trim accelerometer to its sample count)
        accel_data = data_array[:accel_samples, 0]
        ae_data = data_array[:ae_samples, 1]

        return accel_data, ae_data

    except Exception as e:
        print(f"[daq_error] Failed to acquire sensor data: {e}")
        return None, None


# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────
def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_output_filepath(filename_prefix="integrated_tracking"):
    """Generate unique output filepath with timestamp."""
    ensure_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    return os.path.join(OUTPUT_DIR, filename)


def save_tracking_data(filepath, tracking_records, sensor_records):
    """Save integrated tracking and sensor data to CSV."""
    try:
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'frame', 'time_s',
                'tag_x_mm', 'tag_y_mm', 'tag_z_mm',
                'kalman_x_mm', 'kalman_y_mm', 'kalman_z_mm',
                'kalman_vx', 'kalman_vy', 'kalman_vz',
                'grbl_x_mm', 'grbl_y_mm', 'grbl_z_mm',
                'accel_mean', 'accel_rms', 'accel_peak',
                'ae_mean', 'ae_rms', 'ae_peak'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Combine tracking and sensor data by timestamp
            for record in tracking_records:
                timestamp = record['timestamp']

                # Find closest sensor data by timestamp
                closest_sensor = min(sensor_records,
                                   key=lambda x: abs((x['timestamp'] - timestamp).total_seconds()))

                row = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'frame': record.get('frame', ''),
                    'time_s': record.get('time_s', ''),
                    'tag_x_mm': record.get('tag_x_mm', ''),
                    'tag_y_mm': record.get('tag_y_mm', ''),
                    'tag_z_mm': record.get('tag_z_mm', ''),
                    'kalman_x_mm': record.get('kalman_x_mm', ''),
                    'kalman_y_mm': record.get('kalman_y_mm', ''),
                    'kalman_z_mm': record.get('kalman_z_mm', ''),
                    'kalman_vx': record.get('kalman_vx', ''),
                    'kalman_vy': record.get('kalman_vy', ''),
                    'kalman_vz': record.get('kalman_vz', ''),
                    'grbl_x_mm': record.get('grbl_x_mm', ''),
                    'grbl_y_mm': record.get('grbl_y_mm', ''),
                    'grbl_z_mm': record.get('grbl_z_mm', ''),
                    'accel_mean': closest_sensor.get('accel_mean', ''),
                    'accel_rms': closest_sensor.get('accel_rms', ''),
                    'accel_peak': closest_sensor.get('accel_peak', ''),
                    'ae_mean': closest_sensor.get('ae_mean', ''),
                    'ae_rms': closest_sensor.get('ae_rms', ''),
                    'ae_peak': closest_sensor.get('ae_peak', '')
                }
                writer.writerow(row)

        print(f"[save] Integrated data saved: {filepath}")
        return True
    except Exception as e:
        print(f"[save_error] Failed to save data: {e}")
        return False


# ── MAIN TRACKING FUNCTIONS ───────────────────────────────────────────────────
def detect_apriltag(frame, camera_matrix, dist_coeffs, tag_size_mm):
    """Detect AprilTag in frame and return pose."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{TAG_FAMILY.upper()}"))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Use first detected tag
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, tag_size_mm, camera_matrix, dist_coeffs
        )

        # Convert rotation vector to rotation matrix
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        # Get position in camera coordinates (mm)
        tag_position = tvec * 1000  # Convert to mm

        return tag_position, corners[0][0]
    return None, None


def camera_worker(camera_records, kalman_records, start_ref, stop_event, home_event):
    """Camera thread: captures frames, detects AprilTags, runs Kalman filter."""
    # Camera setup
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[camera] Failed to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Camera calibration (simplified - replace with actual calibration)
    focal_length = CAM_W
    center = (CAM_W/2, CAM_H/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Kalman filter setup
    dt = 1.0 / TARGET_FPS
    kf = KalmanFilter3D(dt, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE)

    # Wait for home position
    while not home_event.is_set():
        time.sleep(0.01)

    print("[camera] Starting tracking...")

    frame_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        timestamp = datetime.now()
        elapsed = timestamp - start_ref
        elapsed_sec = elapsed.total_seconds()

        # Detect AprilTag
        tag_position, corners = detect_apriltag(frame, camera_matrix, dist_coeffs, TAG_SIZE_MM)

        # Kalman filter prediction
        kf.predict()

        kalman_pos = kf.get_position()
        kalman_vel = kf.get_velocity()

        # Update with measurement if available
        if tag_position is not None:
            kf.update(tag_position)

        # Store camera record
        record = {
            'timestamp': timestamp,
            'frame': frame_count,
            'time_s': elapsed_sec,
            'tag_x_mm': tag_position[0] if tag_position is not None else None,
            'tag_y_mm': tag_position[1] if tag_position is not None else None,
            'tag_z_mm': tag_position[2] if tag_position is not None else None,
            'kalman_x_mm': kalman_pos[0],
            'kalman_y_mm': kalman_pos[1],
            'kalman_z_mm': kalman_pos[2],
            'kalman_vx': kalman_vel[0],
            'kalman_vy': kalman_vel[1],
            'kalman_vz': kalman_vel[2]
        }
        camera_records.append(record)

        # Draw on frame
        if tag_position is not None:
            # Draw tag corners
            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)

            # Draw position text
            cv2.putText(frame, f"Tag: ({tag_position[0]:.1f}, {tag_position[1]:.1f}, {tag_position[2]:.1f}) mm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Kalman estimate
        cv2.putText(frame, f"Kalman: ({kalman_pos[0]:.1f}, {kalman_pos[1]:.1f}, {kalman_pos[2]:.1f}) mm",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("AprilTag Tracking with Kalman + DAQ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        # Maintain target FPS
        target_time = start_time + frame_count / TARGET_FPS
        sleep_time = max(0, target_time - time.time())
        time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()
    print("[camera] Camera thread stopped")


def grbl_worker(grbl_records, start_ref, stop_event, home_event):
    """Background thread: polls GRBL position via serial."""
    while not home_event.is_set():
        time.sleep(0.01)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        ser.reset_input_buffer()
        print(f"[grbl] Connected to {PORT}@{BAUD}")
    except Exception as e:
        print(f"[grbl] Failed to open port: {e}")
        return

    pattern = re.compile(r"<.*?\|MPos:([\d.,\-]+)")

    while not stop_event.is_set():
        try:
            ser.write(b"?")
            response = ser.readline().decode('utf-8').strip()

            match = pattern.search(response)
            if match:
                pos_str = match.group(1)
                coords = [float(x) for x in pos_str.split(',')]

                record = {
                    'timestamp': datetime.now(),
                    'grbl_x_mm': coords[0],
                    'grbl_y_mm': coords[1],
                    'grbl_z_mm': coords[2]
                }
                grbl_records.append(record)

            time.sleep(1.0 / TARGET_FPS)

        except Exception as e:
            print(f"[grbl] Error: {e}")
            time.sleep(0.1)

    ser.close()
    print("[grbl] GRBL thread stopped")


def daq_worker(sensor_records, start_ref, stop_event, home_event):
    """DAQ thread: acquires sensor data synchronized with tracking."""
    # Initialize DAQ
    task, accel_samples, ae_samples = initialize_daq()
    if task is None:
        return

    # Wait for home position
    while not home_event.is_set():
        time.sleep(0.01)

    print("[daq] Starting sensor acquisition...")

    while not stop_event.is_set():
        timestamp = datetime.now()

        # Acquire sensor data
        accel_data, ae_data = acquire_sensor_data(task, accel_samples, ae_samples)

        if accel_data is not None and ae_data is not None:
            # Calculate statistics
            record = {
                'timestamp': timestamp,
                'accel_mean': float(np.mean(accel_data)),
                'accel_rms': float(np.sqrt(np.mean(accel_data**2))),
                'accel_peak': float(np.max(np.abs(accel_data))),
                'ae_mean': float(np.mean(ae_data)),
                'ae_rms': float(np.sqrt(np.mean(ae_data**2))),
                'ae_peak': float(np.max(np.abs(ae_data)))
            }
            sensor_records.append(record)

        # Sleep to maintain timing
        time.sleep(DAQ_DURATION)

    # Cleanup
    if task is not None:
        task.close()
    print("[daq] DAQ thread stopped")


def main():
    """Main integrated tracking and sensor acquisition routine."""
    print("\n=================================================================")
    print("     AprilTag Tracking with Kalman Filter + DAQ Sensors")
    print("=================================================================\n")

    # Initialize data storage
    camera_records = []
    grbl_records = []
    sensor_records = []

    # Threading events
    stop_event = threading.Event()
    home_event = threading.Event()

    start_ref = datetime.now()

    # Initialize threads
    camera_thread = threading.Thread(
        target=camera_worker,
        args=(camera_records, [], start_ref, stop_event, home_event),
        daemon=True
    )

    grbl_thread = threading.Thread(
        target=grbl_worker,
        args=(grbl_records, start_ref, stop_event, home_event),
        daemon=True
    )

    daq_thread = threading.Thread(
        target=daq_worker,
        args=(sensor_records, start_ref, stop_event, home_event),
        daemon=True
    )

    # Start threads
    camera_thread.start()
    grbl_thread.start()
    daq_thread.start()

    print("  [system] All threads started. Press 'q' in camera window to stop.")

    try:
        # Wait for threads to finish
        while camera_thread.is_alive() or grbl_thread.is_alive() or daq_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[main] Interrupt received, shutting down...")
        stop_event.set()

    # Wait for clean shutdown
    camera_thread.join(timeout=2)
    grbl_thread.join(timeout=2)
    daq_thread.join(timeout=2)

    print("\n[main] Processing results...")

    # Combine and save data
    if SAVE_SENSOR_DATA and (camera_records or sensor_records):
        csv_file = create_output_filepath("integrated_tracking")
        save_tracking_data(csv_file, camera_records, sensor_records)

    # Generate summary
    print("\n[summary] Acquisition complete:")
    print(f"  Camera frames: {len(camera_records)}")
    print(f"  GRBL positions: {len(grbl_records)}")
    print(f"  Sensor readings: {len(sensor_records)}")

    if SAVE_PLOTS and camera_records:
        print("\n[plot] Generating tracking plots...")
        # Extract data for plotting
        timestamps = [(r['timestamp'] - start_ref).total_seconds() for r in camera_records]
        kalman_x = [r['kalman_x_mm'] for r in camera_records]
        kalman_y = [r['kalman_y_mm'] for r in camera_records]
        kalman_z = [r['kalman_z_mm'] for r in camera_records]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Kalman position tracking
        axes[0, 0].plot(timestamps, kalman_x, 'b-', label='X', linewidth=1.5)
        axes[0, 0].plot(timestamps, kalman_y, 'r-', label='Y', linewidth=1.5)
        axes[0, 0].plot(timestamps, kalman_z, 'g-', label='Z', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (mm)')
        axes[0, 0].set_title('Kalman Filter Position Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Sensor data over time (if available)
        if sensor_records:
            sensor_timestamps = [(r['timestamp'] - start_ref).total_seconds() for r in sensor_records]
            accel_rms = [r['accel_rms'] for r in sensor_records]
            ae_rms = [r['ae_rms'] for r in sensor_records]

            axes[0, 1].plot(sensor_timestamps, accel_rms, 'b-', label='Accelerometer RMS', linewidth=1)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('RMS Amplitude (V)')
            axes[0, 1].set_title('Sensor RMS Levels')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(sensor_timestamps, ae_rms, 'r-', label='AE RMS', linewidth=1)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('RMS Amplitude (V)')
            axes[1, 0].set_title('Acoustic Emission RMS')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # GRBL position (if available)
        if grbl_records:
            grbl_timestamps = [(r['timestamp'] - start_ref).total_seconds() for r in grbl_records]
            grbl_x = [r['grbl_x_mm'] for r in grbl_records]
            grbl_y = [r['grbl_y_mm'] for r in grbl_records]
            grbl_z = [r['grbl_z_mm'] for r in grbl_records]

            axes[1, 1].plot(grbl_timestamps, grbl_x, 'b-', label='X', linewidth=1)
            axes[1, 1].plot(grbl_timestamps, grbl_y, 'r-', label='Y', linewidth=1)
            axes[1, 1].plot(grbl_timestamps, grbl_z, 'g-', label='Z', linewidth=1)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Position (mm)')
            axes[1, 1].set_title('GRBL Machine Position')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\n[done] Integrated tracking and sensor acquisition complete!\n")


if __name__ == "__main__":
    main()