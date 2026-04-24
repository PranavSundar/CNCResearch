"""
AprilTagTrack.py
Simultaneously tracks CNC motion via GRBL serial + AprilTag (tag36h11) via webcam.

Camera pose is zeroed at the home position: the first tag detection after the
camera thread starts (which is after homing completes) is used as the (0,0,0)
origin. All subsequent poses are reported relative to that baseline.

Usage:
  python AprilTagTrack.py

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
import os
import sys
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
TAG_SIZE_MM = 30.0      # physical side length of printed AprilTag (in) — measure yours!

EMA_ALPHA       = 0.8       # GRBL position smoothing (0 = heavy, 1 = raw)
POSE_JUMP_MM    = 80.0      # discard detection if centroid jumps more than this (in)
MIN_TAG_SIDE_PX = 20.0      # reject detections where tag side < this many pixels


# ── CAMERA TRACKING THREAD ─────────────────────────────────────────────────────
def camera_worker(cam_records, start_ref, stop_event, home_event):
    """
    Background thread: captures frames, detects AprilTag, appends to cam_records.
    Each record: [elapsed_t, cx_px, cy_px, tx_mm, ty_mm, tag_side_px]

    XY position is derived purely from pixel centroid displacement scaled by the
    tag's apparent size in pixels.  Because the tag's physical side length is
    known (TAG_SIZE_MM), scale = TAG_SIZE_MM / tag_side_px gives mm/px with no
    focal length or camera calibration required.  This makes the XY measurement
    accurate for in-plane motion regardless of camera height or FOV.

    The first detection after the thread starts (post-homing) is the origin.
    """
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # lock focus so pixel-to-mm scale stays constant

    aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params   = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx_origin    = None   # pixel centroid at home
    cy_origin    = None
    scale_origin = None   # TAG_SIZE_MM / tag_side_px at home  =>  mm per pixel
    prev_cx      = None   # for outlier rejection
    prev_cy      = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        t_now = time.time() - start_ref[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = aruco_detector.detectMarkers(gray)
        n_tags = 0 if ids is None else len(ids)

        frame_disp = frame.copy()

        if ids is not None:
            for corners, tag_id in zip(corners_list, ids.flatten()):
                pts = corners[0]   # (4, 2)

                # Apparent tag side length: average of all 4 sides in pixels
                tag_side_px = np.mean([
                    np.linalg.norm(pts[1] - pts[0]),
                    np.linalg.norm(pts[2] - pts[1]),
                    np.linalg.norm(pts[3] - pts[2]),
                    np.linalg.norm(pts[0] - pts[3]),
                ])

                if tag_side_px < MIN_TAG_SIDE_PX:
                    continue   # tag too small / too far / partially occluded

                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()

                # Latch origin on first detection (machine is at home)
                if cx_origin is None:
                    cx_origin    = cx
                    cy_origin    = cy
                    scale_origin = TAG_SIZE_MM / tag_side_px   # mm / px
                    prev_cx, prev_cy = cx, cy
                    print(f"  [camera] Home origin latched at "
                          f"({cx:.1f}, {cy:.1f}) px  "
                          f"tag={tag_side_px:.1f} px  "
                          f"scale={scale_origin:.4f} mm/px")
                    home_event.set()
                    cam_records.append([t_now, cx, cy, 0.0, 0.0, tag_side_px])
                    continue

                # Outlier rejection in pixel space
                jump_px = POSE_JUMP_MM / scale_origin
                if abs(cx - prev_cx) > jump_px or abs(cy - prev_cy) > jump_px:
                    cam_records.append([t_now, cx, cy, np.nan, np.nan, tag_side_px])
                    continue

                # Convert centroid displacement to mm using origin scale.
                # Image X increases rightward  → same as CNC X (+right).
                # Image Y increases downward   → opposite to CNC Y (+up / +away),
                # so negate ty to match CNC axis convention.
                tx =  (cx - cx_origin) * scale_origin
                ty = -(cy - cy_origin) * scale_origin

                prev_cx, prev_cy = cx, cy
                cam_records.append([t_now, cx, cy, tx, ty, tag_side_px])

                # Draw detection overlay
                cv2.polylines(frame_disp, [pts.astype(int)], True, (0, 255, 0), 2)
                for pt in pts.astype(int):
                    cv2.circle(frame_disp, tuple(pt), 5, (0, 255, 0), -1)
                cv2.circle(frame_disp, (int(cx), int(cy)), 8, (0, 0, 255), -1)

                label = f"ID{tag_id}  X:{tx:.1f} Y:{ty:.1f} mm"
                cv2.putText(frame_disp, label,
                            (int(cx) - 80, int(cy) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        status = "ZEROED" if cx_origin is not None else "awaiting home"
        cv2.putText(
            frame_disp,
            f"t={t_now:.2f}s   tags={n_tags}   [{status}]",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
        )
        cv2.imshow("AprilTag Tracking", frame_disp)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# ── HELPERS ────────────────────────────────────────────────────────────────────
def safe_range(arr):
    """Returns max-min, avoiding zero division downstream."""
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


def align_camera_to_grbl(cam_t, cam_tx, cam_ty, cam_tag_px, grbl_t, grbl_x, grbl_y, grbl_z):
    """
    Recover CNC XYZ from camera data via least-squares regression.

    Feature set (6 terms):
      cam_tx              — horizontal pixel shift  (encodes CNC X + perspective bleed from Y)
      cam_ty              — vertical pixel shift    (encodes CNC Z + camera tilt)
      depth_ratio         — tag_px_home/tag_px - 1  (encodes CNC Y depth change)
      cam_tx*depth_ratio  — perspective interaction (X-shift varies with distance)
      cam_ty*depth_ratio  — vertical-perspective interaction
      1                   — bias

    Returns: (tx_corrected, ty_corrected, tz_corrected)
    """
    vp = np.isfinite(cam_tx) & np.isfinite(cam_ty) & np.isfinite(cam_tag_px) & (cam_tag_px > 0)
    if vp.sum() < 20 or len(grbl_t) < 2:
        return cam_tx, cam_ty, np.full(len(cam_tx), np.nan)

    tag_px_home = cam_tag_px[vp][0]

    ct       = cam_t[vp]
    in_range = (ct >= grbl_t[0]) & (ct <= grbl_t[-1])
    if in_range.sum() < 20:
        return cam_tx, cam_ty, np.full(len(cam_tx), np.nan)

    ctx = cam_tx[vp][in_range]
    cty = cam_ty[vp][in_range]
    dr  = tag_px_home / cam_tag_px[vp][in_range] - 1.0

    gx = np.interp(ct[in_range], grbl_t, grbl_x)
    gy = np.interp(ct[in_range], grbl_t, grbl_y)
    gz = np.interp(ct[in_range], grbl_t, grbl_z)

    A = np.column_stack([ctx, cty, dr, ctx * dr, cty * dr, np.ones(len(ctx))])
    coef_x, _, _, _ = np.linalg.lstsq(A, gx, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, gy, rcond=None)
    coef_z, _, _, _ = np.linalg.lstsq(A, gz, rcond=None)

    all_ctx = cam_tx[vp]
    all_cty = cam_ty[vp]
    all_dr  = tag_px_home / cam_tag_px[vp] - 1.0
    A_all   = np.column_stack([all_ctx, all_cty, all_dr,
                                all_ctx * all_dr, all_cty * all_dr,
                                np.ones(vp.sum())])

    tx_out = cam_tx.copy().astype(float)
    ty_out = cam_ty.copy().astype(float)
    tz_out = np.full(len(cam_tx), np.nan)
    tx_out[vp] = A_all @ coef_x
    ty_out[vp] = A_all @ coef_y
    tz_out[vp] = A_all @ coef_z

    print(f"  [align] coef_x: {np.round(coef_x, 4)}")
    print(f"  [align] coef_y: {np.round(coef_y, 4)}")
    print(f"  [align] coef_z: {np.round(coef_z, 4)}")
    return tx_out, ty_out, tz_out


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Connect GRBL ──────────────────────────────────────────────────────────
    s = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    s.write(b"\r\n\r\n")
    time.sleep(2)
    s.reset_input_buffer()

    print("Setting up machine...")
    s.write(b"$X\n");    time.sleep(1)
    s.write(b"$10=0\n"); time.sleep(0.5)   # force WPos reporting in status messages
    s.write(b"$H\n");    time.sleep(5)
    s.write(b"G21 G90 G92 X0 Y0 Z0\n"); time.sleep(1)

    # ── Shared state ──────────────────────────────────────────────────────────
    grbl_positions  = []    # list of [x, y, z] EMA-smoothed
    grbl_timestamps = []    # list of elapsed floats

    cam_records  = []       # list of [t, cx_px, cy_px, tx_mm, ty_mm, tag_side_px]
    ema_pos      = [None, None, None]

    start_time   = time.time()
    start_ref    = [start_time]     # mutable wrapper shared with camera thread
    stop_event   = threading.Event()
    home_event   = threading.Event()
    interval     = 1.0 / TARGET_FPS

    # ── Start camera thread ───────────────────────────────────────────────────
    cam_thread = threading.Thread(
        target=camera_worker,
        args=(cam_records, start_ref, stop_event, home_event),
        daemon=True,
    )
    cam_thread.start()

    print("Waiting for tag detection at home position...")
    if not home_event.wait(timeout=30.0):
        print("WARNING: tag not detected within 30 s — check camera and tag placement")

    print(f"Starting square pattern ({SIDE} mm @ {FEED} mm/min, GRBL at {TARGET_FPS} fps)")

    # ── Move sequence ─────────────────────────────────────────────────────────
    move_sequence = [
        (f"G1 Z-{Z_SAFE:.1f} F{FEED}",   "Plunge Z"),
        (f"G1 X{SIDE} Y0 F{FEED}",        "-> X+"),
        (f"G1 X{SIDE} Y-{SIDE} F{FEED}", "-> Corner"),
        (f"G1 X0 Y-{SIDE} F{FEED}",      "<- X-"),
        (f"G1 X0 Y0 F{FEED}",            "<- Home"),
        (f"G1 Z{Z_SAFE + 5:.1f} F{FEED}","Retract Z"),
    ]

    for cmd, label in move_sequence:
        print(f"  {label}")
        s.write((cmd + "\n").encode())

        deadline       = time.time() + 90.0
        idle_count     = 0
        motion_started = False
        s.reset_input_buffer()

        while time.time() < deadline:
            poll_start = time.time()

            s.write(b"?\n")
            time.sleep(0.02)

            raw = ""
            while s.in_waiting:
                try:
                    raw += s.readline().decode(errors="ignore").strip()
                except Exception:
                    pass

            if raw:
                match = re.search(
                    r"WPos:([\d\.\-]+),([\d\.\-]+),([\d\.\-]+)", raw
                ) or re.search(
                    r"MPos:([\d\.\-]+),([\d\.\-]+),([\d\.\-]+)", raw
                )
                if match:
                    raw_pos = [
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    ]
                    now = time.time() - start_time

                    for ax in range(3):
                        if ema_pos[ax] is None:
                            ema_pos[ax] = raw_pos[ax]
                        else:
                            ema_pos[ax] = (
                                EMA_ALPHA * raw_pos[ax]
                                + (1 - EMA_ALPHA) * ema_pos[ax]
                            )

                    grbl_positions.append(list(ema_pos))
                    grbl_timestamps.append(now)

                if "Run" in raw or "Jog" in raw:
                    motion_started = True
                    idle_count = 0
                elif "Idle" in raw and motion_started:
                    idle_count += 1
                    if idle_count >= 3:
                        break

            elapsed   = time.time() - poll_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    s.write(b"$H\n"); time.sleep(5)
    total_time = time.time() - start_time

    stop_event.set()
    cam_thread.join(timeout=3.0)

    print(f"Done! Complete in {total_time:.1f} seconds")
    s.close()


    # ── Process GRBL data ──────────────────────────────────────────────────────
    pos = np.array(grbl_positions)
    t   = np.array(grbl_timestamps)

    _, uidx = np.unique(t, return_index=True)
    pos, t  = pos[uidx], t[uidx]

    dt_g    = np.diff(t)
    dpos    = np.diff(pos, axis=0)
    valid_g = dt_g > 1e-6
    speed   = np.linalg.norm(dpos[valid_g], axis=1) / dt_g[valid_g]
    t_vel   = t[1:][valid_g]
    MAX_SPEED = FEED / 60.0 * 3
    keep_g  = np.isfinite(speed) & (speed < MAX_SPEED)
    speed, t_vel = speed[keep_g], t_vel[keep_g]

    dt_vel  = np.diff(t_vel)
    valid_a = dt_vel > 1e-6
    accel   = np.diff(speed)[valid_a] / dt_vel[valid_a]
    t_acc   = t_vel[1:][valid_a]
    keep_a  = np.isfinite(accel)
    accel, t_acc = accel[keep_a], t_acc[keep_a]


    # ── Process camera data ────────────────────────────────────────────────────
    cam_arr = np.array(cam_records) if cam_records else np.empty((0, 6))
    # columns: [t, cx_px, cy_px, tx_mm, ty_mm, tag_side_px]

    has_cam      = len(cam_arr) > 0
    has_cam_pose = has_cam and np.any(np.isfinite(cam_arr[:, 3]))

    cam_t  = cam_arr[:, 0] if has_cam else np.array([])
    cam_cx = cam_arr[:, 1] if has_cam else np.array([])
    cam_cy = cam_arr[:, 2] if has_cam else np.array([])
    cam_tx     = cam_arr[:, 3] if has_cam else np.array([])
    cam_ty     = cam_arr[:, 4] if has_cam else np.array([])
    cam_tag_px = cam_arr[:, 5] if has_cam else np.array([])   # apparent tag size (px)

    cam_tz = np.full(len(cam_tx), np.nan)

    # Auto-align camera axes to GRBL using depth-ratio regression
    if has_cam_pose and len(t) > 1:
        cam_tx, cam_ty, cam_tz = align_camera_to_grbl(
            cam_t, cam_tx, cam_ty, cam_tag_px, t, pos[:, 0], pos[:, 1], pos[:, 2]
        )
        print("  [align] camera axes corrected via depth-ratio regression")

    # Camera XY speed from pose
    spd_c = np.array([])
    tv_c  = np.array([])
    if has_cam_pose:
        vp     = np.isfinite(cam_tx) & np.isfinite(cam_ty)
        cp_t   = cam_t[vp]
        cp_xy  = np.column_stack([cam_tx[vp], cam_ty[vp]])
        if len(cp_t) > 1:
            dt_c   = np.diff(cp_t)
            dxy_c  = np.diff(cp_xy, axis=0)
            spd_c  = np.linalg.norm(dxy_c, axis=1) / dt_c
            tv_c   = cp_t[1:]
            keep_c = np.isfinite(spd_c) & (dt_c > 1e-3)
            spd_c, tv_c = spd_c[keep_c], tv_c[keep_c]


    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\nStats (GRBL):")
    print(f"  Points captured : {len(t)}")
    print(f"  Duration        : {t[-1]:.1f} s")
    print(f"  Avg sample rate : {len(t) / t[-1]:.1f} Hz")
    if len(speed):
        print(f"  Max speed       : {speed.max():.2f} mm/s")
    if len(accel):
        print(f"  Max |accel|     : {np.abs(accel).max():.2f} mm/s²")

    if has_cam:
        print(f"\nStats (Camera):")
        print(f"  Frames with tag : {len(cam_arr)}")
        if cam_t[-1] > 0:
            print(f"  Avg tag rate    : {len(cam_arr) / cam_t[-1]:.1f} Hz")
        valid_tag_px = cam_tag_px[np.isfinite(cam_tag_px)]
        if len(valid_tag_px) > 0:
            print(f"  Avg tag size    : {valid_tag_px.mean():.1f} px")


    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"CNC + AprilTag Run  —  {len(t)} GRBL pts / {len(cam_arr)} cam frames  —  {total_time:.1f} s",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    # Row 0 — toolpaths ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(pos[:, 0], pos[:, 1], "b.-", lw=1.5, ms=2)
    ax.plot(pos[0, 0],  pos[0, 1],  "go", ms=8)
    ax.plot(pos[-1, 0], pos[-1, 1], "ro", ms=8)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="GRBL Toolpath")
    ax.set_aspect("equal"); ax.grid(True)
    ax.legend(["Path", "Start", "End"], fontsize=8)
    pub_fig(ax, fig)

    ax = fig.add_subplot(gs[0, 1])
    if has_cam:
        if has_cam_pose:
            vp = np.isfinite(cam_tx)
            ax.plot(cam_tx[vp], cam_ty[vp], "m.-", lw=1.5, ms=2)
            ax.set(xlabel="X cam (mm)", ylabel="Y cam (mm)", title="Camera Pose Path")
        else:
            ax.plot(cam_cx, cam_cy, "m.-", lw=1.5, ms=2)
            ax.invert_yaxis()
            ax.set(xlabel="X (px)", ylabel="Y (px)", title="Camera Pixel Path")
    else:
        ax.set_title("Camera Path (no data)")
    ax.grid(True)
    pub_fig(ax, fig)

    # Normalised overlay
    ax = fig.add_subplot(gs[0, 2])
    if len(pos) > 1:
        gx = (pos[:, 0] - pos[:, 0].min()) / safe_range(pos[:, 0])
        gy = (pos[:, 1] - pos[:, 1].min()) / safe_range(pos[:, 1])
        ax.plot(gx, gy, "b-", lw=1.5, alpha=0.8, label="GRBL")
    if has_cam_pose:
        vp = np.isfinite(cam_tx)
        if vp.sum() > 1:
            nx = (cam_tx[vp] - cam_tx[vp].min()) / safe_range(cam_tx[vp])
            ny = (cam_ty[vp] - cam_ty[vp].min()) / safe_range(cam_ty[vp])
            ax.plot(nx, ny, "m-", lw=1.5, alpha=0.8, label="Camera")
    ax.set(xlabel="Norm X", ylabel="Norm Y", title="Path Overlay (normalised)")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    # Row 1 — X / Y / Z vs time ────────────────────────────────────────────────
    cam_cols = [cam_tx, cam_ty, cam_tz] if has_cam_pose else [None, None, None]
    for i, (col, label, color, cam_col) in enumerate([
        (0, "X", "r", cam_cols[0]),
        (1, "Y", "m", cam_cols[1]),
        (2, "Z", "g", cam_cols[2]),
    ]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t, pos[:, col], color=color, lw=1.5, label="GRBL")
        if cam_col is not None and len(cam_col):
            vp = np.isfinite(cam_col)
            if vp.sum() > 0:
                ax.plot(cam_t[vp], cam_col[vp], "k--", lw=1, alpha=0.7, label="Camera")
        ax.set(xlabel="Time (s)", ylabel=f"{label} (mm)", title=f"{label} Position")
        ax.legend(fontsize=8); ax.grid(True)
        pub_fig(ax, fig)

    # ── Position error stats (camera vs GRBL) ─────────────────────────────────
    err_labels  = ["X", "Y", "Z"]
    err_avg     = []
    err_max     = []
    err_rms     = []
    cam_aligned = [cam_tx, cam_ty, cam_tz]
    for ci, grbl_col in enumerate([pos[:, 0], pos[:, 1], pos[:, 2]]):
        cc = cam_aligned[ci]
        if has_cam_pose and cc is not None:
            vp = np.isfinite(cc)
            if vp.sum() > 1:
                ct_valid  = cam_t[vp]
                grbl_interp = np.interp(ct_valid, t, grbl_col)
                err = np.abs(cc[vp] - grbl_interp)
                err_avg.append(float(err.mean()))
                err_max.append(float(err.max()))
                err_rms.append(float(np.sqrt((err**2).mean())))
            else:
                err_avg.append(np.nan); err_max.append(np.nan); err_rms.append(np.nan)
        else:
            err_avg.append(np.nan); err_max.append(np.nan); err_rms.append(np.nan)

    print("\nPosition Error (camera vs GRBL):")
    for i, lbl in enumerate(err_labels):
        print(f"  {lbl}  Avg={err_avg[i]:.2f}  Max={err_max[i]:.2f}  RMS={err_rms[i]:.2f}  (mm)")

    # Row 2 — velocity, acceleration, error bar chart ─────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    if len(speed):
        ax.plot(t_vel, speed, "b-", lw=1.5, label="GRBL")
    if len(spd_c):
        ax.plot(tv_c, spd_c, "m--", lw=1, alpha=0.8, label="Camera XY")
    ax.set(xlabel="Time (s)", ylabel="Speed (mm/s)", title="Velocity")
    ax.legend(fontsize=8); ax.grid(True)
    pub_fig(ax, fig)

    ax = fig.add_subplot(gs[2, 1])
    if len(accel):
        ax.plot(t_acc, accel, "c-", lw=1.5)
    ax.set(xlabel="Time (s)", ylabel="Accel (mm/s²)", title="GRBL Acceleration")
    ax.grid(True)
    pub_fig(ax, fig)

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


    # ── Save CSVs ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    vel_col = nearest_value(t, t_vel, speed)
    acc_col = nearest_value(t, t_acc, accel)

    grbl_file = f"grbl_{ts}.csv"
    with open(grbl_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "x_mm", "y_mm", "z_mm", "velocity_mms", "acceleration_mms2"])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.4f}",
                f"{pos[i, 0]:.4f}", f"{pos[i, 1]:.4f}", f"{pos[i, 2]:.4f}",
                f"{vel_col[i]:.4f}" if np.isfinite(vel_col[i]) else "",
                f"{acc_col[i]:.4f}" if np.isfinite(acc_col[i]) else "",
            ])
    print(f"GRBL data  -> {grbl_file}")

    if has_cam:
        cam_file = f"camera_{ts}.csv"
        with open(cam_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "cx_px", "cy_px", "tx_mm", "ty_mm", "tz_mm"])
            for row in cam_arr:
                w.writerow([
                    f"{row[0]:.4f}",
                    f"{row[1]:.2f}", f"{row[2]:.2f}",
                    f"{row[3]:.4f}" if np.isfinite(row[3]) else "",
                    f"{row[4]:.4f}" if np.isfinite(row[4]) else "",
                    f"{row[5]:.4f}" if np.isfinite(row[5]) else "",
                ])
        print(f"Camera data -> {cam_file}")
