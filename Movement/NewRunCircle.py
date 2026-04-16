import serial
import time
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


#  CONFIGURATION
PORT       = "COM3"
BAUD       = 115200
FEED       = 1200     # mm/min
Z_SAFE     = 73.9394  # mm
TARGET_FPS = 80       # samples per second

# Circle parameters
CX      = 140.0    # circle centre X (mm)
CY      = -140.0   # circle centre Y (mm)
RADIUS  = 70.0    # radius (mm)
# Start point is at (CX + RADIUS, CY) — rightmost point of the circle
START_X = CX + RADIUS   # = 70.0
START_Y = CY             # = -50.0


EMA_ALPHA = 0.3   # smoothing factor: 0 = max smooth, 1 = raw passthrough

# EMA state per axis (X, Y, Z); None = not yet initialized
ema_pos = [None, None, None]


#  CONNECT & SETUP
s = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
s.write(b"\r\n\r\n")   # wake GRBL
time.sleep(2)
s.reset_input_buffer()

print("Setting up machine...")
s.write(b"$X\n"); time.sleep(1)                    # unlock
s.write(b"$H\n"); time.sleep(5)                    # home
s.write(b"G21 G90 G92 X0 Y0 Z0\n"); time.sleep(1) # mm, absolute, zero here


#  INITIALIZE
positions  = []   # list of [x, y, z] smoothed values
timestamps = []   # list of elapsed-time floats
start_time = time.time()
interval   = 1.0 / TARGET_FPS   # seconds between polls

print(f"Starting circle  (centre=({CX},{CY}) mm  r={RADIUS} mm  @ {FEED} mm/min,  tracking at {TARGET_FPS} fps)")


#  MOVE SEQUENCE
move_sequence = [
    (f"G1 Z-{Z_SAFE:.4f} F{FEED}",                               "Plunge Z"),
    (f"G1 X{START_X:.3f} Y{START_Y:.3f} F{FEED}",               "-> Circle start"),
    (f"G2 X{START_X:.3f} Y{START_Y:.3f} I{-RADIUS:.3f} J0.000 F{FEED}", "Circle CW"),
    (f"G1 X0 Y0 F{FEED}",                                        "<- Home"),
    (f"G1 Z{Z_SAFE + 5:.4f} F{FEED}",                            "Retract Z"),
]

for cmd, label in move_sequence:
    print(f"  {label}")
    s.write((cmd + "\n").encode())

    # Tracking loop for this move 
    deadline   = time.time() + 90.0
    in_motion  = False
    idle_count = 0   # require consecutive Idle responses to avoid early breaks

    s.reset_input_buffer()  # clear stale bytes before polling

    while time.time() < deadline:
        poll_start = time.time()

        s.write(b"?\n")
        time.sleep(0.02)   # give GRBL slightly more time to respond

        # drain all available bytes into one response string
        raw = ""
        while s.in_waiting:
            try:
                raw += s.readline().decode(errors="ignore").strip()
            except Exception:
                pass

        if raw:
            match = re.search(r"WPos:([\d\.\-]+),([\d\.\-]+),([\d\.\-]+)", raw)

            if match:
                raw_pos = [float(match.group(1)),
                           float(match.group(2)),
                           float(match.group(3))]

                now = time.time() - start_time
                dt  = (now - timestamps[-1]) if timestamps else interval

                # ── EMA smoothing for each axis ──
                for ax in range(3):
                    if ema_pos[ax] is None:
                        ema_pos[ax] = raw_pos[ax]
                    else:
                        ema_pos[ax] = EMA_ALPHA * raw_pos[ax] + (1 - EMA_ALPHA) * ema_pos[ax]
                smoothed = list(ema_pos)

                positions.append(smoothed)
                timestamps.append(now)

            if "Run" in raw or "Jog" in raw:
                in_motion = True
                idle_count = 0
            elif "Idle" in raw:
                idle_count += 1
                # require 3 consecutive Idle responses before moving to next step
                # avoids breaking on a brief Idle blip between buffered moves
                if idle_count >= 3:
                    break
            else:
                idle_count = 0

        # Sleep only the remaining time in this frame window
        elapsed   = time.time() - poll_start
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

s.write(b"$H\n"); time.sleep(5)   # final home

total_time = time.time() - start_time
print(f"Done! Complete in {total_time:.1f} seconds")
s.close()


#  ANALYZE DATA
pos = np.array(positions)   # shape (N, 3)
t   = np.array(timestamps)  # shape (N,)

# Remove duplicate timestamps
_, unique_idx = np.unique(t, return_index=True)
pos = pos[unique_idx]
t   = t[unique_idx]

# Velocity (mm/s) — skip near-zero dt, strip non-finite, clip physically impossible speeds
dt    = np.diff(t)
dpos  = np.diff(pos, axis=0)
valid = dt > 1e-6
speed = np.linalg.norm(dpos[valid], axis=1) / dt[valid]
t_vel = t[1:][valid]
MAX_SPEED     = FEED / 60.0 * 3          # 3× feed rate ceiling (mm/s)
keep          = np.isfinite(speed) & (speed < MAX_SPEED)
speed, t_vel  = speed[keep], t_vel[keep]

# Acceleration (mm/s²)
dt_vel        = np.diff(t_vel)
valid2        = dt_vel > 1e-6
accel         = np.diff(speed)[valid2] / dt_vel[valid2]
t_acc         = t_vel[1:][valid2]
keep2         = np.isfinite(accel)
accel, t_acc  = accel[keep2], t_acc[keep2]

# Stats
print("\nStats:")
print(f"  Points captured : {len(t)}")
print(f"  Duration        : {t[-1]:.1f} s")
print(f"  Avg sample rate : {len(t) / t[-1]:.1f} fps")
print(f"  Max speed       : {speed.max():.2f} mm/s")
print(f"  Max |accel|     : {np.abs(accel).max():.2f} mm/s²")


#  PLOT RESULTS  (styled to match MATLAB reference)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(
    f"CNC Run  –  {len(t)} pts  –  {total_time:.1f} s  –  Max: {speed.max():.1f} mm/s",
    fontsize=14, fontweight="bold"
)

# XY toolpath
axes[0, 0].plot(pos[:, 0], pos[:, 1], "b.-", linewidth=1.5, markersize=3)
axes[0, 0].plot(pos[0, 0],  pos[0, 1],  "go", markersize=10, linewidth=2, label="Start")
axes[0, 0].plot(pos[-1, 0], pos[-1, 1], "ro", markersize=10, linewidth=2, label="End")
axes[0, 0].set_xlabel("X (mm)"); axes[0, 0].set_ylabel("Y (mm)")
axes[0, 0].set_title("Toolpath")
axes[0, 0].set_aspect("equal"); axes[0, 0].grid(True)
axes[0, 0].legend(["Path", "Start", "End"], loc="best")

# X, Y, Z positions
for i, (col, label, color) in enumerate([
        (0, "X", "r"), (1, "Y", "m"), (2, "Z", "g")]):
    ax = axes[0, 1] if i == 0 else axes[0, 2] if i == 1 else axes[1, 0]
    ax.plot(t, pos[:, col], color=color, linewidth=1.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{label} (mm)")
    ax.set_title(f"{label} Position"); ax.grid(True)

# Velocity
axes[1, 1].plot(t_vel, speed, "b-", linewidth=1.5)
axes[1, 1].set_xlabel("Time (s)"); axes[1, 1].set_ylabel("Speed (mm/s)")
axes[1, 1].set_title(f"Velocity (max: {speed.max():.1f} mm/s)")
axes[1, 1].set_ylim(0, speed.max() * 1.1); axes[1, 1].grid(True)

# Acceleration
axes[1, 2].plot(t_acc, accel, "c-", linewidth=1.5)
axes[1, 2].set_xlabel("Time (s)"); axes[1, 2].set_ylabel("Accel (mm/s²)")
axes[1, 2].set_title(f"Acceleration (max: ±{np.abs(accel).max():.1f} mm/s²)")
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig("cnc_run.png", dpi=150)
plt.show()


#  SAVE CSV  — velocity/accel mapped back to position timestamps via nearest lookup
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
filename   = f"cnc_run_{timestamp}.csv"

# Build per-row vel/accel by matching each t[i] to nearest t_vel / t_acc sample
def nearest_value(query_times, data_times, data_values):
    """Return data_values sampled at the nearest data_times for each query time."""
    out = np.full(len(query_times), np.nan)
    if len(data_times) == 0:
        return out
    idx = np.searchsorted(data_times, query_times)
    idx = np.clip(idx, 0, len(data_times) - 1)
    out[:] = data_values[idx]
    return out

vel_col = nearest_value(t, t_vel, speed)
acc_col = nearest_value(t, t_acc, accel)

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "x_mm", "y_mm", "z_mm", "velocity_mms", "acceleration_mms2"])
    for i in range(len(t)):
        writer.writerow([
            f"{t[i]:.4f}",
            f"{pos[i, 0]:.4f}",
            f"{pos[i, 1]:.4f}",
            f"{pos[i, 2]:.4f}",
            f"{vel_col[i]:.4f}" if np.isfinite(vel_col[i]) else "",
            f"{acc_col[i]:.4f}" if np.isfinite(acc_col[i]) else "",
        ])

print(f"Saved to {filename}")