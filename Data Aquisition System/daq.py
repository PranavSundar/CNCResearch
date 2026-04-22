"""
NI cDAQ Data Acquisition - Accelerometer and AE Sensor
Acquires data from accelerometer and AE sensor through NI cDAQ (Ethernet connected)
Connected via dual separate modules in cDAQ-9185 CompactDAQ chassis

Dependencies:
  pip install nidaqmx numpy matplotlib scipy

Usage:
  python daq.py
"""

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.fft import fft, fftfreq
import csv
import os
import sys
from datetime import datetime
import time


# CONFIGURATION PARAMETERS

# DAQ Device Configuration (from Measurement & Automation Explorer)
DAQ_DEVICE = "cDAQ9185_22C6F90"                   # Ethernet cDAQ-9185 chassis S/N

# Accelerometer Channel (NI 9215 in Module 1)
ACCEL_MODULE = 1                                   # Accelerometer module slot
ACCEL_CHANNEL = "ai0"                              # Accelerometer channel (port 0)
ACCEL_SAMPLE_RATE = 8192.5243                      # Sampling rate in scans/s (from Recorder config)
ACCEL_RANGE = 10.0                                 # Input range (±10V)

# AE Sensor Channel (NI 9223 in Module 2)
AE_MODULE = 2                                      # AE sensor module slot
AE_CHANNEL = "ai0"                                 # AE sensor channel (port 0)
AE_SAMPLE_RATE = 131147.541                        # Sampling rate in scans/s (from Recorder config)
AE_RANGE = 10.0                                    # Input range (±10V)

# Acquisition Parameters
DURATION = 1.0                                     # Acquisition duration in seconds
TERMINAL_CONFIG = TerminalConfiguration.DIFFERENTIAL  # Differential input configuration

# Analysis Parameters
AE_BURST_THRESHOLD_SIGMA = 3.0                     # AE burst detection threshold (sigma)
AE_BURST_WINDOW_MS = 1.0                           # Window for RMS calculation (ms)

# File Output
OUTPUT_DIR = "Data"                                # Output directory for CSV files
SAVE_NPZ = False                                   # Save as .npz (NumPy archive)

# Plotting
PLOT_ENABLE = True                                 # Enable plotting
PLOT_DPI = 100                                     # Plot resolution


# UTILITY FUNCTIONS

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_output_filepath(filename_prefix="sensor_data"):
    """Generate unique output filepath with timestamp."""
    ensure_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    return os.path.join(OUTPUT_DIR, filename)


def save_data_to_csv(filepath, accel_data, ae_data, time_vec):
    """Save acquired data to CSV file."""
    try:
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['time_s', 'Accelerometer', 'AE_Sensor']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, t in enumerate(time_vec):
                row = {
                    'time_s': f"{t:.6f}",
                    'Accelerometer': f"{accel_data[i]:.6f}",
                    'AE_Sensor': f"{ae_data[i]:.6f}"
                }
                writer.writerow(row)
        
        print(f"[save] Saved: {filepath}")
        return True
    except Exception as e:
        print(f"[error] Failed to save CSV: {e}", file=sys.stderr)
        return False


def detect_ae_bursts(ae_signal, threshold_factor, sample_rate):
    """
    Detect AE events based on RMS threshold.
    
    Args:
        ae_signal: AE sensor data array
        threshold_factor: Number of standard deviations for threshold
        sample_rate: Sampling rate in Hz
    
    Returns:
        Array of sample indices where AE events occur
    """
    window_size = max(1, int(AE_BURST_WINDOW_MS * sample_rate / 1000))
    
    # Calculate RMS using moving window
    ae_rms = np.sqrt(np.convolve(ae_signal**2, np.ones(window_size)/window_size, mode='same'))
    
    threshold = threshold_factor * np.std(ae_signal)
    ae_events = np.where(ae_rms > threshold)[0]
    
    if len(ae_events) > 0:
        print(f"[analysis] Detected {len(ae_events)} potential AE events")
    else:
        print(f"[analysis] No AE events detected above threshold")
    
    return ae_events


def analyze_vibration(accel_data, sample_rate):
    """
    Analyze vibration metrics from acceleration data.
    
    Args:
        accel_data: Acceleration data array
        sample_rate: Sampling rate in Hz
    """
    # Integration to get velocity and displacement
    velocity = integrate.cumtrapz(accel_data, dx=1/sample_rate, initial=0)
    displacement = integrate.cumtrapz(velocity, dx=1/sample_rate, initial=0)
    
    print(f"\n[vibration] Vibration Analysis:")
    print(f"  Peak Acceleration: {np.max(np.abs(accel_data)):.4f} V")
    print(f"  Peak Velocity: {np.max(np.abs(velocity)):.4f} V·s")
    print(f"  Peak Displacement: {np.max(np.abs(displacement)):.4f} V·s²")




def initialize_daq():
    """
    Initialize dual DAQ channels for accelerometer (Module 1) and AE sensor (Module 2).
    Each module has independent sample rate configuration.
    
    Returns:
        Tuple of (task, accel_samples, ae_samples) or (None, None, None) on error
    """
    try:
        task = nidaqmx.Task()
        
        # Add accelerometer channel (NI 9215, Module 1, slower sample rate)
        accel_phys_ch = f"{DAQ_DEVICE}Mod{ACCEL_MODULE}/{ACCEL_CHANNEL}"
        accel_ch = task.ai_channels.add_ai_voltage_chan(
            physical_channel=accel_phys_ch,
            min_val=-ACCEL_RANGE,
            max_val=ACCEL_RANGE,
            terminal_config=TERMINAL_CONFIG,
            name_to_assign_to_channel="Accelerometer"
        )
        print(f"[init] Added Accelerometer: {accel_phys_ch}")
        print(f"       NI 9215 Module, {ACCEL_SAMPLE_RATE:.1f} scans/s, Differential")
        
        # Add AE sensor channel (NI 9223, Module 2, faster sample rate)
        ae_phys_ch = f"{DAQ_DEVICE}Mod{AE_MODULE}/{AE_CHANNEL}"
        ae_ch = task.ai_channels.add_ai_voltage_chan(
            physical_channel=ae_phys_ch,
            min_val=-AE_RANGE,
            max_val=AE_RANGE,
            terminal_config=TERMINAL_CONFIG,
            name_to_assign_to_channel="AE_Sensor"
        )
        print(f"[init] Added AE Sensor: {ae_phys_ch}")
        print(f"       NI 9223 Module, {AE_SAMPLE_RATE:.1f} scans/s, Differential")
        
        # Calculate samples for each sensor
        accel_samples = int(DURATION * ACCEL_SAMPLE_RATE)
        ae_samples = int(DURATION * AE_SAMPLE_RATE)
        
        # Configure timing using the faster sample rate (AE sensor)
        # Both channels will acquire simultaneously at the AE sensor rate
        task.timing.cfg_samp_clk_timing(
            rate=AE_SAMPLE_RATE,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=ae_samples
        )
        
        print(f"[config] Acquisition Duration: {DURATION} second(s)")
        print(f"[config] Accelerometer: {accel_samples:,} samples")
        print(f"[config] AE Sensor: {ae_samples:,} samples")
        print(f"[config] Master Clock Rate: {AE_SAMPLE_RATE:.1f} scans/s")
        
        return task, accel_samples, ae_samples
    
    except Exception as e:
        print(f"[error] Failed to initialize DAQ: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, None, None


def acquire_dual_sensor_data(task, accel_samples, ae_samples):
    """
    Acquire data from both accelerometer and AE sensor simultaneously.
    
    Args:
        task: Configured nidaqmx.Task
        accel_samples: Number of accelerometer samples to acquire
        ae_samples: Number of AE sensor samples to acquire
    
    Returns:
        Tuple of (accel_data, ae_data, accel_time_vec, ae_time_vec) or (None, None, None, None) on error
    """
    try:
        print(f"\n[acq] Starting data acquisition ({DURATION} second(s))...")
        start_time = time.time()
        
        task.start()
        # Read maximum samples (AE sensor has more samples due to higher rate)
        raw_data = task.read(number_of_samples_per_channel=ae_samples, timeout=DURATION+5)
        elapsed = time.time() - start_time
        
        # Convert to numpy array and separate channels
        data_array = np.array(raw_data).T
        
        # Extract channels
        accel_data = data_array[:accel_samples, 0]  # Trim to accelerometer sample count
        ae_data = data_array[:ae_samples, 1]
        
        # Generate time vectors for each sensor
        accel_time_vec = np.arange(accel_samples) / ACCEL_SAMPLE_RATE
        ae_time_vec = np.arange(ae_samples) / AE_SAMPLE_RATE
        
        print(f"[acq] Acquisition complete ({elapsed:.2f}s elapsed)")
        print(f"[acq] Accelerometer samples: {len(accel_data):,}")
        print(f"[acq] AE Sensor samples: {len(ae_data):,}")
        
        return accel_data, ae_data, accel_time_vec, ae_time_vec
    
    except Exception as e:
        print(f"[error] Failed to acquire data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    """Main data acquisition and analysis routine."""
    print("\n" + "="*70)
    print("  NI cDAQ Data Acquisition - Accelerometer and AE Sensor")
    print("  Ethernet-Connected cDAQ-9185 CompactDAQ Chassis")
    print("="*70 + "\n")
    
    task = None
    
    try:
        # Initialize DAQ
        print("[init] Initializing dual sensor DAQ...")
        task, accel_samples, ae_samples = initialize_daq()
        if task is None:
            return False
        
        # Acquire data
        accel_data, ae_data, accel_time_vec, ae_time_vec = acquire_dual_sensor_data(task, accel_samples, ae_samples)
        task.stop()
        
        if accel_data is None:
            return False
        
        # Data statistics
        print(f"\n[stats] --- Data Statistics ---")
        print(f"Accelerometer - Mean: {np.mean(accel_data):.4f} V, "
              f"RMS: {np.sqrt(np.mean(accel_data**2)):.4f} V, "
              f"Peak: {np.max(np.abs(accel_data)):.4f} V")
        print(f"AE Sensor - Mean: {np.mean(ae_data):.4f} V, "
              f"RMS: {np.sqrt(np.mean(ae_data**2)):.4f} V, "
              f"Peak: {np.max(np.abs(ae_data)):.4f} V")
        
        # Visualization using the shorter accelerometer time vector for consistency
        if PLOT_ENABLE:
            print(f"\n[plot] Generating time and frequency domain plots...")
            # For plotting, we'll plot the full AE data but accelerometer is shorter
            # So we'll create subplots that show each appropriately
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PLOT_DPI)
            
            # Time domain - accelerometer
            axes[0, 0].plot(accel_time_vec, accel_data, 'b-', linewidth=0.8)
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude (V)')
            axes[0, 0].set_title(f'Accelerometer - Time Domain ({len(accel_data):,} samples @ {ACCEL_SAMPLE_RATE:.1f} Hz)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Time domain - AE sensor
            axes[0, 1].plot(ae_time_vec, ae_data, 'r-', linewidth=0.5)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude (V)')
            axes[0, 1].set_title(f'AE Sensor - Time Domain ({len(ae_data):,} samples @ {AE_SAMPLE_RATE:.1f} Hz)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Frequency domain - accelerometer
            n_accel = len(accel_data)
            n_fft_accel = 2**int(np.ceil(np.log2(n_accel)))
            accel_fft = fft(accel_data, n_fft_accel)
            accel_psd = (np.abs(accel_fft[:n_fft_accel//2+1])**2) / (ACCEL_SAMPLE_RATE * n_accel)
            accel_psd[1:-1] *= 2
            freq_accel = np.fft.fftfreq(n_fft_accel, 1/ACCEL_SAMPLE_RATE)[:n_fft_accel//2+1]
            
            axes[1, 0].semilogy(freq_accel, accel_psd, 'b-', linewidth=0.8)
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('PSD (V²/Hz)')
            axes[1, 0].set_title('Accelerometer - Frequency Domain')
            axes[1, 0].set_xlim([0, ACCEL_SAMPLE_RATE/2])
            axes[1, 0].grid(True, alpha=0.3, which='both')
            
            # Frequency domain - AE sensor
            n_ae = len(ae_data)
            n_fft_ae = 2**int(np.ceil(np.log2(n_ae)))
            ae_fft = fft(ae_data, n_fft_ae)
            ae_psd = (np.abs(ae_fft[:n_fft_ae//2+1])**2) / (AE_SAMPLE_RATE * n_ae)
            ae_psd[1:-1] *= 2
            freq_ae = np.fft.fftfreq(n_fft_ae, 1/AE_SAMPLE_RATE)[:n_fft_ae//2+1]
            
            axes[1, 1].semilogy(freq_ae, ae_psd, 'r-', linewidth=0.8)
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('PSD (V²/Hz)')
            axes[1, 1].set_title('AE Sensor - Frequency Domain')
            axes[1, 1].set_xlim([0, AE_SAMPLE_RATE/2])
            axes[1, 1].grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            plt.show()
            print("[plot] Time and frequency domain plots displayed")
        
        # Advanced analysis
        print(f"\n[analysis] Performing advanced analysis...")
        ae_events = detect_ae_bursts(ae_data, AE_BURST_THRESHOLD_SIGMA, AE_SAMPLE_RATE)
        analyze_vibration(accel_data, ACCEL_SAMPLE_RATE)
        
        # Save data to CSV
        print(f"\n[save] Writing data to CSV...")
        csv_file = create_output_filepath("sensor_data")
        # For CSV, use the shorter accelerometer vector length
        save_data_to_csv(csv_file, accel_data, ae_data[:len(accel_data)], accel_time_vec)
        
        # Optional: Save as NPZ archive
        save_opt = input("\nSave data as NumPy archive? (y/n): ").lower()
        if save_opt == 'y':
            npz_file = create_output_filepath("sensor_data").replace('.csv', '.npz')
            np.savez(npz_file, 
                    accel_data=accel_data, 
                    ae_data=ae_data,
                    accel_time_vec=accel_time_vec,
                    ae_time_vec=ae_time_vec,
                    accel_sample_rate=ACCEL_SAMPLE_RATE,
                    ae_sample_rate=AE_SAMPLE_RATE)
            print(f"[save] NPZ archive saved: {npz_file}")
        
        print(f"\n[done] Acquisition and analysis complete!\n")
        return True
    
    except nidaqmx.errors.DaqError as e:
        print(f"\n[error] DAQmx Error: {e}", file=sys.stderr)
        return False
    
    except Exception as e:
        print(f"\n[error] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Always close task properly
        if task is not None:
            try:
                task.stop()
                task.close()
                print("[cleanup] DAQ task closed")
            except Exception as e:
                print(f"[warning] Error closing task: {e}", file=sys.stderr)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
