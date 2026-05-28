# -*- coding: utf-8 -*-
"""
Real-time respiration-rate demo - MATLAB-equivalent implementation
Matches Siheng Li's Method (2021 & 2024) with 0.03m displacement threshold

Behaviour:
- Radar frames saved at full FPS.
- Respiration rate (RR) computed once every θ seconds (30 s window) via ACF-based estimation.
- Between computations, RR column is marked as "NA".
- Handles USB disconnection gracefully by exiting the process
"""

import sys, os, time, math, csv, collections, multiprocessing as mp
import numpy as np
from datetime import datetime

# --- NEW IMPORT ---
import argparse

from pymoduleconnector import ModuleConnector
from scipy.signal import savgol_filter, find_peaks
from scipy.fftpack import fft


class RespirationEstimator:
    # ... (This class is unchanged)
    def __init__(self, radar_resolution, fs, bed_loc=1.5, theta=30, beta=0.4,
                 min_bin_setting=20, max_bin_setting=120): # <-- MODIFIED: Added new parameters
        self.rad_res = radar_resolution
        self.fs      = fs
        self.theta   = theta        # analysis window [s]
        self.delta   = 15           # step size [s] - compute every 15s
        self.beta    = beta         # stationarity/ACF threshold
        self.c       = 3e8          # speed of light [m/s]
        self.fc      = 7.3e9        # carrier [Hz]

        # locate chest-range bins
        # bed_idx      = int(round(bed_loc / self.rad_res)) # <-- Old logic (bed_loc still passed but unused for bins)
        # self.min_bin = max(0, bed_idx - 20)              # <-- Old logic
        # self.max_bin = min(187, bed_idx + 20)            # <-- Old logic

        # --- MODIFIED: New, easily modifiable bin range ---
        self.min_bin = min_bin_setting
        self.max_bin = max_bin_setting
        # Ensure bins are within valid radar range (0-187 based on original code)
        self.min_bin = max(0, self.min_bin)
        self.max_bin = min(187, self.max_bin) # Assuming 187 is max possible bin
        # --- End of new logic ---

        # buffer holds (timestamp, complex_frame)
        self.buffer  = collections.deque()

        # gate compute to once per δ-second window
        self.last_compute_ts = None

        # Add diagnostic counters
        self.frame_count = 0
        self.compute_count = 0

        print(f"[RespEst]  fs={self.fs} Hz, θ={self.theta}s, δ={self.delta}s, β={self.beta}, "
              f"rad_res={self.rad_res:.4f} m, bins={self.min_bin}–{self.max_bin}")

    def update(self, timestamp, complex_frame):
        """Add one frame; return new RR only every θ seconds, else NaN."""
        self.frame_count += 1


        self.buffer.append((timestamp, complex_frame))
        EPS = 1.0 / self.fs

        # Trim buffer if it grows beyond θ + EPS
        while (self.buffer and
               (self.buffer[-1][0] - self.buffer[0][0] > self.theta + EPS)):
            self.buffer.popleft()

        span = self.buffer[-1][0] - self.buffer[0][0]
        if span + EPS < self.theta:
            return np.nan

        # Skip until next computation time (every δ seconds)
        if (self.last_compute_ts is not None and
            timestamp - self.last_compute_ts < self.delta - EPS):
            return np.nan

        # Compute with ACF-based estimator
        mats      = np.stack([f for _, f in self.buffer], axis=0)   # (T,B)
        radar_mat = mats.T                                         # (B,T)

        self.compute_count += 1
        # Use sys.stdout.write to avoid interfering with the updating distance line
        # sys.stdout.write('\r' + ' ' * 50 + '\r')
        # print(f"\n[RespEst] === Compute #{self.compute_count} at t={timestamp:.2f} ===")

        rr = self._compute_rr(radar_mat)
        self.last_compute_ts = timestamp
        return rr

    def _compute_rr(self, radar_mat):
        """MATLAB-equivalent respiratory-rate estimation."""
        num_bins, num_slow = radar_mat.shape

        # Check input data validity
        data_mag = np.abs(radar_mat)
        # print(f"[RespEst]    Input data: shape={radar_mat.shape}, "
        #       f"mag_mean={data_mag.mean():.6f}, mag_std={data_mag.std():.6f}")

        if data_mag.max() < 1e-10:
            # print("[RespEst]    WARNING: Near-zero input data detected!")
            return np.nan

        # 1) BSL FILTERING (Baseline removal)
        alpha = 0.9
        clutter_est = np.zeros_like(radar_mat, dtype=np.complex128)
        target_est = np.zeros_like(radar_mat, dtype=np.complex128)

        for k in range(num_slow):
            if k == 0:
                clutter_est[:, k] = alpha * radar_mat[:, k]
            else:
                clutter_est[:, k] = alpha * radar_mat[:, k] + (1 - alpha) * clutter_est[:, k-1]
            target_est[:, k] = radar_mat[:, k] - clutter_est[:, k]

        # print(f"[RespEst]    BSL filtering done (α={alpha})")

        # 2) Crop range bins
        target_est_cut = target_est[self.min_bin:self.max_bin+1, :]

        # 3) Doppler FFT
        Nfft = 2 ** int(np.ceil(np.log2(num_slow)))  # nextpow2
        doppler_fft = np.fft.fft(target_est_cut, Nfft, axis=1)
        doppler_mag = np.abs(doppler_fft[:, :Nfft//2])
        f_axis = np.arange(Nfft//2) * (self.fs / Nfft)

        # 4) Locate bin with max energy in breathing frequency range (0.1-0.85 Hz)
        freq_mask = (f_axis >= 0.1) & (f_axis <= 0.85)
        doppler_mag_filtered = doppler_mag[:, freq_mask]

        # Find max value in each range bin within frequency range
        peak_values = np.max(doppler_mag_filtered, axis=1)

        # --- MODIFIED LOGIC: START ---
        # 5) Extract respiratory signal from a weighted average of the top N bins
        num_bins_to_combine = 3
        # Get indices of the top N bins by energy
        top_indices_local = np.argsort(peak_values)[-num_bins_to_combine:]
        top_energies = peak_values[top_indices_local]
        top_indices_global = top_indices_local + self.min_bin

        # Calculate weights based on energy (normalized)
        total_energy = np.sum(top_energies)
        weights = top_energies / total_energy if total_energy > 0 else np.ones(num_bins_to_combine) / num_bins_to_combine

        # Create a weighted sum of the complex signals from the top bins
        weighted_signal = np.zeros(num_slow, dtype=np.complex128)
        for i, global_idx in enumerate(top_indices_global):
            weighted_signal += radar_mat[global_idx, :] * weights[i]

        resp_signal = weighted_signal
        # print(f"[RespEst]    Doppler FFT bin selection → weighted avg of bins: {top_indices_global}")
        # --- MODIFIED LOGIC: END ---

        # 6) Denoise using Savitzky-Golay filter
        smoothed_signal = savgol_filter(resp_signal.real, 21, 3) + \
                          1j * savgol_filter(resp_signal.imag, 21, 3)

        # 7) Extract phase
        phase_denoised = np.unwrap(np.angle(smoothed_signal))
        amplitude_denoised = np.abs(smoothed_signal)

        # 8) Calculate displacement
        max_phase = np.max(phase_denoised)
        min_phase = np.min(phase_denoised)
        delta_phase = abs(max_phase - min_phase)
        window_depth = (delta_phase * self.c) / (4 * np.pi * self.fc)

        # print(f"[RespEst]    Δφ={delta_phase:5.2f} rad ⇒ window_depth={window_depth*1000:5.2f} mm")

        # 9) Check stationarity - using 0.03m threshold
        if window_depth > 0.025:
            # print("\n[RespEst]    window depth >25 mm → reject (motion detected)")
            return np.nan

        # 10) Compute normalized autocorrelation
        # --- MODIFIED LOGIC: START ---
        # Define lag range for normal breathing based on physiological frequencies (0.1 - 0.85 Hz)
        # This fixes the original issue of being capped at 30 BPM.
        min_lag = math.ceil((1 / 0.85) * self.fs)   # Corresponds to max freq of 0.85 Hz (~51 BPM)
        max_lag = math.floor((1 / 0.1) * self.fs)   # Corresponds to min freq of 0.1 Hz (6 BPM)
        # --- MODIFIED LOGIC: END ---

        # Compute normalized autocorrelation using numpy
        def normalized_xcorr(x):
            """Compute normalized autocorrelation matching MATLAB's xcorr with 'coeff' option"""
            x = x - np.mean(x)
            # Normalize the signal
            norm_factor = np.sqrt(np.sum(x * np.conj(x)))
            if norm_factor > 0:
                x = x / norm_factor
            # Compute correlation
            corr = np.correlate(x, x, mode='full')
            # Return only positive lags
            return corr[len(corr)//2:]

        acf_phase = normalized_xcorr(phase_denoised)
        acf_amp = normalized_xcorr(amplitude_denoised)

        # Restrict to lags in our physiological range
        acf_phase_crop = acf_phase[min_lag:max_lag+1]
        acf_amp_crop = acf_amp[min_lag:max_lag+1]

        # Find maximum correlation
        max_phase = np.max(acf_phase_crop) if len(acf_phase_crop) > 0 else 0
        max_amp = np.max(acf_amp_crop) if len(acf_amp_crop) > 0 else 0
        max_acf = max(max_phase, max_amp)

        # print(f"[RespEst]    ACF maxima — phase={max_phase:4.2f}  amp={max_amp:4.2f}  "
        #       f"(threshold β={self.beta})")

        if max_acf < self.beta:
            # print(f"\n[RespEst]    max correlation {max_acf:.2f} < β={self.beta} → reject")
            return np.nan

        # --- MODIFIED LOGIC: START ---
        # 11) Determine which signal to use and find the FIRST significant peak for robustness
        target_acf = None
        if max_phase > max_amp:
            target_acf = acf_phase_crop
            # print("[RespEst]    Using phase ACF")
        else:
            target_acf = acf_amp_crop
            # print("[RespEst]    Using amplitude ACF")

        # Use find_peaks instead of argmax to avoid selecting harmonics
        # A peak must be at least 80% of the threshold to be considered significant
        peaks, _ = find_peaks(target_acf, height=self.beta * 0.8)

        if len(peaks) == 0:
            # print("\n[RespEst]    ACF signal is strong, but no distinct peaks found → reject")
            return np.nan

        # The first peak corresponds to the fundamental breathing period
        lag_idx = peaks[0]
        # --- MODIFIED LOGIC: END ---

        # Convert lag index to respiration rate
        N = lag_idx + min_lag  # actual lag in samples
        RR = 60 * self.fs / N

        # print(f"[RespEst]    lag={N} samples ⇒ RR={RR:5.2f} bpm")
        return RR


class CollectionThreadX4MP(mp.Process):
    # comport
    def __init__(self, stopEvent, radarSettings, baseband=True,
                 fs=17, radarPort='COM4', dataQueue=None):
        super().__init__()
        self.exit          = mp.Event()
        self.stopEvent     = stopEvent
        self.radarDataQ    = dataQueue
        self.radarPort     = radarPort
        self.radarSettings = radarSettings
        self.fs            = fs
        self.baseband      = baseband
        self.daemon        = True  # Make thread daemon so it dies with main
        self.error_count   = 0
        self.max_errors    = 5
        print('[Radar]  collection thread initialised')

    def run(self):
        print('[Radar]  initialising module …')
        try:
            self.mc    = ModuleConnector(self.radarPort) # <-- Removed log_file=None
            self.radar = self.mc.get_xep()

            # --- MOVED SETUP INSIDE TRY BLOCK ---
            # apply settings
            self.radar.x4driver_set_iterations(self.radarSettings['Iterations'])
            self.radar.x4driver_set_dac_min(self.radarSettings['DACMin'])
            self.radar.x4driver_set_dac_max(self.radarSettings['DACMax'])
            self.radar.x4driver_set_pulses_per_step(self.radarSettings['PulsesPerStep'])
            self.radar.x4driver_set_frame_area(self.radarSettings['FrameStart'],
                                               self.radarSettings['FrameStop'])
            self.radar.x4driver_set_downconversion(1 if self.baseband else 0)
            self.radar.x4driver_set_fps(self.fs)

            self.clearBuffer()
            print('[Radar]  started data collection')
            # --- END OF MOVED CODE ---

        except Exception as e:
            print(f'[Radar] ERROR: Failed to connect or configure radar on {self.radarPort}: {e}')
            sys.exit(1)  # Exit with error code to trigger watchdog

        frame_count = 0
        while not self.exit.is_set() and not self.stopEvent.is_set():
            try:
                ts    = time.time()
                frame = self.radar.read_message_data_float().get_copy()
                frame_count += 1

                self.radarDataQ.put((ts, frame))
                self.error_count = 0  # Reset error count on successful read

            except Exception as e:
                # Check if it's a USB/IO error
                error_str = str(e).lower()
                if any(x in error_str for x in ['i/o operation', 'aborted', 'io_read',
                                               'overlapped', 'usb', 'reader is bailing']):
                    print(f'\n[Radar] FATAL: USB disconnection detected: {e}')
                    print('[Radar] Exiting to allow watchdog restart...')
                    sys.exit(1)  # Exit with error code

                # For other errors, count them
                self.error_count += 1
                print(f'\n[Radar] ERROR reading frame ({self.error_count}/{self.max_errors}): {e}')

                if self.error_count >= self.max_errors:
                    print('\n[Radar] FATAL: Too many consecutive errors, exiting...')
                    sys.exit(1)  # Exit with error code

                time.sleep(0.1)

        # Clean shutdown
        try:
            self.mc.close()
        except:
            pass
        print('[Radar]  collection thread stopped cleanly')

    def clearBuffer(self):
        while self.radar.peek_message_data_float():
            _ = self.radar.read_message_data_float()

    def shutdown(self):
        self.exit.set()


class Main:
    # comport
    def __init__(self, shared_data_dir):
        self.shared_data_dir = shared_data_dir 
        self.port     = 'COM4'
        self.fs       = 17
        
        # --- MODIFIED: Use standard YYYY-MM-DD format for filenames ---
        self.today    = time.strftime('%Y-%m-%d')

        self.min_bin_setting = 20
        self.max_bin_setting = 120
        
        self.csv_buffer = []
        self.CSV_BUFFER_SIZE = 100
        
        self.bpm_csv_path = None 
        self.bpm_csv_buffer = []
        self.BPM_CSV_BUFFER_SIZE = 50 
        
        self.dist_csv_path = None 
        self.dist_csv_buffer = []
        self.DIST_CSV_BUFFER_SIZE = 50 
        
        self.make_radar_settings()
        self.setup_log_files() 
        res            = self.radarSettings['RADAR_RESOLUTION']
        
        self.respEst   = RespirationEstimator(res, self.fs, bed_loc=1.5, theta=30, beta=0.4,
                                              min_bin_setting=self.min_bin_setting, 
                                              max_bin_setting=self.max_bin_setting)
        
        self.dataQ       = mp.Queue()
        self.stopEvent   = mp.Event()
        self.radarThread = CollectionThreadX4MP(
            self.stopEvent, self.radarSettings, True,
            fs=self.fs, radarPort=self.port, dataQueue=self.dataQ)
        
        self.last_distance_print_ts = 0
        self.latest_distance = np.nan
        
        self.background_frame = None
        self.background_avg_count = 0
        self.INITIAL_BG_FRAMES = 50

    def make_radar_settings(self):
        # ... (This method is unchanged)
        self.radarSettings = {
            'Iterations'      : 32,
            'DACMin'          : 949,
            'DACMax'          : 1100,
            'PulsesPerStep'   : 52,
            'FrameStart'      : 0.5,
            'FrameStop'       : 9.75,
            'RADAR_RESOLUTION': 51.8617 / 1000.0,
            'RadarType'       : 'X4'
        }

    def setup_log_files(self): 
        # ... (This method is unchanged)
        self.dir = os.path.join(self.shared_data_dir, 'logs', 'Novelda_Data')
        os.makedirs(self.dir, exist_ok=True)
        
        # --- MODIFIED: Append date to filenames (file + date) ---
        self.csv_path = os.path.join(self.dir, f'radar_data_log_{self.today}.csv')
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                num_bins = int((self.radarSettings['FrameStop'] - self.radarSettings['FrameStart'])
                              / self.radarSettings['RADAR_RESOLUTION'])
                hdr = (['unix_timestamp'] +
                       [f'frame_{i}' for i in range(1, num_bins*2 + 1)] +
                       ['distance', 'respiration_rate'])
                csv.writer(f).writerow(hdr)
        
        # --- MODIFIED: Append date to filenames ---
        self.bpm_csv_path = os.path.join(self.dir, f'bpm_{self.today}.csv')
        if not os.path.exists(self.bpm_csv_path):
            with open(self.bpm_csv_path, 'w', newline='') as f:
                hdr = ['unix_timestamp', 'bpm']
                csv.writer(f).writerow(hdr)

        # --- MODIFIED: Append date to filenames ---
        self.dist_csv_path = os.path.join(self.dir, f'distance_{self.today}.csv')
        if not os.path.exists(self.dist_csv_path):
            with open(self.dist_csv_path, 'w', newline='') as f:
                hdr = ['unix_timestamp', 'distance_m']
                csv.writer(f).writerow(hdr)


    def _flush_csv_buffer(self):
        # ... (This method is unchanged)
        if not self.csv_buffer:
            return
        try:
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerows(self.csv_buffer)
            self.csv_buffer.clear()
        except Exception as e:
            print(f"\n[ERROR] Could not write to CSV file: {e}")

    def _flush_bpm_csv_buffer(self):
        # ... (This method is unchanged)
        if not self.bpm_csv_buffer:
            return
        try:
            with open(self.bpm_csv_path, 'a', newline='') as f:
                csv.writer(f).writerows(self.bpm_csv_buffer)
            self.bpm_csv_buffer.clear()
        except Exception as e:
            print(f"\n[ERROR] Could not write to BPM CSV file: {e}")

    def _flush_dist_csv_buffer(self):
        # ... (This method is unchanged)
        if not self.dist_csv_buffer:
            return
        try:
            with open(self.dist_csv_path, 'a', newline='') as f:
                csv.writer(f).writerows(self.dist_csv_buffer)
            self.dist_csv_buffer.clear()
        except Exception as e:
            print(f"\n[ERROR] Could not write to Distance CSV file: {e}")

    def rollover(self):
        # --- MODIFIED: Active rollover logic ---
        now = time.strftime('%Y-%m-%d')
        if now != self.today:
            self._flush_csv_buffer()
            self._flush_bpm_csv_buffer() 
            self._flush_dist_csv_buffer() 
            print(f"\n[Main]  date rollover {self.today} → {now}")
            self.today = now
            self.setup_log_files() # This will create new file paths with the new date

    def radar_to_np(self, frame):
        # ... (This method is unchanged)
        a = np.asarray(frame, dtype=np.float64)
        n = len(a)
        if n == 0:
            print("[Main] WARNING: Empty frame received!")
            return np.zeros(1, dtype=np.complex128)
        return a[:n//2] + 1j * a[n//2:]

    def _calculate_distance(self, complex_frame):
        # ... (This method is unchanged)
        if len(complex_frame) <= 1:
            return np.nan

        if self.background_avg_count < self.INITIAL_BG_FRAMES:
            if self.background_frame is None:
                self.background_frame = complex_frame.copy()
            else:
                self.background_frame = (self.background_frame * self.background_avg_count + complex_frame) / (self.background_avg_count + 1)
            self.background_avg_count += 1
            return np.nan

        frame_no_bg = complex_frame - self.background_frame
        
        start_bin = 2
        peak_bin_index_local = np.argmax(np.abs(frame_no_bg[start_bin:]))
        peak_bin_index = peak_bin_index_local + start_bin

        frame_start = self.radarSettings['FrameStart']
        radar_res = self.radarSettings['RADAR_RESOLUTION']
        distance = frame_start + (peak_bin_index * radar_res)
        
        return distance

    def save_frame(self, buf):
        # --- THIS METHOD IS MODIFIED ---
        ts, raw = buf
        complex_frame = self.radar_to_np(raw)

        distance = self._calculate_distance(complex_frame)
        self.latest_distance = distance
        
        rr = self.respEst.update(ts, complex_frame)

        # --- STRINGS FOR FULL LOG + CONSOLE (floats, full timestamp) ---
        dist_str = f"{distance:.3f}" if np.isfinite(distance) else "NA"
        rr_str = f"{rr:.2f}" if np.isfinite(rr) else "NA"
        
        # --- MODIFIED: High precision timestamp string for CSVs ---
        ts_str = f"{ts:.4f}"
        
        # --- 1) This saves to 'radar_data_log.csv' (full raw data, full timestamp) ---
        row_to_save = [ts] + list(raw) + [dist_str, rr_str]
        self.csv_buffer.append(row_to_save)
        
        if len(self.csv_buffer) >= self.CSV_BUFFER_SIZE:
            self._flush_csv_buffer()

        # --- 2) NEW LOGIC: Save to distance.csv (PER FRAME, float, rounded timestamp) ---
        self.dist_csv_buffer.append([ts_str, dist_str])
        if len(self.dist_csv_buffer) >= self.DIST_CSV_BUFFER_SIZE:
            self._flush_dist_csv_buffer()
        
        # --- 3) MODIFIED LOGIC: Save to bpm.csv (PER COMPUTATION, float, rounded timestamp) ---
        # We save if the computation was successful (finite rr) OR if it was attempted but failed (NA)
        
        # This flag is true ONLY on the frame a computation is attempted
        is_computation_attempt = hasattr(self.respEst, 'last_compute_ts') and self.respEst.last_compute_ts == ts
        
        # --- MODIFICATION IS HERE: ---
        # Save if computation is successful (np.isfinite) OR if it was a failed attempt
        if np.isfinite(rr) or (rr_str == "NA" and is_computation_attempt):
            # In both successful (rr_str) and failed ("NA") cases, 
            # rr_str already holds the correct string value we want to save.
            self.bpm_csv_buffer.append([ts_str, rr_str])
            
            # We flush (write) to the file immediately on every computation attempt (success or fail)
            self._flush_bpm_csv_buffer()
        # --- END OF MODIFICATION ---

        # --- 4) Print messages (uses float strings for better console info) ---
        # Muted as per user request to hide detailed calculation info
        # if np.isfinite(rr):
        #     sys.stdout.write('\r' + ' ' * 50 + '\r') 
        #     print(f"\n[Main] ★★★ NEW RR computed v1.1.0: {rr_str} bpm | Distance: {dist_str} m at {datetime.fromtimestamp(ts).strftime('%H:%M:%S')} ★★★")
            
        # elif rr_str == "NA" and is_computation_attempt: # Use the flag we just made
        #     sys.stdout.write('\r' + ' ' * 50 + '\r')
        #     print(f"\n[Main] ⚠️  RR computation attempted but NOT VALID v1.1.0 | Distance: {dist_str} m at {datetime.fromtimestamp(ts).strftime('%H:%M:%S')} ⚠️")
        # --- END MODIFIED LOGIC ---

    def main(self):
        # ... (This method is unchanged)
        print(f"[Main]  resetting module on {self.port}")
        try:
            mc = ModuleConnector(self.port) # <-- Removed log_file=None
            mc.get_xep().module_reset()
            mc.close()
            time.sleep(2)
        except Exception as e:
            print(f"[Main] ERROR: Failed to reset module: {e}")
            print("[Main] Continuing anyway...")

        self.radarThread.start()
        print("[Main] Starting main loop. Press Ctrl+C to stop.")
        print("[Main] Learning background for distance calculation, please wait...")

        last_health_check = time.time()
        health_check_interval = 5.0

        try:
            while True:
                self.rollover() #<-- ENABLED date rollover logic
                current_time = time.time()

                if current_time - last_health_check > health_check_interval:
                    if not self.radarThread.is_alive():
                        print("\n[Main] ERROR: Radar thread died unexpectedly!")
                        print("[Main] Exiting to trigger watchdog restart...")
                        sys.exit(1)
                    last_health_check = current_time

                # --- MUTED DISTANCE PRINTING AS REQUESTED ---
                # if current_time - self.last_distance_print_ts >= 2:
                #     if np.isfinite(self.latest_distance):
                #         print(f"\r[Info] Current Distance: {self.latest_distance:.3f} m   ", end="")
                #     else:
                #         print(f"\r[Info] Current Distance: Calculating...   ", end="")
                #     sys.stdout.flush()
                #     self.last_distance_print_ts = current_time
                
                try:
                    if not self.dataQ.empty():
                        self.save_frame(self.dataQ.get_nowait())
                except collections.deque.Empty:
                    pass
                except Exception as e:
                    print(f"Error in main loop processing queue: {e}")

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n[Main]  Ctrl-C received, initiating graceful shutdown...")
        finally:
            self.shutdown()

    def shutdown(self):
        # ... (This method is unchanged)
        print("\n[Main]  initiating shutdown sequence...")
        self.stopEvent.set()

        print("[Main]  Processing remaining queued data...")
        while not self.dataQ.empty():
            try:
                self.save_frame(self.dataQ.get_nowait())
            except:
                break
        
        print("[Main]  Flushing final data to CSV...")
        self._flush_csv_buffer()
        self._flush_bpm_csv_buffer() 
        self._flush_dist_csv_buffer() 

        time.sleep(0.1)
        self.radarThread.exit.set()

        print("[Main]  waiting for radar thread to stop...")
        self.radarThread.join(timeout=1.0)

        if self.radarThread.is_alive():
            print("[Main]  radar thread still running, forcing termination")
            self.radarThread.terminate()

        print("[Main]  shutdown complete")


if __name__ == '__main__':
    # ... (This section is unchanged)
    parser = argparse.ArgumentParser(description="Novelda Radar Respiration and Distance Monitor")
    parser.add_argument(
        "--shared_data_dir",
        type=str,
        required=True,
        help="The absolute path to the shared_data directory for logging."
    )
    # --- MODIFIED: Added version argument to fix the crash ---
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        help="The application version string."
    )
    args = parser.parse_args()

    Main(shared_data_dir=args.shared_data_dir).main()