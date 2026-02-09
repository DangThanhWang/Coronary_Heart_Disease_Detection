"""
Load and preprocess ECG waveforms from CSV files
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.signal import find_peaks

class WaveformLoader:
    def __init__(self, csv_root: str):
        self.csv_root = Path(csv_root)
    
    def load_waveform(self, filename: str, env: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Load waveform from CSV file
        Returns: (time, voltage, peak_indices)
        """
        csv_path = self.csv_root / env / 'csv' / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        time = df['Time'].values
        voltage = df['Voltage'].values
        peaks = df['Peak'].values
        
        # Get R-peak indices from annotation
        peak_indices = np.where(peaks == 1)[0].tolist()
        
        # If no peaks annotated, detect them
        if len(peak_indices) == 0:
            peak_indices = self.detect_r_peaks(voltage)
        
        return time, voltage, peak_indices
    
    def detect_r_peaks(self, voltage: np.ndarray, fs: int = 500) -> List[int]:
        """
        Detect R-peaks in ECG signal
        fs: sampling frequency in Hz
        """
        # Normalize voltage
        voltage_norm = (voltage - np.mean(voltage)) / (np.std(voltage) + 1e-8)
        
        # Find peaks with typical ECG parameters
        # Minimum distance between peaks: ~0.4s = 200 samples @ 500Hz
        # Prominence: at least 0.5 std above baseline
        peaks, properties = find_peaks(
            voltage_norm,
            distance=fs // 3,  # ~333ms minimum between peaks
            prominence=0.3,     # At least 0.3 std above baseline
            height=0.0          # Must be above mean
        )
        
        return peaks.tolist()
    
    def extract_single_beat(self, voltage: np.ndarray, peak_idx: int, 
                           window_before: int = 100, window_after: int = 200) -> Optional[np.ndarray]:
        """
        Extract a single heartbeat centered around R-peak
        window_before: samples before R-peak (P wave region)
        window_after: samples after R-peak (T wave region)
        """
        start = max(0, peak_idx - window_before)
        end = min(len(voltage), peak_idx + window_after)
        
        # Check if we have enough samples
        if (peak_idx - start) < window_before * 0.5 or (end - peak_idx) < window_after * 0.5:
            return None
        
        beat = voltage[start:end]
        
        # Pad if necessary to ensure consistent length
        target_length = window_before + window_after
        if len(beat) < target_length:
            pad_before = window_before - (peak_idx - start)
            pad_after = window_after - (end - peak_idx)
            beat = np.pad(beat, (max(0, pad_before), max(0, pad_after)), mode='edge')
        
        return beat[:target_length]  # Ensure exact length
    
    def extract_all_beats(self, voltage: np.ndarray, peak_indices: List[int],
                         window_before: int = 100, window_after: int = 200) -> List[np.ndarray]:
        """Extract all complete heartbeats from a waveform"""
        beats = []
        for peak_idx in peak_indices:
            beat = self.extract_single_beat(voltage, peak_idx, window_before, window_after)
            if beat is not None:
                beats.append(beat)
        return beats
    
    def get_average_beat(self, voltage: np.ndarray, peak_indices: List[int],
                        window_before: int = 100, window_after: int = 200) -> Optional[np.ndarray]:
        """Get average heartbeat (template)"""
        beats = self.extract_all_beats(voltage, peak_indices, window_before, window_after)
        
        if len(beats) == 0:
            return None
        
        # Stack and average
        beats_array = np.array(beats)
        avg_beat = np.mean(beats_array, axis=0)
        
        return avg_beat
    
    def estimate_heart_rate(self, peak_indices: List[int], fs: int = 500) -> float:
        """
        Estimate heart rate from R-peak intervals
        Returns: heart rate in BPM
        """
        if len(peak_indices) < 2:
            return 75.0  # Default
        
        # RR intervals in samples
        rr_intervals = np.diff(peak_indices)
        
        # Convert to seconds
        rr_seconds = rr_intervals / fs
        
        # Heart rate = 60 / mean_RR
        mean_rr = np.mean(rr_seconds)
        
        if mean_rr <= 0:
            return 75.0
        
        hr = 60.0 / mean_rr
        
        # Clip to reasonable range
        hr = np.clip(hr, 40, 180)
        
        return float(hr)


def load_prototype_waveform(prototype_row: pd.Series, csv_root: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Helper to load prototype waveform from prototype CSV row"""
    loader = WaveformLoader(csv_root)
    
    filename = prototype_row['filename']
    env = prototype_row['env']
    
    return loader.load_waveform(filename, env)

