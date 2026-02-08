"""
Segment-wise Attribution for ECG Waveforms
Compare unknown waveform with prototype, segment by segment
"""
import numpy as np
from typing import Dict, Tuple, List
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr

class SegmentAttributor:
    def __init__(self, window_before: int = 100, window_after: int = 200, heart_rate: float = 75.0):
        """
        window_before: samples before R-peak (P wave + PR segment)
        window_after: samples after R-peak (QRS + ST + T wave)
        heart_rate: estimated heart rate in BPM (for adaptive segmentation)
        Total beat length: window_before + window_after
        """
        self.window_before = window_before
        self.window_after = window_after
        self.total_length = window_before + window_after
        self.heart_rate = heart_rate
        
        # Segment boundaries (approximate, relative to R-peak at index window_before)
        self.r_peak_idx = window_before
        
        # ADAPTIVE SEGMENTATION based on heart rate
        # At 60 BPM (1 beat/sec), segments are at normal duration
        # Higher HR → shorter segments; Lower HR → longer segments
        hr_factor = 60.0 / heart_rate  # Scale factor
        
        # Typical ECG segments (in samples @ 500 Hz, normalized to 60 BPM):
        # P wave: ~80ms = 40 samples
        # PR segment: ~60ms = 30 samples
        # QRS: ~80-120ms = 40-60 samples (relatively constant)
        # ST segment: ~100ms = 50 samples
        # T wave: ~160ms = 80 samples
        
        # Apply heart rate scaling (QRS less affected)
        p_width = int(40 * hr_factor)
        pr_width = int(30 * hr_factor)
        qrs_width = int(60 * min(hr_factor, 1.2))  # QRS less variable
        st_width = int(50 * hr_factor)
        t_width = int(80 * hr_factor)
        
        self.segments = {
            'P_wave': (self.r_peak_idx - p_width - pr_width, self.r_peak_idx - pr_width),
            'PR_segment': (self.r_peak_idx - pr_width, self.r_peak_idx),
            'QRS_complex': (self.r_peak_idx - 20, self.r_peak_idx + qrs_width),
            'ST_segment': (self.r_peak_idx + qrs_width, self.r_peak_idx + qrs_width + st_width),
            'T_wave': (self.r_peak_idx + qrs_width + st_width, 
                      self.r_peak_idx + qrs_width + st_width + t_width)
        }
    
    def compute_similarity(self, unknown_beat: np.ndarray, prototype_beat: np.ndarray,
                          metric: str = 'correlation') -> float:
        """
        Compute similarity between two beats
        metric: 'correlation', 'euclidean', 'cosine'
        Returns: similarity score [0, 1] where 1 = identical
        """
        if len(unknown_beat) != len(prototype_beat):
            raise ValueError(f"Beat lengths must match: {len(unknown_beat)} vs {len(prototype_beat)}")
        
        if metric == 'correlation':
            corr, _ = pearsonr(unknown_beat, prototype_beat)
            # Convert to [0, 1]: correlation is [-1, 1]
            return (corr + 1) / 2
        
        elif metric == 'euclidean':
            dist = euclidean(unknown_beat, prototype_beat)
            # Normalize by length
            max_possible_dist = np.sqrt(len(unknown_beat) * 4)  # Assuming values in [-1, 1]
            return 1 - min(dist / max_possible_dist, 1.0)
        
        elif metric == 'cosine':
            sim = 1 - cosine(unknown_beat, prototype_beat)
            return max(0, sim)  # Cosine similarity in [0, 1]
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_segment_attribution(self, unknown_beat: np.ndarray, prototype_beat: np.ndarray,
                                   metric: str = 'correlation') -> Dict[str, float]:
        """
        Compute similarity for each ECG segment
        Returns dict: {segment_name: similarity_score}
        """
        if len(unknown_beat) != self.total_length or len(prototype_beat) != self.total_length:
            raise ValueError(f"Beats must be {self.total_length} samples long")
        
        attributions = {}
        
        for seg_name, (start, end) in self.segments.items():
            unknown_seg = unknown_beat[start:end]
            proto_seg = prototype_beat[start:end]
            
            similarity = self.compute_similarity(unknown_seg, proto_seg, metric)
            attributions[seg_name] = similarity
        
        # Overall similarity
        overall = self.compute_similarity(unknown_beat, prototype_beat, metric)
        attributions['Overall'] = overall
        
        return attributions
    
    def compute_segment_importance(self, attributions: Dict[str, float]) -> Dict[str, float]:
        """
        Compute how much each segment contributes to overall similarity
        Returns: {segment_name: importance_score} where sum = 1.0
        """
        # Get segment similarities (exclude Overall)
        seg_sims = {k: v for k, v in attributions.items() if k != 'Overall'}
        
        # Compute importance as proportion of total
        total = sum(seg_sims.values())
        
        if total == 0:
            # All segments have 0 similarity
            n = len(seg_sims)
            return {k: 1.0/n for k in seg_sims.keys()}
        
        importance = {k: v / total for k, v in seg_sims.items()}
        
        return importance
    
    def get_segment_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Get segment boundaries for visualization"""
        return self.segments.copy()


def compare_waveforms(unknown_beat: np.ndarray, prototype_beat: np.ndarray,
                     window_before: int = 100, window_after: int = 200,
                     heart_rate: float = 75.0) -> Dict:
    """
    Full comparison: similarity + attribution + importance
    Returns dict with all metrics
    heart_rate: estimated heart rate for adaptive segmentation
    """
    attributor = SegmentAttributor(window_before, window_after, heart_rate)
    
    # Compute segment-wise similarity
    attributions = attributor.compute_segment_attribution(unknown_beat, prototype_beat)
    
    # Compute importance
    importance = attributor.compute_segment_importance(attributions)
    
    return {
        'similarity': attributions,
        'importance': importance,
        'segments': attributor.get_segment_ranges(),
        'heart_rate': heart_rate
    }

