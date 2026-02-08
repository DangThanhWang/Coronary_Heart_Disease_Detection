"""
Visualize waveform-to-waveform attribution
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class AttributionVisualizer:
    def __init__(self):
        self.segment_colors = {
            'P_wave': '#FFB6C1',       # Light pink
            'PR_segment': '#FFA07A',   # Light salmon
            'QRS_complex': '#FF6347',  # Tomato red
            'ST_segment': '#FFD700',   # Gold
            'T_wave': '#87CEEB'        # Sky blue
        }
    
    def plot_comparison(self, unknown_beat: np.ndarray, prototype_beat: np.ndarray,
                       attributions: Dict[str, float], importance: Dict[str, float],
                       segments: Dict[str, Tuple[int, int]],
                       save_path: str = None, title: str = "Waveform Attribution"):
        """
        Plot unknown vs prototype with segment-wise attribution
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2, 1]})
        
        time_axis = np.arange(len(unknown_beat))
        
        # --- Plot 1: Waveforms with segment highlights ---
        ax1 = axes[0]
        
        # Draw segment backgrounds
        for seg_name, (start, end) in segments.items():
            color = self.segment_colors.get(seg_name, '#CCCCCC')
            ax1.axvspan(start, end, alpha=0.2, color=color, label=seg_name)
        
        # Plot waveforms
        ax1.plot(time_axis, prototype_beat, 'b-', linewidth=2, label='Prototype', alpha=0.7)
        ax1.plot(time_axis, unknown_beat, 'r--', linewidth=2, label='Unknown', alpha=0.7)
        
        ax1.set_ylabel('Voltage (normalized)', fontsize=12)
        ax1.set_title(f'{title}\nOverall Similarity: {attributions["Overall"]:.3f}', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Difference heatmap ---
        ax2 = axes[1]
        
        difference = np.abs(unknown_beat - prototype_beat)
        
        # Create 2D array for heatmap
        diff_2d = np.tile(difference, (20, 1))
        
        im = ax2.imshow(diff_2d, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                       extent=[0, len(unknown_beat), 0, 1])
        
        # Add segment boundaries
        for seg_name, (start, end) in segments.items():
            ax2.axvline(start, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axvline(end, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add segment labels
            mid = (start + end) / 2
            ax2.text(mid, 0.5, seg_name.replace('_', '\n'), 
                    ha='center', va='center', fontsize=9, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax2.set_ylabel('Difference', fontsize=12)
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2, label='|Unknown - Prototype|')
        
        # --- Plot 3: Segment-wise similarity bar chart ---
        ax3 = axes[2]
        
        seg_names = [k for k in attributions.keys() if k != 'Overall']
        seg_similarities = [attributions[k] for k in seg_names]
        seg_importances = [importance[k] for k in seg_names]
        
        # Create bars
        bars = ax3.barh(seg_names, seg_similarities, color=[self.segment_colors.get(k, '#CCCCCC') for k in seg_names])
        
        # Add importance annotations
        for i, (name, sim, imp) in enumerate(zip(seg_names, seg_similarities, seg_importances)):
            ax3.text(sim + 0.02, i, f'{sim:.2f} ({imp*100:.1f}%)', 
                    va='center', fontsize=10)
        
        ax3.set_xlabel('Similarity Score', fontsize=12)
        ax3.set_xlim(0, 1.1)
        ax3.set_title('Segment-wise Similarity & Importance', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_attribution_heatmap(self, attribution_matrix: np.ndarray, 
                                 prototype_ids: list, sample_ids: list,
                                 save_path: str = None):
        """
        Plot heatmap of attributions between multiple samples and prototypes
        attribution_matrix: (n_samples, n_prototypes)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(attribution_matrix, aspect='auto', cmap='RdYlGn', 
                      vmin=0, vmax=1, interpolation='nearest')
        
        ax.set_xticks(np.arange(len(prototype_ids)))
        ax.set_yticks(np.arange(len(sample_ids)))
        ax.set_xticklabels(prototype_ids, rotation=45, ha='right')
        ax.set_yticklabels(sample_ids)
        
        ax.set_xlabel('Prototype ID', fontsize=12)
        ax.set_ylabel('Sample ID', fontsize=12)
        ax.set_title('Waveform Attribution Matrix', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Overall Similarity')
        
        # Add text annotations
        for i in range(len(sample_ids)):
            for j in range(len(prototype_ids)):
                text = ax.text(j, i, f'{attribution_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

