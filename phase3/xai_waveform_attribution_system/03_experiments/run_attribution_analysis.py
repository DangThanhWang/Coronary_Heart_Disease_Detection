"""
Run Waveform-to-Waveform Attribution Analysis
Test on both HardOOD3 and Chapman datasets
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from waveform_loader import WaveformLoader, load_prototype_waveform
from segment_attributor import SegmentAttributor, compare_waveforms
from visualizer import AttributionVisualizer

def analyze_dataset(dataset_name: str, proto_csv: str, csv_root: str, 
                   output_dir: str, n_samples: int = 50):
    """
    Analyze waveform attribution for a dataset
    """
    print(f"\n{'='*80}")
    print(f"WAVEFORM ATTRIBUTION ANALYSIS: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load prototypes
    proto_df = pd.read_csv(proto_csv)
    print(f"[1/5] Loaded {len(proto_df)} prototypes")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    loader = WaveformLoader(csv_root)
    attributor = SegmentAttributor()
    visualizer = AttributionVisualizer()
    
    print(f"[2/5] Extracting prototype templates...")
    
    # Get one prototype per class
    proto_templates = {}
    proto_info = {}
    
    for label in sorted(proto_df['label'].unique()):
        # Get highest purity prototype for this class
        class_protos = proto_df[proto_df['label'] == label].sort_values('purity', ascending=False)
        
        if len(class_protos) == 0:
            continue
        
        best_proto = class_protos.iloc[0]
        
        try:
            time, voltage, peaks = load_prototype_waveform(best_proto, csv_root)
            
            # Get average beat
            avg_beat = loader.get_average_beat(voltage, peaks)
            
            if avg_beat is not None:
                proto_templates[label] = avg_beat
                proto_info[label] = {
                    'proto_id': best_proto['node_id'],
                    'purity': best_proto['purity'],
                    'wins': best_proto.get('wins_total', best_proto.get('wins', 0)),
                    'filename': best_proto['filename']
                }
                print(f"  Class {label}: Prototype {best_proto['node_id']} (purity={best_proto['purity']:.3f})")
        except Exception as e:
            print(f"  Class {label}: ERROR - {e}")
    
    print(f"\n[3/5] Comparing unknown samples with prototypes...")
    
    # Get some unknown samples from Eval set
    eval_csv_path = Path(csv_root) / 'Eval' / 'csv'
    
    if not eval_csv_path.exists():
        print(f"  ERROR: Eval directory not found at {eval_csv_path}")
        return
    
    eval_files = sorted(list(eval_csv_path.glob('*.csv')))[:n_samples]
    
    if len(eval_files) == 0:
        print(f"  ERROR: No CSV files found in {eval_csv_path}")
        return
    
    print(f"  Found {len(eval_files)} eval samples")
    
    results = []
    
    for i, eval_file in enumerate(eval_files):
        print(f"\n  Sample {i+1}/{len(eval_files)}: {eval_file.name}")
        
        try:
            # Load unknown sample
            time, voltage, peaks = loader.load_waveform(eval_file.name, 'Eval')
            unknown_beat = loader.get_average_beat(voltage, peaks)
            
            if unknown_beat is None:
                print(f"    SKIP: Could not extract beat")
                continue
            
            # Estimate heart rate for adaptive segmentation
            heart_rate = loader.estimate_heart_rate(peaks)
            
            # Compare with all prototype templates
            sample_results = {
                'filename': eval_file.name,
                'heart_rate': heart_rate,
                'comparisons': {}
            }
            
            for label, proto_beat in proto_templates.items():
                comparison = compare_waveforms(unknown_beat, proto_beat, heart_rate=heart_rate)
                
                sample_results['comparisons'][label] = comparison
                
                if i < 3:  # Only print first 3 for brevity
                    print(f"    vs Class {label}: Overall={comparison['similarity']['Overall']:.3f}, " +
                          f"QRS={comparison['similarity']['QRS_complex']:.3f}, " +
                          f"T={comparison['similarity']['T_wave']:.3f}")
            
            results.append(sample_results)
            
            # Generate PNG visualization for ALL samples
            best_label = max(sample_results['comparisons'].keys(), 
                           key=lambda l: sample_results['comparisons'][l]['similarity']['Overall'])
            
            best_comparison = sample_results['comparisons'][best_label]
            
            viz_path = output_path / f"{dataset_name}_{eval_file.stem}_vs_proto{best_label}.png"
            
            visualizer.plot_comparison(
                unknown_beat,
                proto_templates[best_label],
                best_comparison['similarity'],
                best_comparison['importance'],
                best_comparison['segments'],
                save_path=str(viz_path),
                title=f"{dataset_name} - {eval_file.stem} vs Prototype Class {best_label} (HR={heart_rate:.0f})"
            )
            
            if i < 3:
                print(f"    Saved visualization: {viz_path.name}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    print(f"\n[4/5] Computing summary statistics...")
    
    # Aggregate results
    summary = []
    
    for res in results:
        for label, comp in res['comparisons'].items():
            summary.append({
                'sample': res['filename'],
                'prototype_class': label,
                'overall_similarity': comp['similarity']['Overall'],
                'P_wave_similarity': comp['similarity']['P_wave'],
                'QRS_similarity': comp['similarity']['QRS_complex'],
                'T_wave_similarity': comp['similarity']['T_wave'],
                'QRS_importance': comp['importance']['QRS_complex'],
                'T_importance': comp['importance']['T_wave']
            })
    
    summary_df = pd.DataFrame(summary)
    summary_csv = output_path / f"{dataset_name}_attribution_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"  Saved summary: {summary_csv.name}")
    
    print(f"\n[5/5] Summary statistics:")
    
    # Report statistics objectively
    print(f"\n  Sample size: n={len(summary_df)} comparisons")
    print(f"\n  Similarity scores (mean ± std):")
    print(f"    Overall: {summary_df['overall_similarity'].mean():.3f} ± {summary_df['overall_similarity'].std():.3f}")
    print(f"    QRS:     {summary_df['QRS_similarity'].mean():.3f} ± {summary_df['QRS_similarity'].std():.3f}")
    print(f"    T-wave:  {summary_df['T_wave_similarity'].mean():.3f} ± {summary_df['T_wave_similarity'].std():.3f}")
    print(f"    P-wave:  {summary_df['P_wave_similarity'].mean():.3f} ± {summary_df['P_wave_similarity'].std():.3f}")
    
    print(f"\n  Importance weights:")
    print(f"    QRS:   {summary_df['QRS_importance'].mean():.3f} ± {summary_df['QRS_importance'].std():.3f}")
    print(f"    T-wave: {summary_df['T_importance'].mean():.3f} ± {summary_df['T_importance'].std():.3f}")
    
    # Report which segment has highest importance (objective fact)
    avg_importance = summary_df[['QRS_importance', 'T_importance']].mean()
    dominant_segment = 'QRS' if avg_importance['QRS_importance'] > avg_importance['T_importance'] else 'T-wave'
    print(f"\n  Dominant segment: {dominant_segment}")
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {dataset_name}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    return summary_df


def main():
    """Run attribution analysis on both datasets"""
    
    print("\n" + "="*80)
    print("WAVEFORM-TO-WAVEFORM ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # HardOOD3
    print("\n[DATASET 1] HardOOD3")
    hardood3_proto = "artifacts/phase3/dual_memory/dual_memory_prototypes.csv"
    hardood3_csv = "Data/Generated_HardOOD3"
    hardood3_out = "phase3/xai_waveform_attribution_system/04_results/hardood3"
    
    try:
        h3_summary = analyze_dataset("HardOOD3", hardood3_proto, hardood3_csv, 
                                    hardood3_out, n_samples=50)
    except Exception as e:
        print(f"ERROR: {e}")
        h3_summary = None
    
    # Chapman
    print("\n[DATASET 2] Chapman-Shaoxing")
    chapman_proto = "artifacts/phase3/chapman_experiment/dual_memory_prototypes.csv"
    chapman_csv = "Data/Generated_Chapman"
    chapman_out = "phase3/xai_waveform_attribution_system/04_results/chapman"
    
    try:
        ch_summary = analyze_dataset("Chapman", chapman_proto, chapman_csv,
                                    chapman_out, n_samples=50)
    except Exception as e:
        print(f"ERROR: {e}")
        ch_summary = None
    
    # Cross-dataset comparison
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    
    if h3_summary is not None and ch_summary is not None:
        print("\nQRS importance:")
        print(f"  HardOOD3: {h3_summary['QRS_importance'].mean():.3f} ± {h3_summary['QRS_importance'].std():.3f}")
        print(f"  Chapman:  {ch_summary['QRS_importance'].mean():.3f} ± {ch_summary['QRS_importance'].std():.3f}")
        print(f"  Difference: {abs(h3_summary['QRS_importance'].mean() - ch_summary['QRS_importance'].mean()):.3f}")
        
        print("\nT-wave importance:")
        print(f"  HardOOD3: {h3_summary['T_importance'].mean():.3f} ± {h3_summary['T_importance'].std():.3f}")
        print(f"  Chapman:  {ch_summary['T_importance'].mean():.3f} ± {ch_summary['T_importance'].std():.3f}")
        print(f"  Difference: {abs(h3_summary['T_importance'].mean() - ch_summary['T_importance'].mean()):.3f}")
        
        print("\nP-wave similarity:")
        print(f"  HardOOD3: {h3_summary['P_wave_similarity'].mean():.3f} ± {h3_summary['P_wave_similarity'].std():.3f}")
        print(f"  Chapman:  {ch_summary['P_wave_similarity'].mean():.3f} ± {ch_summary['P_wave_similarity'].std():.3f}")
        
        print("\nQRS similarity:")
        print(f"  HardOOD3: {h3_summary['QRS_similarity'].mean():.3f} ± {h3_summary['QRS_similarity'].std():.3f}")
        print(f"  Chapman:  {ch_summary['QRS_similarity'].mean():.3f} ± {ch_summary['QRS_similarity'].std():.3f}")
        
        print("\nT-wave similarity:")
        print(f"  HardOOD3: {h3_summary['T_wave_similarity'].mean():.3f} ± {h3_summary['T_wave_similarity'].std():.3f}")
        print(f"  Chapman:  {ch_summary['T_wave_similarity'].mean():.3f} ± {ch_summary['T_wave_similarity'].std():.3f}")
        
        # Export comparison report
        comparison_report = {
            'Dataset': ['HardOOD3', 'Chapman'],
            'n_samples': [len(h3_summary), len(ch_summary)],
            'QRS_similarity_mean': [h3_summary['QRS_similarity'].mean(), ch_summary['QRS_similarity'].mean()],
            'QRS_similarity_std': [h3_summary['QRS_similarity'].std(), ch_summary['QRS_similarity'].std()],
            'P_wave_similarity_mean': [h3_summary['P_wave_similarity'].mean(), ch_summary['P_wave_similarity'].mean()],
            'P_wave_similarity_std': [h3_summary['P_wave_similarity'].std(), ch_summary['P_wave_similarity'].std()],
            'T_wave_similarity_mean': [h3_summary['T_wave_similarity'].mean(), ch_summary['T_wave_similarity'].mean()],
            'T_wave_similarity_std': [h3_summary['T_wave_similarity'].std(), ch_summary['T_wave_similarity'].std()],
            'QRS_importance_mean': [h3_summary['QRS_importance'].mean(), ch_summary['QRS_importance'].mean()],
            'T_importance_mean': [h3_summary['T_importance'].mean(), ch_summary['T_importance'].mean()],
        }
        
        comparison_df = pd.DataFrame(comparison_report)
        comparison_df.to_csv('phase3/xai_waveform_attribution_system/04_results/cross_dataset_comparison.csv', index=False)
        print("\nComparison report saved to: cross_dataset_comparison.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()