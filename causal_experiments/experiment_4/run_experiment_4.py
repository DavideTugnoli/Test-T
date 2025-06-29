"""
Script to run Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This experiment creates a CPDAG from the true DAG with controlled ambiguity,
then tests TabPFN with different levels of causal knowledge.

Usage:
    python run_experiment_4.py                    # Run full experiment
    python run_experiment_4.py --no-resume       # Start fresh
    python run_experiment_4.py --quick           # Run quick test
"""

import argparse
from experiment_4 import run_experiment_4
from utils.scm_data import get_dag_and_config
from utils.dag_utils import print_dag_info


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 4: Causal Knowledge Level Impact')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced configurations')
    
    args = parser.parse_args()
    
    # Show experiment info
    print("=" * 60)
    print("EXPERIMENT 4: Causal Knowledge Level Impact")
    print("=" * 60)
    
    # Configuration
    if args.quick:
        print("\n\nRunning QUICK experiment...")
        config = {
            'train_sizes': [50, 200],
            'n_repetitions': 3,
            'test_size': 1000,
            'n_permutations': 2,
            'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
            'include_categorical': False,
            'n_estimators': 2,
            'random_seed_base': 42,
            'cpdag_ambiguity_level': 0.3
        }
        output_dir = args.output or "experiment_4_results_quick"
    else:
        print("\n\nRunning FULL experiment...")
        config = {
            'train_sizes': [50, 100, 200, 500],
            'n_repetitions': 10,
            'test_size': 2000,
            'n_permutations': 3,
            'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
            'include_categorical': False,
            'n_estimators': 3,
            'random_seed_base': 42,
            'cpdag_ambiguity_level': 0.3
        }
        output_dir = args.output or "experiment_4_results"
    
    # Calculate total configurations (will be determined after causal discovery)
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  CPDAG ambiguity level: {config['cpdag_ambiguity_level']}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    print(f"  Quick mode: {args.quick}")
    
    print("\nNote: Total configurations will be determined after CPDAG creation")
    print("      based on the number of DAGs found in the CPDAG equivalence class.")
    
    # Run experiment
    results = run_experiment_4(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print detailed summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Show DAG levels discovered
        dag_levels = results['dag_level'].unique()
        print(f"\nDAG Levels Tested: {list(dag_levels)}")
        
        # Overall comparison
        for metric in config['metrics']:
            print(f"\n{metric.upper()} Results:")
            print("-" * 40)
            
            # Mean by DAG level
            mean_by_level = results.groupby('dag_level')[metric].mean()
            
            # Sort by performance (lower is better)
            sorted_levels = mean_by_level.sort_values()
            
            print("Performance ranking (best to worst):")
            for i, (level, value) in enumerate(sorted_levels.items(), 1):
                print(f"  {i}. {level}: {value:.4f}")
            
            # Compare to vanilla (no_dag)
            if 'no_dag' in mean_by_level:
                vanilla_value = mean_by_level['no_dag']
                print(f"\nComparison to vanilla TabPFN ({vanilla_value:.4f}):")
                
                for level in dag_levels:
                    if level != 'no_dag':
                        diff = mean_by_level[level] - vanilla_value
                        pct_change = (diff / vanilla_value) * 100
                        direction = "better" if diff < 0 else "worse"
                        print(f"  {level}: {diff:+.4f} ({pct_change:+.1f}% {direction})")
            
            # Performance by training size
            print(f"\nPerformance by training size:")
            for train_size in sorted(config['train_sizes']):
                subset = results[results['train_size'] == train_size]
                mean_by_level_size = subset.groupby('dag_level')[metric].mean()
                
                print(f"  Training size {train_size}:")
                for level in dag_levels:
                    if level in mean_by_level_size:
                        print(f"    {level}: {mean_by_level_size[level]:.4f}")
        
        # DAG complexity analysis
        print(f"\nDAG Complexity Analysis:")
        print("-" * 40)
        
        # Show edge counts for each level
        dag_edge_info = results.groupby('dag_level')['dag_edges'].first()
        print("DAG edge counts by level:")
        
        for level in sorted(dag_edge_info.index, key=lambda x: dag_edge_info[x]):
            edge_count = dag_edge_info[level]
            print(f"  {level}: {edge_count} edges")
        
        # Correlation between edges and performance
        print(f"\nCorrelation between DAG edges and performance:")
        for metric in config['metrics']:
            correlation = results['dag_edges'].corr(results[metric])
            print(f"  {metric}: {correlation:.3f}")


if __name__ == "__main__":
    main() 