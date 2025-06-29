"""
Script to run Experiment 3: Robustness to incorrect DAGs.

Usage:
    python run_experiment_3.py                    # Run full experiment
    python run_experiment_3.py --no-resume       # Start fresh
"""

import argparse
from experiment_3 import run_experiment_3
from utils.dag_utils import print_dag_info, create_dag_variations
from utils.scm_data import get_dag_and_config


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 3: Robustness to incorrect DAGs')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show experiment info
    print("=" * 60)
    print("EXPERIMENT 3: Robustness to Incorrect DAGs")
    print("=" * 60)
    print("\nResearch Question:")
    print("Is providing an incorrect DAG better or worse than providing")
    print("no DAG at all? How robust is TabPFN to DAG misspecification?")
    
    # Show correct DAG
    dag, col_names, _ = get_dag_and_config(False)
    print("\nCorrect SCM structure:")
    print_dag_info(dag, col_names)
    
    # Show DAG variations
    dag_variations = create_dag_variations(dag)
    print("\n\nDAG variations to test:")
    print("-" * 40)
    print("1. correct: The true DAG")
    print("2. no_dag: No DAG provided (vanilla TabPFN)")
    print("3. wrong_parents: Completely wrong parent relationships")
    print("4. missing_edges: Some true edges removed")
    print("5. extra_edges: Spurious edges added")
    
    # Configuration
    print("\n\nRunning FULL experiment...")
    config = {
        'train_sizes': [20, 50, 100, 200, 500],
        'dag_types': ['correct', 'no_dag', 'wrong_parents', 'missing_edges', 'extra_edges'],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42
    }
    output_dir = args.output or "experiment_3_results"
    
    # Calculate total configurations
    total_configs = (len(config['train_sizes']) * 
                    len(config['dag_types']) * 
                    config['n_repetitions'])
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  DAG types: {config['dag_types']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    
    # Run experiment
    results = run_experiment_3(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print detailed summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Overall comparison
        for metric in config['metrics']:
            print(f"\n{metric.upper()} Results:")
            print("-" * 40)
            
            # Mean by DAG type
            mean_by_dag = results.groupby('dag_type')[metric].mean()
            
            # Sort by performance (lower is better)
            sorted_dags = mean_by_dag.sort_values()
            
            print("Performance ranking (best to worst):")
            for i, (dag_type, value) in enumerate(sorted_dags.items(), 1):
                print(f"  {i}. {dag_type}: {value:.4f}")
            
            # Compare to correct DAG
            correct_value = mean_by_dag['correct']
            print(f"\nComparison to correct DAG ({correct_value:.4f}):")
            
            for dag_type in config['dag_types']:
                if dag_type != 'correct':
                    diff = mean_by_dag[dag_type] - correct_value
                    pct_worse = (diff / correct_value) * 100
                    print(f"  {dag_type}: {diff:+.4f} ({pct_worse:+.1f}%)")


if __name__ == "__main__":
    main()