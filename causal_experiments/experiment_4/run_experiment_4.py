"""
Script to run Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This experiment first runs causal discovery on a simulated dataset to obtain a
CPDAG, then uses this CPDAG to test TabPFN's generative performance with
varying levels of causal knowledge derived from the CPDAG's equivalence class.

Usage examples:
    # Run experiment with categorical variables included
    python run_experiment_4.py --include-categorical

    # Run experiment with only continuous variables
    python run_experiment_4.py

    # Start a fresh run, ignoring any previous checkpoints
    python run_experiment_4.py --include-categorical --no-resume
"""

import argparse
from experiment_4 import run_experiment_4
from run_pc_discovery import run_pc_discovery_on_dataset
from utils.scm_data import generate_scm_data, get_dag_and_config


def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 4: Causal Knowledge Level Impact',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--no-resume', action='store_true',
                       help='Start a fresh run (ignores any existing checkpoint).')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (a default name will be generated if not specified).')
    parser.add_argument('--include-categorical', action='store_true',
                        help='Include categorical variables in the SCM (default: only continuous variables).')

    args = parser.parse_args()
    
    # Show experiment info
    print("=" * 60)
    print("EXPERIMENT 4: Causal Knowledge Level Impact")
    print(f"Data type: {'Mixed (continuous + categorical)' if args.include_categorical else 'Continuous only'}")
    print("=" * 60)
    
    # Configuration for the experiment
    print("\n\nRunning experiment with full configuration...")
    config = {
        'train_sizes': [50, 100, 200, 500],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
        'include_categorical': args.include_categorical,
        'n_estimators': 3,
        'random_seed_base': 42,
        'max_dags_to_test': 5
    }
    
    output_dir = args.output or f"experiment_4_results_{'mixed' if args.include_categorical else 'continuous'}"
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Resume enabled: {not args.no_resume}")
    print(f"  Output directory: {output_dir}")

    # --- Causal Discovery Step ---
    print("\n" + "-" * 50)
    print("STEP 1: Causal Discovery")
    print("-" * 50)
    
    true_dag, col_names, categorical_cols = get_dag_and_config(
        include_categorical=args.include_categorical
    )
    
    # Generate data for discovery
    n_discovery_samples = 2000
    print(f"Generating {n_discovery_samples} samples for PC discovery...")
    X_discovery = generate_scm_data(
        n_samples=n_discovery_samples,
        random_state=config['random_seed_base'],
        include_categorical=args.include_categorical
    )
    
    print("Discovering CPDAG from data using PC algorithm...")
    cpdag = run_pc_discovery_on_dataset(
        dataset_name="mixed" if args.include_categorical else "continuous",
        data=X_discovery,
        true_dag=true_dag,
        task_type="classification" if "target" in col_names else "unsupervised",
        target_column="target" if "target" in col_names else None,
        verbose=False,
        output_dir=None,
    )
    # The CPDAG from run_pc_discovery is a numpy array
    print(f"CPDAG discovered successfully.")
    
    # --- Experiment Execution Step ---
    print("\n" + "-" * 50)
    print("STEP 2: Running Experiment 4 with Discovered CPDAG")
    print("-" * 50)

    run_experiment_4(
        cpdag=cpdag,
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )

    print("\n" + "=" * 50)
    print("All experiments finished.")
    print(f"Results saved in: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main() 