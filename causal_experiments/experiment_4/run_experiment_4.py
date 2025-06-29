"""
Script to run Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This experiment first runs causal discovery on a simulated dataset to obtain a
CPDAG, then uses this CPDAG to test TabPFN's generative performance with
varying levels of causal knowledge derived from the CPDAG's equivalence class.

Usage examples:
    # Run experiment with mixed (continuous + categorical) data for discovery
    python run_experiment_4.py --dataset-name mixed

    # Run experiment with only continuous data for discovery
    python run_experiment_4.py --dataset-name continuous

    # Start a fresh run, ignoring any previous checkpoints
    python run_experiment_4.py --dataset-name mixed --no-resume
"""

import argparse
from experiment_4 import run_experiment_4
from run_pc_discovery import run_pc_discovery_on_dataset
from utils.scm_data import SCMGenerator, get_dag_and_config


def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 4: Causal Knowledge Level Impact',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--no-resume', action='store_true',
                       help='Start a fresh run (ignores any existing checkpoint).')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (a default name will be generated if not specified).')
    parser.add_argument('--dataset-name', type=str, default="mixed", choices=["mixed", "continuous"],
                        help='Name of the SCM configuration to use for the causal discovery phase.\n'
                             '"mixed": Generates data with both continuous and categorical features.\n'
                             '"continuous": Generates data with only continuous features.')

    args = parser.parse_args()
    
    # Show experiment info
    print("=" * 60)
    print("EXPERIMENT 4: Causal Knowledge Level Impact")
    print(f"Causal discovery dataset type: {args.dataset_name}")
    print("=" * 60)
    
    # Configuration for the experiment
    print("\n\nRunning experiment with full configuration...")
    config = {
        'train_sizes': [50, 100, 200, 500],
        'n_repetitions': 10,
        'test_size': 2000,
        'n_permutations': 3,
        'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
        'include_categorical': args.dataset_name == "mixed",
        'n_estimators': 3,
        'random_seed_base': 42,
    }
    
    output_dir = args.output or f"experiment_4_results_{args.dataset_name}"
    
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
        include_categorical=(args.dataset_name == "mixed")
    )
    
    scm_gen = SCMGenerator(
        config_name=args.dataset_name, 
        seed=config['random_seed_base']
    )
    
    # Generate data for discovery
    n_discovery_samples = 2000
    print(f"Generating {n_discovery_samples} samples for PC discovery...")
    X_discovery, _, _, _ = scm_gen.generate_data(n_samples=n_discovery_samples)
    
    print("Discovering CPDAG from data using PC algorithm...")
    cpdag = run_pc_discovery_on_dataset(
        dataset_name=args.dataset_name,
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