"""
Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This experiment tests how different levels of causal knowledge affect TabPFN's
synthetic data generation. We create a CPDAG from the true DAG with controlled
ambiguity, generate all possible DAGs from this CPDAG, and test TabPFN
with DAGs of increasing complexity/completeness.

The CPDAG should be provided as input (e.g., from external causal discovery).
"""
import sys
import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
from typing import Optional

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised
from utils.scm_data import SCMGenerator
from utils.metrics import SyntheticDataEvaluator
from utils.dag_utils import cpdag_to_dags

# No longer need causal discovery imports - using dummy CPDAG generation


def categorize_dags_by_complexity(dags):
    """
    Categorize DAGs by their complexity (number of edges).
    
    Args:
        dags: List of DAG dictionaries
        
    Returns:
        Dictionary with categories: {category_name: dag}
    """
    if not dags:
        return {'no_dag': None}
    
    # Calculate edge counts for all DAGs
    edge_counts = []
    for dag in dags:
        edge_count = sum(len(parents) for parents in dag.values())
        edge_counts.append((edge_count, dag))
    
    # Sort by edge count
    edge_counts.sort(key=lambda x: x[0])
    
    # Create categories
    n_dags = len(edge_counts)
    categories = {'no_dag': None}  # Always include vanilla case
    
    if n_dags == 1:
        categories['discovered_dag'] = edge_counts[0][1]
    elif n_dags == 2:
        categories['minimal_dag'] = edge_counts[0][1]
        categories['full_dag'] = edge_counts[1][1]
    elif n_dags >= 3:
        categories['minimal_dag'] = edge_counts[0][1]
        categories['medium_dag'] = edge_counts[n_dags // 2][1]
        categories['full_dag'] = edge_counts[-1][1]
        
        # If we have many DAGs, add more categories
        if n_dags >= 5:
            categories['quarter_dag'] = edge_counts[n_dags // 4][1]
            categories['threequarter_dag'] = edge_counts[3 * n_dags // 4][1]
    
    return categories


def generate_synthetic_data_quiet(model, n_samples, dag=None, n_permutations=3):
    """Generate synthetic data with TabPFN, suppressing output."""
    plt.ioff()
    plt.close('all')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        X_synthetic = model.generate_synthetic_data(
            n_samples=n_samples,
            t=1.0,
            n_permutations=n_permutations,
            dag=dag
        ).cpu().numpy()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close('all')
    
    return X_synthetic


def run_single_configuration(train_size, dag_level, repetition, config, 
                           X_test, dag_categories, col_names, categorical_cols):
    """
    Run one configuration: train_size + dag_level + repetition.
    """
    print(f"    DAG level: {dag_level}, Rep: {repetition+1}/{config['n_repetitions']}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data
    scm_gen = SCMGenerator(
        config_name="mixed" if config['include_categorical'] else "continuous",
        seed=seed
    )
    X_train, _, _, _ = scm_gen.generate_data(n_samples=train_size)
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Get the DAG to use
    dag_to_use = dag_categories[dag_level]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train model
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfen_reg=reg)
    
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    
    model.fit(X_train_tensor)
    
    # Generate synthetic data
    X_synthetic = generate_synthetic_data_quiet(
        model, 
        n_samples=X_test.shape[0], 
        dag=dag_to_use,
        n_permutations=config['n_permutations']
    )
    
    # Evaluate synthetic data
    evaluator = SyntheticDataEvaluator(X_test, categorical_cols)
    metrics = evaluator.evaluate(X_synthetic, plot=False)  # Disable plotting
    
    # Prepare result dictionary
    result = {
        'train_size': train_size,
        'dag_level': dag_level,
        'repetition': repetition,
        'categorical': config['include_categorical'],
        **metrics
    }
    
    # Add DAG structure info for analysis
    if dag_to_use is not None:
        result['dag_edges'] = sum(len(parents) for parents in dag_to_use.values())
        result['dag_nodes'] = len(dag_to_use)
        # Add the actual DAG structure in index notation for clarity
        result['dag_structure'] = str(dag_to_use)
    else:
        result['dag_edges'] = 0
        result['dag_nodes'] = 0
        result['dag_structure'] = 'None'
    
    return result


def save_checkpoint(results_so_far, current_config_idx, output_dir):
    checkpoint_path = Path(output_dir) / "experiment_4_checkpoint.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump((results_so_far, current_config_idx), f)


def load_checkpoint(output_dir):
    checkpoint_path = Path(output_dir) / "experiment_4_checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None, -1


def run_experiment_4(cpdag, config=None, output_dir="experiment_4_results", resume=True):
    """
    Run Experiment 4: Test TabPFN with different DAGs from a given CPDAG.
    
    Args:
        cpdag: CPDAG adjacency matrix from causal discovery.
        config: Dictionary with experiment settings.
        output_dir: Directory to save results.
        resume: Whether to resume from a checkpoint.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Load config and setup ---
    print("\n--- Step 1: Loading configuration ---")
    
    if config is None:
        # Provide a default config if none is given
        config = {
            'train_sizes': [50, 100, 200],
            'n_repetitions': 5,
            'test_size': 1000,
            'n_permutations': 3,
            'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
            'include_categorical': True,
            'n_estimators': 3,
            'random_seed_base': 42
        }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
        
    # Check for resumption
    checkpoint = load_checkpoint(output_dir) if resume else None
    if resume and checkpoint[0] is not None:
        results, last_completed_idx = checkpoint
        print(f"Resuming from checkpoint. Last completed configuration index: {last_completed_idx}")
    else:
        results = []
        last_completed_idx = -1

    # --- CPDAG to DAGs ---
    print("\n--- Step 2: Generating all DAGs from the discovered CPDAG ---")
    
    # The CPDAG is a numpy array, cpdag_to_dags will handle it
    all_dags = cpdag_to_dags(cpdag)
    
    if not all_dags:
        print("No valid DAGs could be generated from the CPDAG. Exiting.")
        final_df = pd.DataFrame(results)
        final_df.to_csv(Path(output_dir) / "raw_results_final.csv", index=False)
        if os.path.exists(Path(output_dir) / "experiment_4_checkpoint.pkl"):
            os.remove(Path(output_dir) / "experiment_4_checkpoint.pkl")
        return
        
    print(f"Generated {len(all_dags)} possible DAGs from the CPDAG.")
    
    dag_categories = categorize_dags_by_complexity(all_dags)
    print(f"DAGs categorized into {len(dag_categories)} levels: {list(dag_categories.keys())}")

    # --- Step 3: Prepare shared test data ---
    print("\n--- Step 3: Generating shared test data ---")
    scm_gen = SCMGenerator(
        config_name="mixed" if config['include_categorical'] else "continuous",
        seed=config['random_seed_base'] - 1  # Use a different seed for test set
    )
    X_test, _, categorical_cols, col_names = scm_gen.generate_data(n_samples=config['test_size'])
    print(f"Generated test data of size {X_test.shape}")
    if categorical_cols:
        print(f"Categorical columns identified: {categorical_cols}")

    # --- Step 4: Run experiment configurations ---
    print("\n--- Step 4: Running experiment configurations ---")
    
    configurations = []
    for train_size in config['train_sizes']:
        for dag_level in dag_categories.keys():
            for rep in range(config['n_repetitions']):
                configurations.append((train_size, dag_level, rep))

    total_configs = len(configurations)
    print(f"Total configurations to run: {total_configs}")

    for i, (train_size, dag_level, repetition) in enumerate(configurations):
        if i <= last_completed_idx:
            continue
            
        print(f"\nRunning config {i+1}/{total_configs}: Train size={train_size}, DAG level='{dag_level}', Rep={repetition+1}")
        
        result = run_single_configuration(
            train_size=train_size,
            dag_level=dag_level,
            repetition=repetition,
            config=config,
            X_test=X_test,
            dag_categories=dag_categories,
            col_names=col_names,
            categorical_cols=categorical_cols
        )
        results.append(result)
        
        # Checkpoint after each result
        save_checkpoint(results, i, output_dir)
        print(f"  Saved checkpoint for config {i+1}")

    # --- Step 5: Save final results ---
    print("\n--- Step 5: Saving final results ---")
    final_df = pd.DataFrame(results)
    final_df.to_csv(Path(output_dir) / "raw_results_final.csv", index=False)
    print(f"Final results saved to {Path(output_dir) / 'raw_results_final.csv'}")

    # Clean up checkpoint file
    checkpoint_path = Path(output_dir) / "experiment_4_checkpoint.pkl"
    if checkpoint_path.exists():
        os.remove(checkpoint_path)

    print("\nExperiment 4 finished successfully!") 