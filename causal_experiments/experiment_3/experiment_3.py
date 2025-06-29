"""
Experiment 3: Robustness to incorrect DAGs.

This experiment tests whether providing an incorrect DAG is better or worse
than providing no DAG at all. We compare multiple DAG conditions:
- correct: The true DAG
- no_dag: No DAG provided (vanilla TabPFN)
- wrong_parents: DAG with incorrect parent relationships
- missing_edges: DAG missing some true edges
- extra_edges: DAG with spurious edges added
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

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import SyntheticDataEvaluator
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info, create_dag_variations


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


def run_single_configuration(train_size, dag_type, repetition, config, 
                           X_test, correct_dag, col_names, categorical_cols,
                           dag_variations):
    """
    Run one configuration: train_size + dag_type + repetition.
    """
    print(f"    DAG type: {dag_type}, Rep: {repetition+1}/{config['n_repetitions']}")
    
    # Set seeds
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate training data
    X_train = generate_scm_data(train_size, seed, config['include_categorical'])
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Get the DAG to use
    dag_to_use = dag_variations[dag_type]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train model
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    
    model.fit(X_train_tensor)
    
    # Generate synthetic data with specified DAG
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag_to_use, config['n_permutations']
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate
    evaluator = SyntheticDataEvaluator(config['metrics'])
    metrics = evaluator.evaluate(X_test, X_synth, col_names, categorical_cols)
    
    # Build result
    result = {
        'train_size': train_size,
        'dag_type': dag_type,
        'repetition': repetition,
        'categorical': config['include_categorical']
    }
    
    # Add all metrics
    for metric, value in metrics.items():
        result[metric] = value
    
    # Add some DAG structure info for debugging
    if dag_to_use is not None:
        result['dag_edges'] = sum(len(parents) for parents in dag_to_use.values())
    else:
        result['dag_edges'] = 0
    
    return result


def save_checkpoint(results_so_far, current_config_idx, output_dir):
    """Save checkpoint for resuming."""
    checkpoint = {
        'results': results_so_far,
        'current_config_idx': current_config_idx
    }
    
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(output_dir):
    """Load checkpoint if exists."""
    checkpoint_file = Path(output_dir) / "checkpoint.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None


def run_experiment_3(config=None, output_dir="experiment_3_results", resume=True):
    """
    Main experiment function for testing robustness to incorrect DAGs.
    """
    # Default config
    if config is None:
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
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 3 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Show correct DAG structure
    print("\nCorrect DAG structure:")
    print_dag_info(correct_dag, col_names)
    
    # Create incorrect DAG variations using generic function
    dag_variations = create_dag_variations(correct_dag, random_state=config['random_seed_base'])
    
    # Show incorrect DAG variations
    print("\nDAG variations created:")
    for dag_type, dag in dag_variations.items():
        if dag_type != 'correct' and dag is not None:
            print(f"\n{dag_type}:")
            print_dag_info(dag, col_names)
    
    # Generate all configurations
    configurations = []
    for train_size in config['train_sizes']:
        for dag_type in config['dag_types']:
            for rep in range(config['n_repetitions']):
                configurations.append({
                    'train_size': train_size,
                    'dag_type': dag_type,
                    'repetition': rep
                })
    
    total_configs = len(configurations)
    print(f"\nTotal configurations: {total_configs}")
    
    # Check for checkpoint
    results_so_far = []
    start_idx = 0
    
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            print("Resuming from checkpoint!")
            results_so_far = checkpoint['results']
            start_idx = checkpoint['current_config_idx']
            print(f"  Resuming from configuration {start_idx}/{total_configs}")
    
    # Run experiment
    print(f"\nStarting experiment...")
    completed = len(results_so_far)
    
    try:
        for idx in range(start_idx, total_configs):
            cfg = configurations[idx]
            
            print(f"\n[{idx+1}/{total_configs}] Train size: {cfg['train_size']}, "
                  f"DAG: {cfg['dag_type']}, Rep: {cfg['repetition']+1}")
            
            result = run_single_configuration(
                cfg['train_size'], 
                cfg['dag_type'], 
                cfg['repetition'],
                config, 
                X_test, 
                correct_dag, 
                col_names, 
                categorical_cols,
                dag_variations
            )
            
            results_so_far.append(result)
            
            # Save to CSV incrementally
            df_current = pd.DataFrame(results_so_far)
            df_current.to_csv(output_dir / "raw_results.csv", index=False)
            
            # Save checkpoint
            save_checkpoint(results_so_far, idx + 1, output_dir)
            
            # Progress
            completed += 1
            print(f"    Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    # Experiment completed
    print("\nExperiment completed!")
    
    # Clean up checkpoint
    checkpoint_file = output_dir / "checkpoint.pkl"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    # Final results
    df_results = pd.DataFrame(results_so_far)
    df_results.to_csv(output_dir / "raw_results_final.csv", index=False)
    
    # Basic summary statistics
    print("\nBasic results summary:")
    print("=" * 60)
    
    # Group by DAG type and compute mean metrics
    for metric in config['metrics']:
        print(f"\n{metric} (mean ± std) by DAG type:")
        summary = df_results.groupby('dag_type')[metric].agg(['mean', 'std'])
        print(summary.round(4))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    return df_results


if __name__ == "__main__":
    # Test with standard config
    test_config = {
        'train_sizes': [50, 100],
        'dag_types': ['correct', 'no_dag', 'wrong_parents'],
        'n_repetitions': 2,
        'test_size': 500,
        'n_permutations': 2,
        'metrics': ['max_corr_diff', 'propensity_mse'],
        'include_categorical': False,
        'n_estimators': 3,
        'random_seed_base': 42
    }
    
    results = run_experiment_3(test_config, "test_exp3")