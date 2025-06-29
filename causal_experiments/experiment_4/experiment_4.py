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

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports - add parent directory to path for utils
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import SyntheticDataEvaluator
from utils.dag_utils import (
    cpdag_to_dags, 
    dag_belongs_to_cpdag, 
    convert_named_dag_to_indices,
    print_dag_info
)

# No longer need causal discovery imports - using dummy CPDAG generation


def create_cpdag_from_true_dag(true_dag, ambiguity_level=0.3, random_state=42):
    """
    Create a CPDAG from the true DAG by introducing controlled ambiguity.
    
    This creates a realistic CPDAG that contains the true DAG in its equivalence 
    class, plus other plausible DAGs with different edge orientations.
    
    Args:
        true_dag: True DAG structure {node: [parents]}
        ambiguity_level: Fraction of edges to make undirected (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        CPDAG adjacency matrix (numpy array) in causallearn format
    """
    print(f"    Creating CPDAG from true DAG (ambiguity={ambiguity_level})...")
    
    rng = np.random.default_rng(random_state)
    
    # Get all nodes
    all_nodes = set(true_dag.keys())
    for parents in true_dag.values():
        all_nodes.update(parents)
    all_nodes = sorted(list(all_nodes))
    n_nodes = len(all_nodes)
    
    # Initialize CPDAG matrix (all zeros)
    cpdag = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Collect all edges from true DAG
    edges = []
    for child, parents in true_dag.items():
        for parent in parents:
            edges.append((parent, child))
    
    if not edges:
        print("    No edges in true DAG - returning empty CPDAG")
        return cpdag
    
    # Decide which edges to make ambiguous (undirected)
    n_ambiguous = int(len(edges) * ambiguity_level)
    ambiguous_edges = rng.choice(len(edges), size=n_ambiguous, replace=False)
    
    print(f"    Making {n_ambiguous}/{len(edges)} edges undirected for ambiguity")
    
    # Fill CPDAG matrix
    for i, (parent, child) in enumerate(edges):
        if i in ambiguous_edges:
            # Make this edge undirected (both directions = -1)
            cpdag[parent, child] = -1
            cpdag[child, parent] = -1
        else:
            # Keep this edge directed (parent->child: parent=-1, child=1)
            cpdag[parent, child] = -1
            cpdag[child, parent] = 1
    
    print(f"    Created CPDAG with {np.sum(cpdag != 0) // 2} edges ({n_ambiguous} undirected)")
    return cpdag


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
    X_train = generate_scm_data(train_size, seed, config['include_categorical'])
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Get the DAG to use
    dag_to_use = dag_categories[dag_level]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train model
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    
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


def run_experiment_4(config=None, output_dir="experiment_4_results", resume=True):
    """
    Main experiment function for testing causal knowledge level impact.
    """
    # Default config
    if config is None:
        config = {
            'train_sizes': [50, 100, 200, 500],
            'n_repetitions': 10,
            'test_size': 2000,
            'n_permutations': 3,
            'metrics': ['max_corr_diff', 'propensity_mse', 'kmarginal'],
            'include_categorical': False,
            'n_estimators': 3,
            'random_seed_base': 42,
            'cpdag_ambiguity_level': 0.3  # Fraction of edges to make undirected in CPDAG
        }
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 4 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup: Get true DAG and generate test data
    true_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    print("\nTrue DAG structure (for reference):")
    print_dag_info(true_dag, col_names)
    
    # CPDAG Creation: Create CPDAG from true DAG with controlled ambiguity
    print(f"\nStep 1: Create CPDAG from True DAG")
    print("-" * 50)
    
    cpdag_matrix = create_cpdag_from_true_dag(
        true_dag, 
        config['cpdag_ambiguity_level'],
        config['random_seed_base']
    )
    
    print(f"    CPDAG shape: {cpdag_matrix.shape}")
    print(f"    CPDAG edges: {np.sum(cpdag_matrix != 0) // 2}")
    
    # Generate all possible DAGs from CPDAG
    print("\nStep 2: Generate DAGs from CPDAG")
    print("-" * 50)
    
    try:
        possible_dags = cpdag_to_dags(cpdag_matrix)
        print(f"    Found {len(possible_dags)} possible DAGs")
    except Exception as e:
        print(f"    Failed to generate DAGs: {e}")
        print("    Using only true DAG as fallback")
        possible_dags = [true_dag]
    
    # Show discovered DAGs
    for i, dag in enumerate(possible_dags[:5]):  # Show first 5
        edge_count = sum(len(parents) for parents in dag.values())
        print(f"    DAG {i+1}: {edge_count} edges")
        if i < 3:  # Detailed info for first 3
            print_dag_info(dag, col_names)
    
    if len(possible_dags) > 5:
        print(f"    ... and {len(possible_dags) - 5} more DAGs")
    
    # Categorize DAGs by complexity
    print("\nStep 3: Categorize DAGs by complexity")
    print("-" * 50)
    
    dag_categories = categorize_dags_by_complexity(possible_dags)
    
    print("DAG categories created:")
    for category, dag in dag_categories.items():
        if dag is None:
            print(f"  {category}: No DAG (vanilla TabPFN)")
        else:
            edge_count = sum(len(parents) for parents in dag.values())
            print(f"  {category}: {edge_count} edges")
    
    # Check if true DAG belongs to created CPDAG (for validation)
    if len(possible_dags) > 0:
        belongs = dag_belongs_to_cpdag(true_dag, cpdag_matrix)
        print(f"\nValidation: True DAG belongs to created CPDAG: {belongs}")
    
    # Load checkpoint if resuming
    start_config_idx = 0
    all_results = []
    
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint is not None:
            all_results = checkpoint['results']
            start_config_idx = checkpoint['current_config_idx']
            print(f"\nResuming from configuration {start_config_idx}")
            print(f"Already completed: {len(all_results)} results")
    
    # Generate all configurations
    configurations = []
    dag_levels = list(dag_categories.keys())
    
    for train_size in config['train_sizes']:
        for dag_level in dag_levels:
            for rep in range(config['n_repetitions']):
                configurations.append({
                    'train_size': train_size,
                    'dag_level': dag_level,
                    'repetition': rep
                })
    
    total_configs = len(configurations)
    print(f"\nStep 4: Run Experiment")
    print("-" * 50)
    print(f"Total configurations: {total_configs}")
    print(f"DAG levels: {dag_levels}")
    print(f"Training sizes: {config['train_sizes']}")
    print(f"Repetitions: {config['n_repetitions']}")
    
    # Run configurations
    try:
        for config_idx in range(start_config_idx, total_configs):
            config_data = configurations[config_idx]
            
            print(f"\nConfiguration {config_idx + 1}/{total_configs}:")
            print(f"  Train size: {config_data['train_size']}")
            
            result = run_single_configuration(
                config_data['train_size'],
                config_data['dag_level'],
                config_data['repetition'],
                config,
                X_test,
                dag_categories,
                col_names,
                categorical_cols
            )
            
            all_results.append(result)
            
            # Save checkpoint every 10 configurations
            if (config_idx + 1) % 10 == 0:
                save_checkpoint(all_results, config_idx + 1, output_dir)
                print(f"    Checkpoint saved (completed {config_idx + 1}/{total_configs})")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        save_checkpoint(all_results, config_idx, output_dir)
        print(f"Progress saved. Completed {len(all_results)} configurations.")
        return None
    
    # Save final results
    print(f"\nExperiment completed! Total results: {len(all_results)}")
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    
    # Save detailed results
    results_file = output_dir / "raw_results_final.csv"
    df_results.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    # Generate summary report
    report_file = output_dir / "experiment_4_report.txt"
    with open(report_file, 'w') as f:
        f.write("EXPERIMENT 4: Causal Knowledge Level Impact\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Research Question:\n")
        f.write("How does the level of causal knowledge (derived from CPDAG) affect TabPFN's performance?\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Create CPDAG from true DAG with controlled ambiguity\n")
        f.write("2. Generate all possible DAGs from the CPDAG\n")
        f.write("3. Categorize DAGs by complexity (number of edges)\n")
        f.write("4. Test TabPFN with different levels of causal knowledge\n\n")
        
        f.write(f"CPDAG Ambiguity Level: {config['cpdag_ambiguity_level']}\n")
        f.write(f"Generated DAGs: {len(possible_dags)}\n\n")
        
        f.write("DAG Categories:\n")
        for category, dag in dag_categories.items():
            if dag is None:
                f.write(f"  {category}: No DAG (vanilla)\n")
            else:
                edge_count = sum(len(parents) for parents in dag.values())
                f.write(f"  {category}: {edge_count} edges\n")
        f.write("\n")
        
        f.write("Performance Summary:\n")
        f.write("-" * 30 + "\n")
        
        for metric in config['metrics']:
            f.write(f"\n{metric.upper()}:\n")
            
            # Mean performance by DAG level
            mean_by_level = df_results.groupby('dag_level')[metric].mean()
            sorted_levels = mean_by_level.sort_values()
            
            f.write("Performance ranking (best to worst):\n")
            for i, (level, value) in enumerate(sorted_levels.items(), 1):
                f.write(f"  {i}. {level}: {value:.4f}\n")
            
            # Performance vs vanilla
            if 'no_dag' in mean_by_level:
                vanilla_perf = mean_by_level['no_dag']
                f.write(f"\nComparison to vanilla ({vanilla_perf:.4f}):\n")
                
                for level in dag_categories.keys():
                    if level != 'no_dag':
                        diff = mean_by_level[level] - vanilla_perf
                        pct_change = (diff / vanilla_perf) * 100
                        f.write(f"  {level}: {diff:+.4f} ({pct_change:+.1f}%)\n")
    
    print(f"Report saved to: {report_file}")
    
    # Clean up checkpoint
    checkpoint_file = output_dir / "checkpoint.pkl"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return df_results 