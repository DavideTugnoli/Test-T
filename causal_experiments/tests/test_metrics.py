import pytest
import numpy as np
import pandas as pd

# Adjust the path to import from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from causal_experiments.utils.metrics import SyntheticDataEvaluator

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_data():
    """Provides a consistent set of real and synthetic data for testing."""
    real_data = pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 100),
        'numeric2': np.random.uniform(0, 10, 100),
        'categorical1': np.random.randint(0, 3, 100),
        'categorical2': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # For simplicity, let's map 'A', 'B', 'C' to integers for synthetic data generation
    cat2_map = {'A': 0, 'B': 1, 'C': 2}
    real_data['categorical2'] = real_data['categorical2'].map(cat2_map)

    # Synthetic data that is slightly different
    synthetic_data = real_data.copy()
    synthetic_data['numeric1'] += np.random.normal(0, 0.1, 100)
    synthetic_data['categorical1'] = np.random.randint(0, 3, 100)
    
    categorical_columns = [2, 3]
    
    return real_data, synthetic_data, categorical_columns


# --- Test Suite ---

class TestSyntheticDataEvaluator:
    """Test suite for the SyntheticDataEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization with default, custom, and invalid metrics."""
        # Default initialization
        evaluator = SyntheticDataEvaluator()
        assert set(evaluator.metrics) == {'max_corr_diff', 'propensity_mse', 'kmarginal'}

        # Custom metrics
        evaluator_custom = SyntheticDataEvaluator(metrics=['kmarginal'])
        assert evaluator_custom.metrics == ['kmarginal']

        # Invalid metric should raise ValueError
        with pytest.raises(ValueError, match="Invalid metrics: {'invalid_metric'}"):
            SyntheticDataEvaluator(metrics=['invalid_metric'])

    def test_ensure_dataframe(self, sample_data):
        """Test the conversion of numpy arrays to pandas DataFrames."""
        real_data, _, _ = sample_data
        evaluator = SyntheticDataEvaluator()
        
        # Test with numpy array
        numpy_data = real_data.values
        df = evaluator._ensure_dataframe(numpy_data, column_names=real_data.columns.tolist())
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == real_data.columns.tolist()
        
        # Test with existing DataFrame
        df_copy = evaluator._ensure_dataframe(real_data, None)
        assert isinstance(df_copy, pd.DataFrame)
        assert real_data is not df_copy # Should be a copy

    def test_kmarginal_perfect_match(self, sample_data):
        """Test k-marginal distance with identical data, expecting a score of 0."""
        real_data, _, cat_cols = sample_data
        evaluator = SyntheticDataEvaluator()
        
        distance = evaluator._compute_kmarginal_distance(
            real_data, real_data.copy(), categorical_indices=cat_cols
        )
        assert np.isclose(distance, 0.0), "Distance for identical datasets should be 0."

    def test_kmarginal_total_mismatch(self):
        """Test k-marginal with completely disjoint data, expecting a high score."""
        real = pd.DataFrame({'A': [0]*100, 'B': [0]*100})
        synth = pd.DataFrame({'A': [1]*100, 'B': [1]*100})
        evaluator = SyntheticDataEvaluator()

        distance = evaluator._compute_kmarginal_distance(real, synth, k=2)
        assert np.isclose(distance, 1.0), "Distance for completely disjoint datasets should be 1.0."

    def test_kmarginal_k_too_large(self, sample_data):
        """Test that k-marginal raises an error if k is larger than the number of features."""
        real_data, synth_data, _ = sample_data
        evaluator = SyntheticDataEvaluator()
        
        with pytest.raises(ValueError, match="Cannot compute 5-marginals with only 4 features"):
            evaluator._compute_kmarginal_distance(real_data, synth_data, k=5) 