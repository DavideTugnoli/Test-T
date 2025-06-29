"""
Tests for DAG utility functions.

This test suite covers all the DAG manipulation and analysis
functions defined in `causal_experiments.utils.dag_utils`.
"""

import pytest
import numpy as np

from ..utils.dag_utils import (
    topological_sort,
    get_worst_ordering,
    count_violations,
    get_random_ordering,
    reorder_data_and_dag,
    validate_dag,
    convert_named_dag_to_indices,
    convert_indices_dag_to_named,
    create_wrong_parents_dag,
    create_missing_edges_dag,
    create_extra_edges_dag,
    create_dag_variations,
    cpdag_to_dags,
    convert_cpdag_to_named_dags,
    dag_belongs_to_cpdag,
)

# --- Test Fixtures ---

@pytest.fixture
def simple_dag():
    """A simple DAG: 0 -> 1 <- 2"""
    return {1: [0, 2], 0: [], 2: []}


@pytest.fixture
def chain_dag():
    """A chain DAG: 3 -> 2 -> 1 -> 0"""
    return {0: [1], 1: [2], 2: [3], 3: []}


@pytest.fixture
def named_dag_and_cols():
    """A more complex DAG with names and corresponding column list."""
    named_dag = {
        "X1": [],
        "X2": ["X1", "X3"],
        "X3": ["X4"],
        "X4": []
    }
    columns = ["X1", "X2", "X3", "X4"]
    return named_dag, columns


@pytest.fixture
def indexed_dag(named_dag_and_cols):
    """Index-based version of the complex named_dag."""
    named_dag, columns = named_dag_and_cols
    return convert_named_dag_to_indices(named_dag, columns)


@pytest.fixture
def test_cpdag():
    """A test CPDAG: 0 -> 1 -- 2, with 3 isolated.
    
    This represents the equivalence class containing:
    - DAG1: 0 -> 1 -> 2, with 3 isolated  
    - DAG2: 0 -> 1 <- 2, with 3 isolated
    """
    # CPDAG adjacency matrix from causallearn.pc format
    # 0 -> 1: adj[0,1]=-1, adj[1,0]=1 (directed)
    # 1--2: adj[1,2]=-1, adj[2,1]=-1 (undirected)
    # 3 isolated: all zeros
    return np.array([
        [0, -1, 0, 0],  # 0: points to 1
        [1, 0, -1, 0],  # 1: receives from 0, undirected with 2
        [0, -1, 0, 0],  # 2: undirected with 1
        [0, 0, 0, 0]    # 3: isolated
    ])


@pytest.fixture
def cpdag_column_names():
    """Column names for the test CPDAG."""
    return ["X0", "X1", "X2", "X3"]


# --- Test Functions ---

class TestDAGUtils:
    """Test suite for DAG utility functions."""

    def test_topological_sort(self, chain_dag):
        """Test that topological sort produces a valid ordering."""
        topo_order = topological_sort(chain_dag)
        assert topo_order == [3, 2, 1, 0]
        
        # Verify that for every edge, parent comes before child
        pos = {node: i for i, node in enumerate(topo_order)}
        for child, parents in chain_dag.items():
            for parent in parents:
                assert pos[parent] < pos[child]

    def test_topological_sort_cycle(self):
        """Test that topological sort raises an error for cyclic graphs."""
        cyclic_dag = {0: [1], 1: [0]}
        with pytest.raises(ValueError, match="DAG contains cycles!"):
            topological_sort(cyclic_dag)

    def test_get_worst_ordering(self, chain_dag):
        """Test that worst ordering is the reverse of topological."""
        worst_order = get_worst_ordering(chain_dag)
        assert worst_order == [0, 1, 2, 3]

    def test_count_violations(self, chain_dag):
        """Test the counting of causal violations."""
        assert count_violations(chain_dag, [3, 2, 1, 0]) == 0  # Topological
        assert count_violations(chain_dag, [0, 1, 2, 3]) == 3  # Worst
        assert count_violations(chain_dag, [3, 1, 2, 0]) == 1  # 2 should be before 1

    def test_get_random_ordering(self, simple_dag):
        """Test random ordering generation."""
        random_order_1 = get_random_ordering(simple_dag, random_state=42)
        random_order_2 = get_random_ordering(simple_dag, random_state=1337)
        assert set(random_order_1) == {0, 1, 2}
        assert random_order_1 != random_order_2

    def test_reorder_data_and_dag(self, indexed_dag):
        """Test reordering of data and DAG indices."""
        data = np.random.randn(10, 4)
        
        # This topological sort is not guaranteed to be unique.
        # Original ordering of columns was ["X1", "X2", "X3", "X4"]
        # One valid topological sort is [0, 3, 2, 1] -> [X1, X4, X3, X2]
        # another is [3, 0, 2, 1] -> [X4, X1, X3, X2]
        new_ordering = topological_sort(indexed_dag)
        
        reordered_data, reordered_dag = reorder_data_and_dag(data, indexed_dag, new_ordering)

        # Check data columns are correctly reordered
        assert np.array_equal(reordered_data, data[:, new_ordering])
        
        # To validate the reordered DAG, we convert it back to its named representation
        # which should be invariant to the topological sort used for reordering.
        original_columns = ["X1", "X2", "X3", "X4"] # Based on the fixture
        new_columns = [original_columns[i] for i in new_ordering]
        
        # Convert the reordered (indexed) DAG back to a named DAG
        named_reordered_dag = convert_indices_dag_to_named(reordered_dag, new_columns)
        
        # The named DAG should be identical to the original, regardless of internal indexing
        original_named_dag = {
            "X1": [],
            "X2": ["X1", "X3"],
            "X3": ["X4"],
            "X4": []
        }
        
        # Sort parent lists to ensure comparison is order-independent
        for k in original_named_dag:
            original_named_dag[k].sort()
        for k in named_reordered_dag:
            named_reordered_dag[k].sort()
            
        assert named_reordered_dag == original_named_dag

    def test_validate_dag(self, simple_dag):
        """Test DAG validation for acyclicity."""
        assert validate_dag(simple_dag)
        cyclic_dag = {0: [1], 1: [0]}
        assert not validate_dag(cyclic_dag)

    def test_dag_conversions(self, named_dag_and_cols, indexed_dag):
        """Test conversion between named and indexed DAGs."""
        named_dag, columns = named_dag_and_cols
        
        # Test forward conversion
        converted_indexed = convert_named_dag_to_indices(named_dag, columns)
        assert converted_indexed == indexed_dag

        # Test reverse conversion
        converted_back_named = convert_indices_dag_to_named(indexed_dag, columns)
        assert converted_back_named == named_dag

    def test_create_wrong_parents_dag(self, indexed_dag):
        """Test creation of a DAG with shuffled parent relationships."""
        wrong_dag = create_wrong_parents_dag(indexed_dag, random_state=42)
        
        assert validate_dag(wrong_dag)
        original_edges = sum(len(p) for p in indexed_dag.values())
        wrong_edges = sum(len(p) for p in wrong_dag.values())
        assert original_edges == wrong_edges
        assert wrong_dag != indexed_dag

    def test_create_missing_edges_dag(self, indexed_dag):
        """Test creation of a DAG with removed edges."""
        missing_dag = create_missing_edges_dag(indexed_dag, removal_fraction=0.5, random_state=42)
        
        assert validate_dag(missing_dag)
        original_edges = sum(len(p) for p in indexed_dag.values()) # 3 edges
        missing_edges = sum(len(p) for p in missing_dag.values())
        assert missing_edges < original_edges
        # int(3 * 0.5) = 1 edge removed, 2 should remain
        assert missing_edges == 2

    def test_create_extra_edges_dag(self, indexed_dag):
        """Test creation of a DAG with added edges."""
        extra_dag = create_extra_edges_dag(indexed_dag, addition_fraction=0.5, random_state=42)
        
        assert validate_dag(extra_dag)
        original_edges = sum(len(p) for p in indexed_dag.values()) # 3 edges
        extra_edges = sum(len(p) for p in extra_dag.values())
        assert extra_edges > original_edges
        # int(3 * 0.5) = 1 edge added, 4 total
        assert extra_edges == 4

    def test_create_dag_variations(self, indexed_dag):
        """Test the generation of all standard DAG variations."""
        variations = create_dag_variations(indexed_dag, random_state=42)
        
        assert 'correct' in variations
        assert 'no_dag' in variations
        assert 'wrong_parents' in variations
        assert 'missing_edges' in variations
        assert 'extra_edges' in variations
        
        assert variations['correct'] == indexed_dag
        assert variations['no_dag'] is None
        assert validate_dag(variations['wrong_parents'])
        assert validate_dag(variations['missing_edges'])
        assert validate_dag(variations['extra_edges'])

    def test_cpdag_to_dags(self):
        """Test the expansion of a CPDAG into all possible DAGs."""
        # Represents: 0 -> 1 -- 2, with 3 isolated
        cpdag_adj = np.array([
            [0, -1, 0, 0],
            [1, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0]
        ])
        
        dags = cpdag_to_dags(cpdag_adj)
        
        # Expected DAGs are: 0->1->2 and 0->1<-2
        expected_dag1 = {0: [], 1: [0], 2: [1], 3: []}
        expected_dag2 = {0: [], 1: [0, 2], 2: [], 3: []}
        
        # Use a comparable format (set of frozensets of items) to ignore dict key order
        def dag_to_comparable(dag):
            return frozenset((k, frozenset(v)) for k, v in dag.items())

        found_dags_set = {dag_to_comparable(d) for d in dags}
        
        assert len(dags) == 2
        assert dag_to_comparable(expected_dag1) in found_dags_set
        assert dag_to_comparable(expected_dag2) in found_dags_set

    def test_convert_cpdag_to_named_dags(self):
        """Test the conversion of a CPDAG to named DAGs."""
        cpdag_adj = np.array([
            [0, -1, 0, 0],
            [1, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0]
        ])
        col_names = ["X0", "X1", "X2", "X3"]
        
        named_dags = convert_cpdag_to_named_dags(cpdag_adj, col_names)
        
        expected_named1 = {'X0': [], 'X1': ['X0'], 'X2': ['X1'], 'X3': []}
        expected_named2 = {'X0': [], 'X1': ['X0', 'X2'], 'X2': [], 'X3': []}

        def named_dag_to_comparable(dag):
             return frozenset((k, frozenset(v)) for k, v in dag.items())

        found_named_dags_set = {named_dag_to_comparable(d) for d in named_dags}

        assert len(named_dags) == 2
        assert named_dag_to_comparable(expected_named1) in found_named_dags_set
        assert named_dag_to_comparable(expected_named2) in found_named_dags_set

    def test_dag_belongs_to_cpdag_positive_case(self, test_cpdag):
        """Test that a DAG that belongs to the CPDAG is correctly identified."""
        # DAG1: 0 -> 1 -> 2, with 3 isolated
        dag1 = {0: [], 1: [0], 2: [1], 3: []}
        assert dag_belongs_to_cpdag(dag1, test_cpdag)
        
        # DAG2: 0 -> 1 <- 2, with 3 isolated
        dag2 = {0: [], 1: [0, 2], 2: [], 3: []}
        assert dag_belongs_to_cpdag(dag2, test_cpdag)

    def test_dag_belongs_to_cpdag_negative_case(self, test_cpdag):
        """Test that a DAG that doesn't belong to the CPDAG is correctly rejected."""
        # Wrong DAG: 1 -> 0 -> 2 (reverses the required 0 -> 1 direction)
        wrong_dag1 = {0: [1], 1: [], 2: [0], 3: []}
        assert not dag_belongs_to_cpdag(wrong_dag1, test_cpdag)
        
        # Wrong DAG: 0 -> 2 -> 1 (wrong edge structure)
        wrong_dag2 = {0: [], 1: [2], 2: [0], 3: []}
        assert not dag_belongs_to_cpdag(wrong_dag2, test_cpdag)
        
        # Wrong DAG: extra edge 0 -> 3
        wrong_dag3 = {0: [], 1: [0], 2: [1], 3: [0]}
        assert not dag_belongs_to_cpdag(wrong_dag3, test_cpdag)

    def test_dag_belongs_to_cpdag_with_named_dag(self, test_cpdag, cpdag_column_names):
        """Test the function with named DAGs."""
        # Named DAG that should belong: X0 -> X1 -> X2
        named_dag1 = {'X0': [], 'X1': ['X0'], 'X2': ['X1'], 'X3': []}
        assert dag_belongs_to_cpdag(named_dag1, test_cpdag, cpdag_column_names)
        
        # Named DAG that should belong: X0 -> X1 <- X2
        named_dag2 = {'X0': [], 'X1': ['X0', 'X2'], 'X2': [], 'X3': []}
        assert dag_belongs_to_cpdag(named_dag2, test_cpdag, cpdag_column_names)
        
        # Named DAG that shouldn't belong
        wrong_named_dag = {'X0': ['X1'], 'X1': [], 'X2': ['X0'], 'X3': []}
        assert not dag_belongs_to_cpdag(wrong_named_dag, test_cpdag, cpdag_column_names)

    def test_dag_belongs_to_cpdag_reordered_nodes(self, test_cpdag):
        """Test that function handles DAGs with nodes in different order."""
        # Same DAG but with nodes defined in different order
        dag_reordered = {3: [], 2: [1], 1: [0], 0: []}
        assert dag_belongs_to_cpdag(dag_reordered, test_cpdag)

    def test_dag_belongs_to_cpdag_missing_nodes(self, test_cpdag):
        """Test behavior when DAG has missing or extra nodes."""
        # DAG missing node 3 (should still match if other structure is correct)
        dag_missing_node = {0: [], 1: [0], 2: [1]}
        # This should fail because the CPDAG expects 4 nodes
        assert not dag_belongs_to_cpdag(dag_missing_node, test_cpdag)
        
        # DAG with extra node
        dag_extra_node = {0: [], 1: [0], 2: [1], 3: [], 4: []}
        assert not dag_belongs_to_cpdag(dag_extra_node, test_cpdag)

    def test_dag_belongs_to_cpdag_empty_cases(self):
        """Test edge cases with empty DAGs and CPDAGs."""
        # Empty DAG and empty CPDAG
        empty_dag = {}
        empty_cpdag = np.zeros((0, 0))
        assert dag_belongs_to_cpdag(empty_dag, empty_cpdag)
        
        # Single isolated node
        single_node_dag = {0: []}
        single_node_cpdag = np.zeros((1, 1))
        assert dag_belongs_to_cpdag(single_node_dag, single_node_cpdag)

    def test_dag_belongs_to_cpdag_all_isolated(self):
        """Test with CPDAG containing only isolated nodes."""
        # All isolated nodes
        isolated_dag = {0: [], 1: [], 2: []}
        isolated_cpdag = np.zeros((3, 3))
        assert dag_belongs_to_cpdag(isolated_dag, isolated_cpdag)
        
        # DAG with edges when CPDAG has none - should fail
        dag_with_edges = {0: [], 1: [0], 2: []}
        assert not dag_belongs_to_cpdag(dag_with_edges, isolated_cpdag)

    def test_dag_belongs_to_cpdag_complex_case(self):
        """Test with a more complex CPDAG with multiple undirected edges."""
        # CPDAG: 0 -- 1 -- 2 (all undirected)
        complex_cpdag = np.array([
            [0, -1, 0],
            [-1, 0, -1],
            [0, -1, 0]
        ])
        
        # All 8 possible orientations should be valid:
        # 0->1->2, 0->1<-2, 0<-1->2, 0<-1<-2, etc.
        valid_dags = [
            {0: [], 1: [0], 2: [1]},      # 0->1->2
            {0: [], 1: [0, 2], 2: []},    # 0->1<-2  
            {0: [1], 1: [], 2: [1]},      # 0<-1->2
            {0: [1], 1: [2], 2: []},      # 0<-1<-2
        ]
        
        for dag in valid_dags:
            assert dag_belongs_to_cpdag(dag, complex_cpdag)
        
        # Invalid DAG: has cycle 0->1->2->0
        invalid_dag = {0: [2], 1: [0], 2: [1]}
        assert not dag_belongs_to_cpdag(invalid_dag, complex_cpdag) 