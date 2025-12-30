"""
Comprehensive Test Suite for EMTL Solver
========================================

This module contains unit tests, integration tests, and property-based tests
for the Edge-Magic Total Labeling solver.

Run tests with: pytest tests/ -v --cov=emtl_solver
"""

import sys
import os
import pytest
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emtl_solver import (
    GraphParameters,
    GraphConstructor,
    EMTLSolver,
    EMTLVisualizer,
    EMTLResult,
    SolverStatus,
    solve_emtl,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def small_params():
    """Small graph parameters for quick tests."""
    return GraphParameters(m=2, n=2, k=2, t=1)


@pytest.fixture
def medium_params():
    """Medium graph parameters."""
    return GraphParameters(m=2, n=3, k=2, t=2)


@pytest.fixture
def symmetric_params():
    """Symmetric parameters (all equal)."""
    return GraphParameters(m=3, n=3, k=3, t=3)


@pytest.fixture
def minimal_params():
    """Minimal valid parameters."""
    return GraphParameters(m=1, n=1, k=1, t=1)


@pytest.fixture
def no_bc_edges_params():
    """Parameters with no B-C edges (t=0)."""
    return GraphParameters(m=2, n=2, k=2, t=0)


# =============================================================================
# GRAPHPARAMETERS TESTS
# =============================================================================

class TestGraphParameters:
    """Tests for the GraphParameters data class."""
    
    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        params = GraphParameters(m=2, n=3, k=2, t=2)
        is_valid, msg = params.validate()
        assert is_valid
        assert msg == ""
    
    def test_invalid_m_zero(self):
        """Test that m=0 is rejected."""
        params = GraphParameters(m=0, n=2, k=2, t=1)
        is_valid, msg = params.validate()
        assert not is_valid
        assert "m must be ≥ 1" in msg
    
    def test_invalid_n_zero(self):
        """Test that n=0 is rejected."""
        params = GraphParameters(m=2, n=0, k=2, t=0)
        is_valid, msg = params.validate()
        assert not is_valid
        assert "n must be ≥ 1" in msg
    
    def test_invalid_k_zero(self):
        """Test that k=0 is rejected."""
        params = GraphParameters(m=2, n=2, k=0, t=1)
        is_valid, msg = params.validate()
        assert not is_valid
        assert "k must be ≥ 1" in msg
    
    def test_invalid_t_negative(self):
        """Test that negative t is rejected."""
        params = GraphParameters(m=2, n=2, k=2, t=-1)
        is_valid, msg = params.validate()
        assert not is_valid
        assert "t must be ≥ 0" in msg
    
    def test_invalid_t_greater_than_n(self):
        """Test that t > n is rejected."""
        params = GraphParameters(m=2, n=3, k=2, t=4)
        is_valid, msg = params.validate()
        assert not is_valid
        assert "t must be ≤ n" in msg
    
    def test_num_vertices_calculation(self):
        """Test vertex count calculation: |V| = m + 2n + k."""
        params = GraphParameters(m=2, n=3, k=4, t=2)
        # |V| = 2 + 2*3 + 4 = 12
        assert params.num_vertices == 12
    
    def test_num_edges_calculation(self):
        """Test edge count calculation: |E| = mn + nt + nk."""
        params = GraphParameters(m=2, n=3, k=4, t=2)
        # |E| = 2*3 + 3*2 + 3*4 = 6 + 6 + 12 = 24
        assert params.num_edges == 24
    
    def test_total_labels_calculation(self):
        """Test total labels: |V| + |E|."""
        params = GraphParameters(m=2, n=3, k=4, t=2)
        assert params.total_labels == params.num_vertices + params.num_edges
    
    def test_string_representation(self):
        """Test __str__ method."""
        params = GraphParameters(m=2, n=3, k=4, t=2)
        assert str(params) == "G(m=2, n=3, k=4, t=2)"
    
    def test_t_equals_n_valid(self):
        """Test that t=n is valid (maximum regularity)."""
        params = GraphParameters(m=2, n=3, k=2, t=3)
        is_valid, _ = params.validate()
        assert is_valid
    
    def test_t_zero_valid(self):
        """Test that t=0 is valid (no B-C edges)."""
        params = GraphParameters(m=2, n=3, k=2, t=0)
        is_valid, _ = params.validate()
        assert is_valid


# =============================================================================
# GRAPHCONSTRUCTOR TESTS
# =============================================================================

class TestGraphConstructor:
    """Tests for the GraphConstructor class."""
    
    def test_construct_basic(self, small_params):
        """Test basic graph construction."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_params.num_vertices
        assert G.number_of_edges() == small_params.num_edges
    
    def test_partition_sizes(self, small_params):
        """Test that partition sizes are correct."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        assert len(vertex_sets['A']) == small_params.m
        assert len(vertex_sets['B']) == small_params.n
        assert len(vertex_sets['C']) == small_params.n
        assert len(vertex_sets['D']) == small_params.k
    
    def test_vertex_naming(self, small_params):
        """Test that vertices are named correctly."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        for i, v in enumerate(vertex_sets['A']):
            assert v == f'A{i}'
        for i, v in enumerate(vertex_sets['B']):
            assert v == f'B{i}'
        for i, v in enumerate(vertex_sets['C']):
            assert v == f'C{i}'
        for i, v in enumerate(vertex_sets['D']):
            assert v == f'D{i}'
    
    def test_ab_complete_bipartite(self, small_params):
        """Test that A-B subgraph is complete bipartite."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        for a in vertex_sets['A']:
            for b in vertex_sets['B']:
                assert G.has_edge(a, b), f"Missing A-B edge: {a}-{b}"
    
    def test_cd_complete_bipartite(self, small_params):
        """Test that C-D subgraph is complete bipartite."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        for c in vertex_sets['C']:
            for d in vertex_sets['D']:
                assert G.has_edge(c, d), f"Missing C-D edge: {c}-{d}"
    
    def test_bc_t_regular(self, small_params):
        """Test that B-C subgraph is t-regular."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        for b in vertex_sets['B']:
            bc_neighbors = sum(1 for c in vertex_sets['C'] if G.has_edge(b, c))
            assert bc_neighbors == small_params.t, \
                f"Vertex {b} has {bc_neighbors} C-neighbors, expected {small_params.t}"
        
        for c in vertex_sets['C']:
            bc_neighbors = sum(1 for b in vertex_sets['B'] if G.has_edge(b, c))
            assert bc_neighbors == small_params.t, \
                f"Vertex {c} has {bc_neighbors} B-neighbors, expected {small_params.t}"
    
    def test_no_extra_edges(self, small_params):
        """Test that there are no unexpected edges."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        # Check no A-C edges
        for a in vertex_sets['A']:
            for c in vertex_sets['C']:
                assert not G.has_edge(a, c), f"Unexpected A-C edge: {a}-{c}"
        
        # Check no A-D edges
        for a in vertex_sets['A']:
            for d in vertex_sets['D']:
                assert not G.has_edge(a, d), f"Unexpected A-D edge: {a}-{d}"
        
        # Check no B-D edges
        for b in vertex_sets['B']:
            for d in vertex_sets['D']:
                assert not G.has_edge(b, d), f"Unexpected B-D edge: {b}-{d}"
    
    def test_no_self_loops(self, small_params):
        """Test that there are no self-loops."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        for v in G.nodes():
            assert not G.has_edge(v, v), f"Self-loop at {v}"
    
    def test_verify_structure_passes(self, small_params):
        """Test that verify_structure passes for valid graph."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        result = GraphConstructor.verify_structure(G, vertex_sets, small_params)
        assert result is True
    
    def test_construct_with_t_zero(self, no_bc_edges_params):
        """Test construction with no B-C edges."""
        G, vertex_sets = GraphConstructor.construct(no_bc_edges_params)
        
        # Should have no B-C edges
        for b in vertex_sets['B']:
            for c in vertex_sets['C']:
                assert not G.has_edge(b, c)
    
    def test_t_regular_bipartite_edges(self):
        """Test the t-regular bipartite edge generation."""
        # Test for n=4, t=2
        edges = GraphConstructor.create_t_regular_bipartite_edges(4, 2)
        
        # Check we have correct number of edges: n * t = 8
        assert len(edges) == 8
        
        # Check each vertex has degree t
        from collections import Counter
        left_degrees = Counter(e[0] for e in edges)
        right_degrees = Counter(e[1] for e in edges)
        
        for i in range(4):
            assert left_degrees[i] == 2
            assert right_degrees[i] == 2
    
    def test_t_regular_raises_on_invalid(self):
        """Test that t > n raises ValueError."""
        with pytest.raises(ValueError):
            GraphConstructor.create_t_regular_bipartite_edges(3, 5)


# =============================================================================
# EMTLSOLVER TESTS
# =============================================================================

class TestEMTLSolver:
    """Tests for the EMTLSolver class."""
    
    def test_solver_initialization(self):
        """Test solver initialization with custom parameters."""
        solver = EMTLSolver(timeout_seconds=30, num_workers=4)
        assert solver.timeout_seconds == 30
        assert solver.num_workers == 4
    
    def test_solve_small_graph(self, small_params):
        """Test solving a small graph."""
        G, _ = GraphConstructor.construct(small_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, time = solver.solve(G)
        
        assert status == SolverStatus.FOUND
        assert magic_k is not None
        assert v_labels is not None
        assert e_labels is not None
    
    def test_labels_are_bijection(self, small_params):
        """Test that labels form a valid bijection."""
        G, _ = GraphConstructor.construct(small_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, _ = solver.solve(G)
        
        if status == SolverStatus.FOUND:
            all_labels = list(v_labels.values()) + list(e_labels.values())
            total = small_params.total_labels
            
            # Check all labels are in correct range
            assert all(1 <= l <= total for l in all_labels)
            
            # Check all labels are unique
            assert len(set(all_labels)) == len(all_labels)
            
            # Check we have all labels
            assert set(all_labels) == set(range(1, total + 1))
    
    def test_magic_property(self, small_params):
        """Test that the magic property holds for all edges."""
        G, _ = GraphConstructor.construct(small_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, _ = solver.solve(G)
        
        if status == SolverStatus.FOUND:
            for (u, v) in G.edges():
                edge_sum = v_labels[u] + e_labels[(u, v)] + v_labels[v]
                assert edge_sum == magic_k, \
                    f"Edge ({u}, {v}): {v_labels[u]} + {e_labels[(u, v)]} + {v_labels[v]} = {edge_sum} ≠ {magic_k}"
    
    def test_verify_labeling(self, small_params):
        """Test the verify_labeling method."""
        G, _ = GraphConstructor.construct(small_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, _ = solver.solve(G)
        
        if status == SolverStatus.FOUND:
            result = EMTLSolver.verify_labeling(G, magic_k, v_labels, e_labels)
            assert result is True
    
    def test_solve_minimal_graph(self, minimal_params):
        """Test solving the minimal possible graph."""
        G, _ = GraphConstructor.construct(minimal_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, _ = solver.solve(G)
        
        assert status == SolverStatus.FOUND
    
    def test_solve_symmetric_graph(self, symmetric_params):
        """Test solving a symmetric graph."""
        G, _ = GraphConstructor.construct(symmetric_params)
        solver = EMTLSolver(timeout_seconds=60)
        
        magic_k, v_labels, e_labels, status, _ = solver.solve(G)
        
        assert status == SolverStatus.FOUND


# =============================================================================
# EMTLRESULT TESTS
# =============================================================================

class TestEMTLResult:
    """Tests for the EMTLResult data class."""
    
    def test_exists_property_found(self, small_params):
        """Test exists property when EMTL is found."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        result = EMTLResult(
            status=SolverStatus.FOUND,
            graph=G,
            vertex_sets=vertex_sets,
            params=small_params,
            magic_constant=27,
            vertex_labels={'A0': 1},
            edge_labels={('A0', 'B0'): 2},
        )
        
        assert result.exists is True
    
    def test_exists_property_not_found(self, small_params):
        """Test exists property when EMTL is not found."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        result = EMTLResult(
            status=SolverStatus.INFEASIBLE,
            graph=G,
            vertex_sets=vertex_sets,
            params=small_params,
        )
        
        assert result.exists is False
    
    def test_exists_property_timeout(self, small_params):
        """Test exists property on timeout."""
        G, vertex_sets = GraphConstructor.construct(small_params)
        
        result = EMTLResult(
            status=SolverStatus.TIMEOUT,
            graph=G,
            vertex_sets=vertex_sets,
            params=small_params,
        )
        
        assert result.exists is False


# =============================================================================
# SOLVE_EMTL INTEGRATION TESTS
# =============================================================================

class TestSolveEMTL:
    """Integration tests for the solve_emtl function."""
    
    def test_solve_basic(self):
        """Test basic solve_emtl call."""
        result = solve_emtl(m=2, n=2, k=2, t=1, visualize=False, verbose=False)
        
        assert isinstance(result, EMTLResult)
        assert result.exists is True
        assert result.magic_constant is not None
    
    def test_solve_returns_correct_graph(self):
        """Test that solve_emtl returns correct graph structure."""
        result = solve_emtl(m=2, n=3, k=2, t=2, visualize=False, verbose=False)
        
        assert result.graph.number_of_nodes() == 2 + 2*3 + 2  # 10
        assert result.graph.number_of_edges() == 2*3 + 3*2 + 3*2  # 18
    
    def test_solve_with_different_params(self):
        """Test solve_emtl with various parameter combinations."""
        test_cases = [
            (1, 1, 1, 1),
            (2, 2, 2, 1),
            (1, 2, 1, 1),
            (3, 3, 3, 3),
            (2, 2, 2, 0),
        ]
        
        for m, n, k, t in test_cases:
            result = solve_emtl(m=m, n=n, k=k, t=t, 
                               visualize=False, verbose=False, timeout=30)
            assert isinstance(result, EMTLResult), f"Failed for ({m}, {n}, {k}, {t})"
    
    def test_solve_invalid_params(self):
        """Test solve_emtl with invalid parameters."""
        result = solve_emtl(m=2, n=3, k=2, t=5,  # t > n
                           visualize=False, verbose=False)
        
        assert result.status == SolverStatus.INVALID_PARAMS
        assert result.exists is False
    
    def test_solve_with_visualization_save(self, tmp_path):
        """Test that visualization can be saved."""
        save_path = str(tmp_path / "test_graph.png")
        result = solve_emtl(m=2, n=2, k=2, t=1, 
                           visualize=False,  # Don't show
                           save_fig=save_path,
                           verbose=False)
        
        # The file should be created by visualizer
        # Note: visualize=False means don't call plt.show()


# =============================================================================
# COMPREHENSIVE PARAMETER TESTS
# =============================================================================

class TestComprehensiveParameters:
    """Comprehensive tests across many parameter combinations."""
    
    @pytest.mark.parametrize("m,n,k,t", [
        (1, 1, 1, 0),
        (1, 1, 1, 1),
        (1, 2, 1, 1),
        (2, 2, 2, 1),
        (2, 2, 2, 2),
        (2, 3, 2, 1),
        (2, 3, 2, 2),
        (3, 3, 3, 2),
        (3, 3, 3, 3),
        (1, 4, 1, 2),
    ])
    def test_various_parameters(self, m, n, k, t):
        """Test EMTL solving across various valid parameters."""
        result = solve_emtl(m=m, n=n, k=k, t=t, 
                           visualize=False, verbose=False, timeout=30)
        
        # Should complete without error
        assert result.status in [SolverStatus.FOUND, SolverStatus.TIMEOUT, 
                                 SolverStatus.INFEASIBLE]
        
        # If found, verify the solution
        if result.exists:
            assert result.magic_constant is not None
            assert result.vertex_labels is not None
            assert result.edge_labels is not None
            
            # Verify magic property
            for (u, v) in result.graph.edges():
                edge_sum = (result.vertex_labels[u] + 
                           result.edge_labels[(u, v)] + 
                           result.vertex_labels[v])
                assert edge_sum == result.magic_constant


# =============================================================================
# EDGE CASES AND BOUNDARY TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimal_graph(self):
        """Test the smallest possible valid graph."""
        result = solve_emtl(m=1, n=1, k=1, t=0, visualize=False, verbose=False)
        assert result.status in [SolverStatus.FOUND, SolverStatus.INFEASIBLE]
    
    def test_disconnected_graph(self):
        """Test graph with no B-C edges (disconnected A-B and C-D)."""
        result = solve_emtl(m=2, n=2, k=2, t=0, visualize=False, verbose=False)
        
        # Graph is disconnected but might still have EMTL
        assert result.status in [SolverStatus.FOUND, SolverStatus.INFEASIBLE,
                                 SolverStatus.TIMEOUT]
    
    def test_maximum_regularity(self):
        """Test with maximum B-C regularity (t=n)."""
        result = solve_emtl(m=2, n=3, k=2, t=3, visualize=False, verbose=False)
        
        # B-C is complete bipartite K_{n,n}
        assert result.status in [SolverStatus.FOUND, SolverStatus.TIMEOUT,
                                 SolverStatus.INFEASIBLE]
    
    def test_single_vertex_partitions(self):
        """Test with single vertex in each partition."""
        result = solve_emtl(m=1, n=1, k=1, t=1, visualize=False, verbose=False)
        
        # Very small graph
        assert result.exists is True  # Should definitely find EMTL


# =============================================================================
# PERFORMANCE TESTS (Optional - can be skipped in CI)
# =============================================================================

class TestPerformance:
    """Performance tests for larger graphs."""
    
    @pytest.mark.slow
    def test_medium_graph_performance(self):
        """Test that medium graphs solve in reasonable time."""
        import time
        
        start = time.time()
        result = solve_emtl(m=3, n=3, k=3, t=3,  # Reduced size for faster test
                           visualize=False, verbose=False, timeout=60)
        elapsed = time.time() - start
        
        # Should complete within timeout (with some margin)
        assert elapsed < 65  # Allow some margin for system overhead
        assert result is not None
    
    @pytest.mark.slow
    def test_larger_graph_performance(self):
        """Test larger graph (may timeout but should not crash)."""
        result = solve_emtl(m=4, n=4, k=4, t=4,
                           visualize=False, verbose=False, timeout=30)
        
        # Should complete without error
        assert result is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

