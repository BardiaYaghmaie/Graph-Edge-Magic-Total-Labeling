#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                    EDGE-MAGIC TOTAL LABELING (EMTL) SOLVER
================================================================================

A complete implementation for constructing specific graph structures and finding
Edge-Magic Total Labelings using Constraint Programming.

This solver addresses the following mathematical problem:
    Given a graph G = (V, E) with a specific partition structure, find a bijection
    f: V ∪ E → {1, 2, ..., |V| + |E|} such that for every edge uv ∈ E:
        f(u) + f(uv) + f(v) = k  (constant magic sum)

Graph Structure:
    - Vertex set V is partitioned into four disjoint subsets: A, B, C, D
    - |A| = m, |B| = n, |C| = n, |D| = k vertices
    - Complete bipartite graph K_{m,n} between A and B
    - Complete bipartite graph K_{n,k} between C and D  
    - t-regular bipartite graph between B and C

Algorithm:
    Uses Google OR-Tools CP-SAT solver for constraint satisfaction.
    The EMTL problem is formulated as a Constraint Satisfaction Problem (CSP):
    - Variables: Labels for each vertex and edge (domain: 1 to |V|+|E|)
    - Constraints: All-different + magic sum constraint for each edge

Author: EMTL Research Implementation
Version: 2.0.0
License: MIT
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import networkx as nx                    # Graph data structures and algorithms
import matplotlib.pyplot as plt          # Visualization and plotting
import matplotlib.patches as mpatches    # Custom legend patches
import numpy as np                       # Numerical operations
from ortools.sat.python import cp_model  # Google OR-Tools constraint solver
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# DATA CLASSES AND ENUMERATIONS
# =============================================================================

class SolverStatus(Enum):
    """
    Enumeration representing the possible outcomes of the EMTL solver.
    
    Values:
        FOUND: A valid EMTL was discovered
        INFEASIBLE: Proven that no EMTL exists for this graph
        TIMEOUT: Search exceeded time limit without conclusion
        INVALID_PARAMS: Input parameters are mathematically invalid
    """
    FOUND = "found"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    INVALID_PARAMS = "invalid_parameters"


@dataclass
class GraphParameters:
    """
    Data class encapsulating the graph construction parameters.
    
    Attributes:
        m: Number of vertices in set A
        n: Number of vertices in sets B and C (must be equal)
        k: Number of vertices in set D
        t: Regularity degree of the B-C bipartite subgraph
        
    Mathematical Constraints:
        - All values must be positive integers (m, n, k ≥ 1)
        - t must satisfy 0 ≤ t ≤ n (regularity cannot exceed partition size)
    """
    m: int  # |A| - vertices in partition A
    n: int  # |B| = |C| - vertices in partitions B and C
    k: int  # |D| - vertices in partition D
    t: int  # Regularity of B-C bipartite subgraph
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validates that parameters satisfy mathematical constraints.
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if self.m < 1:
            return False, f"m must be ≥ 1, got {self.m}"
        if self.n < 1:
            return False, f"n must be ≥ 1, got {self.n}"
        if self.k < 1:
            return False, f"k must be ≥ 1, got {self.k}"
        if self.t < 0:
            return False, f"t must be ≥ 0, got {self.t}"
        if self.t > self.n:
            return False, f"t must be ≤ n for t-regular bipartite graph (t={self.t}, n={self.n})"
        return True, ""
    
    @property
    def num_vertices(self) -> int:
        """Total number of vertices: |V| = m + 2n + k"""
        return self.m + 2 * self.n + self.k
    
    @property
    def num_edges(self) -> int:
        """Total number of edges: |E| = mn + nk + nt"""
        return self.m * self.n + self.n * self.k + self.n * self.t
    
    @property
    def total_labels(self) -> int:
        """Total labels needed: |V| + |E|"""
        return self.num_vertices + self.num_edges
    
    def __str__(self) -> str:
        return f"G(m={self.m}, n={self.n}, k={self.k}, t={self.t})"


@dataclass
class EMTLResult:
    """
    Data class containing the complete results of an EMTL computation.
    
    Attributes:
        status: The outcome of the solver (FOUND, INFEASIBLE, TIMEOUT, etc.)
        graph: The NetworkX graph object
        vertex_sets: Dictionary mapping set names ('A', 'B', 'C', 'D') to vertex lists
        params: The GraphParameters used to construct the graph
        magic_constant: The magic sum k (None if no EMTL found)
        vertex_labels: Mapping of vertex names to their labels (None if no EMTL)
        edge_labels: Mapping of edge tuples to their labels (None if no EMTL)
        solve_time: Time taken by the solver in seconds
        message: Human-readable status message
    """
    status: SolverStatus
    graph: nx.Graph
    vertex_sets: Dict[str, List[str]]
    params: GraphParameters
    magic_constant: Optional[int] = None
    vertex_labels: Optional[Dict[str, int]] = None
    edge_labels: Optional[Dict[Tuple[str, str], int]] = None
    solve_time: float = 0.0
    message: str = ""
    
    @property
    def exists(self) -> bool:
        """Returns True if a valid EMTL was found."""
        return self.status == SolverStatus.FOUND


# =============================================================================
# GRAPH CONSTRUCTION MODULE
# =============================================================================

class GraphConstructor:
    """
    Handles the construction of the specialized graph structure.
    
    This class builds a graph G = (V, E) with the following properties:
    
    1. VERTEX PARTITION:
       V = A ∪ B ∪ C ∪ D (disjoint union)
       |A| = m, |B| = |C| = n, |D| = k
    
    2. EDGE STRUCTURE:
       - E(A,B): Complete bipartite K_{m,n} (every vertex in A connects to every vertex in B)
       - E(C,D): Complete bipartite K_{n,k} (every vertex in C connects to every vertex in D)
       - E(B,C): t-regular bipartite (each vertex has exactly t neighbors in the other set)
    
    The t-regular bipartite subgraph is constructed using a circulant pattern:
    vertex B_i connects to C_j where j ∈ {i, i+1, ..., i+t-1} (mod n)
    
    This construction guarantees:
    - Each vertex in B has exactly t neighbors in C
    - Each vertex in C has exactly t neighbors in B
    - The subgraph is simple (no multi-edges or self-loops)
    """
    
    @staticmethod
    def create_t_regular_bipartite_edges(n: int, t: int) -> List[Tuple[int, int]]:
        """
        Generates edges for a t-regular bipartite graph between two sets of n vertices.
        
        Algorithm (Circulant Construction):
            For each vertex i in the first set (0 to n-1):
                Connect to vertices (i + j) mod n in the second set
                for j = 0, 1, ..., t-1
        
        This guarantees t-regularity because:
            - Each vertex i in set 1 has exactly t outgoing edges
            - Each vertex k in set 2 receives edges from vertices 
              (k), (k-1), ..., (k-t+1) mod n, which is exactly t vertices
        
        Args:
            n: Number of vertices in each partition
            t: Desired regularity (degree of each vertex)
        
        Returns:
            List of (i, j) tuples representing edges between vertex i in 
            first set and vertex j in second set (0-indexed)
        
        Raises:
            ValueError: If t > n (impossible to achieve t-regularity)
        
        Example:
            >>> create_t_regular_bipartite_edges(4, 2)
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 0)]
        """
        if t > n:
            raise ValueError(
                f"Cannot create {t}-regular bipartite graph with {n} vertices per side. "
                f"Maximum regularity is n={n}."
            )
        
        if t == 0:
            return []  # No edges needed for 0-regular graph
        
        edges = []
        for i in range(n):
            for offset in range(t):
                j = (i + offset) % n
                edges.append((i, j))
        
        return edges
    
    @staticmethod
    def construct(params: GraphParameters) -> Tuple[nx.Graph, Dict[str, List[str]]]:
        """
        Constructs the complete graph from parameters.
        
        Construction Steps:
            1. Create vertex labels for each partition (A0, A1, ..., B0, B1, ..., etc.)
            2. Add all vertices to the graph with partition metadata
            3. Add complete bipartite edges between A-B
            4. Add complete bipartite edges between C-D
            5. Add t-regular bipartite edges between B-C
        
        Args:
            params: GraphParameters defining the graph structure
        
        Returns:
            Tuple of:
                - NetworkX Graph object
                - Dictionary mapping partition names to vertex lists
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters first
        is_valid, error_msg = params.validate()
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Initialize empty graph
        G = nx.Graph()
        
        # Create vertex identifiers for each partition
        # Using format: PartitionName + Index (e.g., A0, A1, B0, B1, ...)
        A = [f'A{i}' for i in range(params.m)]
        B = [f'B{i}' for i in range(params.n)]
        C = [f'C{i}' for i in range(params.n)]
        D = [f'D{i}' for i in range(params.k)]
        
        # Add vertices with partition metadata (useful for visualization)
        for v in A:
            G.add_node(v, partition='A')
        for v in B:
            G.add_node(v, partition='B')
        for v in C:
            G.add_node(v, partition='C')
        for v in D:
            G.add_node(v, partition='D')
        
        # =================================================================
        # EDGE SET 1: Complete Bipartite K_{m,n} between A and B
        # =================================================================
        # Every vertex in A connects to every vertex in B
        # Total edges: m × n
        for a in A:
            for b in B:
                G.add_edge(a, b, edge_type='A-B')
        
        # =================================================================
        # EDGE SET 2: Complete Bipartite K_{n,k} between C and D
        # =================================================================
        # Every vertex in C connects to every vertex in D
        # Total edges: n × k
        for c in C:
            for d in D:
                G.add_edge(c, d, edge_type='C-D')
        
        # =================================================================
        # EDGE SET 3: t-Regular Bipartite between B and C
        # =================================================================
        # Each vertex in B and C has exactly t neighbors in the other set
        # Total edges: n × t
        bc_edges = GraphConstructor.create_t_regular_bipartite_edges(params.n, params.t)
        for i, j in bc_edges:
            G.add_edge(B[i], C[j], edge_type='B-C')
        
        vertex_sets = {'A': A, 'B': B, 'C': C, 'D': D}
        
        return G, vertex_sets
    
    @staticmethod
    def verify_structure(G: nx.Graph, vertex_sets: Dict[str, List[str]], 
                        params: GraphParameters) -> bool:
        """
        Verifies that the constructed graph matches the expected structure.
        
        Verification Checks:
            1. Partition sizes match parameters
            2. A-B subgraph is complete bipartite
            3. C-D subgraph is complete bipartite
            4. B-C subgraph is t-regular
            5. No other edges exist
        
        Args:
            G: The constructed graph
            vertex_sets: Partition mapping
            params: Expected parameters
        
        Returns:
            True if all checks pass
        
        Raises:
            AssertionError: If any verification check fails
        """
        A, B, C, D = (vertex_sets['A'], vertex_sets['B'], 
                      vertex_sets['C'], vertex_sets['D'])
        
        # Check partition sizes
        assert len(A) == params.m, f"Expected |A|={params.m}, got {len(A)}"
        assert len(B) == params.n, f"Expected |B|={params.n}, got {len(B)}"
        assert len(C) == params.n, f"Expected |C|={params.n}, got {len(C)}"
        assert len(D) == params.k, f"Expected |D|={params.k}, got {len(D)}"
        
        # Verify A-B is complete bipartite
        for a in A:
            for b in B:
                assert G.has_edge(a, b), f"Missing edge {a}-{b} in A-B complete bipartite"
        
        # Verify C-D is complete bipartite
        for c in C:
            for d in D:
                assert G.has_edge(c, d), f"Missing edge {c}-{d} in C-D complete bipartite"
        
        # Verify B-C is t-regular
        for b in B:
            deg_bc = sum(1 for c in C if G.has_edge(b, c))
            assert deg_bc == params.t, \
                f"Vertex {b} has degree {deg_bc} in B-C, expected {params.t}"
        
        for c in C:
            deg_bc = sum(1 for b in B if G.has_edge(b, c))
            assert deg_bc == params.t, \
                f"Vertex {c} has degree {deg_bc} in B-C, expected {params.t}"
        
        # Verify total edge count
        expected_edges = params.num_edges
        assert G.number_of_edges() == expected_edges, \
            f"Expected {expected_edges} edges, got {G.number_of_edges()}"
        
        return True


# =============================================================================
# EMTL SOLVER MODULE
# =============================================================================

class EMTLSolver:
    """
    Constraint Programming solver for Edge-Magic Total Labeling.
    
    MATHEMATICAL FORMULATION:
    =========================
    
    Given graph G = (V, E), find a bijection f: V ∪ E → {1, 2, ..., |V|+|E|}
    such that ∃k: ∀(u,v) ∈ E: f(u) + f(uv) + f(v) = k
    
    CONSTRAINT SATISFACTION PROBLEM (CSP) FORMULATION:
    ==================================================
    
    Variables:
        - x_v ∈ {1, ..., |V|+|E|} for each vertex v ∈ V
        - x_e ∈ {1, ..., |V|+|E|} for each edge e ∈ E
        - k ∈ {k_min, ..., k_max} (magic constant)
    
    Constraints:
        1. ALL-DIFFERENT: All variables must have distinct values
           (ensures f is a bijection)
        
        2. MAGIC SUM: For each edge (u,v) ∈ E:
           x_u + x_{(u,v)} + x_v = k
    
    BOUNDS ON MAGIC CONSTANT:
    ========================
    
    For a valid EMTL:
        - Minimum edge sum: 1 + 2 + 3 = 6 (smallest possible labels)
        - But with bijection, minimum for any edge ≈ n + some small values
        - Maximum edge sum: Uses three largest labels
    
    Theoretical bounds:
        k_min = |V| + 4  (approximate lower bound)
        k_max = 3(|V|+|E|) - 3  (three largest labels)
    
    SOLVER: Google OR-Tools CP-SAT
    ==============================
    
    OR-Tools CP-SAT is a state-of-the-art constraint programming solver that uses:
        - Lazy clause generation
        - Clause learning
        - Efficient propagation
        - Multi-threaded search
    
    It's particularly effective for:
        - Combinatorial optimization
        - Scheduling problems
        - Assignment problems (like EMTL)
    """
    
    def __init__(self, timeout_seconds: int = 60, num_workers: int = 8):
        """
        Initialize the EMTL solver.
        
        Args:
            timeout_seconds: Maximum time to search for a solution
            num_workers: Number of parallel search workers (threads)
        """
        self.timeout_seconds = timeout_seconds
        self.num_workers = num_workers
    
    def solve(self, G: nx.Graph) -> Tuple[Optional[int], 
                                          Optional[Dict[str, int]], 
                                          Optional[Dict[Tuple[str, str], int]],
                                          SolverStatus,
                                          float]:
        """
        Attempts to find an Edge-Magic Total Labeling for the given graph.
        
        Algorithm Steps:
            1. Extract vertices and edges from graph
            2. Create CP-SAT model with variables for labels
            3. Add ALL-DIFFERENT constraint (bijection requirement)
            4. Add MAGIC SUM constraints for each edge
            5. Run solver with timeout
            6. Extract solution if found
        
        Args:
            G: NetworkX graph to find EMTL for
        
        Returns:
            Tuple of:
                - magic_constant: The magic sum k (None if not found)
                - vertex_labels: Dict mapping vertices to labels (None if not found)
                - edge_labels: Dict mapping edges to labels (None if not found)
                - status: SolverStatus indicating outcome
                - solve_time: Time taken in seconds
        """
        import time
        start_time = time.time()
        
        # Extract graph structure
        vertices = list(G.nodes())
        edges = list(G.edges())
        
        num_vertices = len(vertices)
        num_edges = len(edges)
        total = num_vertices + num_edges
        
        # =================================================================
        # STEP 1: Create the Constraint Programming Model
        # =================================================================
        model = cp_model.CpModel()
        
        # =================================================================
        # STEP 2: Create Variables
        # =================================================================
        
        # Vertex label variables: x_v ∈ {1, ..., total}
        vertex_vars = {}
        for v in vertices:
            vertex_vars[v] = model.NewIntVar(1, total, f'label_vertex_{v}')
        
        # Edge label variables: x_e ∈ {1, ..., total}
        edge_vars = {}
        for e in edges:
            edge_vars[e] = model.NewIntVar(1, total, f'label_edge_{e}')
        
        # Magic constant variable with theoretical bounds
        # Lower bound: vertex count + 4 (heuristic)
        # Upper bound: sum of three largest possible labels
        k_min = num_vertices + 4
        k_max = 3 * total - 3  # total + (total-1) + (total-2)
        magic_k = model.NewIntVar(k_min, k_max, 'magic_constant')
        
        # =================================================================
        # STEP 3: Add ALL-DIFFERENT Constraint (Bijection)
        # =================================================================
        # This ensures f: V ∪ E → {1, ..., |V|+|E|} is a bijection
        # Every label must be used exactly once
        all_vars = list(vertex_vars.values()) + list(edge_vars.values())
        model.AddAllDifferent(all_vars)
        
        # =================================================================
        # STEP 4: Add MAGIC SUM Constraints
        # =================================================================
        # For each edge (u,v): f(u) + f(uv) + f(v) = k
        for u, v in edges:
            model.Add(
                vertex_vars[u] + edge_vars[(u, v)] + vertex_vars[v] == magic_k
            )
        
        # =================================================================
        # STEP 5: Configure and Run Solver
        # =================================================================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds
        solver.parameters.num_search_workers = self.num_workers
        
        # Optional: Add search hints for better performance
        # The solver can use these as starting points
        solver.parameters.log_search_progress = False
        
        status = solver.Solve(model)
        solve_time = time.time() - start_time
        
        # =================================================================
        # STEP 6: Process Results
        # =================================================================
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Solution found! Extract the labeling
            magic_constant = solver.Value(magic_k)
            vertex_labels = {v: solver.Value(vertex_vars[v]) for v in vertices}
            edge_labels = {e: solver.Value(edge_vars[e]) for e in edges}
            
            return (magic_constant, vertex_labels, edge_labels, 
                    SolverStatus.FOUND, solve_time)
        
        elif status == cp_model.INFEASIBLE:
            # Proven that no solution exists
            return None, None, None, SolverStatus.INFEASIBLE, solve_time
        
        else:
            # Timeout or unknown status
            return None, None, None, SolverStatus.TIMEOUT, solve_time
    
    @staticmethod
    def verify_labeling(G: nx.Graph, magic_constant: int,
                       vertex_labels: Dict[str, int],
                       edge_labels: Dict[Tuple[str, str], int]) -> bool:
        """
        Verifies that a labeling is a valid EMTL.
        
        Verification Checks:
            1. BIJECTION: Labels form a bijection to {1, ..., |V|+|E|}
            2. MAGIC PROPERTY: Every edge sum equals the magic constant
        
        Args:
            G: The graph
            magic_constant: The claimed magic sum
            vertex_labels: Vertex labeling
            edge_labels: Edge labeling
        
        Returns:
            True if the labeling is valid
        
        Raises:
            AssertionError: If verification fails (with descriptive message)
        """
        vertices = list(G.nodes())
        edges = list(G.edges())
        total = len(vertices) + len(edges)
        
        # Check bijection property
        all_labels = list(vertex_labels.values()) + list(edge_labels.values())
        expected_labels = set(range(1, total + 1))
        actual_labels = set(all_labels)
        
        assert actual_labels == expected_labels, \
            f"Labels don't form bijection to {{1, ..., {total}}}.\n" \
            f"Missing: {expected_labels - actual_labels}\n" \
            f"Extra: {actual_labels - expected_labels}"
        
        # Check magic property for each edge
        for u, v in edges:
            edge_sum = vertex_labels[u] + edge_labels[(u, v)] + vertex_labels[v]
            assert edge_sum == magic_constant, \
                f"Edge ({u}, {v}): {vertex_labels[u]} + {edge_labels[(u, v)]} + " \
                f"{vertex_labels[v]} = {edge_sum} ≠ {magic_constant}"
        
        return True


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class EMTLVisualizer:
    """
    Creates publication-quality visualizations of graphs with EMTL labelings.
    
    Visual Design:
        - Vertices arranged in four columns (A, B, C, D from left to right)
        - Color-coded partitions for easy identification
        - Edge colors indicate subgraph type (A-B, B-C, C-D)
        - Labels displayed on vertices and edges
        - Legend and statistics included
    """
    
    # Color scheme (accessible and visually distinct)
    COLORS = {
        'A': '#E63946',  # Red - Partition A
        'B': '#2A9D8F',  # Teal - Partition B
        'C': '#457B9D',  # Blue - Partition C
        'D': '#8AB17D',  # Green - Partition D
        'edge_AB': '#FFCCD5',  # Light pink - A-B edges
        'edge_BC': '#BDE0FE',  # Light blue - B-C edges
        'edge_CD': '#C7F9CC',  # Light green - C-D edges
    }
    
    @staticmethod
    def create_layout(vertex_sets: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
        """
        Creates a 4-column layout for the graph visualization.
        
        Layout:
            A vertices at x = -3
            B vertices at x = -1
            C vertices at x = +1
            D vertices at x = +3
        
        Vertices are vertically centered within their column.
        
        Args:
            vertex_sets: Dictionary mapping partition names to vertex lists
        
        Returns:
            Dictionary mapping vertex names to (x, y) positions
        """
        pos = {}
        x_positions = {'A': -3, 'B': -1, 'C': 1, 'D': 3}
        
        for partition, x in x_positions.items():
            vertices = vertex_sets[partition]
            n = len(vertices)
            for i, v in enumerate(vertices):
                # Center vertices vertically
                y = (n - 1) / 2 - i
                pos[v] = (x, y)
        
        return pos
    
    @staticmethod
    def visualize(result: EMTLResult, 
                  figsize: Tuple[int, int] = (16, 10),
                  save_path: Optional[str] = None,
                  show: bool = True) -> plt.Figure:
        """
        Creates a comprehensive visualization of the graph and its EMTL.
        
        Features:
            - Color-coded vertex partitions
            - Color-coded edge types
            - Vertex labels showing name and EMTL label
            - Edge labels showing EMTL label
            - Legend explaining the color scheme
            - Statistics panel
            - Title with magic constant (if found)
        
        Args:
            result: EMTLResult containing graph and labeling
            figsize: Figure dimensions (width, height)
            save_path: Optional path to save the figure
            show: Whether to display the figure
        
        Returns:
            matplotlib Figure object
        """
        G = result.graph
        vertex_sets = result.vertex_sets
        params = result.params
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#FAFAFA')
        
        # Get layout
        pos = EMTLVisualizer.create_layout(vertex_sets)
        
        # Prepare node colors
        node_colors = []
        for node in G.nodes():
            partition = node[0]  # First character indicates partition
            node_colors.append(EMTLVisualizer.COLORS[partition])
        
        # Prepare edge colors
        edge_colors = []
        for u, v in G.edges():
            pu, pv = u[0], v[0]
            if (pu == 'A' and pv == 'B') or (pu == 'B' and pv == 'A'):
                edge_colors.append(EMTLVisualizer.COLORS['edge_AB'])
            elif (pu == 'B' and pv == 'C') or (pu == 'C' and pv == 'B'):
                edge_colors.append(EMTLVisualizer.COLORS['edge_BC'])
            else:
                edge_colors.append(EMTLVisualizer.COLORS['edge_CD'])
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=2.0,
            alpha=0.7,
            style='solid'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=900,
            edgecolors='#333333',
            linewidths=2.0
        )
        
        # Prepare and draw node labels
        if result.vertex_labels:
            node_labels = {v: f"{v}\n({result.vertex_labels[v]})" 
                          for v in G.nodes()}
        else:
            node_labels = {v: v for v in G.nodes()}
        
        nx.draw_networkx_labels(
            G, pos, node_labels, ax=ax,
            font_size=9,
            font_weight='bold',
            font_color='white'
        )
        
        # Draw edge labels if EMTL exists
        if result.edge_labels:
            edge_label_dict = {e: str(result.edge_labels[e]) 
                              for e in result.edge_labels}
            nx.draw_networkx_edge_labels(
                G, pos, edge_label_dict, ax=ax,
                font_size=8,
                font_color='#1a1a2e',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='#cccccc',
                    alpha=0.9
                ),
                rotate=False
            )
        
        # Create title
        if result.exists:
            title = f"Graph {params} with Edge-Magic Total Labeling\n" \
                    f"Magic Constant k = {result.magic_constant}"
        else:
            status_msg = "No EMTL Found" if result.status == SolverStatus.INFEASIBLE \
                        else f"Solver Status: {result.status.value}"
            title = f"Graph {params}\n{status_msg}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color=EMTLVisualizer.COLORS['A'], 
                          label=f'Set A ({len(vertex_sets["A"])} vertices)'),
            mpatches.Patch(color=EMTLVisualizer.COLORS['B'], 
                          label=f'Set B ({len(vertex_sets["B"])} vertices)'),
            mpatches.Patch(color=EMTLVisualizer.COLORS['C'], 
                          label=f'Set C ({len(vertex_sets["C"])} vertices)'),
            mpatches.Patch(color=EMTLVisualizer.COLORS['D'], 
                          label=f'Set D ({len(vertex_sets["D"])} vertices)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
                  framealpha=0.9, edgecolor='#cccccc')
        
        # Add statistics text box
        stats_text = (
            f"|V| = {params.num_vertices}\n"
            f"|E| = {params.num_edges}\n"
            f"Solve time: {result.solve_time:.2f}s"
        )
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', 
                     edgecolor='#cccccc', alpha=0.9)
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        if show:
            plt.show()
        
        return fig


# =============================================================================
# MAIN SOLVER INTERFACE
# =============================================================================

def solve_emtl(m: int, n: int, k: int, t: int,
               timeout: int = 60,
               visualize: bool = True,
               save_fig: Optional[str] = None,
               verbose: bool = True) -> EMTLResult:
    """
    Main function to solve the EMTL problem for a given graph configuration.
    
    This function orchestrates the complete pipeline:
        1. Parameter validation
        2. Graph construction
        3. Structure verification
        4. EMTL solving via constraint programming
        5. Solution verification (if found)
        6. Visualization (optional)
    
    USAGE EXAMPLE:
    ==============
    
    >>> result = solve_emtl(m=2, n=3, k=2, t=2)
    >>> if result.exists:
    ...     print(f"Magic constant: {result.magic_constant}")
    ...     print(f"Vertex labels: {result.vertex_labels}")
    
    Args:
        m: Number of vertices in partition A
        n: Number of vertices in partitions B and C
        k: Number of vertices in partition D
        t: Regularity of the B-C bipartite subgraph
        timeout: Maximum solver time in seconds
        visualize: Whether to display visualization
        save_fig: Optional file path to save visualization
        verbose: Whether to print progress messages
    
    Returns:
        EMTLResult containing all computation results
    """
    def log(msg: str):
        if verbose:
            print(msg)
    
    # =================================================================
    # STEP 1: Create and Validate Parameters
    # =================================================================
    log("=" * 70)
    log(f"EMTL SOLVER - Graph G(m={m}, n={n}, k={k}, t={t})")
    log("=" * 70)
    
    params = GraphParameters(m=m, n=n, k=k, t=t)
    is_valid, error_msg = params.validate()
    
    if not is_valid:
        log(f"\n✗ Invalid parameters: {error_msg}")
        # Return result indicating invalid parameters
        G = nx.Graph()
        return EMTLResult(
            status=SolverStatus.INVALID_PARAMS,
            graph=G,
            vertex_sets={'A': [], 'B': [], 'C': [], 'D': []},
            params=params,
            message=error_msg
        )
    
    # =================================================================
    # STEP 2: Construct Graph
    # =================================================================
    log(f"\n[1/5] Constructing graph...")
    log(f"      Expected: |V|={params.num_vertices}, |E|={params.num_edges}")
    
    G, vertex_sets = GraphConstructor.construct(params)
    
    # =================================================================
    # STEP 3: Verify Graph Structure
    # =================================================================
    log(f"\n[2/5] Verifying graph structure...")
    
    try:
        GraphConstructor.verify_structure(G, vertex_sets, params)
        log(f"      ✓ Structure verified: {G.number_of_nodes()} vertices, "
            f"{G.number_of_edges()} edges")
    except AssertionError as e:
        log(f"      ✗ Structure verification failed: {e}")
        return EMTLResult(
            status=SolverStatus.INVALID_PARAMS,
            graph=G,
            vertex_sets=vertex_sets,
            params=params,
            message=str(e)
        )
    
    # =================================================================
    # STEP 4: Solve for EMTL
    # =================================================================
    log(f"\n[3/5] Searching for Edge-Magic Total Labeling...")
    log(f"      Variables: {params.total_labels} labels to assign")
    log(f"      Timeout: {timeout} seconds")
    
    solver = EMTLSolver(timeout_seconds=timeout)
    magic_constant, vertex_labels, edge_labels, status, solve_time = solver.solve(G)
    
    # =================================================================
    # STEP 5: Process Results
    # =================================================================
    if status == SolverStatus.FOUND:
        log(f"\n[4/5] ✓ EMTL FOUND!")
        log(f"      Magic constant k = {magic_constant}")
        log(f"      Solve time: {solve_time:.3f} seconds")
        
        # Verify the solution
        log(f"\n[5/5] Verifying solution...")
        try:
            EMTLSolver.verify_labeling(G, magic_constant, vertex_labels, edge_labels)
            log(f"      ✓ Verification passed!")
        except AssertionError as e:
            log(f"      ✗ Verification failed: {e}")
        
        # Print labeling details
        if verbose:
            log(f"\n{'─' * 70}")
            log("EDGE-MAGIC TOTAL LABELING DETAILS")
            log(f"{'─' * 70}")
            
            log("\nVertex Labels:")
            for v in sorted(G.nodes()):
                log(f"    f({v}) = {vertex_labels[v]}")
            
            log("\nEdge Labels (with verification):")
            for (u, v) in sorted(G.edges()):
                label = edge_labels[(u, v)]
                total = vertex_labels[u] + label + vertex_labels[v]
                log(f"    f({u}—{v}) = {label:3d}    "
                    f"[{vertex_labels[u]} + {label} + {vertex_labels[v]} = {total}]")
    
    elif status == SolverStatus.INFEASIBLE:
        log(f"\n[4/5] ✗ NO EMTL EXISTS (proven infeasible)")
        log(f"      Solve time: {solve_time:.3f} seconds")
    
    else:
        log(f"\n[4/5] ⚠ SOLVER TIMEOUT")
        log(f"      Could not determine if EMTL exists within {timeout}s")
        log(f"      Try increasing timeout for larger graphs")
    
    # Create result object
    result = EMTLResult(
        status=status,
        graph=G,
        vertex_sets=vertex_sets,
        params=params,
        magic_constant=magic_constant,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
        solve_time=solve_time,
        message=f"Solver completed in {solve_time:.3f}s"
    )
    
    # =================================================================
    # STEP 6: Visualization
    # =================================================================
    if visualize:
        log(f"\n[Visualization] Creating graph visualization...")
        EMTLVisualizer.visualize(result, save_path=save_fig, show=True)
        if save_fig:
            log(f"      Saved to: {save_fig}")
    
    log("\n" + "=" * 70)
    
    return result


# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def run_examples():
    """
    Runs a series of example configurations to demonstrate the solver.
    
    Examples are chosen to show:
        1. A simple case that definitely has an EMTL
        2. Various parameter combinations
        3. Different graph sizes
    """
    examples = [
        # (m, n, k, t, description)
        (2, 2, 2, 1, "Small graph - basic example"),
        (1, 2, 1, 1, "Minimal connected structure"),
        (2, 3, 2, 2, "Medium graph with 2-regular B-C"),
        (3, 3, 3, 3, "Symmetric structure (all 3s)"),
        (2, 2, 2, 0, "No B-C edges (disconnected components)"),
    ]
    
    print("\n" + "█" * 70)
    print("         EDGE-MAGIC TOTAL LABELING - DEMONSTRATION")
    print("█" * 70)
    
    results_summary = []
    
    for m, n, k, t, desc in examples:
        print(f"\n{'━' * 70}")
        print(f"EXAMPLE: {desc}")
        print(f"Parameters: m={m}, n={n}, k={k}, t={t}")
        print(f"{'━' * 70}")
        
        result = solve_emtl(
            m=m, n=n, k=k, t=t,
            timeout=60,
            visualize=True,
            save_fig=f"images/output/emtl_m{m}_n{n}_k{k}_t{t}.png",
            verbose=True
        )
        
        results_summary.append((m, n, k, t, result.exists, result.magic_constant))
    
    # Print summary table
    print("\n" + "█" * 70)
    print("                    RESULTS SUMMARY")
    print("█" * 70)
    print(f"\n{'Parameters':<20} {'|V|':>6} {'|E|':>6} {'EMTL':>10} {'Magic k':>10}")
    print("─" * 60)
    
    for m, n, k, t, exists, magic_k in results_summary:
        v = m + 2*n + k
        e = m*n + n*k + n*t
        status = "✓ Yes" if exists else "✗ No"
        k_str = str(magic_k) if magic_k else "—"
        print(f"(m={m},n={n},k={k},t={t}){'':<6} {v:>6} {e:>6} {status:>10} {k_str:>10}")
    
    print("\n" + "█" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the EMTL solver demonstration.
    
    When run directly, this script demonstrates the solver with
    several example configurations and produces visualizations.
    """
    run_examples()
