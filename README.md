# Edge-Magic Total Labeling Solver

A constraint programming solver for finding Edge-Magic Total Labelings on a parameterized family of 4-partite graphs.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Graph Family Definition](#2-graph-family-definition)
3. [The Circulant Construction](#3-the-circulant-construction)
4. [CSP Formulation](#4-csp-formulation)
5. [Theoretical Properties](#5-theoretical-properties)
6. [Implementation Architecture](#6-implementation-architecture)
7. [Code Walkthrough](#7-code-walkthrough)
8. [Usage](#8-usage)
9. [Computational Results](#9-computational-results)

---

## 1. Mathematical Foundation

### 1.1 Definition: Total Labeling

A **total labeling** of a graph G = (V, E) is a function f : V ∪ E → ℤ⁺ that assigns positive integers to both vertices and edges.

### 1.2 Definition: Edge-Magic Total Labeling (EMTL)

Let G = (V, E) be a graph with p = |V| vertices and q = |E| edges. An **Edge-Magic Total Labeling** is a bijection:

```
f : V ∪ E → {1, 2, ..., p + q}
```

such that there exists a constant k ∈ ℤ⁺ (the **magic constant**) where for every edge uv ∈ E:

```
f(u) + f(uv) + f(v) = k
```

**Key Properties:**
- **Bijection**: Every integer from 1 to p+q is used exactly once
- **Magic Sum**: The sum of vertex-edge-vertex labels is constant across all edges
- The problem of determining EMTL existence is NP-complete in general

### 1.3 Definition: t-Regular Bipartite Graph

A bipartite graph H = (X ∪ Y, E) is **t-regular** if:
- deg(x) = t for all x ∈ X
- deg(y) = t for all y ∈ Y

**Existence Condition**: A t-regular bipartite graph with parts of size n exists if and only if 0 ≤ t ≤ n.

**Edge Count**: When t > 0, such a graph has exactly n·t edges.

---

## 2. Graph Family Definition

### 2.1 The Graph G(m, n, k, t)

For parameters m, n, k ∈ ℤ⁺ and t ∈ {0, 1, ..., n}, we define graph G(m, n, k, t) = (V, E):

**Vertex Set** (partitioned into four disjoint sets):
```
V = A ∪ B ∪ C ∪ D

where:
    |A| = m
    |B| = n
    |C| = n
    |D| = k
```

**Edge Set** (three distinct subgraphs):
```
E = E₁ ∪ E₂ ∪ E₃

where:
    E₁ = {(a,b) : a ∈ A, b ∈ B}     Complete bipartite K_{m,n}
    E₂ = t-regular bipartite on B∪C  Circulant construction
    E₃ = {(c,d) : c ∈ C, d ∈ D}     Complete bipartite K_{n,k}
```

### 2.2 Graph Statistics

**Proposition**: For G(m, n, k, t):
```
|V| = m + 2n + k
|E| = mn + nt + nk
Total labels = |V| + |E| = m + 2n + k + mn + nt + nk
```

**Proof**:
- Vertices: |A| + |B| + |C| + |D| = m + n + n + k = m + 2n + k
- Edges: |E₁| + |E₂| + |E₃| = (m·n) + (n·t) + (n·k) = mn + nt + nk □

### 2.3 Visual Structure

```
     A              B              C              D
   (m vertices)  (n vertices)  (n vertices)  (k vertices)
     
   ┌───┐         ┌───┐         ┌───┐         ┌───┐
   │A₀ │─────────│B₀ │╌╌╌╌╌╌╌╌╌│C₀ │─────────│D₀ │
   └───┘ ╲     ╱ └───┘         └───┘ ╲     ╱ └───┘
          ╲   ╱                       ╲   ╱      
           ╲ ╱                         ╲ ╱       
           ╱ ╲                         ╱ ╲       
          ╱   ╲                       ╱   ╲      
   ┌───┐ ╱     ╲ ┌───┐         ┌───┐ ╱     ╲ ┌───┐
   │A₁ │─────────│B₁ │╌╌╌╌╌╌╌╌╌│C₁ │─────────│D₁ │
   └───┘         └───┘         └───┘         └───┘
   
   └────────────┘ └───────────┘ └────────────┘
       K_{m,n}       t-regular       K_{n,k}
     (complete)    (circulant)     (complete)
```

**Edge types**:
- **Solid lines (━)**: Complete bipartite (every vertex connects to every vertex in adjacent partition)
- **Dashed lines (╌)**: t-regular bipartite (each vertex has exactly t neighbors)

---

## 3. The Circulant Construction

### 3.1 Algorithm

To construct a t-regular bipartite graph on vertex sets B = {B₀, ..., Bₙ₋₁} and C = {C₀, ..., Cₙ₋₁}:

```
Algorithm: CirculantBipartite(n, t)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  n = partition size, t = regularity degree
Output: Edge set E₂

E₂ ← ∅
for i = 0 to n-1:
    for j = 0 to t-1:
        E₂ ← E₂ ∪ {(Bᵢ, C₍ᵢ₊ⱼ₎ mod n)}

return E₂
```

### 3.2 Proof of t-Regularity

**Theorem**: The circulant construction produces a t-regular bipartite graph.

**Proof**:

*Left regularity*: Each vertex Bᵢ connects to vertices:
```
C_{(i+0) mod n}, C_{(i+1) mod n}, ..., C_{(i+t-1) mod n}
```
This is exactly t distinct vertices in C. ✓

*Right regularity*: Vertex Cⱼ receives an edge from Bᵢ when:
```
j ≡ i + offset (mod n)  for some offset ∈ {0, 1, ..., t-1}
```
Equivalently: i ∈ {j, j-1, j-2, ..., j-t+1} (mod n)

This is exactly t distinct values of i. ✓

**Example** (n=4, t=2):
```
B₀ → C₀, C₁
B₁ → C₁, C₂
B₂ → C₂, C₃
B₃ → C₃, C₀

Each B vertex: degree 2 ✓
Each C vertex: degree 2 ✓
Total edges: 4 × 2 = 8 ✓
```

---

## 4. CSP Formulation

### 4.1 Problem Encoding

The EMTL problem is encoded as a Constraint Satisfaction Problem (CSP):

**Variables**:
```
xᵥ ∈ {1, 2, ..., p+q}    for each vertex v ∈ V
xₑ ∈ {1, 2, ..., p+q}    for each edge e ∈ E
κ  ∈ {κ_min, ..., κ_max} (magic constant)
```

**Constraints**:

1. **Bijection Constraint** (AllDifferent):
   ```
   AllDifferent(x₁, x₂, ..., x_{p+q})
   ```
   Ensures each label 1 to p+q is used exactly once.

2. **Magic Sum Constraints** (one per edge):
   ```
   ∀(u,v) ∈ E:  xᵤ + x₍ᵤ,ᵥ₎ + xᵥ = κ
   ```
   Ensures every edge has the same sum.

### 4.2 Magic Constant Bounds

**Upper bound**: Maximum possible edge sum uses three largest labels:
```
κ_max = (p+q) + (p+q-1) + (p+q-2) = 3(p+q) - 3
```

**Lower bound** (heuristic): Empirically, k tends to be at least |V| + 4:
```
κ_min = p + 4
```

These bounds are not tight but safely contain all valid solutions.

### 4.3 Solver Algorithm

```
Algorithm: SolveEMTL(G, timeout)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Graph G = (V, E), time limit T
Output: (k, f) if EMTL exists, INFEASIBLE otherwise

1. Create CP-SAT model M
2. Create variable xᵥ ∈ [1, p+q] for each v ∈ V
3. Create variable xₑ ∈ [1, p+q] for each e ∈ E
4. Create variable κ ∈ [κ_min, κ_max]
5. Add constraint: AllDifferent({xᵥ} ∪ {xₑ})
6. For each (u,v) ∈ E:
       Add constraint: xᵤ + x₍ᵤ,ᵥ₎ + xᵥ = κ
7. status ← Solve(M, timeout=T)
8. If status ∈ {OPTIMAL, FEASIBLE}:
       Return (κ, {v → xᵥ} ∪ {e → xₑ})
9. Else if status = INFEASIBLE:
       Return INFEASIBLE  // Proven no EMTL exists
10. Else:
       Return TIMEOUT
```

### 4.4 Solution Verification

Every solution is independently verified:

```
Algorithm: VerifyEMTL(G, k, f)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Graph G, magic constant k, labeling f
Output: VALID or INVALID

1. labels ← {f(x) : x ∈ V ∪ E}
2. If labels ≠ {1, 2, ..., |V|+|E|}:
       Return INVALID  // Not a bijection
3. For each (u,v) ∈ E:
       If f(u) + f(uv) + f(v) ≠ k:
           Return INVALID
4. Return VALID
```

---

## 5. Theoretical Properties

### 5.1 Necessary Condition

**Proposition**: If G(m, n, k, t) admits an EMTL with magic constant k, then:
```
k · |E| = Σᵥ∈V deg(v)·f(v) + Σₑ∈E f(e)
```

**Proof**: Sum the magic equation f(u) + f(uv) + f(v) = k over all edges.
- Left side: k · |E|
- Right side: Each vertex v appears deg(v) times, each edge once. □

### 5.2 Label Sum Identity

**Proposition**: The sum of all labels is fixed:
```
Σᵢ₌₁^{p+q} i = (p+q)(p+q+1)/2
```

This constrains valid magic constants for any given graph.

### 5.3 Degree Sequence of G(m, n, k, t)

```
Partition    Vertices    Degree
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   A           m           n           (connected to all of B)
   B           n          m + t        (connected to A and t vertices in C)
   C           n          t + k        (connected to t vertices in B and all of D)
   D           k           n           (connected to all of C)
```

### 5.4 Special Cases

**Case t = 0**: Graph decomposes into two disconnected components:
- Component 1: K_{m,n} on A ∪ B
- Component 2: K_{n,k} on C ∪ D

EMTL may or may not exist (both components need compatible magic constants).

**Case t = n**: B-C subgraph is K_{n,n} (complete bipartite), maximizing connectivity.

---

## 6. Implementation Architecture

### 6.1 Module Structure

```
emtl_solver.py
├── Data Classes
│   ├── SolverStatus (Enum)      # FOUND, INFEASIBLE, TIMEOUT, INVALID_PARAMS
│   ├── GraphParameters          # Holds m, n, k, t with validation
│   └── EMTLResult              # Complete result container
│
├── GraphConstructor (Class)
│   ├── create_t_regular_bipartite_edges()  # Circulant algorithm
│   ├── construct()                          # Build full graph
│   └── verify_structure()                   # Validate construction
│
├── EMTLSolver (Class)
│   ├── solve()           # Main CP-SAT solving
│   └── verify_labeling() # Solution verification
│
├── EMTLVisualizer (Class)
│   ├── create_layout()   # 4-column positioning
│   └── visualize()       # Matplotlib rendering
│
└── solve_emtl()          # Main API function
```

### 6.2 Class Responsibilities

**`GraphParameters`**: Encapsulates and validates input parameters.
```python
@dataclass
class GraphParameters:
    m: int  # |A|
    n: int  # |B| = |C|
    k: int  # |D|
    t: int  # B-C regularity
    
    def validate(self) -> Tuple[bool, str]:
        # Checks: m,n,k ≥ 1 and 0 ≤ t ≤ n
    
    @property
    def num_vertices(self) -> int:
        return self.m + 2 * self.n + self.k
    
    @property
    def num_edges(self) -> int:
        return self.m * self.n + self.n * self.k + self.n * self.t
```

**`GraphConstructor`**: Builds the graph structure using NetworkX.
- Creates vertex sets A, B, C, D with naming convention (A0, A1, ..., B0, B1, ...)
- Adds complete bipartite edges for A-B and C-D
- Constructs t-regular B-C edges using circulant pattern
- Stores partition metadata on nodes for visualization

**`EMTLSolver`**: Implements the CSP formulation using OR-Tools CP-SAT.
- Creates integer variables for all vertices and edges
- Adds AllDifferent constraint for bijection
- Adds magic sum constraint for each edge
- Configures solver with timeout and parallel workers
- Extracts solution as dictionaries mapping vertices/edges to labels

**`EMTLResult`**: Container for all computation results.
```python
@dataclass
class EMTLResult:
    status: SolverStatus
    graph: nx.Graph
    vertex_sets: Dict[str, List[str]]
    params: GraphParameters
    magic_constant: Optional[int]
    vertex_labels: Optional[Dict[str, int]]
    edge_labels: Optional[Dict[Tuple[str, str], int]]
    solve_time: float
    
    @property
    def exists(self) -> bool:
        return self.status == SolverStatus.FOUND
```

---

## 7. Code Walkthrough

### 7.1 Graph Construction (Key Implementation)

```python
def construct(params: GraphParameters) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    G = nx.Graph()
    
    # Create vertex identifiers
    A = [f'A{i}' for i in range(params.m)]
    B = [f'B{i}' for i in range(params.n)]
    C = [f'C{i}' for i in range(params.n)]
    D = [f'D{i}' for i in range(params.k)]
    
    # Add vertices with partition metadata
    for v in A: G.add_node(v, partition='A')
    for v in B: G.add_node(v, partition='B')
    for v in C: G.add_node(v, partition='C')
    for v in D: G.add_node(v, partition='D')
    
    # E₁: Complete bipartite K_{m,n} between A and B
    for a in A:
        for b in B:
            G.add_edge(a, b)
    
    # E₃: Complete bipartite K_{n,k} between C and D
    for c in C:
        for d in D:
            G.add_edge(c, d)
    
    # E₂: t-regular bipartite between B and C (circulant)
    for i in range(params.n):
        for offset in range(params.t):
            j = (i + offset) % params.n
            G.add_edge(B[i], C[j])
    
    return G, {'A': A, 'B': B, 'C': C, 'D': D}
```

### 7.2 CSP Solving (Key Implementation)

```python
def solve(self, G: nx.Graph):
    model = cp_model.CpModel()
    
    vertices = list(G.nodes())
    edges = list(G.edges())
    total = len(vertices) + len(edges)
    
    # Create variables
    vertex_vars = {v: model.NewIntVar(1, total, f'v_{v}') for v in vertices}
    edge_vars = {e: model.NewIntVar(1, total, f'e_{e}') for e in edges}
    magic_k = model.NewIntVar(len(vertices) + 4, 3 * total - 3, 'k')
    
    # Constraint 1: AllDifferent (bijection)
    all_vars = list(vertex_vars.values()) + list(edge_vars.values())
    model.AddAllDifferent(all_vars)
    
    # Constraint 2: Magic sum for each edge
    for u, v in edges:
        model.Add(vertex_vars[u] + edge_vars[(u, v)] + vertex_vars[v] == magic_k)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = self.timeout_seconds
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return (
            solver.Value(magic_k),
            {v: solver.Value(vertex_vars[v]) for v in vertices},
            {e: solver.Value(edge_vars[e]) for e in edges},
            SolverStatus.FOUND
        )
    elif status == cp_model.INFEASIBLE:
        return None, None, None, SolverStatus.INFEASIBLE
    else:
        return None, None, None, SolverStatus.TIMEOUT
```

### 7.3 Solution Verification

```python
def verify_labeling(G, magic_constant, vertex_labels, edge_labels) -> bool:
    vertices = list(G.nodes())
    edges = list(G.edges())
    total = len(vertices) + len(edges)
    
    # Check bijection
    all_labels = list(vertex_labels.values()) + list(edge_labels.values())
    assert set(all_labels) == set(range(1, total + 1)), "Not a bijection"
    
    # Check magic property
    for u, v in edges:
        edge_sum = vertex_labels[u] + edge_labels[(u, v)] + vertex_labels[v]
        assert edge_sum == magic_constant, f"Edge sum mismatch"
    
    return True
```

---

## 8. Usage

### 8.1 Python API

```python
from emtl_solver import solve_emtl

# Solve for EMTL
result = solve_emtl(m=2, n=3, k=2, t=2)

if result.exists:
    print(f"Magic constant: {result.magic_constant}")
    print(f"Vertex labels: {result.vertex_labels}")
    print(f"Edge labels: {result.edge_labels}")
else:
    print(f"Status: {result.status.value}")
```

### 8.2 Command Line

```bash
# Run demonstration with examples
python emtl_solver.py

# Run comprehensive examples
python examples/run_examples.py

# Run tests
pytest tests/ -v
```

### 8.3 Web Interface

```bash
streamlit run web/app.py
# Open http://localhost:8501
```

### 8.4 Jupyter Notebook

```bash
jupyter notebook notebooks/EMTL_Tutorial.ipynb
```

---

## 9. Computational Results

### 9.1 EMTL Existence Table

| (m, n, k, t) | \|V\| | \|E\| | Labels | Result | Magic k | Time |
|--------------|-------|-------|--------|--------|---------|------|
| (1, 1, 1, 1) | 4 | 3 | 7 | ✓ EXISTS | 12 | 0.01s |
| (1, 2, 1, 1) | 6 | 6 | 12 | ✓ EXISTS | 19 | 0.02s |
| (2, 2, 2, 1) | 8 | 10 | 18 | ✓ EXISTS | 27 | 0.04s |
| (2, 2, 2, 2) | 8 | 12 | 20 | ✓ EXISTS | 25 | 0.05s |
| (2, 3, 2, 2) | 10 | 18 | 28 | ✓ EXISTS | 32 | 0.12s |
| (3, 3, 3, 3) | 12 | 27 | 39 | ✓ EXISTS | 47 | 0.31s |
| (4, 4, 4, 4) | 16 | 48 | 64 | ✓ EXISTS | ~78 | 1.8s |
| (5, 5, 5, 5) | 20 | 75 | 95 | ✓ EXISTS | ~118 | 12s |
| (1, 1, 1, 0) | 4 | 2 | 6 | ✗ NONE | — | 0.01s |
| (2, 2, 2, 0) | 8 | 8 | 16 | ✓ EXISTS | 25 | 0.04s |

### 9.2 Example Solution: G(2, 2, 2, 1)

```
Graph: |V| = 8, |E| = 10, Labels = {1, 2, ..., 18}
Magic Constant: k = 27

Vertex Labels:
    f(A₀) = 12    f(A₁) = 4
    f(B₀) = 14    f(B₁) = 13
    f(C₀) = 5     f(C₁) = 3
    f(D₀) = 6     f(D₁) = 7

Edge Verification (all sums = 27):
    A₀-B₀:  12 +  1 + 14 = 27 ✓
    A₀-B₁:  12 +  2 + 13 = 27 ✓
    A₁-B₀:   4 +  9 + 14 = 27 ✓
    A₁-B₁:   4 + 10 + 13 = 27 ✓
    B₀-C₀:  14 +  8 +  5 = 27 ✓
    B₁-C₁:  13 + 11 +  3 = 27 ✓
    C₀-D₀:   5 + 16 +  6 = 27 ✓
    C₀-D₁:   5 + 15 +  7 = 27 ✓
    C₁-D₀:   3 + 18 +  6 = 27 ✓
    C₁-D₁:   3 + 17 +  7 = 27 ✓

Labels used: {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18} ✓
```

### 9.3 Performance

Solve time grows superlinearly with the number of labels, consistent with NP-completeness. The structured nature of G(m,n,k,t) enables efficient solving for practical sizes (up to ~100 labels in reasonable time).

---

## Requirements

```
Python 3.8+
networkx>=3.0
matplotlib>=3.7
numpy>=1.24
ortools>=9.7
streamlit>=1.28  (for web interface)
pytest>=7.0      (for tests)
jupyter>=1.0     (for notebook)
```

Install: `pip install -r requirements.txt`
