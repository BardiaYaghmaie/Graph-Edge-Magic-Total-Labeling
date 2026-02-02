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

A **total labeling** of a graph $G = (V, E)$ is a function $f : V \cup E \to \mathbb{Z}^+$ that assigns positive integers to both vertices and edges.

### 1.2 Definition: Edge-Magic Total Labeling (EMTL)

Let $G = (V, E)$ be a graph with $p = |V|$ vertices and $q = |E|$ edges. An **Edge-Magic Total Labeling** is a bijection:

$$f : V \cup E \to \{1, 2, \dots, p + q\}$$

such that there exists a constant $k \in \mathbb{Z}^+$ (the **magic constant**) where for every edge $uv \in E$:

$$f(u) + f(uv) + f(v) = k$$

**Key Properties:**
- **Bijection**: Every integer from 1 to $p+q$ is used exactly once.
- **Magic Sum**: The sum of vertex-edge-vertex labels is constant across all edges.
- **Complexity**: The problem of determining EMTL existence is NP-complete.

### 1.3 Definition: $t$-Regular Bipartite Graph

A bipartite graph $H = (X \cup Y, E)$ is **$t$-regular** if:
- $\deg(x) = t$ for all $x \in X$
- $\deg(y) = t$ for all $y \in Y$

**Existence Condition**: A $t$-regular bipartite graph with parts of size $n$ exists if and only if $0 \le t \le n$.
**Edge Count**: When $t > 0$, such a graph has exactly $n \cdot t$ edges.

---

## 2. Graph Family Definition

### 2.1 The Graph $G(m, n, k, t)$

For parameters $m, n, k \in \mathbb{Z}^+$ and $t \in \{0, 1, \dots, n\}$, we define graph $G(m, n, k, t) = (V, E)$:

**Vertex Set** (partitioned into four disjoint sets):
$$V = A \cup B \cup C \cup D$$

where:
*   $|A| = m$
*   $|B| = n$
*   $|C| = n$
*   $|D| = k$

**Edge Set** (three distinct subgraphs):
$$E = E_1 \cup E_2 \cup E_3$$

where:
*   $E_1 = \{(a,b) : a \in A, b \in B\}$ (Complete bipartite $K_{m,n}$)
*   $E_2$ is a $t$-regular bipartite graph on $B \cup C$ (constructed via circulant method)
*   $E_3 = \{(c,d) : c \in C, d \in D\}$ (Complete bipartite $K_{n,k}$)

### 2.2 Graph Statistics

For $G(m, n, k, t)$:
*   $|V| = m + 2n + k$
*   $|E| = mn + nt + nk$
*   Total labels = $|V| + |E|$

**Proof**:
- Vertices: $|A| + |B| + |C| + |D| = m + n + n + k = m + 2n + k$
- Edges: $|E_1| + |E_2| + |E_3| = (mn) + (nt) + (nk) = mn + nt + nk$ $\square$

### 2.3 Visual Structure

```
     A              B              C              D
   (m vertices)  (n vertices)  (n vertices)  (k vertices)
     
   ┌───┐         ┌───┐         ┌───┐         ┌───┐
   │A₀ │─────────│B₀ │╌╌╌╌╌╌╌╌╌│C₀ │─────────│D₀ │
   └───┘ ╲     ╱ └───┘         └───┘ ╲     ╱ └───┘
          ╲   ╱                       ╲   ╱      
           ╲ ╱      t-regular          ╲ ╱       
           ╱ ╲      bipartite          ╱ ╲       
          ╱   ╲                       ╱   ╲      
   ┌───┐ ╱     ╲ ┌───┐         ┌───┐ ╱     ╲ ┌───┐
   │A₁ │─────────│B₁ │╌╌╌╌╌╌╌╌╌│C₁ │─────────│D₁ │
   └───┘         └───┘         └───┘         └───┘
   
       K_{m,n}       Circulant       K_{n,k}
     (Complete)    Construction    (Complete)
```

**Edge types**:
- **Solid lines (━)**: Complete bipartite (every vertex connects to every vertex in adjacent partition)
- **Dashed lines (╌)**: $t$-regular bipartite (each vertex has exactly $t$ neighbors)

---

## 3. The Circulant Construction

### 3.1 Algorithm

To construct a $t$-regular bipartite graph on vertex sets $B = \{B_0, \dots, B_{n-1}\}$ and $C = \{C_0, \dots, C_{n-1}\}$:

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

### 3.2 Proof of $t$-Regularity

**Theorem**: The circulant construction produces a $t$-regular bipartite graph.

**Proof**:

*Left regularity*: Each vertex $B_i$ connects to vertices:
$$C_{(i+0) \bmod n}, C_{(i+1) \bmod n}, \dots, C_{(i+t-1) \bmod n}$$
This is exactly $t$ distinct vertices in $C$. ✓

*Right regularity*: Vertex $C_j$ receives an edge from $B_i$ when:
$$j \equiv i + \text{offset} \pmod n \quad \text{for some offset} \in \{0, \dots, t-1\}$$
Equivalently: $i \equiv j - \text{offset} \pmod n$.
This yields exactly $t$ distinct values of $i$. ✓

---

## 4. CSP Formulation

### 4.1 Problem Encoding

The EMTL problem is encoded as a Constraint Satisfaction Problem (CSP):

**Variables**:
*   $x_v \in \{1, 2, \dots, p+q\}$ for each vertex $v \in V$
*   $x_e \in \{1, 2, \dots, p+q\}$ for each edge $e \in E$
*   $\kappa \in [\kappa_{\min}, \kappa_{\max}]$ (magic constant)

**Constraints**:

1.  **Bijection Constraint** (AllDifferent):
    $$\text{AllDifferent}(\{x_v\}_{v \in V} \cup \{x_e\}_{e \in E})$$
    Ensures each label is used exactly once.

2.  **Magic Sum Constraints** (one per edge):
    $$\forall (u,v) \in E: x_u + x_{(u,v)} + x_v = \kappa$$
    Ensures every edge has the same sum.

### 4.2 Solver Algorithm

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
9. Else:
       Return INFEASIBLE or TIMEOUT
```

---

## 5. Theoretical Properties

### 5.1 Necessary Condition (Fundamental Identity)

**Proposition**: If $G(m, n, k, t)$ admits an EMTL with magic constant $k$, then:

$$k \cdot |E| = \sum_{v \in V} f(v)(\deg(v) - 1) + \sum_{i=1}^{|V|+|E|} i$$

**Proof**:
1. Sum the magic equation $f(u) + f(uv) + f(v) = k$ over all edges:
   $$\sum_{e \in E} (f(u) + f(e) + f(v)) = k \cdot |E|$$
2. In this sum, every edge label $f(e)$ appears exactly once. Every vertex label $f(v)$ appears exactly $\deg(v)$ times.
   $$k \cdot |E| = \sum_{v \in V} \deg(v)f(v) + \sum_{e \in E} f(e)$$
3. We know that the set of all labels is $\{1, \dots, |V|+|E|\}$. Let $S$ be their sum.
   $$S = \sum_{v \in V} f(v) + \sum_{e \in E} f(e)$$
4. Substituting $\sum f(e) = S - \sum f(v)$ into the equation from step 2:
   $$k \cdot |E| = \sum_{v \in V} \deg(v)f(v) + S - \sum_{v \in V} f(v)$$
   $$k \cdot |E| = \sum_{v \in V} f(v)(\deg(v) - 1) + S$$ $\square$

This identity is powerful for bounding $k$.

### 5.2 Degree Sequence of $G(m, n, k, t)$

| Partition | Vertices | Degree | Connections |
|-----------|----------|--------|-------------|
| **A** | $m$ | $n$ | Connected to all of B |
| **B** | $n$ | $m + t$ | Connected to A and $t$ vertices in C |
| **C** | $n$ | $t + k$ | Connected to $t$ vertices in B and all of D |
| **D** | $k$ | $n$ | Connected to all of C |

---

## 6. Implementation Architecture

### 6.1 Module Structure

The project is structured to separate graph theory logic from constraint solving.

- **`GraphParameters`**: Encapsulates $m, n, k, t$ and validates mathematical invariants ($m,n,k \ge 1$, $0 \le t \le n$).
- **`GraphConstructor`**:
    - Creates vertex sets with metadata.
    - Implements the circulant algorithm for $E_2$.
    - Verifies the structure (degrees, edges) before solving.
- **`EMTLSolver`**:
    - Maps the graph topology to CP-SAT variables.
    - Implements the CSP model.
    - Configures parallel search (8 workers).
- **`EMTLVisualizer`**:
    - Calculates positions for the 4-column layout.
    - Renders publication-quality figures using Matplotlib.

---

## 7. Code Walkthrough

### 7.1 Graph Construction (Key Implementation)

```python
def construct(params: GraphParameters) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    G = nx.Graph()
    
    # E₁: Complete bipartite K_{m,n} between A and B
    for a in A:
        for b in B:
            G.add_edge(a, b)
    
    # E₂: t-regular bipartite between B and C (circulant)
    for i in range(params.n):
        for offset in range(params.t):
            j = (i + offset) % params.n
            G.add_edge(B[i], C[j])
            
    # E₃: Complete bipartite K_{n,k} between C and D
    for c in C:
        for d in D:
            G.add_edge(c, d)
            
    return G, vertex_sets
```

### 7.2 Solution Verification

Every solution is independently verified to ensure correctness:
```python
def verify_labeling(G, k, f):
    # 1. Check Bijection
    labels = set(f.values())
    expected = set(range(1, len(G.nodes) + len(G.edges) + 1))
    assert labels == expected, "Not a bijection"
    
    # 2. Check Magic Sum
    for u, v in G.edges:
        s = f[u] + f[(u,v)] + f[v]
        assert s == k, f"Edge {u}-{v} sum {s} != {k}"
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
```

### 8.2 Command Line

```bash
# Run demonstration with examples
python emtl_solver.py

# Run web interface
streamlit run web/app.py
```

---

## 9. Computational Results

Empirical results from the solver:

| Parameters $(m, n, k, t)$ | $|V|$ | $|E|$ | Search Space | Result | Magic $k$ | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $(1, 1, 1, 1)$ | 4 | 3 | $5 \times 10^3$ | **Found** | 12 | 0.01s |
| $(2, 2, 2, 1)$ | 8 | 10 | $6 \times 10^{15}$ | **Found** | 27 | 0.04s |
| $(2, 3, 2, 2)$ | 10 | 18 | $3 \times 10^{29}$ | **Found** | 32 | 0.12s |
| $(3, 3, 3, 3)$ | 12 | 27 | $2 \times 10^{46}$ | **Found** | 47 | 0.31s |
| $(1, 1, 1, 0)$ | 4 | 2 | $720$ | **None** | — | 0.01s |

**Example Solution: G(2, 2, 2, 1)**
Magic Constant $k = 27$
*   $A$: $\{12, 4\}$
*   $B$: $\{14, 13\}$
*   $C$: $\{5, 3\}$
*   $D$: $\{6, 7\}$
*   Edge $A_0-B_0$: $12 + 1 + 14 = 27$ ✓

---

## Requirements

```
Python 3.8+
networkx>=3.0
matplotlib>=3.7
numpy>=1.24
ortools>=9.7
streamlit>=1.28
```
