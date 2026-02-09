# Edge-Magic Total Labeling Solver

A constraint-programming solver for Edge-Magic Total Labelings (EMTL) on a
parameterized family of 4-partite graphs. It builds the graph, solves EMTL via
Google OR-Tools CP-SAT, verifies solutions, and optionally visualizes results
or serves them in a Streamlit app.

## Overview

This repo answers the question: for a graph in the family $G(m,n,k,t)$, does an
Edge-Magic Total Labeling exist, and if so what is the magic constant and the
labels?

Key features:
- Graph family construction with a $t$-regular bipartite middle layer.
- CP-SAT formulation with all-different and magic-sum constraints.
- Solution verification and visualization.
- Streamlit web UI and example runners.

## Quick Start

Requirements: Python 3.8+ (3.11 recommended), pip.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If PyPI is blocked, use a mirror:
`pip install -e . --index-url https://mirror-pypi.runflare.com/simple`

Run the solver (demo examples and saved figures):

```bash
python emtl_solver.py
```

Run the Streamlit app:

```bash
streamlit run web/app.py
```

Run tests:

```bash
pytest tests -v
```

## Mathematical Model

### Graph Family $G(m,n,k,t)$

The vertex set is partitioned into four parts:

- $A$ with $|A|=m$
- $B$ with $|B|=n$
- $C$ with $|C|=n$
- $D$ with $|D|=k$

Edges are the union of three subgraphs:

- Complete bipartite $K_{m,n}$ between $A$ and $B$.
- Complete bipartite $K_{n,k}$ between $C$ and $D$.
- A $t$-regular bipartite graph between $B$ and $C$.

Visual structure:

```
     A            B            C            D
  (m vertices) (n vertices) (n vertices) (k vertices)

   +---+         +---+         +---+         +---+
   |A0 |---------|B0 |.........|C0 |---------|D0 |
   +---+  \   /  +---+         +---+  \   /  +---+
           \ /                 \ /
           / \                 / \
   +---+  /   \  +---+         +---+  /   \  +---+
   |A1 |---------|B1 |.........|C1 |---------|D1 |
   +---+         +---+         +---+         +---+
   
       K_{m,n}       t-regular       K_{n,k}
     (Complete)                    (Complete)
```

Edge types:
- Solid lines: complete bipartite (A-B, C-D).
- Dotted lines: t-regular bipartite (B-C).

Counts:

$$|V| = m + 2n + k$$
$$|E| = mn + nk + nt$$

Constraints: $m,n,k \ge 1$ and $0 \le t \le n$.

### Edge-Magic Total Labeling (EMTL)

An EMTL is a bijection

$$f: V \cup E \to \{1,2,\dots,|V|+|E|\}$$

such that a constant $\kappa$ exists where for every edge $uv$:

$$f(u) + f(uv) + f(v) = \kappa.$$

Why this is hard (mathematically):
The decision problem is in NP because a proposed labeling verifies in $O(|E|)$
time. For general graphs it is NP-complete (hence NP-hard), so unless P=NP no
polynomial-time algorithm exists; brute force over $N=|V|+|E|$ labels is
$O(N! \cdot |E|)$ in the worst case, and any exact solver has exponential
worst-case behavior.

### Circulant Construction for $B$-$C$

The $t$-regular bipartite subgraph is built with a circulant rule:

$$E_{BC} = \{(B_i, C_{(i+j) \bmod n}) : i \in [0,n-1], j \in [0,t-1]\}.$$

Each $B_i$ and each $C_j$ has degree $t$ in the $B$-$C$ subgraph.

## CP-SAT Formulation

We pose EMTL as a constraint satisfaction problem (no optimization objective):

Variables:
- $x_v \in [1, |V|+|E|]$ for each vertex $v$.
- $x_e \in [1, |V|+|E|]$ for each edge $e$.
- $\kappa$ (magic constant).

Constraints:
- All-different: all vertex and edge labels are distinct.
- Magic sum: for every edge $(u,v)$, $x_u + x_{(u,v)} + x_v = \kappa$.

Bounds used in code:

- $\kappa_{min}=6$ (smallest possible sum of three distinct positive labels).
- $\kappa_{max}=3(|V|+|E|)-3$ (sum of the three largest labels).

About CP-SAT:
- OR-Tools CP-SAT combines SAT solving with constraint propagation.
- It maintains variable domains, prunes infeasible values, and searches with
  backtracking plus learned conflicts.
- The solver returns FEASIBLE or OPTIMAL when it finds a solution, INFEASIBLE
  if it proves none exist, or TIMEOUT if the limit is hit.

We set a time limit and use multiple workers (see `EMTLSolver`).

### How CP-SAT Solves This Model

Propagation:
- Each constraint narrows variable domains.
- The all-different constraint enforces a bijection across all labels.
- Magic-sum equations link vertex and edge labels to a shared $\kappa$.

Conflict learning:
- When a partial assignment leads to a contradiction, CP-SAT analyzes it.
- It learns a nogood clause so the same dead end is not revisited.

Search heuristics:
- The solver automatically chooses variable and value ordering.
- It uses restarts and a portfolio of strategies to find feasible solutions.
- Parallel workers explore different parts of the search space when enabled.

### Solver Algorithm (High Level)

```
Algorithm: SolveEMTL(G, timeout)
Input:  Graph G = (V, E), time limit T
Output: (k, f) if EMTL exists, INFEASIBLE or TIMEOUT otherwise

1. Create CP-SAT model M
2. Create variable x_v in [1, |V|+|E|] for each v in V
3. Create variable x_e in [1, |V|+|E|] for each e in E
4. Create variable k in [k_min, k_max]
5. Add constraint: AllDifferent({x_v} union {x_e})
6. For each (u, v) in E:
       Add constraint: x_u + x_(u,v) + x_v = k
7. status <- Solve(M, timeout=T)
8. If status in {OPTIMAL, FEASIBLE}:
       Return (k, {v -> x_v} union {e -> x_e})
9. Else:
       Return INFEASIBLE or TIMEOUT
```

## Implementation Map

Main solver code lives in `emtl_solver.py`:

- `GraphParameters`: validates input and provides |V|, |E|, and label counts.
- `GraphConstructor`: builds $G(m,n,k,t)$ and can verify its structure.
- `EMTLSolver`: builds the CP-SAT model, solves it, and verifies solutions.
- `EMTLResult`: container for the solve outcome and labels.
- `EMTLVisualizer`: draws the graph and labels using Matplotlib.
- `solve_emtl(...)`: end-to-end orchestration and optional visualization.

Other relevant files:

- `web/app.py`: Streamlit UI for interactive runs.
- `examples/run_examples.py`: batch runs with saved figures.
- `tests/test_emtl.py`: unit and integration tests for construction and solving.

## Visualization

`EMTLVisualizer` renders the graph with a fixed 4-column layout (A, B, C, D),
color-coded partitions, and edge coloring by subgraph type (A-B, B-C, C-D).
Vertex labels show both vertex IDs and EMTL labels. Edge labels show EMTL
labels. A small legend and statistics panel are included in the figure.

Saved figures typically go under `images/output/`.

## Streamlit App

The Streamlit UI allows you to:

- Choose parameters m, n, k, t.
- Set a solver timeout.
- Run the solver interactively and view results.

Start it with:

```bash
streamlit run web/app.py
```

## Examples

Run curated example sets and save figures:

```bash
python examples/run_examples.py
```

## Tests

```bash
pytest tests -v
```
