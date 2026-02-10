# Edge-Magic Total Labelings on a Four-Partite Graph Family $G(m,n,k,t)$  
*Mathematical model, constraint-programming formulation, and reference implementation*

**Author:** Bardia Yaghmaie  
**Date:** 2026-02-10  
**Repository:** Graph-Edge-Magic-Total-Labeling

---

## Abstract

An **edge-magic total labeling** (EMTL) of a finite simple graph $G=(V,E)$ is a
bijection $f:V\cup E\to \{1,2,\dots,|V|+|E|\}$ such that a constant $\kappa$
exists with
$$
f(u)+f(uv)+f(v)=\kappa \quad \text{for every edge } uv\in E.
$$
This document specifies the EMTL problem studied in this repository for a
parameterized family of four-partite graphs $G(m,n,k,t)$, derives basic
counting identities relevant for correctness checks, and presents a complete
constraint-programming formulation solved via Google OR-Tools CP-SAT. The
article also maps the mathematical objects directly to the project’s Python
implementation, including graph construction, model generation, solution
verification, and visualization.

**Keywords:** graph labeling, edge-magic total labeling, constraint programming,
CP-SAT, OR-Tools.

---

## 1. Introduction

Graph labelings assign integers to vertices and/or edges subject to structural
constraints. **Edge-magic total labeling** is a “global bijection + local
equation” labeling: we must use each integer exactly once across all vertices
and edges, while simultaneously forcing every edge to have the same induced
sum.

From an algorithmic perspective, EMTL instances are challenging because:

1. The search space is the set of bijections on $N=|V|+|E|$ objects, which is
   $N!$ in the worst case.
2. The magic condition couples vertex and edge labels through $|E|$ linear
   equations sharing a single constant $\kappa$.

Accordingly, this repository uses **constraint programming** (CP) to solve EMTL
instances exactly (when feasible within a time limit). In particular, we use
OR-Tools **CP-SAT**, which combines SAT-style conflict learning with CP-style
propagation on integer variables.

The project is specialized to a structured family of graphs $G(m,n,k,t)$ with
four vertex partitions and three edge layers, which is large enough to be
interesting yet structured enough to support systematic experimentation.

---

## 2. Definitions and notation

Throughout, $G=(V,E)$ denotes a finite simple undirected graph.

### 2.1. Total labelings

Let $N = |V|+|E|$. A **total labeling** is a function
$$
f:V\cup E \to \{1,2,\dots,N\}.
$$
It is **bijective** if it is one-to-one and onto; equivalently, the multiset of
assigned labels is exactly $\{1,2,\dots,N\}$.

### 2.2. Edge-magic total labeling

A bijective total labeling $f$ is an **edge-magic total labeling** if there
exists a constant $\kappa\in\mathbb{Z}$ such that
$$
\forall\, uv\in E:\quad f(u)+f(uv)+f(v)=\kappa.
$$
The integer $\kappa$ is called the **magic constant** (or **magic sum**).

---

## 3. The graph family $G(m,n,k,t)$

### 3.1. Vertex partitions

Fix integers $m,n,k\ge 1$ and an integer $t$ with $0\le t\le n$.
The vertex set is partitioned into four disjoint parts:

- $A$ with $|A|=m$,
- $B$ with $|B|=n$,
- $C$ with $|C|=n$,
- $D$ with $|D|=k$.

Hence
$$
|V| = m + 2n + k.
$$

### 3.2. Edge layers

The edge set is the union of three bipartite layers:

1. **Left complete bipartite layer** between $A$ and $B$: all edges $ab$ with
   $a\in A$ and $b\in B$ (i.e., $K_{m,n}$).
2. **Middle $t$-regular bipartite layer** between $B$ and $C$.
3. **Right complete bipartite layer** between $C$ and $D$: all edges $cd$ with
   $c\in C$ and $d\in D$ (i.e., $K_{n,k}$).

Therefore
$$
|E| = mn + nt + nk.
$$

### 3.3. Circulant construction of the $B$–$C$ layer

Index $B=\{B_0,\dots,B_{n-1}\}$ and $C=\{C_0,\dots,C_{n-1}\}$.
The project constructs the $t$-regular bipartite subgraph by the circulant rule
$$
E_{BC}=\{(B_i,C_{(i+j)\bmod n}) : i\in\{0,\dots,n-1\},\, j\in\{0,\dots,t-1\}\}.
$$

**Proposition 3.1 (Regularity of the middle layer).**  
For $0\le t\le n$, the bipartite graph $(B\cup C, E_{BC})$ is $t$-regular on both
sides.

*Sketch.* Each $B_i$ is adjacent to exactly the $t$ vertices
$C_{(i+0)\bmod n},\dots,C_{(i+t-1)\bmod n}$. Fix a $C_r$. It is adjacent to
$B_{(r-0)\bmod n},\dots,B_{(r-(t-1))\bmod n}$, which are $t$ distinct vertices. ∎

### 3.4. Degrees in $G(m,n,k,t)$

The full graph degrees are constant within each partition:

- For $a\in A$, $\deg(a)=n$ (connected to all of $B$).
- For $b\in B$, $\deg(b)=m+t$ (all of $A$ plus $t$ neighbors in $C$).
- For $c\in C$, $\deg(c)=k+t$ (all of $D$ plus $t$ neighbors in $B$).
- For $d\in D$, $\deg(d)=n$ (connected to all of $C$).

These identities are useful in aggregate sum arguments (Section 4).

---

## 4. Algebraic identities for EMTL (sanity checks)

This section derives a global identity satisfied by every EMTL. It is *not*
required to solve the instance (the CP model already enforces everything), but
it is a helpful correctness invariant when inspecting solutions.

Let $f$ be an EMTL of $G=(V,E)$ with magic constant $\kappa$. Summing the magic
equations over all edges yields:
$$
\sum_{uv\in E}\bigl(f(u)+f(uv)+f(v)\bigr) = \sum_{uv\in E}\kappa = |E|\kappa.
$$
Split the left-hand side into edge-label and vertex-label contributions.

### 4.1. Endpoint-sum as a degree-weighted vertex sum

Each vertex label $f(v)$ appears once per incident edge in the sum
$\sum_{uv\in E}(f(u)+f(v))$, so:
$$
\sum_{uv\in E}\bigl(f(u)+f(v)\bigr) = \sum_{v\in V}\deg(v)\,f(v).
$$
Therefore:
$$
|E|\kappa = \sum_{e\in E} f(e) \;+\; \sum_{v\in V}\deg(v)\,f(v).
$$

### 4.2. Using bijectivity to eliminate edge-label sum

Let $N=|V|+|E|$, and let
$$
S_N := \sum_{i=1}^{N} i = \frac{N(N+1)}{2}.
$$
Since $f$ is a bijection on $V\cup E$,
$$
\sum_{e\in E} f(e) + \sum_{v\in V} f(v) = S_N,
$$
so $\sum_{e\in E}f(e)=S_N-\sum_{v\in V}f(v)$. Substituting:
$$
|E|\kappa = S_N + \sum_{v\in V}\bigl(\deg(v)-1\bigr) f(v).
$$

**Interpretation.** The magic constant is not arbitrary; it is constrained by
the vertex labels and degrees. For the family $G(m,n,k,t)$, the coefficients
$(\deg(v)-1)$ take only four values:

- $n-1$ on $A$ and $D$,
- $(m+t)-1$ on $B$,
- $(k+t)-1$ on $C$.

In code, this identity is a convenient post-solve check in addition to per-edge
verification.

**Corollary 4.1 (Divisibility / congruence constraint).**  
Rearranging (4.2) gives an explicit expression for the magic constant:
$$
\kappa \;=\; \frac{S_N + \sum_{v\in V}(\deg(v)-1)f(v)}{|E|}.
$$
In particular, the numerator must be divisible by $|E|$, yielding a necessary
congruence condition:
$$
S_N + \sum_{v\in V}(\deg(v)-1)f(v) \equiv 0 \pmod{|E|}.
$$
This does not by itself guarantee an EMTL exists, but any valid solution must
satisfy it.

### 4.3. Universal bounds on $\kappa$

Because three *distinct* labels are summed on each edge, a safe universal lower
bound is
$$
\kappa \ge 1+2+3=6.
$$
Similarly, the maximum possible sum of three distinct labels from
$\{1,\dots,N\}$ is $N+(N-1)+(N-2)=3N-3$, so
$$
\kappa \le 3N-3.
$$
The implementation uses these bounds as the domain of the magic-constant
variable.

---

## 5. Constraint-programming formulation (CP-SAT)

### 5.1. Decision variables

Let $N=|V|+|E|$.

- For every vertex $v\in V$, introduce an integer variable $x_v\in\{1,\dots,N\}$.
- For every edge $e\in E$, introduce an integer variable $x_e\in\{1,\dots,N\}$.
- Introduce the magic constant variable $\kappa\in\{6,\dots,3N-3\}$.

### 5.2. Constraints

**(C1) Bijection (AllDifferent).**  
Enforce that all labels are distinct:
$$
\mathrm{AllDifferent}\bigl(\{x_v : v\in V\}\cup\{x_e : e\in E\}\bigr).
$$
Since the domains are all $\{1,\dots,N\}$ and the number of variables is $N$,
this implies the labels are a bijection onto $\{1,\dots,N\}$.

**(C2) Magic sum constraints.**  
For every edge $uv\in E$:
$$
x_u + x_{uv} + x_v = \kappa.
$$

There is no optimization objective; the model is purely feasibility.

### 5.3. Model size for $G(m,n,k,t)$

For the family in Section 3:
$$
|V| = m+2n+k,\qquad |E|=mn+nt+nk,\qquad N=|V|+|E|.
$$

- Integer variables: $N$ label variables plus $1$ magic-constant variable.
- Constraints: $1$ AllDifferent constraint plus $|E|$ linear equations.

Even for moderate parameters, $N$ grows quadratically in $n$ due to the complete
bipartite layers, so practical solving relies heavily on propagation, learned
conflicts, and effective search.

### 5.4. Why CP-SAT works well here

The constraints create strong coupling:

- Each magic equation reduces the allowed combinations of the three incident
  labels and the shared $\kappa$.
- AllDifferent propagates global inconsistency quickly when a partial labeling
  starts to “use up” too many numbers.
- CP-SAT learns *nogoods* (conflict clauses) to avoid revisiting impossible
  partial assignments.

### 5.5. What CP-SAT is (mathematically)

At the mathematical level, CP-SAT is a procedure for solving a **constraint
satisfaction problem (CSP)**:

- Variables $X_1,\dots,X_r$,
- Each variable has a finite **domain** $D(X_i)\subset\mathbb{Z}$,
- Constraints restrict the allowed tuples of values.

For EMTL we have the variables $\{x_v\}_{v\in V}$, $\{x_e\}_{e\in E}$, and
$\kappa$, with initial domains:
$$
D(x_v)=D(x_e)=\{1,\dots,N\}, \qquad D(\kappa)=\{6,\dots,3N-3\}.
$$

CP-SAT maintains (and repeatedly tightens) these domains. When the solver makes
a **decision** (e.g., “$x_{A_0}=11$”), that is equivalent to adding an extra
constraint and then re-running the same domain-tightening logic.

When CP-SAT detects that a set of decisions forces a contradiction, it does not
only backtrack; it also tries to derive a *new* constraint (“this particular
combination can never happen”), so it will not waste time reaching the same dead
end again. That is the core idea behind **conflict-driven learning**.

### 5.6. Propagation on the magic constraints (domain filtering)

Consider a single edge constraint for $uv\in E$:
$$
x_u + x_{uv} + x_v = \kappa.
$$

Suppose at some point the solver knows only *bounds* (intervals) for each
variable:
$$
x_u\in[\ell_u, r_u],\quad x_v\in[\ell_v, r_v],\quad x_{uv}\in[\ell_e, r_e],\quad
\kappa\in[\ell_\kappa, r_\kappa].
$$
Then the equality implies the following necessary bounds (each is just
rearranging the equality and using worst-case values):

**Bounds for the edge label from endpoints and $\kappa$:**
$$
x_{uv} = \kappa - x_u - x_v
\;\Rightarrow\;
\begin{cases}
x_{uv}\ge \ell_\kappa - r_u - r_v,\\
x_{uv}\le r_\kappa - \ell_u - \ell_v.
\end{cases}
$$

**Bounds for an endpoint from the other endpoint, edge label, and $\kappa$:**
$$
x_u = \kappa - x_{uv} - x_v
\;\Rightarrow\;
\begin{cases}
x_u\ge \ell_\kappa - r_e - r_v,\\
x_u\le r_\kappa - \ell_e - \ell_v,
\end{cases}
\qquad
x_v \text{ similarly.}
$$

**Bounds for the magic constant from the three labels:**
$$
\kappa = x_u + x_{uv} + x_v
\;\Rightarrow\;
\begin{cases}
\kappa \ge \ell_u + \ell_e + \ell_v,\\
\kappa \le r_u + r_e + r_v.
\end{cases}
$$

In CP terminology, applying these implications to shrink intervals is called
**propagation** (or **domain filtering**). It is *sound*: it never removes a
value that could appear in a real solution. It is also often powerful: once any
two of the three labels on an edge become fixed, the third becomes fixed.

For EMTL this matters because every edge constraint has the “functional” form
$x_{uv}=\kappa-x_u-x_v$. So, as the solver fixes vertex labels, many edge-label
variables become forced (or get very tight bounds), which then interacts with
AllDifferent.

### 5.7. Propagation from AllDifferent (global reasoning)

The constraint
$$
\mathrm{AllDifferent}(X_1,\dots,X_N)
$$
means no two variables may take the same value.

The simplest propagation rule is:

- If some $X_i$ is fixed to value $a$, then remove $a$ from every other domain
  $D(X_j)$ for $j\ne i$.

But AllDifferent can also imply **set-based** deductions. A standard idea is a
*Hall set*:

> If a set of $h$ variables can only take values from a set of $h$ numbers, then
> those $h$ numbers are “reserved” for those variables and can be removed from
> all other variables’ domains.

Example: if two variables have domains $\{1,2\}$ and $\{1,2\}$, then they must
use $\{1,2\}$ in some order, so every other variable cannot use $1$ or $2$.

For EMTL, AllDifferent couples *every* vertex label and *every* edge label,
which is why a single forced edge-label value can trigger a cascade of domain
removals elsewhere in the model.

### 5.8. Search: decisions, backtracking, and the “proof of infeasibility”

Propagation alone usually does not finish the problem; there can remain many
choices. CP-SAT then performs a systematic search (conceptually a backtracking
tree):

1. **Propagate** all constraints until no further domain reduction is possible.
2. If some domain becomes empty, this branch is impossible (**conflict**).
3. If all variables are fixed, a solution is found.
4. Otherwise, pick a variable/value decision and recurse.

Mathematically, a “branch” is the addition of a disjunctive restriction such as
$$
x = a \quad\text{or}\quad x \ne a,
$$
or (for interval domains) a split like
$$
x \le c \quad\text{or}\quad x \ge c+1.
$$

If CP-SAT explores the entire search tree and every branch leads to a conflict,
then it has produced a *proof* that the CSP is infeasible. That corresponds to
the `INFEASIBLE` status in the implementation.

### 5.9. Conflict-driven learning (why CP-SAT avoids repeating mistakes)

A key difference between modern CP-SAT and naive backtracking is **learning**:
when a conflict occurs, CP-SAT analyzes *which decisions caused it* and records
a new constraint that forbids that same combination in the future. This learned
constraint is often called a **nogood**.

For EMTL, conflicts commonly arise like this:

- an edge constraint forces an edge label $x_{uv}$ to equal some value $a$, but
  AllDifferent has already forced $a$ to be used elsewhere; or
- the forced $x_{uv}$ falls outside $\{1,\dots,N\}$ (domain becomes empty); or
- forcing many edges through the same $\kappa$ leaves no consistent way to keep
  all labels distinct.

Learning is powerful because EMTL has many similar substructures (many edges
with the same form), so “bad patterns” repeat across different parts of the
search. A learned nogood prevents CP-SAT from rediscovering the same pattern
again under a different exploration order.

### 5.10. An equivalent “vertex-first” view (useful intuition)

From the magic equations,
$$
x_{uv} = \kappa - x_u - x_v.
$$
So, once $\kappa$ and the vertex labels $\{x_v\}_{v\in V}$ are chosen, *all edge
labels are determined*. The EMTL existence problem can therefore be viewed as:

Find $\kappa$ and a bijection $x:V\to\{1,\dots,N\}$ such that the induced values
$$
\{\kappa - x_u - x_v : uv\in E\}
$$
are (i) all integers in $\{1,\dots,N\}$, (ii) all distinct, and (iii) disjoint
from the vertex-label set.

The implementation keeps explicit variables for edge labels because it makes
the AllDifferent constraint and verification straightforward. Conceptually,
however, CP-SAT is exploiting this same structure: many edge labels become
forced quickly, and AllDifferent then propagates strongly.

### 5.11. The SAT layer (how CP-SAT “learns”)

The name **CP-SAT** reflects a hybrid viewpoint:

- **CP part:** integer variables with domains + propagators (Sections 5.6–5.7).
- **SAT part:** a Boolean search engine that records decisions, detects
  conflicts, and learns clauses.

Internally, CP-SAT reasons about atomic *facts* as Boolean literals such as:

- “$x \le c$” (an upper-bound fact),
- “$x \ge c$” (a lower-bound fact),
- sometimes “$x=a$” (which is equivalent to $x\le a$ and $x\ge a$).

Domain shrinking is then the process of *setting* some of these literals to true
and others to false. For example, fixing $x=a$ can be seen as asserting:
$$
x \le a \quad\text{and}\quad x \ge a,
$$
which removes every other value from the domain.

When CP-SAT branches, it is effectively deciding one such literal (e.g.,
$x\le c$) and then letting propagation derive the consequences. This is why the
search can be described as a SAT-style tree of Boolean decisions, even though
the “meaning” of each decision is about integer bounds.

### 5.12. From propagation to learned nogoods (one concrete EMTL pattern)

Learning requires that propagations come with reasons. Conceptually:

- Propagators deduce new bound/value facts from existing ones.
- If a contradiction is reached, CP-SAT extracts a **nogood**: a Boolean clause
  that forbids the combination of facts that caused the contradiction.

In EMTL, a very common pattern is:

1. Some decisions fix $\kappa$, $x_u$, and $x_v$.
2. The edge equation forces $x_{uv}=\kappa-x_u-x_v$ to a specific value $a$.
3. AllDifferent (or the domain $1\le x_{uv}\le N$) makes that forced value
   impossible.

At a high level, CP-SAT learns something of the form:
$$
\neg(\kappa=\kappa_0 \wedge x_u=p \wedge x_v=q) \quad\Longleftrightarrow\quad
(\kappa\ne \kappa_0)\ \vee\ (x_u\ne p)\ \vee\ (x_v\ne q),
$$
meaning: “do not try that same combination of assignments again.”

The practical effect is that CP-SAT does not merely *backtrack*; it accumulates
constraints that progressively carve away large regions of the search space.

---

## 6. Implementation mapping (math → code)

The reference implementation is in **Python** and centers around
`emtl_solver.py`.

### 6.1. Graph construction

- **Mathematical object:** the graph $G(m,n,k,t)$ and its partitions
  $(A,B,C,D)$.
- **Code:** `GraphParameters` validates $(m,n,k,t)$ and provides:
  - `num_vertices = m + 2n + k`,
  - `num_edges = mn + nk + nt`,
  - `total_labels = num_vertices + num_edges`.
- **Code:** `GraphConstructor.construct(params)` creates a `networkx.Graph` with
  vertices named `A0..A(m-1)`, `B0..B(n-1)`, `C0..C(n-1)`, `D0..D(k-1)`. It adds:
  - all edges between `A` and `B`,
  - all edges between `C` and `D`,
  - middle edges via `create_t_regular_bipartite_edges(n,t)` implementing the
    circulant rule from Section 3.3.

### 6.2. CP-SAT model generation and solving

- **Mathematical object:** variables $x_v$, $x_e$, and $\kappa$, with constraints
  (C1)–(C2).
- **Code:** `EMTLSolver.solve(G)` builds an `ortools.sat.python.cp_model.CpModel`
  as follows:
  1. `NewIntVar(1, N, ...)` for every vertex and edge label,
  2. `NewIntVar(6, 3N-3, ...)` for the magic constant,
  3. `AddAllDifferent([...])`,
  4. for each edge `(u,v)`, add `x_u + x_(u,v) + x_v == kappa`,
  5. run `CpSolver().Solve(model)` with a time limit and (optionally) multiple
     search workers.

The solver returns one of:
`FOUND` (a labeling was found), `INFEASIBLE` (no labeling exists, proved), or
`TIMEOUT` (no conclusion within the time limit).

### 6.3. Verification

Given a purported solution $(\kappa, f)$, verification is linear in $|E|$:

1. Check bijection: labels used by vertices and edges are exactly
   $\{1,\dots,N\}$.
2. Check magic: for every edge $uv$, verify $f(u)+f(uv)+f(v)=\kappa$.

**Code:** `EMTLSolver.verify_labeling(G, magic_constant, vertex_labels, edge_labels)`
implements these checks with explicit assertions.

### 6.4. Visualization and UI

The project includes:

- **Matplotlib visualization:** `EMTLVisualizer.visualize(result, ...)` draws a
  fixed four-column layout (A–B–C–D), colors partitions, colors edge layers, and
  annotates vertices/edges with the obtained labels.
- **Streamlit UI:** `web/app.py` provides an interactive frontend to choose
  $(m,n,k,t)$, set a timeout, run the solver, and view results.

---

## 7. Worked example: $G(2,2,2,1)$

Consider parameters $(m,n,k,t)=(2,2,2,1)$. Then:

$$
|V| = 2 + 2\cdot 2 + 2 = 8,\qquad
|E| = (2)(2) + (2)(2) + (2)(1) = 10,\qquad
N=18.
$$

The implementation (one run) finds an EMTL with magic constant $\kappa=27$.
One valid labeling (not necessarily unique) is:

- Vertex labels:
  - $f(A_0)=11,\; f(A_1)=7$
  - $f(B_0)=15,\; f(B_1)=4$
  - $f(C_0)=10,\; f(C_1)=6$
  - $f(D_0)=8,\; f(D_1)=3$
- Edge labels (selected):
  - $f(A_0B_0)=1$ since $11+1+15=27$
  - $f(B_0C_0)=2$ since $15+2+10=27$
  - $f(C_0D_0)=9$ since $10+9+8=27$

In practice, the solver returns complete dictionaries for all vertex and edge
labels, and `verify_labeling` checks every edge sum.

---

## 8. Practical notes for experimentation

### 8.1. Time limits and scaling

Because the number of label variables is $N=|V|+|E|$ and $|E|$ has quadratic
terms (e.g., $mn$ and $nk$), solve times can grow quickly with $m,n,k$.
The repository’s solver supports a user-chosen timeout and multiple CP-SAT
workers.

### 8.2. Reproducible runs

The main entry point `emtl_solver.py` runs a small demonstration suite and
saves figures under `images/output/`. For programmatic use, call:

```python
from emtl_solver import solve_emtl

result = solve_emtl(m=2, n=2, k=2, t=1, timeout=60, visualize=False, verbose=False)
if result.exists:
    print(result.magic_constant)
```

---

## 9. Conclusion

This project studies edge-magic total labelings on a structured four-partite
graph family $G(m,n,k,t)$ and solves the associated feasibility problem using a
CP-SAT formulation. The mathematical model is compact (a bijection plus one
linear constraint per edge), but the resulting combinatorial search can be
large; CP-SAT’s propagation and conflict-driven learning provide a practical
exact method for exploring EMTL existence across parameter regimes.

