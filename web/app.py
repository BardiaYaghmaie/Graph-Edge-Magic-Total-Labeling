"""
EMTL Web Interface - Streamlit Application
==========================================

An interactive web application for exploring Edge-Magic Total Labelings.

Run with: streamlit run web/app.py
"""

import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import io
import base64

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emtl_solver import (
    solve_emtl,
    GraphParameters,
    SolverStatus,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="EMTL Solver",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<p class="main-header">🔢 Edge-Magic Total Labeling Solver</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive tool for finding EMTLs on partitioned graphs</p>', 
            unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - PARAMETERS
# =============================================================================

st.sidebar.header("📊 Graph Parameters")

st.sidebar.markdown("""
**Graph Structure:**
- A ↔ B: Complete bipartite K_{m,n}
- B ↔ C: t-regular bipartite
- C ↔ D: Complete bipartite K_{n,k}
""")

st.sidebar.markdown("---")

# Parameter inputs
m = st.sidebar.number_input(
    "**m** (vertices in set A)", 
    min_value=1, max_value=10, value=2,
    help="Number of vertices in partition A"
)

n = st.sidebar.number_input(
    "**n** (vertices in sets B and C)", 
    min_value=1, max_value=10, value=2,
    help="Number of vertices in partitions B and C"
)

k = st.sidebar.number_input(
    "**k** (vertices in set D)", 
    min_value=1, max_value=10, value=2,
    help="Number of vertices in partition D"
)

t = st.sidebar.number_input(
    "**t** (B-C regularity)", 
    min_value=0, max_value=n, value=min(1, n),
    help=f"Regularity degree of B-C subgraph (0 ≤ t ≤ n={n})"
)

st.sidebar.markdown("---")

# Advanced options
st.sidebar.header("⚙️ Solver Options")

timeout = st.sidebar.slider(
    "Timeout (seconds)",
    min_value=5, max_value=120, value=30,
    help="Maximum time to search for a solution"
)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📈 Graph Statistics")
    
    # Calculate statistics
    params = GraphParameters(m=m, n=n, k=k, t=t)
    is_valid, error_msg = params.validate()
    
    if is_valid:
        # Display metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Vertices |V|", params.num_vertices)
        with metric_col2:
            st.metric("Edges |E|", params.num_edges)
        with metric_col3:
            st.metric("Labels Needed", params.total_labels)
        
        # Edge breakdown
        st.subheader("Edge Distribution")
        edge_data = {
            "A-B (complete)": m * n,
            "B-C (t-regular)": n * t,
            "C-D (complete)": n * k,
        }
        
        for edge_type, count in edge_data.items():
            st.write(f"- **{edge_type}**: {count} edges")
    else:
        st.error(f"⚠️ Invalid parameters: {error_msg}")

with col2:
    st.header("🎯 Solve EMTL")
    
    if is_valid:
        if st.button("🚀 Find EMTL", type="primary", use_container_width=True):
            with st.spinner("Searching for Edge-Magic Total Labeling..."):
                # Solve
                result = solve_emtl(
                    m=m, n=n, k=k, t=t,
                    timeout=timeout,
                    visualize=False,
                    verbose=False
                )
            
            # Store result in session state
            st.session_state['result'] = result
            st.session_state['params'] = (m, n, k, t)

# =============================================================================
# RESULTS SECTION
# =============================================================================

if 'result' in st.session_state:
    result = st.session_state['result']
    
    st.markdown("---")
    st.header("📋 Results")
    
    if result.exists:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ EMTL Found!</h3>
            <p><strong>Magic Constant k = {result.magic_constant}</strong></p>
            <p>Solved in {result.solve_time:.3f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🖼️ Visualization", "📊 Vertex Labels", "🔗 Edge Labels"])
        
        with tab1:
            # Generate visualization
            from emtl_solver import EMTLVisualizer
            
            fig = EMTLVisualizer.visualize(result, figsize=(12, 8), show=False)
            st.pyplot(fig)
            plt.close(fig)
            
        with tab2:
            st.subheader("Vertex Labels")
            
            # Create columns for each partition
            v_cols = st.columns(4)
            partitions = ['A', 'B', 'C', 'D']
            
            for i, partition in enumerate(partitions):
                with v_cols[i]:
                    st.markdown(f"**Set {partition}**")
                    for v in result.vertex_sets[partition]:
                        label = result.vertex_labels[v]
                        st.write(f"{v} → **{label}**")
        
        with tab3:
            st.subheader("Edge Labels (with verification)")
            
            # Group edges by type
            edge_groups = {'A-B': [], 'B-C': [], 'C-D': []}
            
            for (u, v) in sorted(result.graph.edges()):
                pu, pv = u[0], v[0]
                label = result.edge_labels[(u, v)]
                total = result.vertex_labels[u] + label + result.vertex_labels[v]
                
                if pu == 'A' or pv == 'A':
                    edge_groups['A-B'].append((u, v, label, total))
                elif pu == 'D' or pv == 'D':
                    edge_groups['C-D'].append((u, v, label, total))
                else:
                    edge_groups['B-C'].append((u, v, label, total))
            
            for group_name, edges in edge_groups.items():
                if edges:
                    st.markdown(f"**{group_name} Edges:**")
                    for u, v, label, total in edges:
                        vu = result.vertex_labels[u]
                        vv = result.vertex_labels[v]
                        st.write(f"  {u}—{v}: **{label}** [{vu} + {label} + {vv} = {total}]")
    
    elif result.status == SolverStatus.INFEASIBLE:
        st.markdown("""
        <div class="error-box">
            <h3>❌ No EMTL Exists</h3>
            <p>The solver has proven that no Edge-Magic Total Labeling exists for this graph configuration.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif result.status == SolverStatus.TIMEOUT:
        st.markdown(f"""
        <div class="info-box">
            <h3>⏱️ Solver Timeout</h3>
            <p>Could not determine if EMTL exists within {timeout} seconds.</p>
            <p>Try increasing the timeout or using smaller parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="error-box">
            <h3>⚠️ Error</h3>
            <p>Status: {result.status.value}</p>
            <p>{result.message}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# INFORMATION SECTION
# =============================================================================

st.markdown("---")

with st.expander("ℹ️ About Edge-Magic Total Labeling"):
    st.markdown("""
    ### What is an Edge-Magic Total Labeling?
    
    An **Edge-Magic Total Labeling (EMTL)** of a graph G = (V, E) is a bijection:
    
    $$f: V \\cup E \\rightarrow \\{1, 2, \\ldots, |V| + |E|\\}$$
    
    such that for every edge $uv \\in E$:
    
    $$f(u) + f(uv) + f(v) = k$$
    
    where $k$ is a constant called the **magic constant**.
    
    ### Graph Structure
    
    This solver works with graphs having four vertex partitions:
    
    | Set | Size | Description |
    |-----|------|-------------|
    | A | m | Connected to all of B |
    | B | n | Connected to A (complete) and C (t-regular) |
    | C | n | Connected to B (t-regular) and D (complete) |
    | D | k | Connected to all of C |
    
    ### Applications
    
    - **Network Design**: Balancing load across connections
    - **Scheduling**: Fair resource allocation
    - **Cryptography**: Secret sharing schemes
    - **Coding Theory**: Error-correcting codes
    """)

with st.expander("🔧 Algorithm Details"):
    st.markdown("""
    ### Constraint Programming Approach
    
    The EMTL problem is solved using **Constraint Satisfaction Programming (CSP)**:
    
    **Variables:**
    - One variable for each vertex label: $x_v \\in \\{1, ..., |V|+|E|\\}$
    - One variable for each edge label: $x_e \\in \\{1, ..., |V|+|E|\\}$
    - One variable for magic constant: $k$
    
    **Constraints:**
    1. **ALL-DIFFERENT**: All labels must be distinct (bijection)
    2. **MAGIC SUM**: For each edge $(u,v)$: $x_u + x_{(u,v)} + x_v = k$
    
    ### Solver: Google OR-Tools CP-SAT
    
    We use Google's CP-SAT solver which employs:
    - Lazy clause generation
    - Clause learning from conflicts
    - Multi-threaded parallel search
    - Efficient constraint propagation
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>EMTL Solver v2.0 | Built with Streamlit & OR-Tools</p>
    <p>For research and education in discrete mathematics and graph theory</p>
</div>
""", unsafe_allow_html=True)

