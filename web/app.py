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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emtl_solver import (
    solve_emtl,
    GraphParameters,
    SolverStatus,
    EMTLVisualizer,
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
# CUSTOM CSS - Clean, Modern Design
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .magic-number {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .result-card p {
        opacity: 0.9;
    }
    
    /* Error/Info cards */
    .error-card {
        background: linear-gradient(135deg, #f87171 0%, #dc2626 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-title">Edge-Magic Total Labeling</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find magic labelings for partitioned graphs using constraint programming</p>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CLEAN PARAMETER INPUT
# =============================================================================

with st.sidebar:
    st.header("⚙️ Parameters")
    
    st.markdown("---")
    
    # Simple parameter inputs with clear descriptions
    m = st.number_input(
        "m — Vertices in set A", 
        min_value=1, max_value=10, value=2,
        help="Size of the first partition"
    )
    
    n = st.number_input(
        "n — Vertices in sets B & C", 
        min_value=1, max_value=10, value=2,
        help="Size of the middle partitions"
    )
    
    k = st.number_input(
        "k — Vertices in set D", 
        min_value=1, max_value=10, value=2,
        help="Size of the last partition"
    )
    
    t = st.number_input(
        "t — B↔C regularity", 
        min_value=0, max_value=int(n), value=min(1, int(n)),
        help=f"Each vertex in B connects to exactly t vertices in C (max {n})"
    )
    
    st.markdown("---")
    
    timeout = st.slider(
        "⏱️ Timeout (seconds)",
        min_value=5, max_value=120, value=30
    )
    
    st.markdown("---")
    
    # Quick explanation
    with st.expander("📖 What is this?"):
        st.markdown("""
        **Edge-Magic Total Labeling** assigns numbers to vertices and edges 
        so that every edge has the same "magic sum":
        
        `vertex₁ + edge + vertex₂ = k`
        
        **Graph Structure:**
        - A ↔ B: Fully connected
        - B ↔ C: Each vertex has t connections  
        - C ↔ D: Fully connected
        """)

# =============================================================================
# MAIN CONTENT - SOLVE BUTTON
# =============================================================================

# Validate parameters
params = GraphParameters(m=int(m), n=int(n), k=int(k), t=int(t))
is_valid, error_msg = params.validate()

if not is_valid:
    st.error(f"❌ Invalid parameters: {error_msg}")
else:
    # Center the solve button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solve_clicked = st.button("🔍 Find Magic Labeling", use_container_width=True)
    
    if solve_clicked:
        with st.spinner("Searching..."):
            result = solve_emtl(
                m=int(m), n=int(n), k=int(k), t=int(t),
                timeout=timeout,
                visualize=False,
                verbose=False
            )
        st.session_state['result'] = result

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if 'result' in st.session_state:
    result = st.session_state['result']
    
    st.markdown("---")
    
    if result.exists:
        # Success - Show magic constant prominently
        st.markdown(f"""
        <div class="result-card">
            <h2>✨ Magic Labeling Found!</h2>
            <div class="magic-number">{result.magic_constant}</div>
            <p>Magic Constant (k)</p>
            <p style="margin-top: 1rem; font-size: 0.9rem;">
                Solved in {result.solve_time:.3f}s · 
                {result.graph.number_of_nodes()} vertices · 
                {result.graph.number_of_edges()} edges
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📊 Graph Visualization")
        
        fig = EMTLVisualizer.visualize(result, figsize=(14, 9), show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Details in expanders
        with st.expander("🏷️ View All Labels"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Vertex Labels:**")
                for partition in ['A', 'B', 'C', 'D']:
                    vertices = result.vertex_sets.get(partition, [])
                    if vertices:
                        labels = [f"{v}={result.vertex_labels[v]}" for v in sorted(vertices)]
                        st.write(f"Set {partition}: " + ", ".join(labels))
            
            with col2:
                st.markdown("**Edge Labels:**")
                for (u, v) in sorted(result.graph.edges())[:15]:
                    label = result.edge_labels[(u, v)]
                    st.write(f"{u}—{v} = {label}")
                if result.graph.number_of_edges() > 15:
                    st.write(f"... and {result.graph.number_of_edges() - 15} more edges")
        
        with st.expander("✓ Verify Magic Property"):
            st.markdown("Every edge sums to the same magic constant:")
            
            verified = []
            for (u, v) in sorted(result.graph.edges())[:10]:
                lu = result.vertex_labels[u]
                le = result.edge_labels[(u, v)]
                lv = result.vertex_labels[v]
                total = lu + le + lv
                verified.append(f"`{lu} + {le} + {lv} = {total}`")
            
            st.write(" · ".join(verified))
            
            if result.graph.number_of_edges() > 10:
                st.write(f"✓ All {result.graph.number_of_edges()} edges verified")
    
    elif result.status == SolverStatus.INFEASIBLE:
        st.markdown("""
        <div class="error-card">
            <h2>❌ No Solution Exists</h2>
            <p>It's mathematically proven that no magic labeling exists for this configuration.</p>
            <p style="margin-top: 1rem;">Try different parameter values.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif result.status == SolverStatus.TIMEOUT:
        st.markdown(f"""
        <div class="info-card">
            <h2>⏱️ Time Limit Reached</h2>
            <p>Couldn't find a solution within {timeout} seconds.</p>
            <p style="margin-top: 1rem;">Try smaller parameters or increase the timeout.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error(f"Error: {result.status.value} - {result.message}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 1rem;">
    EMTL Solver · Built with OR-Tools & Streamlit
</div>
""", unsafe_allow_html=True)
