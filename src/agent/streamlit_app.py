"""
Streamlit Visualization for HypergraphAgent
===========================================

Run with:
    streamlit run src/agent/streamlit_app.py

Features:
- Real-time memory matrix heatmap
- Bistability phase diagram (r vs regime regions)
- Physics parameter controls (λ, μ, γ, n_high)
- Cognitive mode panel with regime indicator
- ODE dynamics viewer
- Chat with memory
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.hypergraph_agent import HypergraphAgent


def make_memory_heatmap(M: np.ndarray, group_labels: list, layer_labels: list):
    """Create a heatmap figure of the memory matrix."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    im = ax.imshow(M.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels)
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel('Memory Groups')
    ax.set_ylabel('Layers')
    ax.set_title('Memory Matrix M[k×L] — Activation Levels')
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(i, j, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation m')
    plt.tight_layout()
    return fig


def make_phase_diagram(lambda_c: float, lambda_val: float, k: int = 4):
    """Create bistability phase diagram."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    r = np.linspace(0, 2.5, 300)
    
    multi = (r < 0.8).astype(float)
    moderate = ((r >= 0.8) & (r < 1.2)).astype(float) * 0.8
    near_wta = (r >= 1.2).astype(float) * 0.6
    
    ax.fill_between(r, 0, multi, alpha=0.25, color='blue', label='Multi-attractor')
    ax.fill_between(r, 0, moderate, alpha=0.25, color='yellow', label='Moderate')
    ax.fill_between(r, 0, near_wta, alpha=0.25, color='red', label='Near-WTA')
    
    current_r = lambda_val / lambda_c if lambda_c > 0 else 0
    ax.axvline(x=current_r, color='lime', linewidth=3, linestyle='--', 
               label=f'Current r={current_r:.2f}')
    ax.axvline(x=1.0, color='black', linewidth=1.5, linestyle=':', 
               alpha=0.7, label='λ_c threshold')
    
    ax.set_xlabel('r = λ / λ_c', fontsize=11)
    ax.set_ylabel('Regime strength', fontsize=11)
    ax.set_title(f'Bistability Phase Diagram (λ_c={lambda_c:.4f})', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right', fontsize=9)
    
    ax.text(0.35, 1.05, 'Multi-attractor\n(bistable)', ha='center', fontsize=9, style='italic')
    ax.text(1.0, 1.05, 'Moderate\n(transitional)', ha='center', fontsize=9, style='italic')
    ax.text(1.8, 1.05, 'Near-WTA\n(synchronized)', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig


def make_group_bars(M: np.ndarray, group_labels: list):
    """Create bar chart of group mean activations."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    mean_act = M.mean(axis=1)
    colors = plt.cm.RdYlBu_r(mean_act)
    
    bars = ax.bar(group_labels, mean_act, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, mean_act):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Mean Activation', fontsize=11)
    ax.set_title('Group Activation Levels', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='m=0.5 threshold')
    ax.legend()
    
    plt.tight_layout()
    return fig


def make_potential_plot(lambda_val: float, lambda_c: float):
    """Create effective potential landscape plot."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    m = np.linspace(0, 1, 300)
    
    bistable = m**3/3 - m**2/2
    r = lambda_val / lambda_c if lambda_c > 0 else 0
    coupling_term = -r * 0.3 * (0.5 - m)**2
    
    V = bistable + coupling_term
    V = V - np.min(V) + 0.05
    
    ax.plot(m, V, 'b-', linewidth=2.5, label=f'V(m), r={r:.2f}')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='unstable (m=0.5)')
    
    m_near_0 = np.exp(-50 * m**2)
    m_near_1 = np.exp(-50 * (1-m)**2)
    
    ax.scatter([0, 1], [0.05, 0.05], color='green', s=120, zorder=5, 
              marker='^', label='stable attractors')
    ax.scatter([0.5], [V[150]], color='red', s=120, marker='x', zorder=5, 
              linewidths=3, label='saddle point')
    
    ax.set_xlabel('m (mean activation)', fontsize=11)
    ax.set_ylabel('Potential V(m)', fontsize=11)
    ax.set_title('Effective Potential Landscape', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def get_regime_name(ratio):
    if ratio < 0.8:
        return "multi-attractor"
    elif ratio < 1.2:
        return "moderate-coupling"
    else:
        return "near-WTA"


def main():
    st.set_page_config(
        page_title="HypergraphAgent Visualizer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 HypergraphAgent — Physics-Based Memory Visualizer")
    st.caption("Multi-layer hypergraph memory with λ_c-controlled bistability")
    
    if 'agent' not in st.session_state:
        st.session_state.agent = HypergraphAgent(
            k=4, L=2,
            use_embeddings=False,
            name="streamlit_agent"
        )
        st.session_state.agent.group_labels = ["work", "personal", "technical", "creative"]
        st.session_state.agent.layer_labels = ["current", "history"]
        st.session_state.messages = []
    
    agent = st.session_state.agent
    lambda_c = agent.memory.get_lambda_c() or 0.044
    
    with st.sidebar:
        st.header("⚙️ Physics Controls")
        
        st.subheader(f"λ_c = {lambda_c:.4f}")
        
        lambda_val = st.slider(
            "λ (global coupling)",
            0.0, lambda_c * 2.5,
            float(agent.memory.lambda_),
            step=0.001,
            help="Synchronizes groups in same layer"
        )
        
        mu_val = st.slider(
            "μ (local coupling)",
            -1.0, 1.0,
            float(agent.memory.mu),
            step=0.01,
            help="Synchronizes same group across layers"
        )
        
        gamma_val = st.slider(
            "γ (decay)",
            0.0, 0.2,
            float(agent.memory.gamma),
            step=0.005,
            help="Forgetting/decay rate"
        )
        
        st.divider()
        
        st.subheader("🎯 n_high Control")
        n_high_current = agent.memory.get_n_high_groups()
        n_high_target = st.slider(
            "Target n_high",
            1, 4,
            n_high_current,
            help="Number of groups to maintain at high activation"
        )
        if st.button("Apply n_high", use_container_width=True):
            agent.memory.set_n_high(n_high_target)
        
        n_high_actual = agent.memory.get_n_high_groups()
        st.progress(n_high_actual / 4.0, text=f"n_high = {n_high_actual}/4")
        
        st.divider()
        
        st.subheader("📊 Regime Indicator")
        current_r = agent.memory.lambda_ / lambda_c if lambda_c > 0 else 0
        regime = get_regime_name(current_r)
        
        regime_color = {"multi-attractor": "blue", "moderate-coupling": "orange", "near-WTA": "red"}
        st.markdown(f"**Regime:** :{regime_color[regime]}[**{regime}**]")
        st.markdown(f"**r = λ/λ_c = {current_r:.3f}**")
        
        regime_descriptions = {
            "multi-attractor": "Groups maintain independent attractors. Memory is distributed.",
            "moderate-coupling": "Crossing regime. Sensitive to perturbations — memory can switch.",
            "near-WTA": "Groups synchronized toward consensus. Single attractor dominant."
        }
        st.caption(regime_descriptions[regime])
        
        st.divider()
        
        st.subheader("🧠 Cognitive Mode")
        mode_options = ["neutral", "exploratory", "focused", "creative"]
        controller_state = agent.controller.get_state_summary()
        current_mode = controller_state.get("mode", "neutral")
        mode_idx = mode_options.index(current_mode) if current_mode in mode_options else 0
        
        selected_mode = st.selectbox("Mode", mode_options, index=mode_idx)
        
        if st.button("Apply Mode", use_container_width=True):
            agent.controller.force_mode(selected_mode)
            lam, mu, _ = agent.controller.update("", 0.5)
            agent.memory.lambda_ = lam
            agent.memory.mu = mu
        
        st.divider()
        
        if st.button("🔄 Reset Memory", use_container_width=True):
            agent.reset_memory()
            st.session_state.messages = []
            st.rerun()
    
    state = agent.get_memory_state()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mode", state["controller"]["mode"])
    with m2:
        st.metric("λ", f"{state['memory']['lambda']:.4f}")
    with m3:
        st.metric("r", f"{state['controller']['lambda_ratio']:.2f}")
    with m4:
        st.metric("Turns", state["conversation_turns"])
    
    tab_heatmap, tab_physics, tab_dynamics = st.tabs([
        "📊 Memory Heatmap",
        "⚡ Bistability Physics",
        "🔄 ODE Dynamics"
    ])
    
    with tab_heatmap:
        col_h1, col_h2 = st.columns([1, 1], gap="large")
        
        with col_h1:
            st.subheader("Memory Matrix M[k×L]")
            fig_heat = make_memory_heatmap(agent.memory.M, agent.group_labels, agent.layer_labels)
            st.pyplot(fig_heat, use_container_width=True)
        
        with col_h2:
            st.subheader("Group Activations")
            fig_bars = make_group_bars(agent.memory.M, agent.group_labels)
            st.pyplot(fig_bars, use_container_width=True)
        
        st.subheader("Per-Layer Activation")
        layer_activations = agent.memory.M.mean(axis=0)
        for layer, label in enumerate(agent.layer_labels):
            with st.expander(f"Layer {layer}: {label} (mean={layer_activations[layer]:.3f})"):
                cols = st.columns(4)
                for group in range(agent.k):
                    with cols[group]:
                        val = agent.memory.M[group, layer]
                        st.metric(
                            f"{agent.group_labels[group]}",
                            f"{val:.3f}",
                            delta=f"{val - 0.5:.2f}" if val != 0.5 else None
                        )
    
    with tab_physics:
        col_p1, col_p2 = st.columns(2, gap="large")
        
        with col_p1:
            st.subheader("Bistability Phase Diagram")
            fig_phase = make_phase_diagram(lambda_c, agent.memory.lambda_, agent.k)
            st.pyplot(fig_phase, use_container_width=True)
        
        with col_p2:
            st.subheader("Effective Potential V(m)")
            fig_pot = make_potential_plot(agent.memory.lambda_, lambda_c)
            st.pyplot(fig_pot, use_container_width=True)
        
        st.divider()
        
        col_params1, col_params2, col_params3 = st.columns(3)
        with col_params1:
            st.metric("λ (global)", f"{agent.memory.lambda_:.4f}")
            st.caption("Synchronizes groups in same layer")
        with col_params2:
            st.metric("μ (local)", f"{agent.memory.mu:.4f}")
            st.caption("Synchronizes same group across layers")
        with col_params3:
            st.metric("γ (decay)", f"{agent.memory.gamma:.4f}")
            st.caption("Pulls activations toward 0")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            ### λ_c Critical Value
            
            **λ_c is computed from k and n_high:**
            - Below λ_c → Multi-attractor regime
            - Near λ_c → Phase transition (moderate coupling)  
            - Above λ_c → Near-WTA (synchronized)
            
            The proximity ratio **r = λ/λ_c** directly maps to the control regime.
            """)
        with col_info2:
            st.markdown("""
            ### ODE Dynamics
            
            ```
            dm/dt = m(1-m)(2m-1)    [bistable term]
                  + λ(mean_col - m) [global coupling]
                  + μ(mean_row - m) [local coupling]
                  - γ·m             [decay]
            ```
            
            The **bistable term** creates the double-well potential with attractors at m=0 and m=1.
            """)
    
    with tab_dynamics:
        st.subheader("ODE State Derivative Analysis")
        
        m_flat = agent.memory.M.flatten()
        dMdt = agent.memory._compute_dMdt(m_flat)
        dMdt_matrix = dMdt.reshape(agent.k, agent.L)
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.subheader("|dm/dt| per Group")
            mean_abs_dMdt = np.mean(np.abs(dMdt_matrix), axis=1)
            for i, (label, val) in enumerate(zip(agent.group_labels, mean_abs_dMdt)):
                st.metric(f"{label}", f"{val:.5f}")
        
        with col_d2:
            st.subheader("Steady State Check")
            max_dMdt = np.max(np.abs(dMdt))
            is_steady = max_dMdt < 1e-4
            
            if is_steady:
                st.success(f"✓ At steady state (max|dm/dt| = {max_dMdt:.6f})")
            else:
                st.warning(f"⚡ Evolving (max|dm/dt| = {max_dMdt:.4f})")
                st.caption("Adjust λ/μ or wait for convergence")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶️ Run 50 ODE steps", use_container_width=True):
                    dt = 0.01
                    for _ in range(50):
                        m_flat = agent.memory.M.flatten()
                        dMdt = agent.memory._compute_dMdt(m_flat)
                        new_m = m_flat + dt * dMdt
                        agent.memory.M = np.clip(new_m, 0, 1).reshape(agent.k, agent.L)
                    st.rerun()
            with col_btn2:
                if st.button("⏩ Run 200 ODE steps", use_container_width=True):
                    dt = 0.01
                    for _ in range(200):
                        m_flat = agent.memory.M.flatten()
                        dMdt = agent.memory._compute_dMdt(m_flat)
                        new_m = m_flat + dt * dMdt
                        agent.memory.M = np.clip(new_m, 0, 1).reshape(agent.k, agent.L)
                    st.rerun()
        
        st.subheader("dM/dt Matrix")
        import pandas as pd
        df_dmdt = pd.DataFrame(
            dMdt_matrix.T,
            index=agent.layer_labels,
            columns=agent.group_labels
        )
        st.dataframe(
            df_dmdt.style.background_gradient(cmap='RdYlBu_r', vmin=-0.1, vmax=0.1),
            use_container_width=True
        )
    
    st.divider()
    
    st.subheader("💬 Chat with Memory")
    
    chat_col1, chat_col2 = st.columns([4, 1])
    with chat_col1:
        user_input = st.text_input(
            "Your message:",
            key="user_input_chat",
            placeholder="Type something... try: 'I want to work on a creative project but also need to finish work tasks'",
            label_visibility="collapsed"
        )
    with chat_col2:
        st.write("")
        send_clicked = st.button("Send ✈️", type="primary", use_container_width=True)
    
    if send_clicked and user_input:
        with st.spinner("Processing..."):
            response = agent.chat(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.session_state.messages:
        for msg in st.session_state.messages[-10:]:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            if msg["role"] == "user":
                st.chat_message("user", avatar=avatar).write(msg["content"])
            else:
                st.chat_message("assistant", avatar=avatar).write(msg["content"])
    
    with st.expander("📜 Full Conversation History"):
        for msg in st.session_state.messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.text(f"{role_icon} {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")


if __name__ == "__main__":
    main()