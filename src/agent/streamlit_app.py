"""
Streamlit Visualization for HypergraphAgent
============================================

Run with:
    streamlit run src/agent/streamlit_app.py

This app provides:
- Real-time memory matrix heatmap
- Physics parameter controls
- Conversation history
- Mode indicator
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.hypergraph_agent import HypergraphAgent


def render_memory_heatmap(M: np.ndarray, group_labels: list, layer_labels: list):
    """Render memory matrix as a heatmap."""
    import pandas as pd
    
    df = pd.DataFrame(
        M.T,
        index=layer_labels,
        columns=group_labels
    )
    
    st.dataframe(
        df.style.background_gradient(cmap='RdYlBu_r', vmin=0, vmax=1),
        use_container_width=True
    )


def main():
    st.title("HypergraphAgent Memory Visualizer")
    
    if 'agent' not in st.session_state:
        st.session_state.agent = HypergraphAgent(
            k=4, L=2,
            use_embeddings=False,
            name="streamlit_agent"
        )
        st.session_state.agent.group_labels = ["work", "personal", "technical", "creative"]
        st.session_state.agent.layer_labels = ["current", "history"]
    
    agent = st.session_state.agent
    
    st.sidebar.header("Physics Controls")
    
    lambda_val = st.sidebar.slider("λ (lambda)", 0.0, 0.2, float(agent.memory.lambda_))
    mu_val = st.sidebar.slider("μ (mu)", -1.0, 1.0, float(agent.memory.mu))
    gamma_val = st.sidebar.slider("γ (decay)", 0.0, 0.1, float(agent.memory.gamma))
    
    agent.memory.lambda_ = lambda_val
    agent.memory.mu = mu_val
    agent.memory.gamma = gamma_val
    
    st.sidebar.header("Mode Controls")
    mode_options = ["balanced", "exploratory", "focused"]
    selected_mode = st.sidebar.selectbox("Cognitive Mode", mode_options)
    
    if st.sidebar.button("Apply Mode"):
        agent.controller.force_mode(selected_mode)
        lam, mu, _ = agent.controller.update("", 0.5)
        agent.memory.lambda_ = lam
        agent.memory.mu = mu
    
    state = agent.get_memory_state()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mode", state["controller"]["mode"])
    with col2:
        st.metric("λ", f"{state['memory']['lambda']:.4f}")
    with col3:
        st.metric("Turns", state["conversation_turns"])
    
    st.subheader("Memory Matrix Heatmap")
    M = agent.memory.M
    render_memory_heatmap(M, agent.group_labels, agent.layer_labels)
    
    st.subheader("Group Activation Levels")
    groups = state["memory"]["groups"]
    for i, (label, val) in enumerate(zip(agent.group_labels, groups)):
        st.progress(float(val), text=f"{label}: {val:.2f}")
    
    st.subheader("Physics State")
    controller = state["controller"]
    cols = st.columns(4)
    with cols[0]:
        st.metric("λ ratio", f"{controller['lambda_ratio']:.2f}")
    with cols[1]:
        st.metric("r", f"{controller['lambda_ratio']:.2f}")
    with cols[2]:
        st.metric("Avg Conflict", f"{controller['avg_conflict']:.2f}")
    with cols[3]:
        st.metric("Oscillating", "Yes" if controller['is_oscillating'] else "No")
    
    st.subheader("Chat")
    user_input = st.text_input("Your message:", key="user_input")
    
    if st.button("Send") and user_input:
        response = agent.chat(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    
    if st.sidebar.button("Reset Memory"):
        agent.reset_memory()
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
