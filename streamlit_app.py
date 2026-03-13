import streamlit as st
import pandas as pd
from pathlib import Path
import json
import numpy as np
import plotly.graph_objs as go

from src.registro_clases import RegistroClases
from src.run_from_dashboard import run_simulation

# Carpeta de experimentos
RUNS_DIR = Path("experiments/results/runs")

rc = RegistroClases()

tab1, tab2 = st.tabs(["📊 Dashboard", "🚀 Run Simulation"])

with tab1:
    st.set_page_config(
        page_title="RL Experiments",
        page_icon="📈",
        layout="wide"
    )

    st.title("RL Experiments Dashboard")
    st.markdown("Visualiza recompensas, moving averages y compara runs fácilmente.")

    # Listar runs disponibles
    run_folders = sorted([f for f in RUNS_DIR.iterdir() if f.is_dir()])
    run_names = [f.name for f in run_folders]

    # ----------------------
    # COLUMNA IZQUIERDA
    # ----------------------

    col_runs, col_main = st.columns([1,3])

    with col_runs:
        st.subheader("Runs")

        selected_run = st.radio(
            "Selecciona una run",
            run_names
        )
        #selected_run = st.selectbox("Run", run_names)

    # selected_runs = st.multiselect("Selecciona uno o más runs:", run_names)

    # ----------------------
    # COLUMNA DERECHA
    # ----------------------

    with col_main:
        
        run_dir = RUNS_DIR / selected_run

        st.header(f"Run: {selected_run}")

        # -------------------
        # LEARNING CURVE NPY
        # -------------------
        with st.container():
            st.subheader("Learning Curve")
            learning_curve_file = run_dir / "learning_curve.npy"

            if learning_curve_file.exists():

                rewards = np.load(learning_curve_file)
                col1, col2, col3 = st.columns(3)

                col1.metric("Episodes", len(rewards))
                col2.metric("Final Reward", round(rewards[-1],2))
                col3.metric("Best Reward", round(max(rewards),2))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=rewards,
                    mode="lines",
                    name="Reward"
                ))

                fig.update_layout(
                    title="Learning Curve",
                    xaxis_title="Episode",
                    yaxis_title="Reward"
                )

                st.plotly_chart(fig, use_container_width=True)
        
        #window = st.slider("Moving Average Window", 1, 100, 20)

        #ma = pd.Series(rewards).rolling(window).mean()

        #fig.add_trace(go.Scatter(
        #    y=ma,
        #    mode="lines",
        #    name="Moving Avg"
        #))

        # -------------------
        # CONFIG TABLE
        # -------------------

        config_file = run_dir / "config.json"

        if config_file.exists():

            with open(config_file) as f:
                config = json.load(f)

            with st.container():
                st.subheader("Config")

                #df_config = pd.DataFrame(
                #    config.items(),
                #    columns=["Parameter", "Value"]
                #)
                df_config = pd.DataFrame([config])
                st.dataframe(df_config)

                st.dataframe(df_config, use_container_width=True)

with tab2: 

    st.header("Run New Simulation")
    env_names, agent_names, policy_names = rc.get_component_names()
    print(env_names, agent_names, policy_names)

    with st.form("run_form"):

        policy =  st.selectbox(
            "Policy algorithm",
            options=policy_names
        )
        env = st.selectbox(
            "Environment",
            options=env_names
        )
        agent = st.selectbox(
            "Agents",
            options=agent_names
        )


        n_episodes = st.number_input("Episodes", 1, 100000, 1000)
        max_steps = st.number_input("Max Steps", 1, 10000, 1200)

        alpha = st.number_input("Alpha", 0.0, 1.0, 0.1)
        gamma = st.number_input("Gamma", 0.0, 1.0, 0.99)

        epsilon = st.number_input("Epsilon", 0.0, 1.0, 1.0)
        epsilon_decay = st.number_input("Epsilon Decay", 0.9, 1.0, 0.995)
        epsilon_min = st.number_input("Epsilon Min", 0.0, 1.0, 0.1)

        seed = st.number_input("Seed", 0, 100000, 42)

        submitted = st.form_submit_button("Run Simulation 🚀")

        config_submit = {
            "algorithm": policy,
            "env": env,
            "agent": agent,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "seed": seed
        }


        if submitted:

            with open("temp_config.json","w") as f:
                json.dump(config_submit,f)

            run_simulation("temp_config.json")

            st.success("Simulation started!")
            st.rerun() # Para que la simulación aparezca automáticamente en el dashboard
