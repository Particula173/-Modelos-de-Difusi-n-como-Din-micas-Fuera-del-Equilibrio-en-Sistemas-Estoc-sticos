import streamlit as st
import numpy as np

from diffusionApp import DiffusionApp
from backwardDiffusion import BackwardDiffusion

st.set_page_config(layout="wide")
st.title("🔥 Diffusion Model Dashboard")

# =========================
# 🔧 SIDEBAR
# =========================
st.sidebar.header("Parámetros")

min_picos = st.sidebar.slider("Min picos", 1, 50, 5)
max_picos = st.sidebar.slider("Max picos", 1, 50, 6)
complejidad = st.sidebar.slider("Complejidad", 0, 5, 0)
target_mean = st.sidebar.slider("Target mean", -100, 100, 40)

epochs = st.sidebar.slider("Epochs", 100, 10000, 3000)
T = st.sidebar.slider("Steps difusión (T)", 50, 1000, 500)

run_button = st.sidebar.button("🚀 Ejecutar")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Datos",
    "⚡ Forward",
    "🔁 Backward",
    "🎥 Dinámica"
])

# =========================
# EJECUCIÓN
# =========================
if run_button:

    # 🔥 correr pipeline
    app = DiffusionApp(n_samples=50000, T=T)

    app.generar_datos(
        min_picos=min_picos,
        max_picos=max_picos,
        complejidad=complejidad,
        target_mean=target_mean
    )

    app.crear_diffusion()

    # 🔥 forward simulation (necesaria para graficar)
    app.diff.simular()

    # 🔥 entrenar + backward
    app.entrenar_modelo(epochs=epochs)
    app.generar_muestras(n_samples=2000)

    # 🔥 desnormalizar
    original = app.data * app.std + app.mean
    generado = app.trayectoria[-1] * app.std + app.mean

    # =========================
    # 📊 TAB 1: DATOS
    # =========================
    with tab1:

        st.subheader("Distribución generada")

        # 👉 usando generador indirectamente (ya está en app.data)
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        fig, ax = plt.subplots()

        ax.hist(original, bins=120, density=True, alpha=0.6)

        kde = gaussian_kde(original)
        x = np.linspace(min(original), max(original), 1000)
        ax.plot(x, kde(x), linewidth=2)

        st.pyplot(fig)

        st.subheader("Resumen")
        st.write({
            "media": float(np.mean(original)),
            "varianza": float(np.var(original)),
            "min": float(np.min(original)),
            "max": float(np.max(original))
        })

    # =========================
    # ⚡ TAB 2: FORWARD
    # =========================
    with tab2:

        st.subheader("Distribuciones en el tiempo")
        st.pyplot(app.diff.fig_distribuciones())

        st.subheader("Test de normalidad")
        st.pyplot(app.diff.fig_normalidad())

        st.subheader("Skewness y Kurtosis")
        st.pyplot(app.diff.fig_skew_kurtosis())

    # =========================
    # 🔁 TAB 3: BACKWARD
    # =========================
    with tab3:

        st.subheader("Comparación final")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.hist(original, bins=100, density=True, alpha=0.5, label="Original")
        ax.hist(generado, bins=100, density=True, alpha=0.5, label="Generado")

        ax.legend()

        st.pyplot(fig)

        # 🔥 evolución backward
        st.subheader("Evolución backward")

        bd = BackwardDiffusion(app.diff, app.model)
        st.pyplot(bd.fig_backward(app.trayectoria))

        # 🔥 evaluación
        st.subheader("Evaluación")

        eval_stats = app.evaluar()
        st.json(eval_stats)
        
    # =========================
    # 🎥 TAB 4: DINÁMICA
    # =========================
    with tab4:

        st.subheader("Evolución del entrenamiento")
        st.pyplot(app.fig_loss())
        
        st.subheader("Fuerza efectiva forward")

        st.pyplot(app.diff.fig_fuerza_forward())
        
        st.subheader("Fuerza efectiva backward")

        bd = BackwardDiffusion(app.diff, app.model)

        st.pyplot(
            bd.fig_fuerza_backward()
        )

        st.subheader("Animación Forward")

        forward_anim = app.diff.animacion_forward()

        forward_gif = app.diff.animacion_forward()

        st.image(forward_gif)

        st.subheader("Animación Backward")

        backward_anim = bd.animacion_backward(
            app.trayectoria
        )

        backward_gif = bd.animacion_backward(
            app.trayectoria
        )

        st.image(backward_gif)