# =========================================
# 🔥 EXPORTADOR DE FIGURAS (SVG VECTORIAL)
# =========================================

import matplotlib.pyplot as plt
import numpy as np

from diffusionApp import DiffusionApp
from backwardDiffusion import BackwardDiffusion

# =========================================
# ⚙️ CONFIGURACIÓN PRO (opcional pero recomendado)
# =========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "figure.dpi": 300
})

# =========================================
# 💾 GUARDAR FIGURA EN SVG
# =========================================
def guardar_figura(fig, nombre):
    fig.savefig(f"{nombre}.svg", bbox_inches='tight')
    print(f"✔ Guardado: {nombre}.svg")

# =========================================
# 📊 FORWARD: evolución de distribuciones
# =========================================
def exportar_forward(diffusion, nombre="forward_diffusion"):

    diffusion.simular()
    fig = diffusion.fig_distribuciones()

    guardar_figura(fig, nombre)

# =========================================
# 📈 NORMALIDAD
# =========================================
def exportar_normalidad(diffusion, nombre="normalidad"):

    fig = diffusion.fig_normalidad()
    guardar_figura(fig, nombre)

# =========================================
# 📉 SKEWNESS & KURTOSIS
# =========================================
def exportar_momentos(diffusion, nombre="momentos"):

    fig = diffusion.fig_skew_kurtosis()
    guardar_figura(fig, nombre)

# =========================================
# 🔁 BACKWARD
# =========================================
def exportar_backward(diffusion, model, nombre="backward"):

    bd = BackwardDiffusion(diffusion, model)
    trayectoria = bd.sample(n_samples=2000)

    fig = bd.fig_backward(trayectoria)
    guardar_figura(fig, nombre)

# =========================================
# ⚖️ COMPARACIÓN FINAL
# =========================================
def exportar_comparacion(app, nombre="comparacion"):

    original = app.data * app.std + app.mean
    generado = app.trayectoria[-1] * app.std + app.mean

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(original, bins=100, density=True, alpha=0.6, label="Original")
    ax.hist(generado, bins=100, density=True, alpha=0.6, label="Generado")

    ax.set_title("Reconstrucción de la distribución")
    ax.legend()

    guardar_figura(fig, nombre)
    
# =========================================
# 📈 LOSS
# =========================================
def exportar_loss(app, nombre="loss_training"):

    fig = app.fig_loss()

    guardar_figura(fig, nombre)
    
# =========================================
# ⚡ FUERZA EFECTIVA
# =========================================
def exportar_fuerza(diffusion, model, nombre="fuerza_backward"):

    bd = BackwardDiffusion(diffusion, model)

    fig = bd.fig_fuerza_backward()

    guardar_figura(fig, nombre)

# =========================================
# ⚡ FUERZA FORWARD
# =========================================
def exportar_fuerza_forward(diffusion, nombre="fuerza_forward"):

    fig = diffusion.fig_fuerza_forward()

    guardar_figura(fig, nombre)

# =========================================
# 🚀 PIPELINE COMPLETO
# =========================================
def ejecutar_y_exportar():

    # 🔧 parámetros (ajústalos si quieres)
    params_generador = {
        "min_picos": 3,
        "max_picos": 6,
        "complejidad": 0,
        "target_mean": 32
    }

    epochs = 3000
    T = 500

    # =====================================
    # 🧪 CREAR APP
    # =====================================
    app = DiffusionApp(n_samples=50000, T=T)

    # 1. datos
    app.generar_datos(**params_generador)

    # 2. forward
    app.crear_diffusion()

    # 3. simular forward
    app.diff.simular()

    # 4. entrenar modelo
    app.entrenar_modelo(epochs=epochs)

    # 5. generar muestras
    app.generar_muestras(n_samples=2000)

    # =====================================
    # 📊 EXPORTAR FIGURAS
    # =====================================
    exportar_forward(app.diff)
    exportar_normalidad(app.diff)
    exportar_momentos(app.diff)
    exportar_backward(app.diff, app.model)
    exportar_comparacion(app)
    exportar_loss(app)
    exportar_fuerza(app.diff, app.model)
    exportar_fuerza_forward(app.diff)

    print("\n🔥 Todas las figuras exportadas en SVG")


# =========================================
# ▶️ EJECUCIÓN
# =========================================
if __name__ == "__main__":
    ejecutar_y_exportar()