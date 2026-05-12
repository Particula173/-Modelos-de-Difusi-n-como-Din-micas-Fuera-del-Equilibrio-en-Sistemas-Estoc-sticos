import matplotlib.pyplot as plt
import numpy as np

from generadorDatos import GeneradorDatos
from forwardDiffusion import ForwardDiffusion
from noisePredictor import train_model
from backwardDiffusion import BackwardDiffusion

from scipy.stats import wasserstein_distance, skew, kurtosis


class DiffusionApp:

    def __init__(self, n_samples=5000, T=200):
        self.n_samples = n_samples
        self.T = T

        self.data = None
        self.diff = None
        self.model = None
        self.trayectoria = None

    # =========================
    # 1. GENERAR DATOS
    # =========================
    def generar_datos(self, **kwargs):

        gen = GeneradorDatos(n_samples=self.n_samples)
        self.data = gen.generar(**kwargs)

        # 🔥 NORMALIZACIÓN
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

        self.data = (self.data - self.mean) / self.std

        print("✔ Datos generados")

    # =========================
    # 2. CREAR FORWARD
    # =========================
    def crear_diffusion(self):

        self.diff = ForwardDiffusion(self.data, T=self.T)
        print("✔ Forward diffusion creado")

    # =========================
    # 3. ANALIZAR FORWARD
    # =========================
    def analizar_forward(self):

        self.diff.simular()
        self.diff.graficar_distribuciones()

    # =========================
    # 4. ENTRENAR MODELO
    # =========================
        # =========================
    # 4. ENTRENAR MODELO
    # =========================
    def entrenar_modelo(self, epochs=2000):

        self.model, self.loss_history = train_model(
            self.diff,
            epochs=epochs
        )

        print("✔ Modelo entrenado")

    # =========================
    # 5. BACKWARD
    # =========================
    def generar_muestras(self, n_samples=2000):

        bd = BackwardDiffusion(self.diff, self.model)
        self.trayectoria = bd.sample(n_samples=n_samples)

        print("✔ Muestras generadas")

    # =========================
    # 6. VISUALIZAR RESULTADOS
    # =========================
    def visualizar(self):

        # 🔥 DESNORMALIZAR
        original = self.data * self.std + self.mean
        generado = self.trayectoria[-1] * self.std + self.mean

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.hist(original, bins=100, density=True, alpha=0.6)
        plt.title("Distribución original (real)")

        plt.subplot(1,2,2)
        plt.hist(generado, bins=100, density=True, alpha=0.6)
        plt.title("Distribución generada")

        plt.show()

    # =========================
    # 7. EVALUACIÓN 🔥
    # =========================
    def evaluar(self):

        # 🔥 DESNORMALIZAR (AQUÍ ESTÁ EL FIX)
        original = self.data * self.std + self.mean
        generado = self.trayectoria[-1] * self.std + self.mean

        stats = {
            "media_original": np.mean(original),
            "media_generada": np.mean(generado),

            "var_original": np.var(original),
            "var_generada": np.var(generado),

            "skew_original": skew(original),
            "skew_generada": skew(generado),

            "kurt_original": kurtosis(original),
            "kurt_generada": kurtosis(generado),

            "wasserstein": wasserstein_distance(original, generado)
        }

        score = stats["wasserstein"]

        if score < 1:
            calidad = "EXCELENTE 🔥"
        elif score < 3:
            calidad = "BUENA 👍"
        elif score < 6:
            calidad = "ACEPTABLE ⚠️"
        else:
            calidad = "MALA ❌"

        txt = f"""
    === EVALUACIÓN REAL ===

    Media:      {stats['media_original']:.2f} vs {stats['media_generada']:.2f}
    Varianza:   {stats['var_original']:.2f} vs {stats['var_generada']:.2f}
    Skewness:   {stats['skew_original']:.2f} vs {stats['skew_generada']:.2f}
    Kurtosis:   {stats['kurt_original']:.2f} vs {stats['kurt_generada']:.2f}

    Distancia Wasserstein: {stats['wasserstein']:.4f}

    Calidad del modelo: {calidad}
    """
        print(txt)

        return stats

    # =========================
    # PIPELINE COMPLETO
    # =========================
    def run(
        self,
        params_generador={},
        epochs=2000,
        n_samples=2000,
        analizar=True
    ):

        self.generar_datos(**params_generador)
        self.crear_diffusion()

        if analizar:
            self.analizar_forward()

        self.entrenar_modelo(epochs=epochs)
        self.generar_muestras(n_samples=n_samples)
        self.visualizar()

        return {
            "data_original": self.data,
            "data_generada": self.trayectoria[-1],
            "trayectoria": self.trayectoria
        }
        
            # =========================
    # 📈 LOSS
    # =========================
    def fig_loss(self):

        fig, ax = plt.subplots()

        ax.plot(self.loss_history)

        ax.set_title("Evolución del Score Matching Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")

        return fig