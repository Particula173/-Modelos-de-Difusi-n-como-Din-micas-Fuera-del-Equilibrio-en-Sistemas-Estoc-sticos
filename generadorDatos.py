import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis

class GeneradorDatos:

    def __init__(self, n_samples=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_samples = n_samples
        self.data = None
        self.config = {}

    # =========================
    # BASE: MIXTURE OF GAUSSIANS
    # =========================
    def _generar_mog(self, n_picos):

        pesos = np.random.dirichlet(np.ones(n_picos))

        # 🔥 mejor separación real
        mus = np.random.uniform(-500, 500, n_picos)
        mus = np.sort(mus)

        # 🔥 evita picos degenerados
        sigmas = np.random.uniform(0.5, 3.0, n_picos)

        data = []
        componentes = []

        for i in range(n_picos):
            n_i = int(self.n_samples * pesos[i])
            samples = np.random.normal(mus[i], sigmas[i], n_i)

            data.append(samples)

            componentes.append({
                "peso": pesos[i],
                "mu": mus[i],
                "sigma": sigmas[i],
                "n": n_i
            })

        return np.concatenate(data), componentes

    # =========================
    # DEFORMACIONES (ESTABLE)
    # =========================
    def _deformar(self, x, complejidad):

        funciones = [
            lambda x: x + np.random.uniform(0.05, 0.2) * np.sin(x),
            lambda x: x + np.random.uniform(0.05, 0.15) * np.tanh(x),

            # 🔥 controlamos explosión
            lambda x: x + np.random.uniform(0.001, 0.005) * (x**2) * np.sign(x),

            lambda x: x + np.random.normal(0, np.random.uniform(0.05, 0.3), size=len(x)),
        ]

        nombres = [
            "sin(x)",
            "tanh(x)",
            "x^2 * sign(x)",
            "ruido gaussiano"
        ]

        # 🔥 control robusto de complejidad
        high = min(6, complejidad + 1)
        complejidad_real = np.random.randint(1, max(2, high))

        # 🔥 límite de seguridad
        complejidad_real = min(complejidad_real, 4)

        idx = np.random.choice(len(funciones), complejidad_real, replace=True)

        deformaciones = []

        for i in idx:
            if np.random.rand() < 0.7:
                # 🔥 mezcla para NO destruir estructura
                x = 0.8 * x + 0.2 * funciones[i](x)
                deformaciones.append(nombres[i])

        # 🔥 evitar colapso
        x = x + np.random.normal(0, 0.1 * np.std(x), size=len(x))

        # 🔥 evitar explosión numérica
        x = np.clip(x, -1e4, 1e4)

        # 🔥 estabilización final (MUY IMPORTANTE)
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        return x, deformaciones

    # =========================
    # GENERADOR PRINCIPAL
    # =========================
    def generar(
        self,
        min_picos=3,
        max_picos=8,
        complejidad=6,
        target_mean=None
    ):

        n_picos = np.random.randint(min_picos, max_picos + 1)

        x, componentes = self._generar_mog(n_picos)

        x, deformaciones = self._deformar(x, complejidad)

        # =========================
        # REALISMO GLOBAL
        # =========================

        escala = np.random.uniform(1, 10)
        x = x * escala

        shift = np.random.uniform(-200, 200)
        x = x + shift

        # outliers
        if np.random.rand() < 0.5:
            n_out = int(0.01 * len(x))
            outliers = np.random.normal(
                loc=np.mean(x),
                scale=5 * np.std(x),
                size=n_out
            )
            x[:n_out] = outliers

        # asimetría
        if np.random.rand() < 0.5:
            x = x + 0.05 * (x**2) * np.sign(x)

        # target mean
        if target_mean is not None:
            x = x + (target_mean - np.mean(x))

        self.data = x

        self.config = {
            "n_picos": n_picos,
            "componentes": componentes,
            "deformaciones": deformaciones,
            "complejidad_real": len(deformaciones),
            "media": np.mean(x),
            "varianza": np.var(x)
        }

        return x

    # =========================
    # RESUMEN
    # =========================
    def resumen(self):

        txt = ["=== RESUMEN ===\n"]

        txt.append(f"Picos generados: {self.config['n_picos']}")
        txt.append(f"Complejidad real: {self.config['complejidad_real']}\n")

        for i, c in enumerate(self.config["componentes"]):
            txt.append(
                f"Pico {i+1}: peso={c['peso']:.3f}, mu={c['mu']:.2f}, sigma={c['sigma']:.2f}"
            )

        txt.append("\nDeformaciones aplicadas:")
        for d in self.config["deformaciones"]:
            txt.append(f" - {d}")

        txt.append(f"\nMedia: {self.config['media']:.2f}")
        txt.append(f"Varianza: {self.config['varianza']:.2f}")

        return "\n".join(txt)

    # =========================
    # ESTADÍSTICAS
    # =========================
    def estadisticas(self):

        return {
            "media": np.mean(self.data),
            "varianza": np.var(self.data),
            "skewness": skew(self.data),
            "kurtosis": kurtosis(self.data)
        }

    # =========================
    # GRAFICAR
    # =========================
    def graficar(self):

        p1, p99 = np.percentile(self.data, [1, 99])
        data_plot = self.data[(self.data > p1) & (self.data < p99)]

        plt.figure(figsize=(10,6))

        plt.hist(data_plot, bins=120, density=True, alpha=0.6)

        kde = gaussian_kde(data_plot)
        kde.set_bandwidth(bw_method=0.05)

        x = np.linspace(min(data_plot), max(data_plot), 2000)
        plt.plot(x, kde(x), linewidth=2)

        plt.title("Distribución compleja (MoG + deformaciones)")
        plt.xlabel("x")
        plt.ylabel("densidad")
        plt.show()
        
    def fig_datos(self):

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import gaussian_kde

        p1, p99 = np.percentile(self.data, [1, 99])
        data_plot = self.data[(self.data > p1) & (self.data < p99)]

        fig, ax = plt.subplots()

        ax.hist(data_plot, bins=120, density=True, alpha=0.6)

        kde = gaussian_kde(data_plot)
        x = np.linspace(min(data_plot), max(data_plot), 1000)

        ax.plot(x, kde(x), linewidth=2)

        ax.set_title("Distribución generada")

        return fig