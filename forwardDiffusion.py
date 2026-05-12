import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro
from matplotlib.animation import FuncAnimation


class ForwardDiffusion:

    def __init__(self, data, T=200, beta_start=1e-4, beta_end=0.02):

        # datos
        self.x0_np = data
        self.x0 = torch.tensor(data, dtype=torch.float32)

        self.T = T

        # schedule
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # análisis
        self.historial = []

    # =========================
    # SAMPLE xt (para entrenamiento)
    # =========================
    def sample_xt(self, x0, t):

        noise = torch.randn_like(x0)

        xt = (
            torch.sqrt(self.alpha_bar[t]) * x0 +
            torch.sqrt(1 - self.alpha_bar[t]) * noise
        )

        return xt, noise

    # =========================
    # SIMULAR (tu versión original)
    # =========================
    def simular(self):

        self.historial = []

        for t in range(self.T):

            xt = (
                np.sqrt(self.alpha_bar[t].item()) * self.x0_np +
                np.sqrt(1 - self.alpha_bar[t].item()) *
                np.random.normal(0, 1, size=self.x0_np.shape)
            )

            sample = xt if len(xt) < 500 else np.random.choice(xt, 500, replace=False)

            p_value = shapiro(sample)[1]

            stats = {
                "t": t,
                "media": np.mean(xt),
                "varianza": np.var(xt),
                "skewness": skew(xt),
                "kurtosis": kurtosis(xt),
                "normalidad_p": p_value
            }

            self.historial.append((xt, stats))

        return self.historial

    # =========================
    # BACKWARD (con modelo)
    # =========================
    def sample_backward(self, model, n_samples=1000):

        device = next(model.parameters()).device

        x = torch.randn(n_samples).to(device)

        trayectoria = [x.detach().cpu().numpy()]

        for t in reversed(range(self.T)):

            t_tensor = torch.full((n_samples,), t, device=device).float() / self.T

            with torch.no_grad():
                noise_pred = model(x, t_tensor)

            coef1 = 1 / torch.sqrt(self.alpha[t])
            coef2 = (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])

            mean = coef1 * (x - coef2 * noise_pred)

            if t > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(self.beta[t])
                x = mean + sigma * z
            else:
                x = mean

            trayectoria.append(x.detach().cpu().numpy())

        return trayectoria

    # =========================
    # GRAFICAR DISTRIBUCIONES
    # =========================
    def fig_distribuciones(self, n_puntos=6):

        fig, ax = plt.subplots()

        pasos = np.linspace(0, self.T - 1, n_puntos, dtype=int)

        for t in pasos:
            xt, _ = self.historial[t]
            ax.hist(xt, bins=80, density=True, alpha=0.4, label=f"t={t}")

        ax.legend()
        ax.set_title("Forward Diffusion")

        return fig

    # =========================
    # NORMALIDAD
    # =========================
    def fig_normalidad(self):

        t_vals = [s[1]["t"] for s in self.historial]
        p_vals = [s[1]["normalidad_p"] for s in self.historial]

        fig, ax = plt.subplots()
        ax.plot(t_vals, p_vals)

        ax.set_yscale("log")
        ax.axhline(0.05, linestyle="--")

        ax.set_title("Test de normalidad")

        return fig
    
    def fig_skew_kurtosis(self):

        t_vals = [s[1]["t"] for s in self.historial]
        skew_vals = [s[1]["skewness"] for s in self.historial]
        kurt_vals = [s[1]["kurtosis"] for s in self.historial]

        fig, ax = plt.subplots()

        ax.plot(t_vals, skew_vals, label="skewness")
        ax.plot(t_vals, kurt_vals, label="kurtosis")

        ax.axhline(0, linestyle="--")

        ax.legend()
        ax.set_title("Convergencia a gaussiana")

        return fig

    # =========================
    # VISUALIZAR BACKWARD
    # =========================
    def graficar_backward(self, trayectoria, pasos=6):

        plt.figure(figsize=(10,6))

        idxs = np.linspace(0, len(trayectoria)-1, pasos, dtype=int)

        for i in idxs:
            plt.hist(trayectoria[i], bins=80, density=True, alpha=0.4, label=f"step {i}")

        plt.legend()
        plt.title("Backward Diffusion")
        plt.show()
    
    # =========================
    # 🎥 ANIMACIÓN FORWARD
    # =========================
    def animacion_forward(self):

        fig, ax = plt.subplots()

        # 🔥 reducir número de frames
        step = max(1, self.T // 5)

        frames = range(
            0,
            len(self.historial),
            step
        )

        def update(frame):

            ax.clear()

            xt, _ = self.historial[frame]

            ax.hist(
                xt,
                bins=80,
                density=True,
                alpha=0.7
            )

            ax.set_title(f"Forward Diffusion t={frame}")

        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=80
        )

        anim.save(
            "forward.gif",
            writer="pillow",
            fps=10
        )

        plt.close(fig)

        return "forward.gif"
        
    # =========================
    # ⚡ FUERZA TEÓRICA FORWARD
    # =========================
    def fig_fuerza_forward(self):

        fuerzas = []
        tiempos = []

        for t in range(self.T):

            xt, _ = self.historial[t]

            beta_t = self.beta[t].item()

            # 🔥 drift teórico
            fuerza = -0.5 * beta_t * xt

            fuerzas.append(
                np.mean(np.abs(fuerza))
            )

            tiempos.append(t)

        fig, ax = plt.subplots()

        ax.plot(tiempos, fuerzas)

        ax.set_title("Fuerza teórica forward")
        ax.set_xlabel("t")
        ax.set_ylabel("|f(x,t)|")

        return fig