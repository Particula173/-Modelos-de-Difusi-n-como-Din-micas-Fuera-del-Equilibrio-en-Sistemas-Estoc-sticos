import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class BackwardDiffusion:

    def __init__(self, diffusion, model):
        """
        diffusion: instancia de tu clase Diffusion1D (tiene alpha, beta, alpha_bar)
        model: NoisePredictor ya entrenado
        """
        self.diff = diffusion
        self.model = model

    def sample(self, n_samples=1000):
        """
        Genera datos desde ruido puro usando el modelo aprendido
        """

        device = next(self.model.parameters()).device

        # 🔥 empezamos desde ruido puro
        x = torch.randn(n_samples).to(device)

        trayectoria = [x.detach().cpu().numpy()]

        # 🔁 backward: T → 0
        for t in reversed(range(self.diff.T)):

            # normalizar tiempo
            t_tensor = torch.full((n_samples,), t, device=device).float() / self.diff.T

            # 🔥 el modelo predice el ruido
            with torch.no_grad():
                noise_pred = self.model(x, t_tensor)

            alpha = self.diff.alpha[t]
            alpha_bar = self.diff.alpha_bar[t]
            beta = self.diff.beta[t]

            # 📌 ecuación clave del backward
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            mean = coef1 * (x - coef2 * noise_pred)

            # 🔥 añadir ruido excepto en t=0
            if t > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * z
            else:
                x = mean

            trayectoria.append(x.detach().cpu().numpy())

        return trayectoria
    

    def fig_backward(self, trayectoria, pasos=6):

        fig, ax = plt.subplots()

        idxs = np.linspace(0, len(trayectoria)-1, pasos, dtype=int)

        for i in idxs:
            ax.hist(trayectoria[i], bins=80, density=True, alpha=0.4, label=f"step {i}")

        ax.legend()
        ax.set_title("Backward Diffusion")

        return fig
    
    # =========================
    # ⚡ FUERZA EFECTIVA
    # =========================
    def fig_fuerza_backward(self, n_samples=4000):

        device = next(self.model.parameters()).device

        x = torch.randn(n_samples).to(device)

        fuerzas = []
        tiempos = []

        for t in reversed(range(self.diff.T)):

            t_tensor = (
                torch.full(
                    (n_samples,),
                    t,
                    device=device
                ).float() / self.diff.T
            )

            with torch.no_grad():
                noise_pred = self.model(x, t_tensor)

            beta_t = self.diff.beta[t]
            alpha_bar_t = self.diff.alpha_bar[t]

            # 🔥 drift forward
            drift = -0.5 * beta_t * x

            # 🔥 score aproximado
            score = (
                -noise_pred /
                torch.sqrt(1 - alpha_bar_t)
            )

            # 🔥 fuerza reconstructiva
            score_force = beta_t * score

            # 🔥 fuerza total backward
            fuerza_total = drift - score_force

            fuerzas.append(
                torch.mean(
                    torch.abs(fuerza_total)
                ).item()
            )

            tiempos.append(t)

        fig, ax = plt.subplots()

        ax.plot(tiempos, fuerzas)

        ax.set_title("Fuerza teórica backward")
        ax.set_xlabel("t")
        ax.set_ylabel("|f_backward|")

        return fig
    
    # =========================
    # 🎥 ANIMACIÓN BACKWARD
    # =========================
    def animacion_backward(self, trayectoria):

        fig, ax = plt.subplots()

        # step = max(1, len(trayectoria) // 50)
        step = 10
        frames = range(
            0,
            len(trayectoria),
            step
        )

        def update(frame):

            ax.clear()

            ax.hist(
                trayectoria[frame],
                bins=80,
                density=True,
                alpha=0.7
            )

            ax.set_title(f"Backward step={frame}")

        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=80
        )

        anim.save(
            "backward.gif",
            writer="pillow",
            fps=20
        )

        plt.close(fig)

        return "backward.gif"