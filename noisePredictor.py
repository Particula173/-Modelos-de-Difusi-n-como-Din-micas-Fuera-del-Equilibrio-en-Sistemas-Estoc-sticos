# noisePredictor.py

import torch
import torch.nn as nn


class NoisePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp).squeeze()



def train_model(diffusion, epochs=2000, batch_size=512):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NoisePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = diffusion.x0.to(device)

    loss_history = []

    for epoch in range(epochs):

        idx = torch.randint(0, len(data), (batch_size,))
        x0 = data[idx]

        t = torch.randint(0, diffusion.T, (batch_size,), device=device)

        xt, noise = diffusion.sample_xt(x0, t)

        t_norm = t.float() / diffusion.T

        pred = model(xt, t_norm)

        loss = ((pred - noise)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model, loss_history