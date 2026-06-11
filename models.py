import torch
import torch.nn as nn
import numpy as np


class KoopmanOperator(nn.Module):
    """Lusch-style structured K: block-diagonal 2×2 rotation matrices.

    Each pair of latent dims gets eigenvalue  mu * e^(±i*omega)
    where mu = sigmoid(mu_raw) ∈ (0, 1), guaranteeing stability.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even for Lusch K"
        n = latent_dim // 2
        # self.mu_raw = nn.Parameter(torch.zeros(n))          # sigmoid → (0,1)
        # self.omega  = nn.Parameter(torch.randn(n) * 0.01)  # frequency
        # randomize mu_raw so modes start at different persistence levels
        # (sigmoid spread roughly over (0.05, 0.95)) instead of all at 0.5
        self.mu_raw = nn.Parameter(torch.randn(n) * 1.5)
        # spread omegas across periods 4 months → 20 years (240 months)
        self.omega  = nn.Parameter(2 * np.pi / torch.linspace(4, 240, n))

    def _build(self):
        mu = torch.sigmoid(self.mu_raw)   # (n,)
        c  = torch.cos(self.omega)         # (n,)
        s  = torch.sin(self.omega)         # (n,)
        # blocks shape: (n, 2, 2)
        blocks = torch.stack([
            torch.stack([ mu * c, -mu * s], dim=1),
            torch.stack([ mu * s,  mu * c], dim=1),
        ], dim=1)
        return torch.block_diag(*blocks.unbind(0))  # (latent_dim, latent_dim)

    @property
    def weight(self):
        """Full K matrix — used by visualize.py for eigendecomposition."""
        return self._build()

    def forward(self, z):
        return z @ self._build().t()  # z: (B, latent_dim)


class Encoder(nn.Module):
    def __init__(self, Y_pad, X_pad, latent_dim, channels=(16, 32, 64)):
        super().__init__()
        n_pools = len(channels)
        flat_dim = channels[-1] * (Y_pad // (2 ** n_pools)) * (X_pad // (2 ** n_pools))

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.view(h.shape[0], -1))  # (B, latent_dim)


class Decoder(nn.Module):
    def __init__(self, Y, X, Y_pad, X_pad, latent_dim, channels=(64, 32, 16)):
        super().__init__()
        n_pools = len(channels)
        self.Yf = Y_pad // (2 ** n_pools)
        self.Xf = X_pad // (2 ** n_pools)
        self.Y = Y
        self.X = X
        self.ch0 = channels[0]
        flat_dim = channels[0] * self.Yf * self.Xf

        self.fc = nn.Linear(latent_dim, flat_dim)

        layers = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        layers += [
            nn.ConvTranspose2d(in_ch, 1, kernel_size=2, stride=2),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        h = self.fc(z).view(B, self.ch0, self.Yf, self.Xf)
        return self.conv(h)[:, :, :self.Y, :self.X]  # (B, 1, Y, X)


class KoopmanNet(nn.Module):
    def __init__(self, Y, X, latent_dim=128, channels=(16, 32, 64)):
        super().__init__()
        n_pools = len(channels)
        factor = 2 ** n_pools
        Y_pad = ((Y + factor - 1) // factor) * factor  # next multiple of factor
        X_pad = ((X + factor - 1) // factor) * factor
        self.pad_y = Y_pad - Y
        self.pad_x = X_pad - X

        self.encoder = Encoder(Y_pad, X_pad, latent_dim, channels)
        self.decoder = Decoder(Y, X, Y_pad, X_pad, latent_dim, tuple(reversed(channels)))
        self.K = KoopmanOperator(latent_dim)

    def encode(self, x):
        x = nn.functional.pad(x, (0, self.pad_x, 0, self.pad_y))
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def advance(self, z, steps=1):
        for _ in range(steps):
            z = self.K(z)
        return z

    def forward(self, x_seq):
        # x_seq: (B, S+1, 1, Y, X) — a short rollout window
        steps = x_seq.shape[1] - 1
        z_seq = [self.encode(x_seq[:, t]) for t in range(steps + 1)]

        x_rec = self.decode(z_seq[0])      # reconstruction of x_t0

        z_preds, x_preds = [], []
        z = z_seq[0]
        for _ in range(steps):
            z = self.K(z)                  # advance one step
            z_preds.append(z)
            x_preds.append(self.decode(z))

        return x_rec, x_preds, z_seq, z_preds
