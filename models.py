import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, Y, X, latent_dim, channels=(16, 32, 64)):
        super().__init__()
        n_pools = len(channels)
        flat_dim = channels[-1] * (Y // (2 ** n_pools)) * (X // (2 ** n_pools))

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
        # x: (B, 1, Y, X)
        h = self.conv(x)
        return self.fc(h.view(h.shape[0], -1))  # (B, latent_dim)


class Decoder(nn.Module):
    def __init__(self, Y, X, latent_dim, channels=(64, 32, 16)):
        super().__init__()
        n_pools = len(channels)
        self.Yf = Y // (2 ** n_pools)
        self.Xf = X // (2 ** n_pools)
        self.Y = Y
        self.X = X
        flat_dim = channels[0] * self.Yf * self.Xf

        self.fc = nn.Linear(latent_dim, flat_dim)
        self.ch0 = channels[0]

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
        # z: (B, latent_dim)
        B = z.shape[0]
        h = self.fc(z).view(B, self.ch0, self.Yf, self.Xf)
        return self.conv(h)[:, :, :self.Y, :self.X]  # (B, 1, Y, X)


class KoopmanNet(nn.Module):
    def __init__(self, Y, X, latent_dim=128, channels=(16, 32, 64)):
        super().__init__()
        self.encoder = Encoder(Y, X, latent_dim, channels)
        self.decoder = Decoder(Y, X, latent_dim, tuple(reversed(channels)))
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def advance(self, z, steps=1):
        for _ in range(steps):
            z = self.K(z)
        return z

    def forward(self, x_t0, x_t1):
        z_t0  = self.encode(x_t0)
        z_t1  = self.encode(x_t1)

        x_rec  = self.decode(z_t0)         # reconstruction
        z_pred = self.advance(z_t0)        # K z_t0
        x_pred = self.decode(z_pred)       # one-step prediction

        return x_rec, x_pred, z_t0, z_t1, z_pred
