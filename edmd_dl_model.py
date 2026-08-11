"""
EDMD-DL model, split into the model(x) -> outputs / loss_fn(outputs, x)
shape so it can be dropped into an existing generic training loop. Loss
functions live in losses.py, which imports EDMDDLOutput from here.

model.forward(x_window) does NOT return a reconstruction -- there is no
decoder. It returns the dictionary embeddings for every snapshot in the
window and the current EDMD-fit K, computed by a differentiable
closed-form regression (torch.linalg.solve), so gradients reach the
dictionary weights via the loss without any hand-rolled alternation
between solving for K and updating the dictionary.

K is never a persisted nn.Parameter. It's refit from psi(x) on every
forward call, so there's no optimizer param group for it and no weight
decay consideration -- if your loop currently sets up a separate
param group for K, that code path is simply unused for this model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Dictionary network: psi(x) -> R^N
# --------------------------------------------------------------------------
class DictionaryNet(nn.Module):
    """
    Maps a masked physical-space field to a finite dictionary of
    observables. Swap the conv body for your existing encoder trunk if
    you want architectural parity with the KAE encoder.

    include_state: concatenates the flattened masked input to the
    learned features so the dictionary always spans the identity
    observables. Needed if you want the optional linear readout
    (physical-space reconstruction) to be well-posed early in training.
    """

    def __init__(
        self,
        in_channels: int,
        grid_hw: Tuple[int, int],
        n_dict: int,
        hidden_channels: int = 64,
        include_state: bool = True,
    ):
        super().__init__()
        H, W = grid_hw
        self.include_state = include_state
        self.state_dim = in_channels * H * W

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            feat_dim = self.conv(dummy).flatten(1).shape[1]

        learned_dim = n_dict - self.state_dim if include_state else n_dict
        if learned_dim <= 0:
            raise ValueError(
                f"n_dict ({n_dict}) must exceed flattened state_dim "
                f"({self.state_dim}) when include_state=True."
            )
        self.readout = nn.Linear(feat_dim, learned_dim)
        self.n_dict = n_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) masked physical field -> psi(x): (B, n_dict)"""
        feat = self.conv(x).flatten(1)
        learned = self.readout(feat)
        if self.include_state:
            state = x.flatten(1)
            return torch.cat([state, learned], dim=1)
        return learned

def edmd_solve(
    psi_t: torch.Tensor,
    psi_tp1: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable ridge-regularized least-squares K such that
    psi(x_{t+1}) ~= K @ psi(x_t) (column-vector convention).
    psi_t, psi_tp1: (M, N), M = however many one-step pairs you pool.
    """
    psi_t = psi_t.float()
    psi_tp1 = psi_tp1.float()
    N = psi_t.shape[1]
    G = psi_t.T @ psi_t
    A = psi_t.T @ psi_tp1
    G_reg = G + ridge * torch.eye(N, device=psi_t.device, dtype=psi_t.dtype)
    K_T = torch.linalg.solve(G_reg, A)
    return K_T.T


def linear_readout_solve(
    psi: torch.Tensor,
    x_flat: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Closed-form C s.t. x_flat ~= psi @ C^T -- the EDMD-DL 'decoder'."""
    psi = psi.float()
    N = psi.shape[1]
    G = psi.T @ psi
    A = psi.T @ x_flat
    G_reg = G + ridge * torch.eye(N, device=psi.device, dtype=psi.dtype)
    # C_T = torch.linalg.solve(G_reg, A)
    G_inv = torch.linalg.inv(G_reg)
    C_T = G_inv @ A
    return C_T.T


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class EDMDDLOutput:
    psi_all: torch.Tensor          # (B, T, N)
    K: torch.Tensor                # (N, N)
    C_readout: Optional[torch.Tensor] = None  # (state_dim, N), only if requested


class EDMDDLModel(nn.Module):
    """
    Wraps DictionaryNet + the EDMD regression into a single model call.
    forward(x_window) -> EDMDDLOutput. No reconstruction is produced
    unless compute_readout=True, in which case a second closed-form
    linear solve (not a network) maps psi back to physical space.
    """

    def __init__(
        self,
        in_channels: int,
        grid_hw: Tuple[int, int],
        n_dict: int,
        hidden_channels: int = 64,
        include_state: bool = True,
        ridge: float = 1e-6,
        compute_readout: bool = False,
    ):
        super().__init__()
        self.dictionary = DictionaryNet(
            in_channels, grid_hw, n_dict, hidden_channels, include_state
        )
        self.ridge = ridge
        self.compute_readout = compute_readout

    def forward(self, x_window: torch.Tensor) -> EDMDDLOutput:
        """x_window: (B, T, C, H, W), T >= 2."""
        B, T, C, H, W = x_window.shape

        psi_all = torch.stack(
            [self.dictionary(x_window[:, t]) for t in range(T)], dim=1
        )  # (B, T, N)

        psi_t = psi_all[:, :-1].reshape(-1, psi_all.shape[-1]).float()
        psi_tp1 = psi_all[:, 1:].reshape(-1, psi_all.shape[-1]).float()
        K = edmd_solve(psi_t, psi_tp1, ridge=self.ridge)
        K = K.to(psi_all.dtype)

        C_readout = None
        if self.compute_readout:
            x_flat = x_window.reshape(B * T, -1)
            psi_flat = psi_all.reshape(B * T, -1)
            C_readout = linear_readout_solve(psi_flat, x_flat, ridge=self.ridge)

        return EDMDDLOutput(
            psi_all=psi_all, K=K, C_readout=C_readout
        )