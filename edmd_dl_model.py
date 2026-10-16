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
        state_channels: Optional[int] = None,
        depth: int = 3
    ):
        super().__init__()
        H, W = grid_hw
        self.include_state = include_state
        self.state_channels = in_channels if state_channels is None else state_channels
        self.state_dim = self.state_channels * H * W


        layers = []
        for i in range(depth):
            layers += [nn.Conv2d(in_channels if i == 0 else hidden_channels, hidden_channels,
                                    3, padding=1, stride=1 if i == 0 else 2),
                        nn.GELU()]
        self.conv = nn.Sequential(*layers)
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
        """x: (B, in_channels, H, W) -- in_channels = C * lookback, most recent frame
        last -> psi(x): (B, n_dict)"""
        feat = self.conv(x).flatten(1)
        learned = self.readout(feat)
        if self.include_state:
            state = x[:, -self.state_channels:].flatten(1)  # most recent frame only
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
    # G_inv = torch.linalg.inv(G_reg)
    # K_T = G_inv @ A
    return K_T.T


def linear_readout_solve(
    psi: torch.Tensor,
    x_flat: torch.Tensor,
    ridge: float = 1e-6,
    chunk_size: int = 100_000,
) -> torch.Tensor:
    """Closed-form C s.t. x_flat ~= psi @ C^T -- the EDMD-DL 'decoder'.

    torch.linalg.solve(G_reg, A) hits a reproducible CUBLAS_STATUS_INTERNAL_ERROR
    in cublasSgetrsBatched when N (dict size) is small (<=8, confirmed via an
    isolated repro against random data) and A's column count (state_dim, in the
    millions) is huge. Solving the identical system in column-chunks avoids that
    pathological (tiny N, huge nrhs) shape without changing the math -- each
    chunk is the same torch.linalg.solve call, just on a narrower slice of A.
    """
    psi = psi.float()
    N = psi.shape[1]
    G = psi.T @ psi
    A = psi.T @ x_flat
    G_reg = G + ridge * torch.eye(N, device=psi.device, dtype=psi.dtype)
    if A.shape[1] > chunk_size:
        C_T = torch.cat(
            [torch.linalg.solve(G_reg, A[:, i:i + chunk_size]) for i in range(0, A.shape[1], chunk_size)],
            dim=1,
        )
    else:
        C_T = torch.linalg.solve(G_reg, A)
    # G_inv = torch.linalg.inv(G_reg)
    # C_T = G_inv @ A
    return C_T.T


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class EDMDDLOutput:
    psi_all: torch.Tensor          # (B, T, N)
    K: torch.Tensor                # (N, N)
    x_target: torch.Tensor         # (B, T, C, H, W) -- physical frame each psi_all[:,t] represents
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
        lookback: int = 1,
        depth: int = 3,
        micro_batch: int = 0,
    ):
        super().__init__()
        self.lookback = lookback
        # windows per dictionary call in lift(); 0 = whole batch at once. Only
        # changes peak activation memory, never the result.
        self.micro_batch = micro_batch
        self.dictionary = DictionaryNet(
            in_channels * lookback, grid_hw, n_dict, hidden_channels, include_state,
            state_channels=in_channels, depth=depth
        )
        self.ridge = ridge
        self.compute_readout = compute_readout

    def lift(self, x_window: torch.Tensor) -> torch.Tensor:
        """x_window: (B, L, C, H, W) -> psi_all: (B, T, N), T = L - lookback + 1.
        Runs the dictionary micro_batch windows at a time (each window's psi
        depends only on that window, so chunking doesn't change the result)."""
        chunk = self.micro_batch or x_window.shape[0]
        return torch.cat([self._lift(x_window[i:i + chunk]) for i in range(0, x_window.shape[0], chunk)])

    def _lift(self, x_window: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x_window.shape
        T = L - self.lookback + 1  # = rollout + 1
        return torch.stack(
            [self.dictionary(x_window[:, t:t + self.lookback].reshape(B, -1, H, W))
             for t in range(T)], dim=1
        )  # (B, T, N)

    def forward(self, x_window: torch.Tensor) -> EDMDDLOutput:
        """x_window: (B, L, C, H, W), L = lookback + rollout."""
        return self.fit(self.lift(x_window), x_window)

    def fit(self, psi_all: torch.Tensor, x_window: torch.Tensor) -> EDMDDLOutput:
        """K and C_readout solved over the whole batch from already-lifted
        psi_all -- the only step that couples windows. See train_step() in
        train.py for why lifting and fitting are separable."""
        B, T = psi_all.shape[:2]
        x_target = x_window[:, self.lookback - 1: self.lookback - 1 + T]  # (B, T, C, H, W)

        psi_t = psi_all[:, :-1].reshape(-1, psi_all.shape[-1]).float()
        psi_tp1 = psi_all[:, 1:].reshape(-1, psi_all.shape[-1]).float()
        K = edmd_solve(psi_t, psi_tp1, ridge=self.ridge)
        K = K.to(psi_all.dtype)

        C_readout = None
        if self.compute_readout:
            x_flat = x_target.reshape(B * T, -1)
            psi_flat = psi_all.reshape(B * T, -1)
            C_readout = linear_readout_solve(psi_flat, x_flat, ridge=self.ridge)

        return EDMDDLOutput(
            psi_all=psi_all, K=K, x_target=x_target, C_readout=C_readout
        )