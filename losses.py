"""
Loss functions for EDMD-DL training, plus the config-driven loss_type
dispatch used by train.py.
"""

import ast

import torch
import torch.nn.functional as F

from edmd_dl_model import EDMDDLOutput, edmd_solve


# --------------------------------------------------------------------------
# EDMD-DL: one-shot multi-lag prediction + autoencoder-style reconstruction
# --------------------------------------------------------------------------
def edmd_dl_loss(
    output: EDMDDLOutput,
    lags, ridge, psi_weight=1.0, recon_weight=1.0, forecast_weight=1.0,
):
    """
    output: whatever EDMDDLModel.forward returned for this batch. Its T
    (window length) must exceed max(lags).

    Three terms, all sharing the same dictionary/C_readout:
      - psi_loss: one-shot lag-N Koopman prediction error in dictionary
        space, averaged over `lags` (see forecast_loss for how each
        one-shot operator is fit).
      - recon_loss: plain autoencoder-style reconstruction -- psi_all
        decoded through C_readout back to physical space, compared to the
        *same* frame it was encoded from (every frame in the window, not
        a prediction). This is what makes C_readout/psi capable of
        representing the input state at all.
      - forecast_loss: physical-space error of each one-shot lag's
        *predicted* frame (psi_0 advanced by K_lag, then decoded),
        compared against the true frame `lag` steps out.

    Returns (loss, logs) -- logs are detached, safe to accumulate/print.
    """
    psi_all = output.psi_all
    C_readout = output.C_readout
    if C_readout is None:
        raise ValueError("edmd_dl_loss needs C_readout -- set MODEL.compute_readout = True")
    B, T, N = psi_all.shape

    max_lag = max(lags)
    if T <= max_lag:
        raise ValueError(f"window length T={T} too short for lags={lags} (need T > {max_lag})")

    x_flat = output.x_target.reshape(B, T, -1)
    psi_0 = psi_all[:, 0]

    # -- autoencoder-style reconstruction: every frame from its own psi --
    recon_pred = psi_all.reshape(B * T, N) @ C_readout.T
    recon_loss = F.mse_loss(recon_pred, x_flat.reshape(B * T, -1))

    # -- one-shot multi-lag prediction, in psi space and (decoded) state space --
    psi_loss = psi_all.new_zeros(())
    forecast_loss = psi_all.new_zeros(())
    logs = {}
    for lag in lags:
        psi_t = psi_all[:, :-lag].reshape(-1, N)
        psi_tp1 = psi_all[:, lag:].reshape(-1, N)
        K_lag = edmd_solve(psi_t, psi_tp1, ridge=ridge)  # one-shot lag-N fit

        z = psi_0 @ K_lag.T
        psi_lag_loss = F.mse_loss(z, psi_all[:, lag])
        psi_loss = psi_loss + psi_lag_loss

        x_pred = z @ C_readout.T
        forecast_lag_loss = F.mse_loss(x_pred, x_flat[:, lag])
        forecast_loss = forecast_loss + forecast_lag_loss

        logs[f"psi_loss_lag{lag}"] = psi_lag_loss.detach()
        logs[f"forecast_loss_lag{lag}"] = forecast_lag_loss.detach()

    psi_loss = psi_loss / len(lags)
    forecast_loss = forecast_loss / len(lags)

    loss = psi_weight * psi_loss + recon_weight * recon_loss + forecast_weight * forecast_loss

    logs["psi_loss"] = psi_loss.detach()
    logs["recon_loss"] = recon_loss.detach()
    logs["forecast_loss"] = forecast_loss.detach()
    logs["K"] = output.K.detach()

    return loss, logs


# --------------------------------------------------------------------------
# ResDMD: residual-based filtering of spurious EDMD eigenpairs
#
# K's eigendecomposition (K v_j = lambda_j v_j) is exact by construction --
# that tells you nothing about whether (lambda_j, v_j) is a real feature of
# the underlying dynamics vs. an artifact of fitting a finite dictionary on
# finite data. ResDMD (Colbrook & Townsend) answers that by going back to
# the *data* the EDMD regression was fit from: for the Koopman EIGENFUNCTION
# candidate g(x) = xi_j . psi(x) associated with lambda_j (xi_j is the
# right eigenvector of K.T, i.e. of the pre-transpose G^-1 A -- NOT of K
# itself; K's right eigenvectors v_j are the DMD *modes* used to decode
# spatial patterns, a different vector from the same eigenvalue), the
# residual measures how well g(x_{t+1}) ~= lambda_j g(x_t) actually holds
# across the training snapshots:
#
#   residual(lambda, xi)^2 = xi^H (L - conj(lambda) A - lambda A^T
#                                    + |lambda|^2 G) xi   /   (xi^H G xi)
#
# with G = psi_t^T psi_t, A = psi_t^T psi_tp1, L = psi_tp1^T psi_tp1 --
# the same Gram statistics already accumulated to fit K, plus L. Small
# residual -> lambda_j is trustworthy; large residual -> likely spurious.
# --------------------------------------------------------------------------
def gram_L(psi_tp1: torch.Tensor) -> torch.Tensor:
    """L = psi_tp1^T @ psi_tp1 -- the forward-snapshot Gram matrix ResDMD
    needs alongside G/A. Accumulate it the same way G/A are accumulated."""
    psi_tp1 = psi_tp1.float()
    return psi_tp1.T @ psi_tp1


def resdmd_residuals(
    K: torch.Tensor,
    G: torch.Tensor,
    A: torch.Tensor,
    L: torch.Tensor,
    eps: float = 1e-8,
):
    """
    K, G, A, L: (N, N), real, from the same fitted operator (K = solve(G,A).T).

    Returns (eigenvalues, mode_eigenvectors, residuals), all length/width N,
    aligned by index -- mode_eigenvectors[:, j] is K's right eigenvector for
    eigenvalues[j] (what koopman_modes() decodes into a spatial pattern),
    and residuals[j] is the ResDMD residual for that same eigenvalue's
    eigenfunction coefficient vector.
    """
    K = K.to(torch.complex64)
    eigenvalues, mode_vecs = torch.linalg.eig(K)          # right eigvecs of K -> spatial modes
    func_eigenvalues, func_vecs = torch.linalg.eig(K.T)   # right eigvecs of K.T -> eigenfunction coeffs

    # Both calls share the same eigenvalue set in exact arithmetic, but
    # torch.linalg.eig gives no ordering guarantee between separate calls
    # (even on K vs K.T of the same matrix) -- match each mode eigenvalue
    # to its nearest eigenfunction eigenvalue so xi_j lines up with v_j.
    dist = (eigenvalues.unsqueeze(1) - func_eigenvalues.unsqueeze(0)).abs()
    match = dist.argmin(dim=1)
    xi = func_vecs[:, match]                               # (N, N), reordered to align with `eigenvalues`

    G = G.to(torch.complex64)
    A = A.to(torch.complex64)
    L = L.to(torch.complex64)
    lam = eigenvalues

    Gxi = G @ xi
    Axi = A @ xi
    ATxi = A.T @ xi
    Lxi = L @ xi

    num = (xi.conj() * (Lxi - lam.conj() * Axi - lam * ATxi + (lam.abs() ** 2) * Gxi)).sum(dim=0)
    den = (xi.conj() * Gxi).sum(dim=0)
    residual_sq = (num / (den + eps)).real.clamp(min=0)
    residuals = residual_sq.sqrt()

    return eigenvalues, mode_vecs, residuals


def resdmd_compute_loss(
    residuals: torch.Tensor,
    singvals: torch.Tensor,
    eps: float = 1e-8,
    loss_threshold: float = 10.0,
    use_cond_penalty: bool = True,
    penalty_coef: float = 1.0,
) -> torch.Tensor:
    """
    Mean squared ResDMD residual, plus a penalty once psi_t's condition
    number (log ratio of largest to smallest singular value) exceeds
    loss_threshold. Residuals alone can be driven near zero by collapsing
    the dictionary onto a rank-deficient span, so the penalty keeps Psi_X
    well-conditioned instead of just chasing small residuals.

    residuals: (N,) from resdmd_residuals.
    singvals: (N,) descending, e.g. torch.linalg.svdvals(psi_t) -- singvals[0]
    is the largest, singvals[-1] the smallest.
    """
    N = residuals.shape[0]

    log_kappa = torch.log(singvals[0] + eps) - torch.log(singvals[-1] + eps)
    log_kappa_thresh = torch.log(torch.tensor(loss_threshold, device=singvals.device))

    cond_penalty = torch.relu(log_kappa - log_kappa_thresh) if use_cond_penalty else torch.zeros_like(log_kappa)

    return (1 / N) * torch.sum(residuals * residuals) + penalty_coef * cond_penalty

def resdmd_batch_loss(
    output: EDMDDLOutput,
    eps: float = 1e-8,
    loss_threshold: float = 10.0,
    use_cond_penalty: bool = True,
    penalty_coef: float = 1.0,
    lag: int = 1,
):
    """
    Second training objective, standalone from edmd_dl_loss: directly
    minimizes the ResDMD spectral residuals of output.K's eigenpairs,
    computed from `lag`-step pairs pooled to fit K (see resdmd_residuals
    above), guarded by resdmd_compute_loss's condition-number penalty on
    Psi_X. lag=1 (default) is the original one-step-ahead operator; lag>1
    fits K as a lag-step transition operator instead, e.g. lag=6 on
    monthly data targets 6-month dynamics directly rather than deriving
    them by iterating a 1-step K.

    Requires psi_t to have at least N = psi_all.shape[-1] pairs
    (drop_last=True in the DataLoader) so svdvals(psi_t) returns N values,
    matching residuals' length, and the window length T must exceed lag.
    """
    psi_all = output.psi_all
    K = output.K
    N = psi_all.shape[-1]
    T = psi_all.shape[1]

    if T <= lag:
        raise ValueError(f"window length T={T} too short for lag={lag}")

    psi_t = psi_all[:, :-lag].reshape(-1, N).float()
    psi_tp1 = psi_all[:, lag:].reshape(-1, N).float()

    G = psi_t.T @ psi_t
    A = psi_t.T @ psi_tp1
    L = gram_L(psi_tp1)

    _, _, residuals = resdmd_residuals(K, G, A, L, eps=eps)
    singvals = torch.linalg.svdvals(psi_t)

    loss = resdmd_compute_loss(
        residuals, singvals,
        eps=eps,
        loss_threshold=loss_threshold,
        use_cond_penalty=use_cond_penalty,
        penalty_coef=penalty_coef,
    )

    logs = {
        "resdmd_loss": loss.detach(),
        "resdmd_residual_mean": residuals.mean().detach(),
        "resdmd_residual_max": residuals.max().detach(),
        "resdmd_log_cond": (torch.log(singvals[0] + eps) - torch.log(singvals[-1] + eps)).detach(),
    }

    return loss, logs


# --------------------------------------------------------------------------
# config.ini [LOSS] loss_type -> loss_fn(cfg, x_seq, output) -> (loss, logs)
# --------------------------------------------------------------------------
def _edmd_dl_loss(cfg, x_seq, output):
    return edmd_dl_loss(
        output,
        ast.literal_eval(cfg.get('LOSS', 'oneshot_lags', fallback='[1]')),
        cfg.getfloat('MODEL', 'train_ridge'),
        cfg.getfloat('LOSS', 'psi_weight', fallback=1.0),
        cfg.getfloat('LOSS', 'recon_weight', fallback=1.0),
        cfg.getfloat('LOSS', 'forecast_weight', fallback=1.0),
    )

def _resdmd_batch_loss(cfg, x_seq, output):
    return resdmd_batch_loss(
        output,
        eps=cfg.getfloat('LOSS', 'resdmd_eps', fallback=1e-8),
        loss_threshold=cfg.getfloat('LOSS', 'resdmd_loss_threshold', fallback=10.0),
        use_cond_penalty=cfg.getboolean('LOSS', 'resdmd_use_cond_penalty', fallback=True),
        penalty_coef=cfg.getfloat('LOSS', 'resdmd_penalty_coef', fallback=1.0),
        lag=cfg.getint('LOSS', 'resdmd_lag', fallback=1),
    )

def _combined_loss(cfg, x_seq, output):
    """edmd_dl_loss (one-shot multi-lag prediction + reconstruction) plus
    resdmd_weight times resdmd_batch_loss (spectral residual penalty), both
    computed off the same forward-pass output. Logs from both are merged;
    no key collisions since resdmd_batch_loss's keys are all
    resdmd_-prefixed."""
    edmd_loss, edmd_logs = _edmd_dl_loss(cfg, x_seq, output)
    resdmd_weight = cfg.getfloat('LOSS', 'resdmd_weight', fallback=1.0)
    resdmd_loss, resdmd_logs = _resdmd_batch_loss(cfg, x_seq, output)
    loss = edmd_loss + resdmd_weight * resdmd_loss
    return loss, {**edmd_logs, **resdmd_logs}

# add new loss types by adding one function above + one entry here
LOSS_FNS = {'edmd_dl': _edmd_dl_loss, 'resdmd_batch': _resdmd_batch_loss, 'combined': _combined_loss}

def get_loss_fn(name):
    return LOSS_FNS[name]
