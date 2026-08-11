"""
Loss functions for EDMD-DL training, plus the config-driven loss_type
dispatch used by train.py.
"""

import torch
import torch.nn.functional as F

from edmd_dl_model import EDMDDLOutput


# --------------------------------------------------------------------------
# EDMD-DL: one-step / rollout / reconstruction loss
# --------------------------------------------------------------------------
def edmd_dl_loss(
    x_window,
    output: EDMDDLOutput,
    rollout_steps, rollout_weight
):
    """
    output: whatever EDMDDLModel.forward returned for this batch.
    Returns (loss, logs) -- logs are detached, safe to accumulate/print.
    """
    psi_all = output.psi_all
    K = output.K
    B, T, N = psi_all.shape

    psi_t = psi_all[:, :-1].reshape(-1, N)
    psi_tp1 = psi_all[:, 1:].reshape(-1, N)
    one_step_pred = psi_t @ K.T
    one_step_loss = F.mse_loss(one_step_pred, psi_tp1)

    rollout_loss = psi_all.new_zeros(())
    if rollout_steps > 1:
        if T < rollout_steps + 1:
            raise ValueError(
                f"window length T={T} too short for rollout_steps={rollout_steps}"
            )
        z = psi_all[:, 0]
        for step in range(1, rollout_steps + 1):
            z = z @ K.T
            rollout_loss += F.mse_loss(z, psi_all[:, step])
        rollout_loss = rollout_loss / rollout_steps

    loss = one_step_loss + rollout_weight * rollout_loss

    logs = {
        "one_step_loss": one_step_loss.detach(),
        "rollout_loss": rollout_loss.detach() if rollout_steps > 1 else None,
        "K": K.detach(),
    }

    x_flat = x_window.reshape(B * T, -1)
    psi_flat = psi_all.reshape(B * T, N)
    recon = psi_flat @ output.C_readout.T
    recon_loss = F.mse_loss(recon, x_flat)
    # if recon_weight > 0:
    #     loss = loss + recon_weight * recon_loss
    logs["recon_loss"] = recon_loss.detach()

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
):
    """
    Second training objective, standalone from edmd_dl_loss: directly
    minimizes the ResDMD spectral residuals of output.K's eigenpairs,
    computed from the same one-step pairs pooled to fit K (see
    resdmd_residuals above), guarded by resdmd_compute_loss's
    condition-number penalty on Psi_X.

    Requires psi_t to have at least N = psi_all.shape[-1] one-step pairs
    (drop_last=True in the DataLoader) so svdvals(psi_t) returns N values,
    matching residuals' length.
    """
    psi_all = output.psi_all
    K = output.K
    N = psi_all.shape[-1]

    psi_t = psi_all[:, :-1].reshape(-1, N).float()
    psi_tp1 = psi_all[:, 1:].reshape(-1, N).float()

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
        x_seq, output,
        cfg.getint('LOSS', 'rollout_steps'), cfg.getfloat('LOSS', 'rollout_weight'),
    )

def _resdmd_batch_loss(cfg, x_seq, output):
    return resdmd_batch_loss(
        output,
        eps=cfg.getfloat('LOSS', 'resdmd_eps', fallback=1e-8),
        loss_threshold=cfg.getfloat('LOSS', 'resdmd_loss_threshold', fallback=10.0),
        use_cond_penalty=cfg.getboolean('LOSS', 'resdmd_use_cond_penalty', fallback=True),
        penalty_coef=cfg.getfloat('LOSS', 'resdmd_penalty_coef', fallback=1.0),
    )

def _combined_loss(cfg, x_seq, output):
    """edmd_dl_loss (one-step + rollout, logged recon) plus resdmd_weight
    times resdmd_batch_loss (spectral residual penalty), both computed off
    the same forward-pass output. Logs from both are merged; no key
    collisions since resdmd_batch_loss's keys are all resdmd_-prefixed."""
    edmd_loss, edmd_logs = _edmd_dl_loss(cfg, x_seq, output)
    resdmd_weight = cfg.getfloat('LOSS', 'resdmd_weight', fallback=1.0)
    resdmd_loss, resdmd_logs = _resdmd_batch_loss(cfg, x_seq, output)
    loss = edmd_loss + resdmd_weight * resdmd_loss
    return loss, {**edmd_logs, **resdmd_logs}

# add new loss types by adding one function above + one entry here
LOSS_FNS = {'edmd_dl': _edmd_dl_loss, 'resdmd_batch': _resdmd_batch_loss, 'combined': _combined_loss}

def get_loss_fn(name):
    return LOSS_FNS[name]
