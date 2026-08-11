"""
ResDMD / EDMD-DL diagnostic plots.

The dictionary (psi) network is the only thing EDMD-DL persists in its
checkpoint -- the Koopman operator K and the physical-space readout
C_readout are refit in closed form from data on every forward call (see
edmd_dl_model.py's module docstring) and are NOT saved. So before plotting,
this script fits one *global* K and C_readout by accumulating the (tiny,
N x N) Gram statistics over many batches of the training split, rather than
solving from a single batch like training does. That gives one fixed,
representative operator to eigendecompose and visualize.

NOTE: N_DICT / HIDDEN / INCLUDE_STATE / TRAIN_RIDGE below are constructor args
that train.py also reads from config.ini's [MODEL] section (not saved in the
checkpoint itself), so they stay in sync automatically as long as this script
is run against the same config.ini the run was trained with.
"""
import os
import sys
import ast
import configparser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from all_data import load_fields, remove_climatology, normalize, split, ORASDataset, smooth
from edmd_dl_model import EDMDDLModel
from losses import resdmd_residuals


def try_cast(value):
    for cast in (int, float, ast.literal_eval):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


if len(sys.argv) == 3 and sys.argv[1] == '-f':
    config_path = sys.argv[2]
else:
    config_path = 'config.ini'

cfg = configparser.ConfigParser()
cfg.read(config_path)

ZARR_PATH     = cfg['DATA']['zarr_path']
VARS          = try_cast(cfg['DATA']['vars'])
START         = cfg['DATA'].get('start', None)
END           = cfg['DATA'].get('end', None)
SMOOTH_WINDOW = cfg.getint('DATA', 'smooth_window')
HIDDEN        = cfg.getint('MODEL', 'latent_dim')      # DictionaryNet conv width (train.py's `latent_dim` arg)
BATCH_SIZE    = cfg.getint('TRAINING', 'batch_size')
ROLLOUT_STEPS = cfg.getint('TRAINING', 'rollout_steps')
out           = cfg['OUTPUT']['dir']
OUT_DIR       = f'{out}/figs'
EVAL_DIR      = f'{out}/eval'
MODEL_PATH    = f'{out}/runs/best_model.pt'
N_MODES       = min(9, cfg.getint('MODEL', 'n_dict'))
N_SAMPLES     = 4

N_DICT        = cfg.getint('MODEL', 'n_dict')
INCLUDE_STATE = cfg.getboolean('MODEL', 'include_state')
TRAIN_RIDGE   = cfg.getfloat('MODEL', 'train_ridge')
EVAL_RIDGE    = cfg.getfloat('MODEL', 'eval_ridge')    # ridge used when refitting the *global* K/C_readout below
COMPUTE_READOUT = cfg.getboolean('MODEL', 'compute_readout')
RESID_TOL     = cfg.getfloat('EVAL', 'resid_tol')      # ResDMD residual above this -> flagged as likely spurious (diagnostic only, not a filter)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arr, months = load_fields(ZARR_PATH, VARS, start=START, end=END)
arr, _clim = remove_climatology(arr, months)
plot_arr = arr
arr, mean, std = normalize(arr)
arr = smooth(arr, SMOOTH_WINDOW)
train_arr, val_arr, test_arr = split(arr)

Y, X = arr.shape[2], arr.shape[3]
C = arr.shape[1]

train_ds_global = ORASDataset(train_arr, steps=1)
val_ds          = ORASDataset(val_arr, steps=1)
val_ds_roll     = ORASDataset(val_arr,  steps=ROLLOUT_STEPS)
test_ds_roll    = ORASDataset(test_arr, steps=ROLLOUT_STEPS)

TMASK_PATH = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/tmask_p.zarr'
try:
    tmask = xr.open_zarr(TMASK_PATH)['tmaskutil'].isel(t=0)
    tmask = tmask.sel(y=slice(0, 682), x=slice(0, 679)).values.astype(float)
    tmask[tmask == 0] = np.nan
except Exception:
    tmask = np.ones((Y, X))


def apply_mask(field):
    return field * tmask


def unnorm(x, channel_idx=0):
    return x * std.squeeze()[channel_idx] + mean.squeeze()[channel_idx]


# UNNORM_DIFF_FIX: helper + all call sites tagged with this marker
def unnorm_diff(x, channel_idx=0):
    """Convert a *difference* of two normalized fields to physical units.
    physical_a - physical_b = (a - b) * std, since the additive `mean` term
    cancels -- unlike unnorm(), this must not add mean back in.
    """
    return x * std.squeeze()[channel_idx]


model = EDMDDLModel(C, (Y, X), N_DICT, HIDDEN, INCLUDE_STATE, TRAIN_RIDGE, COMPUTE_READOUT).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()


class ResDMDModel:
    """A trained EDMDDLModel plus a globally-fit Koopman operator K and
    physical-space readout C_readout, exposing the lift/readout/advance
    operations every plot below needs.

    K and C_readout aren't persisted parameters (see edmd_dl_model.py) --
    `fit` refits them once from a large pool of snapshot pairs so every plot
    below eigendecomposes/reconstructs against the same fixed operator,
    instead of the per-batch refit training does.
    """

    def __init__(self, model, K, C_readout, C, H, W, G, A, L):
        self.model = model
        self.K = K
        self.C_readout = C_readout
        self.C, self.H, self.W = C, H, W
        self.G, self.A, self.L = G, A, L

    @classmethod
    def fit(cls, model, ds, device, ridge, C, H, W, batch_size=64, max_samples=4000):
        """Accumulate the Gram statistics G, A, L over `ds` and solve for K
        and C_readout via ridge regression. G/A/L are kept on the returned
        instance too -- koopman_modes() below reuses them for ResDMD residual
        diagnostics, which need the same large-sample statistics K was fit
        from.
        """
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
        G = A = L = Gc = Ac = None
        N = None
        n_seen = 0
        pairs_seen = 0
        model.eval()
        with torch.no_grad():
            for x_seq in loader:
                if n_seen >= max_samples:
                    break
                x_seq = x_seq.to(device)
                B, T, _, _, _ = x_seq.shape
                psi_all = torch.stack([model.dictionary(x_seq[:, t]) for t in range(T)], dim=1)
                psi_t   = psi_all[:, :-1].reshape(-1, psi_all.shape[-1]).float()
                psi_tp1 = psi_all[:, 1:].reshape(-1, psi_all.shape[-1]).float()
                if N is None:
                    N = psi_t.shape[1]
                    G = torch.zeros(N, N, device=device)
                    A = torch.zeros(N, N, device=device)
                    L = torch.zeros(N, N, device=device)

                G += psi_t.T @ psi_t
                A += psi_t.T @ psi_tp1
                L += psi_tp1.T @ psi_tp1

                x_flat = x_seq.reshape(B * T, -1).float()
                psi_flat = psi_all.reshape(B * T, N).float()
                if Gc is None:
                    state_dim = x_flat.shape[1]
                    Gc = torch.zeros(N, N, device=device)
                    Ac = torch.zeros(N, state_dim, device=device)
                Gc += psi_flat.T @ psi_flat
                Ac += psi_flat.T @ x_flat

                n_seen += B
                pairs_seen += psi_t.shape[0]

        G_reg = G + ridge * torch.eye(N, device=device)
        K = torch.linalg.solve(G_reg, A).T
        Gc_reg = Gc + ridge * torch.eye(N, device=device)
        # C_readout = torch.linalg.solve(Gc_reg, Ac).T
        Gc_inv = torch.linalg.inv(Gc_reg)
        C_readout = (Gc_inv @ Ac).T
        print(f'fit global K/C_readout from {n_seen} windows ({pairs_seen} one-step pairs, N={N})')
        return cls(model, K, C_readout, C, H, W, G, A, L)

    def lift(self, x):
        return self.model.dictionary(x)

    def readout(self, z):
        flat = z @ self.C_readout.T
        return flat.view(-1, self.C, self.H, self.W)

    def advance(self, z, steps=1):
        for _ in range(steps):
            z = z @ self.K.T
        return z


viz = ResDMDModel.fit(model, train_ds_global, DEVICE, EVAL_RIDGE, C, Y, X)


# --------------------------------------------------------------------------
# reconstruction / forecast plots
# --------------------------------------------------------------------------
def forecast(viz, ds, unnorm, apply_mask, out_dir, n_samples, device):
    channel_idx = 0
    sample_indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 4, figsize=(18, n_samples * 3.5))
    col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Forecast $x_{t_1}$', 'Target $x_{t_1}$']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    for row, idx in enumerate(sample_indices):
        x_seq = ds[idx]
        x_t0, x_t1 = x_seq[0], x_seq[1]
        x_seq_b = x_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            z0 = viz.lift(x_seq_b[:, 0])
            x_rec = viz.readout(z0)
            x_pred = viz.readout(viz.advance(z0))

        fields = [
            unnorm(x_t0.numpy()[channel_idx], channel_idx),
            unnorm(x_rec.cpu().numpy()[0, channel_idx], channel_idx),
            unnorm(x_pred.cpu().numpy()[0, channel_idx], channel_idx),
            unnorm(x_t1.numpy()[channel_idx], channel_idx),
        ]

        vmin = np.nanpercentile(fields[0], 2)
        vmax = np.nanpercentile(fields[0], 98)

        for col, field in enumerate(fields):
            ax = axes[row, col]
            im = ax.imshow(apply_mask(field), origin='lower', cmap='RdBu_r',
                            vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(f'sample {idx}', fontsize=9)

    plt.suptitle('Reconstruction & One-Step Forecast (°C anomaly)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved forecast.png')


def plot_loss_vs_anomaly_magnitude(viz, ds, out_dir, device):
    rec_losses, fcst_losses, anomaly_mags = [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            x_seq = ds[idx]
            x_t0, x_t1 = x_seq[0], x_seq[1]
            x_seq_b = x_seq.unsqueeze(0).to(device)

            z0 = viz.lift(x_seq_b[:, 0])
            x_rec = viz.readout(z0)
            x_pred = viz.readout(viz.advance(z0))

            rec_loss = F.mse_loss(x_rec.cpu()[0, 0], x_t0[0]).item()
            fcst_loss = F.mse_loss(x_pred.cpu()[0, 0], x_t1[0]).item()

            rec_losses.append(rec_loss)
            fcst_losses.append(fcst_loss)
            anomaly_mags.append(np.nanmean(np.abs(x_t0.numpy()[0])))

    rec_losses = np.array(rec_losses)
    fcst_losses = np.array(fcst_losses)
    anomaly_mags = np.array(anomaly_mags)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, losses, label in zip(axes, [rec_losses, fcst_losses], ["reconstruction", "forecast"]):
        ax.scatter(anomaly_mags, losses, alpha=0.5, s=15, color="steelblue")
        ax.set_xlabel("|anomaly| magnitude (input frame, normalized)")
        ax.set_ylabel(f"per-sample {label} loss")
        ax.set_title(f"{label} loss vs. anomaly magnitude")
        if len(losses) > 1:
            corr = np.corrcoef(anomaly_mags, losses)[0, 1]
            ax.annotate(f"corr = {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, va='top')

    fig.tight_layout()
    fig.savefig(f"{out_dir}/loss_vs_anomaly_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss_vs_anomaly_magnitude.png')


def plot_dict_variance_vs_eigval(viz, ds, device, out_dir, n_samples=2000):
    """Does the dictionary put the data's variance on persistent (|lambda|
    near 1) or transient/unstable (|lambda| far from 1) modes of the fitted
    K?
    """
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), n_samples), dtype=int)
    Z = []
    with torch.no_grad():
        for idx in idxs:
            x0 = ds[idx][0].unsqueeze(0).to(device)
            Z.append(viz.lift(x0).cpu().numpy()[0])
    Z = np.stack(Z, axis=0)

    K = viz.K.cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(K)
    Vinv = np.linalg.inv(eigenvectors)
    b = Vinv @ Z.T
    mode_var = b.var(axis=1).real
    eig_mag = np.abs(eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(eig_mag, mode_var, alpha=0.6, s=18, color="steelblue")
    axes[0].axvline(1.0, color='crimson', linestyle='--', alpha=0.6, label='|λ|=1 (stability boundary)')
    axes[0].set_xlabel("|eigenvalue| of fitted K")
    axes[0].set_ylabel("mode-coefficient variance")
    axes[0].set_title("Mode variance vs. persistence")
    axes[0].legend()
    if len(mode_var) > 1:
        corr = np.corrcoef(eig_mag, mode_var)[0, 1]
        axes[0].annotate(f"corr = {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                          fontsize=10, va='top')

    order = np.argsort(mode_var)[::-1]
    axes[1].bar(range(len(order)), mode_var[order], color="steelblue", alpha=0.6, label="variance")
    ax2 = axes[1].twinx()
    ax2.plot(range(len(order)), eig_mag[order], color="darkorange", linewidth=1.2, label="|λ|")
    axes[1].set_xlabel("mode (sorted by variance, descending)")
    axes[1].set_ylabel("mode variance", color="steelblue")
    ax2.set_ylabel("|λ|", color="darkorange")
    axes[1].set_title("Variance-ranked modes: does high variance => |λ|≈1?")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/dict_variance_vs_eigval.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved dict_variance_vs_eigval.png')


def plot_physical_variance_vs_eigval(viz, ds, device, out_dir, n_samples=2000, decode_batch=256):
    """For each eigenmode of the fitted K, hold every other mode at its mean
    and let this mode vary with its real per-sample coefficients, decode, and
    measure the induced physical-space variance vs. that mode's |eigenvalue|.
    """
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), n_samples), dtype=int)
    Z = []
    with torch.no_grad():
        for idx in idxs:
            x0 = ds[idx][0].unsqueeze(0).to(device)
            Z.append(viz.lift(x0).cpu().numpy()[0])
    Z = np.stack(Z, axis=0).astype(np.complex64)
    N = Z.shape[1]
    z_mean = Z.mean(axis=0).real

    K = viz.K.cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(K)
    Vinv = np.linalg.inv(eigenvectors)
    b = Vinv @ Z.T
    eig_mag = np.abs(eigenvalues)

    physical_var = np.zeros(N)
    for m in range(N):
        contrib = np.outer(eigenvectors[:, m], b[m]).real.T   # (n, N): this mode's real per-sample contribution
        z_batch = z_mean[None, :] + contrib
        decoded_chunks = []
        for start in range(0, z_batch.shape[0], decode_batch):
            chunk = torch.from_numpy(z_batch[start:start + decode_batch]).float().to(device)
            with torch.no_grad():
                out = viz.readout(chunk).cpu().numpy()
            decoded_chunks.append(out)
        decoded = np.concatenate(decoded_chunks, axis=0)
        decoded = apply_mask(decoded)
        var_map = np.nanvar(decoded, axis=0)
        physical_var[m] = np.nanmean(var_map)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(eig_mag, physical_var, alpha=0.6, s=18, color="steelblue")
    axes[0].axvline(1.0, color='crimson', linestyle='--', alpha=0.6, label='|λ|=1')
    axes[0].set_xlabel("|eigenvalue| of fitted K")
    axes[0].set_ylabel("decoded physical-space variance")
    axes[0].set_title("Physical-space variance vs. mode persistence")
    axes[0].legend()
    if len(physical_var) > 1:
        corr = np.corrcoef(eig_mag, physical_var)[0, 1]
        axes[0].annotate(f"corr = {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                          fontsize=10, va='top')

    order = np.argsort(physical_var)[::-1]
    axes[1].bar(range(len(order)), physical_var[order], color="steelblue", alpha=0.6, label="variance")
    ax2 = axes[1].twinx()
    ax2.plot(range(len(order)), eig_mag[order], color="darkorange", linewidth=1.2, label="|λ|")
    axes[1].set_xlabel("mode (sorted by physical variance, descending)")
    axes[1].set_ylabel("physical-space variance", color="steelblue")
    ax2.set_ylabel("|λ|", color="darkorange")
    axes[1].set_title("Variance-ranked modes: does high physical variance => |λ|≈1?")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/physical_variance_vs_eigval.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved physical_variance_vs_eigval.png')


def koopman_modes(viz, arr, device, out_dir, n_modes, batch_size=BATCH_SIZE):
    K_matrix = viz.K.detach().cpu()
    eigenvalues, eigenvectors, residuals = resdmd_residuals(
        K_matrix, viz.G.cpu(), viz.A.cpu(), viz.L.cpu()
    )
    n_spurious = (residuals > RESID_TOL).sum().item()
    print(f'ResDMD residuals: {n_spurious}/{len(residuals)} eigenpairs above '
          f'RESID_TOL={RESID_TOL} (flagged as likely spurious, not dropped)')

    full_ds = ORASDataset(arr, steps=0)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = viz.lift(x_seq[:, 0].to(device))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)

    V = eigenvectors
    Vinv = torch.linalg.inv(V)
    b = Vinv @ Z.T.to(torch.complex64)
    amplitude = b.abs().std(dim=1)

    order = amplitude.argsort(descending=True)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    amplitude    = amplitude[order]
    residuals    = residuals[order]
    b            = b[order]
    z_mean = Z.mean(dim=0).to(torch.float32).to(device)

    ncols = 3
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()
    for i in range(n_modes):
        v_real = eigenvectors[:, i].real
        scale = amplitude[i].real.item()
        v_scaled = (z_mean + scale * v_real.to(device)).unsqueeze(0).to(device)
        with torch.no_grad():
            mode_pos = viz.readout(v_scaled).cpu()
            mode_neg = viz.readout(z_mean.unsqueeze(0)).cpu()
        # UNNORM_DIFF_FIX: convert normalized-space mode diff to physical units (deg C)
        mode = unnorm_diff((mode_pos - mode_neg).numpy()[0, 0])
        lam = eigenvalues[i]
        angle = lam.angle().abs().item()
        period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
        flag = ' [spurious?]' if residuals[i] > RESID_TOL else ''
        title = f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}  amp={amplitude[i]:.2f}  res={residuals[i]:.3f}{flag}'
        vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
        im = axes[i].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                             vmin=-vext, vmax=vext, aspect='auto')
        axes[i].set_title(title, fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        # UNNORM_DIFF_FIX: colorbar now reflects deg C, not normalized units
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='ΔSST (°C)')
    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Koopman Modes (decoded eigenvectors of fitted K)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/koopman_modes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved koopman_modes.png')

    T = b.shape[1]
    freqs = np.fft.rfftfreq(T, d=1.0)
    periods_yr = np.full_like(freqs, np.nan)
    periods_yr[1:] = 1.0 / (freqs[1:] * 12)

    power = np.abs(np.fft.rfft(b.real.numpy(), axis=1)) ** 2

    enso_band   = (periods_yr >= 2) & (periods_yr <= 7)
    enso_power  = power[:, enso_band].sum(axis=1)
    total_power = power[:, 1:].sum(axis=1)
    enso_frac   = enso_power / (total_power + 1e-12)

    mode_periods = np.array([
        2 * np.pi / eigenvalues[i].angle().abs().item() / 12
        if eigenvalues[i].angle().abs().item() > 1e-4 else np.nan
        for i in range(len(eigenvalues))
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(mode_periods, amplitude.numpy(), c=enso_frac, cmap='viridis', s=30)
    ax.axvspan(2, 7, color='orange', alpha=0.15, label='ENSO band (2-7yr)')
    ax.set_xscale('log')
    ax.set_xlabel('mode period (yr, from eigenvalue angle)')
    ax.set_ylabel('mode amplitude (RMS of b(t))')
    ax.set_title('Koopman modes: amplitude vs. period, colored by ENSO-band power fraction')
    plt.colorbar(sc, ax=ax, label='fraction of b(t) power in 2-7yr band')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/mode_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved mode_spectrum.png')

    n_enso = 4
    enso_order = np.argsort(enso_power)[::-1][:n_enso]

    fig, axes = plt.subplots(1, n_enso, figsize=(n_enso * 5, 4))
    for j, i in enumerate(enso_order):
        v_real = eigenvectors[:, i].real
        scale = amplitude[i].real.item()
        v_scaled = (z_mean + scale * v_real.to(device)).unsqueeze(0)
        with torch.no_grad():
            mode_pos = viz.readout(v_scaled).cpu()
            mode_neg = viz.readout(z_mean.unsqueeze(0)).cpu()
        # UNNORM_DIFF_FIX: convert normalized-space mode diff to physical units (deg C)
        mode = unnorm_diff((mode_pos - mode_neg).numpy()[0, 0])

        lam = eigenvalues[i]
        angle = lam.angle().abs().item()
        period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
        flag = ' [spurious?]' if residuals[i] > RESID_TOL else ''
        title = (f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}\n'
                 f'amp={amplitude[i]:.2f}  ENSO frac={enso_frac[i]:.2f}  res={residuals[i]:.3f}{flag}')

        vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
        im = axes[j].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                             vmin=-vext, vmax=vext, aspect='auto')
        axes[j].set_title(title, fontsize=9)
        axes[j].set_xticks([]); axes[j].set_yticks([])
        # UNNORM_DIFF_FIX: colorbar now reflects deg C, not normalized units
        plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04, label='ΔSST (°C)')

    plt.suptitle('Koopman modes ranked by power in the ENSO band (2-7yr)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/enso_modes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved enso_modes.png')

    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='unit circle')
    eigs = eigenvalues.numpy()
    sc = ax.scatter(eigs.real, eigs.imag, c=residuals.numpy(), cmap='plasma', s=20, zorder=3)
    plt.colorbar(sc, ax=ax, label='ResDMD residual (lower = more trustworthy)')
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_aspect('equal')
    ax.set_title('Koopman eigenvalue spectrum, colored by ResDMD residual')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved eigenvalue_spectrum.png')


def koopman_modes_by_residual(viz, arr, device, out_dir, n_modes, batch_size=BATCH_SIZE):
    """Same panel grid as koopman_modes() above, but ranks eigenpairs by
    ResDMD residual ascending (low -> high) instead of amplitude, so the
    n_modes shown are the most trustworthy rather than the most energetic.
    """
    K_matrix = viz.K.detach().cpu()
    eigenvalues, eigenvectors, residuals = resdmd_residuals(
        K_matrix, viz.G.cpu(), viz.A.cpu(), viz.L.cpu()
    )

    full_ds = ORASDataset(arr, steps=0)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = viz.lift(x_seq[:, 0].to(device))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)

    V = eigenvectors
    Vinv = torch.linalg.inv(V)
    b = Vinv @ Z.T.to(torch.complex64)
    amplitude = b.abs().std(dim=1)

    order = residuals.argsort(descending=False)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    amplitude    = amplitude[order]
    residuals    = residuals[order]
    z_mean = Z.mean(dim=0).to(torch.float32).to(device)

    ncols = 3
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()
    for i in range(n_modes):
        v_real = eigenvectors[:, i].real
        scale = amplitude[i].real.item()
        v_scaled = (z_mean + scale * v_real.to(device)).unsqueeze(0).to(device)
        with torch.no_grad():
            mode_pos = viz.readout(v_scaled).cpu()
            mode_neg = viz.readout(z_mean.unsqueeze(0)).cpu()
        # UNNORM_DIFF_FIX: convert normalized-space mode diff to physical units (deg C)
        mode = unnorm_diff((mode_pos - mode_neg).numpy()[0, 0])
        lam = eigenvalues[i]
        angle = lam.angle().abs().item()
        period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
        flag = ' [spurious?]' if residuals[i] > RESID_TOL else ''
        title = f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}  amp={amplitude[i]:.2f}  res={residuals[i]:.3f}{flag}'
        vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
        im = axes[i].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                             vmin=-vext, vmax=vext, aspect='auto')
        axes[i].set_title(title, fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        # UNNORM_DIFF_FIX: colorbar now reflects deg C, not normalized units
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='ΔSST (°C)')
    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Koopman Modes ranked by ResDMD residual (low to high)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/koopman_modes_by_residual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved koopman_modes_by_residual.png')


def reconstruction_skill(viz, ds, unnorm, apply_mask, tmask, out_dir, n_samples, device):
    from numpy.fft import fft2, fftshift
    sample_indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, n_samples * 3.5))
    col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Difference']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    fig_fft, axes_fft = plt.subplots(n_samples, 1, figsize=(6, n_samples * 3.5))
    if n_samples == 1:
        axes_fft = [axes_fft]

    samples_diff, samples_input, ss_scores = [], [], []

    for row, idx in enumerate(sample_indices):
        x_seq = ds[idx]
        x_t0 = x_seq[0]
        x_seq_b = x_seq.unsqueeze(0).to(device)
        with torch.no_grad():
            x_rec = viz.readout(viz.lift(x_seq_b[:, 0]))

        input_field = unnorm(x_t0.numpy()[0])
        recon_field = unnorm(x_rec.cpu().numpy()[0, 0])
        diff_field  = input_field - recon_field

        diff_field_filled = np.nan_to_num(diff_field, nan=0.0)
        Fimg = fftshift(fft2(diff_field_filled))
        ax_fft = axes_fft[row]
        im_fft = ax_fft.imshow(np.log(np.abs(Fimg) + 1), cmap='viridis')
        ax_fft.set_title(f'FFT of residual | sample {idx}', fontsize=10)
        ax_fft.set_xticks([]); ax_fft.set_yticks([])
        plt.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

        samples_diff.append(diff_field)
        samples_input.append(input_field)
        ss_scores.append(1 - (np.nanmean(diff_field**2) / np.nanvar(input_field)))

        vmin = np.nanpercentile(input_field, 2)
        vmax = np.nanpercentile(input_field, 98)
        abs_max = np.nanpercentile(np.abs(diff_field), 98)

        ax = axes[row, 0]
        im = ax.imshow(apply_mask(input_field), origin='lower', cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row, 1]
        im = ax.imshow(apply_mask(recon_field), origin='lower', cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row, 2]
        im = ax.imshow(apply_mask(diff_field), origin='lower', cmap='RdBu_r',
                        vmin=-abs_max, vmax=abs_max, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(f'sample {idx} | SS={ss_scores[-1]:.3f}', fontsize=9)

    plt.figure(fig.number)
    plt.suptitle('Reconstruction (°C anomaly)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/reconstruction_skill.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    plt.figure(fig_fft.number)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/residual_fft.png', dpi=150, bbox_inches='tight')
    plt.close(fig_fft)

    print(f'saved reconstruction_skill.png and residual_fft.png | mean SS={np.mean(ss_scores):.4f}')

    diff_stack  = np.stack(samples_diff,  axis=0)
    input_stack = np.stack(samples_input, axis=0)

    nlat, nlon = input_field.shape[-2], input_field.shape[-1]
    ocean_mask = ~np.isnan(tmask)
    ocean_mask_flat = ocean_mask.flatten()

    diff_ocean  = diff_stack[:, ocean_mask]
    input_ocean = input_stack[:, ocean_mask]

    r = pearsonr(diff_ocean, input_ocean, axis=0)
    r_values = r.statistic
    print(f"Mean r: {np.nanmean(r_values):.3f}")
    print(f"Median r: {np.nanmedian(r_values):.3f}")
    print(f"Std r: {np.nanstd(r_values):.3f}")
    print(f"% cells with r > 0.5: {(r_values > 0.5).mean() * 100:.1f}%")

    r_map = np.full(nlat * nlon, np.nan)
    r_map[ocean_mask_flat] = r.statistic
    r_spatial = r_map.reshape(nlat, nlon)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    col_titles = ['Pearson R', 'Input Variance']
    for ax, title in zip(axes, col_titles):
        ax.set_title(title, fontsize=12)

    ax = axes[0]
    im = ax.imshow(apply_mask(r_spatial), origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    input_variance = np.nanvar(input_stack, axis=0)
    ax = axes[1]
    im = ax.imshow(apply_mask(input_variance), origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/r_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved r_map.png')


def print_stats(model, viz):
    total_params = sum(p.numel() for p in model.parameters())
    dict_params = sum(p.numel() for p in model.dictionary.parameters())
    print(f"params: dictionary(conv+readout)={dict_params}, total={total_params}")
    print(f"dictionary dim (N): {model.dictionary.n_dict}")


# --------------------------------------------------------------------------
# rollout skill plots
# --------------------------------------------------------------------------
def eval_rollout(model, loader, rollout_steps, device, label):
    step_losses = [0.0] * rollout_steps
    step_counts = [0]   * rollout_steps
    rec_loss_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_seq in loader:
            x_seq = x_seq.to(device)
            z0    = model.lift(x_seq[:, 0])
            x_rec = model.readout(z0)
            rec_loss_total += nn.functional.l1_loss(x_rec, x_seq[:, 0]).item()
            z = z0
            for k in range(rollout_steps):
                if k + 1 >= x_seq.shape[1]:
                    break
                z = model.advance(z)
                x_pred = model.readout(z)
                step_losses[k] += nn.functional.l1_loss(x_pred, x_seq[:, k + 1]).item()
                step_counts[k] += 1
            n_batches += 1
    rec_loss_total /= n_batches
    step_losses = [step_losses[k] / max(step_counts[k], 1) for k in range(rollout_steps)]
    print(f'\n{label} rec loss: {rec_loss_total:.5f}')
    for k, l in enumerate(step_losses):
        print(f'  step {k+1}: {l:.5f}')
    return rec_loss_total, step_losses


def plot_forecast_skill(val_steps, test_steps, val_rec, test_rec, rollout_steps, out_dir):
    steps = list(range(1, rollout_steps + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, val_steps,  marker='o', label='val')
    ax.plot(steps, test_steps, marker='o', label='test')
    ax.axhline(val_rec,  linestyle='--', alpha=0.5, label='val rec baseline')
    ax.axhline(test_rec, linestyle='--', alpha=0.5, label='test rec baseline')
    ax.set_xlabel('forecast step (months)')
    ax.set_ylabel('L1 loss')
    ax.set_title('Forecast skill vs. lead time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/forecast_skill.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved forecast_skill.png')


def rollout_panel(model, ds, unnorm, apply_mask, rollout_steps, n_samples, device, out_dir, channel_idx=0, split_label='test'):
    sample_indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)
    n_cols = 2 + rollout_steps * 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 3.5, n_samples * 3))

    col_titles = ['Input $x_{t_0}$', 'Reconstruction']
    for k in range(rollout_steps):
        col_titles += [f'Forecast +{k+1}mo', f'Target +{k+1}mo']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9)

    for row, idx in enumerate(sample_indices):
        x_seq = ds[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            z0    = model.lift(x_seq[:, 0])
            x_rec = model.readout(z0)
            preds = []
            z = z0
            for _ in range(rollout_steps):
                z = model.advance(z)
                preds.append(model.readout(z))

        inp  = unnorm(x_seq[0, 0, channel_idx].cpu().numpy(), channel_idx)
        rec  = unnorm(x_rec[0, channel_idx].cpu().numpy(), channel_idx)
        vext = np.nanpercentile(np.abs(inp), 98)

        def show(ax, field):
            im = ax.imshow(apply_mask(field), origin='lower', cmap='RdBu_r',
                            vmin=-vext, vmax=vext, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        show(axes[row, 0], inp)
        show(axes[row, 1], rec)
        for k in range(rollout_steps):
            pred = unnorm(preds[k][0, channel_idx].cpu().numpy(), channel_idx)
            show(axes[row, 2 + k * 2], pred)
            if k + 1 < x_seq.shape[1]:
                tgt = unnorm(x_seq[0, k + 1, channel_idx].cpu().numpy(), channel_idx)
                show(axes[row, 2 + k * 2 + 1], tgt)
                tgt_flat = tgt.flatten()
                pred_flat = pred.flatten()
                mask = ~np.isnan(tgt_flat) & ~np.isnan(pred_flat)
                r = pearsonr(tgt_flat[mask], pred_flat[mask])
                print(f'rollout step {k}: var ratio={np.nanvar(tgt) / np.nanvar(pred):.3f}  r={r.statistic:.3f}')
            else:
                axes[row, 2 + k * 2 + 1].set_visible(False)
        axes[row, 0].set_ylabel(f'sample {idx}', fontsize=9)

    plt.suptitle(f'Reconstruction & {rollout_steps}-step rollout ({split_label})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/rollout_panel_{VARS[channel_idx]}_{split_label}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved rollout_panel_{split_label}.png')


if __name__ == "__main__":
    log_path = os.path.join(out, 'output_resdmd_viz.log')
    sys.stdout = open(log_path, 'w', buffering=1)
    sys.stderr = sys.stdout

    print_stats(model, viz)

    # reconstruction / eigenmode plots -> {out}/figs
    plot_dict_variance_vs_eigval(viz, val_ds, DEVICE, OUT_DIR)
    forecast(viz, val_ds, unnorm, apply_mask, OUT_DIR, N_SAMPLES, DEVICE)
    plot_loss_vs_anomaly_magnitude(viz, val_ds, OUT_DIR, DEVICE)
    koopman_modes(viz, arr, DEVICE, OUT_DIR, N_MODES)
    koopman_modes_by_residual(viz, arr, DEVICE, OUT_DIR, N_MODES)
    reconstruction_skill(viz, val_ds, unnorm, apply_mask, tmask, OUT_DIR, N_SAMPLES, DEVICE)
    plot_physical_variance_vs_eigval(viz, val_ds, DEVICE, OUT_DIR)

    # rollout skill plots -> {out}/eval
    val_loader  = DataLoader(val_ds_roll,  batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds_roll, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    val_rec,  val_steps  = eval_rollout(viz, val_loader,  ROLLOUT_STEPS, DEVICE, 'val')
    test_rec, test_steps = eval_rollout(viz, test_loader, ROLLOUT_STEPS, DEVICE, 'test')
    plot_forecast_skill(val_steps, test_steps, val_rec, test_rec, ROLLOUT_STEPS, EVAL_DIR)

    for k in range(len(VARS)):
        rollout_panel(viz, val_ds_roll, unnorm, apply_mask, ROLLOUT_STEPS, N_SAMPLES, DEVICE, EVAL_DIR, k, split_label='val')
