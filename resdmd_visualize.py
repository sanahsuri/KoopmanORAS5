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

NOTE: N_DICT / HIDDEN / INCLUDE_STATE / TRAIN_RIDGE / LOOKBACK below are
constructor args that train.py also reads from config.ini's [MODEL] section
(not saved in the checkpoint itself), so they stay in sync automatically as
long as this script is run against the same config.ini the run was trained
with.
"""
import os
import sys
import ast
import glob
import configparser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import ConcatDataset, DataLoader

from all_data import load_fields, remove_climatology, normalize, split, split_members, ORASDataset, smooth
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
Y_SLICE       = try_cast(cfg['DATA']['y_slice']) if 'y_slice' in cfg['DATA'] else None
X_SLICE       = try_cast(cfg['DATA']['x_slice']) if 'x_slice' in cfg['DATA'] else None
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
LOOKBACK      = cfg.getint('MODEL', 'lookback', fallback=1)
DEPTH         = cfg.getint('MODEL', 'depth', fallback=3)
TRAIN_RIDGE   = cfg.getfloat('MODEL', 'train_ridge')
EVAL_RIDGE    = cfg.getfloat('MODEL', 'eval_ridge')    # ridge used when refitting the *global* K/C_readout below
COMPUTE_READOUT = cfg.getboolean('MODEL', 'compute_readout')
RESID_TOL     = cfg.getfloat('EVAL', 'resid_tol')      # ResDMD residual above this -> flagged as likely spurious (diagnostic only, not a filter)
# one-shot N-month-ahead forecast lags, e.g. `oneshot_lags = [12]`; [] (default) skips it
ONESHOT_LAGS  = try_cast(cfg['EVAL'].get('oneshot_lags', '[]'))
# which variance-ranked modes (1-indexed, matching the "mode N" plot titles)
# get a _plot_mode_phase_grid sweep in koopman_modes() -- [VIZ] is optional,
# missing section/key both fall back to the same [1, 2, 3, 5] default.
PHASE_SWEEP_MODES = [m - 1 for m in try_cast(cfg.get('VIZ', 'phase_sweep_modes', fallback='[1, 2, 3, 5]'))]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_FRAC    = cfg.getfloat('DATA', 'train_frac', fallback=0.8)
VAL_FRAC      = cfg.getfloat('DATA', 'val_frac', fallback=0.1)
STD_FLOOR     = cfg.getfloat('DATA', 'std_floor', fallback=0.0)

# clim/mean/std exactly as train.py saved them with the checkpoint, so the
# model sees the same input scaling it was trained on; recomputed from the
# training months only for a checkpoint that doesn't carry them
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
arr, months = load_fields(ZARR_PATH, VARS, y_slice=Y_SLICE, x_slice=X_SLICE, start=START, end=END)
N_TRAIN_MONTHS = int(arr.shape[0] * TRAIN_FRAC)
arr, _clim = remove_climatology(arr, months, train_frac=TRAIN_FRAC, clim=ckpt.get('clim'), n_vars=len(VARS))
plot_arr = arr
arr, mean, std = normalize(arr, mean=ckpt.get('mean'), std=ckpt.get('std'), n_fit=N_TRAIN_MONTHS,
                           n_vars=len(VARS), std_floor=STD_FLOOR)
arr = smooth(arr, SMOOTH_WINDOW)
# global K/C_readout are fit on every member's training months; the
# single-trajectory plots/diagnostics below all use member 0 (opa0), so
# arr/plot_arr/mean/std become that member's (T, n_vars, Y, X) view
member_train_arrs = [split(a, TRAIN_FRAC, VAL_FRAC)[0] for a in split_members(arr, len(VARS))]
arr, plot_arr, mean, std = (split_members(a, len(VARS))[0] for a in (arr, plot_arr, mean, std))
train_arr, val_arr, test_arr = split(arr, TRAIN_FRAC, VAL_FRAC)
train_months, val_months, test_months = split(months, TRAIN_FRAC, VAL_FRAC)

Y, X = arr.shape[2], arr.shape[3]
C = arr.shape[1]

train_ds_global = ConcatDataset([ORASDataset(a, steps=1, lookback=LOOKBACK) for a in member_train_arrs])
val_ds          = ORASDataset(val_arr, steps=1, lookback=LOOKBACK)
val_ds_roll     = ORASDataset(val_arr,  steps=ROLLOUT_STEPS, lookback=LOOKBACK)
test_ds_roll    = ORASDataset(test_arr, steps=ROLLOUT_STEPS, lookback=LOOKBACK)

TMASK_MATCHES = glob.glob(os.path.join(ZARR_PATH, 'tmask_*.zarr'))
TMASK_PATH = TMASK_MATCHES[0] if TMASK_MATCHES else None
try:
    tmask = xr.open_zarr(TMASK_PATH)['tmaskutil'].isel(t=0)
    tmask = tmask.isel(y=slice(0, Y), x=slice(0, X)).values.astype(float)
    tmask[tmask == 0] = np.nan
except Exception:
    tmask = np.ones((Y, X))


def apply_mask(field):
    return field * tmask


def unnorm(x, channel_idx=0):
    # squeeze(axis=0) only drops the leading (T-reduction) dim -- squeeze()
    # with no args would also drop the channel dim when C==1, silently
    # turning [channel_idx] into an index along Y instead of a channel
    # select (wrong-shaped, wrongly-broadcast per-pixel scale/offset).
    return x * std.squeeze(axis=0)[channel_idx] + mean.squeeze(axis=0)[channel_idx]

def unnorm_diff(x, channel_idx=0):
    """Convert a *difference* of two normalized fields to physical units.
    physical_a - physical_b = (a - b) * std, since the additive `mean` term
    cancels -- unlike unnorm(), this must not add mean back in.
    """
    return x * std.squeeze(axis=0)[channel_idx]


def _peak_loading_ref(V, viz, ref_arr, device):
    """Physically anchors each mode's phase to the real data at wherever that
    mode's own decoded pattern is strongest, instead of a fixed channel/
    location chosen up front. |readout(v)| is phase-invariant (rotating v by
    e^{i*theta} just rotates readout(v) by the same phase), so each mode's
    peak-loading (channel, y, x) is well-defined before any phase choice is
    made -- this is what lets the reference generalize across any number of
    variables/channels instead of hardcoding VARS[0].
    ref_arr controls which time period the reference series is drawn from --
    callers pass a train-only slice so the sign choice never sees held-out
    val/test data.
    Returns ref: (n_modes, T) complex, ref_arr's time series at each mode's
    own peak-loading location.
    """
    C_readout_c = viz.C_readout.to(torch.complex64).to(device)
    S = (V.T.to(device).to(torch.complex64) @ C_readout_c.T).view(-1, viz.C, viz.H, viz.W)
    tmask_t = torch.from_numpy(np.nan_to_num(tmask, nan=0.0)).to(device)  # 0 over land, so land is never picked
    mag = S.abs() * tmask_t
    flat_idx = mag.view(mag.shape[0], -1).argmax(dim=1)                   # (n_modes,) peak index per mode
    c_idx, y_idx, x_idx = np.unravel_index(flat_idx.cpu().numpy(), (viz.C, viz.H, viz.W))
    ref = torch.from_numpy(ref_arr[:, c_idx, y_idx, x_idx].T).to(torch.complex64)  # (n_modes, T_train)
    return ref


# PHASE_FIX: helper + all call sites tagged with this marker
def _pin_mode_phase(eigenvectors, b, ref, n_train):
    """Rotate each eigenvector (and its matching row of b) by the unit-modulus
    phase that maximizes correlation of Re(b_i(t)) with that mode's own
    peak-loading reference series ref[i], so a mode's positive phase
    consistently means "more of the real anomaly at this mode's own peak
    location" rather than an arbitrary convention that can flip between
    eig() calls/backends/runs. Only b[:, :n_train] (the training period) is
    used to choose the phase, so the sign is never fit against held-out
    val/test data -- the resulting rotation is then applied to the full b
    (all timesteps) used for plotting. |b_i(t)| (and hence amplitude and
    variance explained) is unaffected -- only the sign/orientation of the
    decoded spatial pattern changes.
    """
    z = (b[:, :n_train] * ref).sum(dim=1)  # (n_modes,) complex, train period only
    phase = torch.exp(1j * z.angle())
    eigenvectors = eigenvectors * phase.unsqueeze(0)
    b = b * phase.conj().unsqueeze(1)
    return eigenvectors, b


def mode_variance_explained(viz, eigenvalues, eigenvectors, b, arr, device):
    """Per mode: fraction of the data's physical-unit variance it explains,
    plus the typical coefficient size used to scale its plotted pattern.

    Mode i's contribution to the reconstructed field is
        x_i(t) = c_i * Re(b_i(t) * u_i),   u_i = C_readout @ v_i,
    with c_i = 2 for a complex eigenvalue (its conjugate partner contributes
    the identical term) and 1 for a real one. var_frac[i] is the ocean-summed
    time variance of x_i in physical units (all channels) divided by the
    same total for `arr` over the time steps b covers (arr[LB-1:]). The
    modes aren't orthogonal, so var_frac needn't sum to 1. Invariant to
    _pin_mode_phase, since b_i * v_i is.

    amplitude[i] = c_i * RMS|b_i(t)| -- the pattern scale for plotting, so
    decoded patterns show a typical-magnitude anomaly in degC.

    Returns (var_frac, amplitude), both (N,) CPU tensors.
    """
    LB = viz.lookback
    ocean = (~np.isnan(tmask)).astype(np.float64)                       # (Y, X)
    std_c = np.nan_to_num(std.squeeze(axis=0))                          # (C, Y, X), land -> 0
    weight = std_c * ocean                                              # physical scale, land masked
    data = arr[LB - 1:]
    total_var = sum(float((weight[c] ** 2 * data[:, c].var(axis=0)).sum()) for c in range(data.shape[1]))

    w = torch.from_numpy(weight.reshape(-1)).float().to(device)
    C_readout = viz.C_readout.to(device)
    pair = 1.0 + (eigenvalues.imag.abs() > 1e-6).double()               # c_i
    br = b.real.double()
    bi = b.imag.double()
    br = br - br.mean(dim=1, keepdim=True)
    bi = bi - bi.mean(dim=1, keepdim=True)

    var_frac = torch.zeros(b.shape[0], dtype=torch.float64)
    with torch.no_grad():
        for i in range(b.shape[0]):
            v = eigenvectors[:, i].to(device)
            p = ((C_readout @ v.real.float()) * w).double()
            q = ((C_readout @ v.imag.float()) * w).double()
            # x_i(t)/c_i = br(t)*p - bi(t)*q, so its pixel-summed time variance is:
            var_i = ((br[i] ** 2).mean() * (p @ p).cpu()
                     + (bi[i] ** 2).mean() * (q @ q).cpu()
                     - 2 * (br[i] * bi[i]).mean() * (p @ q).cpu())
            var_frac[i] = pair[i] ** 2 * var_i / total_var

    amplitude = (pair * b.abs().double().pow(2).mean(dim=1).sqrt()).float()
    return var_frac, amplitude


model = EDMDDLModel(C, (Y, X), N_DICT, HIDDEN, INCLUDE_STATE, TRAIN_RIDGE, COMPUTE_READOUT, lookback=LOOKBACK,
                    depth=DEPTH).to(DEVICE)
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

    def __init__(self, model, K, C_readout, C, H, W, G, A, L, lookback=1):
        self.model = model
        self.K = K
        self.C_readout = C_readout
        self.C, self.H, self.W = C, H, W
        self.G, self.A, self.L = G, A, L
        self.lookback = lookback

    # lag=1 (default) reproduces the original one-step fit exactly; lag=N
    # fits K directly from N-step-apart pairs instead of compounding the
    # one-step K N times -- see forecast_oneshot.py.
    @classmethod
    def fit(cls, model, ds, device, ridge, C, H, W, batch_size=64, max_samples=4000, lag=1, lookback=1):
        """Accumulate the Gram statistics G, A, L over `ds` and solve for K
        and C_readout via ridge regression. G/A/L are kept on the returned
        instance too -- koopman_modes() below reuses them for ResDMD residual
        diagnostics, which need the same large-sample statistics K was fit
        from.

        lag=1 (default) fits the usual one-step operator from adjacent
        frames. lag=N fits K directly from pairs N steps apart, i.e. a
        genuine N-step transition operator -- distinct from (and generally
        better-behaved than) applying the lag=1 K to itself N times.

        lookback must match the model's own lookback (trailing frames
        stacked as extra input channels per dictionary call) -- `ds` must
        therefore have been built with the same `lookback`. Each dictionary
        call below consumes lookback frames at a time
        (x_seq[:, t:t+lookback]), so the window's T "current states" run
        from physical index lookback-1 to L-1, one fewer than when
        lookback=1.
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
                B, L_win, _, _, _ = x_seq.shape
                T = L_win - lookback + 1
                psi_all = torch.stack(
                    [model.dictionary(x_seq[:, t:t + lookback].reshape(B, -1, H, W)) for t in range(T)], dim=1
                )
                # pairs spaced `lag` steps apart, not hardcoded-adjacent frames
                psi_t   = psi_all[:, :-lag].reshape(-1, psi_all.shape[-1]).float()
                psi_tp1 = psi_all[:, lag:].reshape(-1, psi_all.shape[-1]).float()
                if N is None:
                    N = psi_t.shape[1]
                    G = torch.zeros(N, N, device=device)
                    A = torch.zeros(N, N, device=device)
                    L = torch.zeros(N, N, device=device)

                G += psi_t.T @ psi_t
                A += psi_t.T @ psi_tp1
                L += psi_tp1.T @ psi_tp1

                # readout target is the physical frame each psi_all[:, t] actually
                # represents (the most recent of its lookback-frame input), not the
                # whole window -- matches EDMDDLModel.forward's x_target slicing
                x_target = x_seq[:, lookback - 1: lookback - 1 + T]
                x_flat = x_target.reshape(B * T, -1).float()
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
        # label reflects whichever lag was actually fit, not always "one-step"
        print(f'fit global K/C_readout from {n_seen} windows ({pairs_seen} lag-{lag} pairs, N={N})')
        return cls(model, K, C_readout, C, H, W, G, A, L, lookback=lookback)

    def lift(self, x_window):
        """x_window: (B, lookback, C, H, W) -- the trailing lookback frames
        ending at the state being lifted. Flattened to (B, lookback*C, H, W)
        to match EDMDDLModel.forward's per-step dictionary input (most
        recent frame last)."""
        B, L_win, C, H, W = x_window.shape
        return self.model.dictionary(x_window.reshape(B, L_win * C, H, W))

    def readout(self, z):
        flat = z @ self.C_readout.T
        return flat.view(-1, self.C, self.H, self.W)

    def advance(self, z, steps=1):
        for _ in range(steps):
            z = z @ self.K.T
        return z


viz = ResDMDModel.fit(model, train_ds_global, DEVICE, EVAL_RIDGE, C, Y, X, lookback=LOOKBACK)


# --------------------------------------------------------------------------
# reconstruction / forecast plots
# --------------------------------------------------------------------------
def persistence_corr(rec_field, pred_field):
    """Pearson r and MSE between the reconstruction (the model's read of
    "now") and the forecast/prediction. This is NOT forecast accuracy
    (that's r/var ratio against the *target*, computed elsewhere) -- it
    measures how much the forecast merely persists the reconstructed
    current state forward unchanged. r near 1 (and MSE near 0) means the
    "forecast" is barely distinguishable from persistence (little added
    dynamics from K); lower r / higher MSE means K is actually doing
    something.
    Returns (r, mse), both float('nan') if fewer than 2 valid (non-NaN,
    e.g. land-masked) pixels are shared between the two fields.
    """
    rec_flat = np.asarray(rec_field).flatten()
    pred_flat = np.asarray(pred_field).flatten()
    mask = ~np.isnan(rec_flat) & ~np.isnan(pred_flat)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    r = pearsonr(rec_flat[mask], pred_flat[mask]).statistic
    mse = np.mean((rec_flat[mask] - pred_flat[mask]) ** 2)
    return r, mse


# lead_steps=1 (default) reproduces the original one-step-ahead panel
# exactly, including its output filename; lead_steps=N labels/saves as an
# N-month panel and reads the target frame N steps into the window instead
# of hardcoded index 1 -- pass a `ds` whose window is >= N+1 long (e.g. via
# forecast_oneshot.py) and a `viz` whose K already represents that horizon
# (advance() below still applies K exactly once).
def forecast(viz, ds, unnorm, apply_mask, out_dir, n_samples, device, months, lead_steps=1):
    channel_idx = 0
    LB = viz.lookback
    sample_indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 5, figsize=(22.5, n_samples * 3.5))
    # titled by lead_steps instead of always "$x_{t_1}$"
    col_titles = ['Input $x_{t_0}$', 'Reconstruction', f'Forecast +{lead_steps}mo', f'Target +{lead_steps}mo',
                  'Forecast $-$ Target']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    for row, idx in enumerate(sample_indices):
        x_seq = ds[idx]
        # "current" state is the last of the lookback frames (index LB-1), not
        # always index 0; target is lead_steps past that, not hardcoded index 1
        x_t0, x_t1 = x_seq[LB - 1], x_seq[LB - 1 + lead_steps]
        x_seq_b = x_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            z0 = viz.lift(x_seq_b[:, 0:LB])
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

        err = unnorm_diff(x_pred.cpu().numpy()[0, channel_idx] - x_t1.numpy()[channel_idx], channel_idx)
        err_lim = np.nanpercentile(np.abs(err), 98)
        ax = axes[row, 4]
        im = ax.imshow(apply_mask(err), origin='lower', cmap='RdBu_r',
                       vmin=-err_lim, vmax=err_lim, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f'RMSE={np.sqrt(np.nanmean(err ** 2)):.3f}', fontsize=8)

        persist_r, persist_mse = persistence_corr(fields[1], fields[2])
        axes[row, 2].set_xlabel(f'r(rec,fcst)={persist_r:.3f}  mse={persist_mse:.3f}', fontsize=8)

        # fading check: same var-ratio diagnostic as rollout_panel's per-step
        # print, but for this one-shot lag-N forecast (fields[2]) against its
        # target (fields[3]) -- lets one-shot vs compounding fading be
        # compared on equal footing instead of only ever measuring compounding.
        pred_flat = fields[2].flatten()
        tgt_flat = fields[3].flatten()
        mask = ~np.isnan(tgt_flat) & ~np.isnan(pred_flat)
        r_tf = pearsonr(tgt_flat[mask], pred_flat[mask])
        print(f'lead +{lead_steps}mo sample {idx}: var ratio={np.nanvar(fields[3]) / np.nanvar(fields[2]):.3f}  '
              f'r(tgt,fcst)={r_tf.statistic:.3f}  persistence r(rec,fcst)={persist_r:.3f}  '
              f'persistence mse(rec,fcst)={persist_mse:.3f}')

        axes[row, 0].set_ylabel(f'sample {idx}\nmonth {months[idx + LB - 1]}', fontsize=9)
        axes[row, 3].set_xlabel(f'month {months[idx + LB - 1 + lead_steps]}', fontsize=8)

    plt.suptitle(f'Reconstruction & +{lead_steps}mo Forecast (°C anomaly)', fontsize=13)
    plt.tight_layout()
    # keeps the original filename/behavior at lead_steps=1; distinct filename
    # for other horizons so this never overwrites the default forecast.png
    fname = 'forecast.png' if lead_steps == 1 else f'forecast_{lead_steps}mo.png'
    plt.savefig(f'{out_dir}/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {fname}')


def plot_loss_vs_anomaly_magnitude(viz, ds, out_dir, device):
    LB = viz.lookback
    rec_losses, fcst_losses, anomaly_mags = [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            x_seq = ds[idx]
            x_t0, x_t1 = x_seq[LB - 1], x_seq[LB]
            x_seq_b = x_seq.unsqueeze(0).to(device)

            z0 = viz.lift(x_seq_b[:, 0:LB])
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
    LB = viz.lookback
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), n_samples), dtype=int)
    Z = []
    with torch.no_grad():
        for idx in idxs:
            x0 = ds[idx][0:LB].unsqueeze(0).to(device)
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
    LB = viz.lookback
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), n_samples), dtype=int)
    Z = []
    with torch.no_grad():
        for idx in idxs:
            x0 = ds[idx][0:LB].unsqueeze(0).to(device)
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

def dedupe_conjugate_pairs(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Keep-mask selecting one representative per complex-conjugate pair
    (plus every real eigenvalue). A real K's complex eigenvalues always come
    in conjugate pairs -- same |lambda|, period, amplitude and residual, and
    since koopman_modes() only plots eigenvectors[:, i].real, and
    Re(v) == Re(conj(v)), both halves decode to the literal same panel. This
    is expected structure, not a defect -- the fix is only in which half
    gets a slot when ranking/plotting the top n_modes, not in the dynamics.
    """
    lam = eigenvalues.detach().cpu()
    N = lam.shape[0]
    keep = torch.ones(N, dtype=torch.bool)
    claimed = torch.zeros(N, dtype=torch.bool)
    for i in range(N):
        if claimed[i] or lam[i].imag.abs() < 1e-6:
            continue  # already paired off, or real (no partner to drop)
        dist = (lam - lam[i].conj()).abs()
        dist[i] = float('inf')
        dist[claimed] = float('inf')
        j = int(dist.argmin().item())
        if dist[j] < 1e-3:
            keep[j] = False
            claimed[i] = True
            claimed[j] = True
    return keep


def _plot_mode_grid(viz, eigenvalues, eigenvectors, amplitude, var_frac, residuals, z_mean, device,
                     out_dir, fname, suptitle, n_modes=None):
    """Decode and plot each of the given (already-ranked) eigenmodes as a
    spatial anomaly panel. Shared by every "grid of decoded modes" plot
    (top-by-variance, all-modes) so the panel layout /
    titling / colorbar logic lives in one place. n_modes=None plots every
    mode passed in; callers wanting a top-K subset should pass that count
    (the arrays themselves must already be sliced/ordered by the caller).
    """
    if n_modes is None:
        n_modes = eigenvalues.shape[0]
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
        title = f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}  var={100 * var_frac[i]:.1f}%  res={residuals[i]:.3f}{flag}'
        vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
        im = axes[i].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                             vmin=-vext, vmax=vext, aspect='auto')
        axes[i].set_title(title, fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        # UNNORM_DIFF_FIX: colorbar now reflects deg C, not normalized units
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='ΔSST (°C)')
    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {fname}')


def _plot_mode_phase_grid(viz, eigenvalue, eigenvector, amplitude, z_mean, device,
                           out_dir, fname, mode_label, n_phases=24):
    """Same single mode (same eigenvalue/period), swept through its own
    oscillation cycle in 360/n_phases-degree steps -- unlike _plot_mode_grid,
    which plots each of several modes at one fixed phase, this plots one
    mode at several phases so you can see how the spatial pattern rotates
    over a full period. v_rot = Re(v * e^{i*phase}) generalizes
    _plot_mode_grid's phase=0 case (eigenvector[:, i].real).
    """
    ncols = 4
    nrows = (n_phases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = axes.flatten()
    scale = amplitude.real.item()

    # pass 1: decode every phase first so the colorbar can be fixed across
    # the whole sweep -- per-panel vmin/vmax (98th percentile of that panel
    # alone) makes near-zero phases (e.g. 90/270 deg for a real mode) look
    # just as saturated as the peak-amplitude phases, hiding the actual
    # amplitude modulation/rotation this plot exists to show.
    modes = []
    for i in range(n_phases):
        deg = i * 360 / n_phases
        phase = torch.exp(1j * torch.deg2rad(torch.tensor(float(deg))))
        v_rot = (eigenvector * phase).real
        v_scaled = (z_mean + scale * v_rot.to(device)).unsqueeze(0).to(device)
        with torch.no_grad():
            mode_pos = viz.readout(v_scaled).cpu()
            mode_neg = viz.readout(z_mean.unsqueeze(0)).cpu()
        modes.append(unnorm_diff((mode_pos - mode_neg).numpy()[0, 0]))
    vext = max(np.nanpercentile(np.abs(apply_mask(m)), 98) for m in modes)

    # pass 2: plot every phase against that one shared vext
    for i, (mode, deg) in enumerate(zip(modes, (i * 360 / n_phases for i in range(n_phases)))):
        im = axes[i].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                             vmin=-vext, vmax=vext, aspect='auto')
        axes[i].set_title(f'{deg:.0f}°', fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='ΔSST (°C)')
    for j in range(n_phases, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(mode_label, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {fname}')


def koopman_modes(viz, arr, device, out_dir, n_modes, n_train, batch_size=BATCH_SIZE):
    LB = viz.lookback
    K_matrix = viz.K.detach().cpu()
    eigenvalues, eigenvectors, residuals = resdmd_residuals(
        K_matrix, viz.G.cpu(), viz.A.cpu(), viz.L.cpu()
    )
    n_spurious = (residuals > RESID_TOL).sum().item()
    print(f'ResDMD residuals: {n_spurious}/{len(residuals)} eigenpairs above '
          f'RESID_TOL={RESID_TOL} (flagged as likely spurious, not dropped)')

    full_ds = ORASDataset(arr, steps=0, lookback=LB)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = viz.lift(x_seq[:, 0:LB].to(device))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)

    V = eigenvectors
    Vinv = torch.linalg.inv(V)
    b = Vinv @ Z.T.to(torch.complex64)
    # PHASE_FIX: pin eig()'s arbitrary per-mode phase to this mode's own peak-loading location,
    # using only arr[:n_train] (train period) so the sign choice never sees held-out val/test data.
    # Z's row i is the state at physical time i+LB-1 (full_ds windows of length LB,
    # steps=0), so the train-only slice of b/arr must be shifted by LB-1 to match.
    n_train_eff = n_train - LB + 1
    ref = _peak_loading_ref(V, viz, arr[LB - 1:n_train], device)
    eigenvectors, b = _pin_mode_phase(eigenvectors, b, ref, n_train_eff)
    var_frac, amplitude = mode_variance_explained(viz, eigenvalues, eigenvectors, b, arr, device)

    # rank by variance explained, but only among one representative per
    # conjugate pair -- otherwise both halves of an oscillatory mode get
    # ranked (and plotted) as if they were two distinct modes
    keep_mask = dedupe_conjugate_pairs(eigenvalues)
    ranked = var_frac.clone()
    ranked[~keep_mask] = -float('inf')
    order = ranked.argsort(descending=True)
    n_modes = min(n_modes, int(keep_mask.sum().item()))
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    amplitude    = amplitude[order]
    var_frac     = var_frac[order]
    residuals    = residuals[order]
    b            = b[order]
    z_mean = Z.mean(dim=0).to(torch.float32).to(device)
    print('top modes by variance explained: ' +
          '  '.join(f'{i + 1}:{100 * var_frac[i]:.1f}%' for i in range(n_modes)))

    _plot_mode_grid(viz, eigenvalues, eigenvectors, amplitude, var_frac, residuals, z_mean, device,
                     out_dir, 'koopman_modes.png', 'Koopman Modes (decoded eigenvectors of fitted K)',
                     n_modes=n_modes)

    # [VIZ] phase_sweep_modes (1-indexed) by variance rank, each swept
    # through its own oscillation cycle at 15-degree steps (24 panels) --
    # same mode/period throughout, unlike koopman_modes.png which shows
    # different modes at one fixed phase each
    for i in [idx for idx in PHASE_SWEEP_MODES if idx < n_modes]:
        lam = eigenvalues[i]
        angle = lam.angle().abs().item()
        period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
        label = f'Mode {i+1} phase sweep  |λ|={lam.abs():.3f}  T={period_str}  var={100 * var_frac[i]:.1f}%'
        _plot_mode_phase_grid(viz, lam, eigenvectors[:, i], amplitude[i], z_mean, device,
                               out_dir, f'mode_phase_sweep_mode{i+1}.png', label, n_phases=24)

    # every deduped mode (not just the top-n_modes by variance), so spurious/
    # low-variance modes can be eyeballed too instead of trusting the
    # variance/ENSO-band ranking alone -- eigenvalues/eigenvectors/etc. are
    # already sorted variance-descending with duplicates pushed to the end,
    # so this is just "how many of the front slice are real deduped modes"
    n_modes_all = int(keep_mask.sum().item())
    _plot_mode_grid(viz, eigenvalues, eigenvectors, amplitude, var_frac, residuals, z_mean, device,
                     out_dir, 'koopman_modes_all.png',
                     f'All {n_modes_all} Koopman Modes (ranked by variance explained, decoded)',
                     n_modes=n_modes_all)

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
    sc = ax.scatter(mode_periods, 100 * var_frac.numpy(), c=enso_frac, cmap='viridis', s=30)
    ax.axvspan(2, 7, color='orange', alpha=0.15, label='ENSO band (2-7yr)')
    ax.set_xscale('log')
    ax.set_xlabel('mode period (yr, from eigenvalue angle)')
    ax.set_ylabel('SST anomaly variance explained (%)')
    ax.set_title('Koopman modes: variance explained vs. period, colored by ENSO-band power fraction')
    plt.colorbar(sc, ax=ax, label='fraction of b(t) power in 2-7yr band')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/mode_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved mode_spectrum.png')

    # same conjugate-pair issue as the variance ranking above -- restrict
    # to one representative per pair before picking the top n_enso
    keep_mask_np = keep_mask[order].numpy()
    n_enso = min(4, int(keep_mask_np.sum()))
    enso_power_ranked = np.where(keep_mask_np, enso_power, -np.inf)
    enso_order = np.argsort(enso_power_ranked)[::-1][:n_enso]

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
                 f'var={100 * var_frac[i]:.1f}%  ENSO frac={enso_frac[i]:.2f}  res={residuals[i]:.3f}{flag}')

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

    return n_spurious, len(residuals), residuals.mean().item(), residuals.max().item(), eigenvalues

def reconstruction_skill(viz, ds, unnorm, apply_mask, tmask, out_dir, n_samples, device):
    LB = viz.lookback
    sample_indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    samples_diff, samples_input, ss_scores = [], [], []

    for idx in sample_indices:
        x_seq = ds[idx]
        x_t0 = x_seq[LB - 1]
        x_seq_b = x_seq.unsqueeze(0).to(device)
        with torch.no_grad():
            x_rec = viz.readout(viz.lift(x_seq_b[:, 0:LB]))

        input_field = unnorm(x_t0.numpy()[0])
        recon_field = unnorm(x_rec.cpu().numpy()[0, 0])
        diff_field  = input_field - recon_field

        samples_diff.append(diff_field)
        samples_input.append(input_field)
        ss_scores.append(1 - (np.nanmean(diff_field**2) / np.nanvar(input_field)))

    print(f'mean SS={np.mean(ss_scores):.4f}')

    diff_stack  = np.stack(samples_diff,  axis=0)
    input_stack = np.stack(samples_input, axis=0)

    ocean_mask = ~np.isnan(tmask)

    diff_ocean  = diff_stack[:, ocean_mask]
    input_ocean = input_stack[:, ocean_mask]

    # r is corr(residual, input) per cell -- NOT reconstruction fidelity.
    # It diagnoses whether error still tracks the input signal (model hasn't
    # absorbed it), so LOWER is better; r near 0 means the residual is
    # unstructured noise rather than unexplained signal.
    r = pearsonr(diff_ocean, input_ocean, axis=0)
    r_values = r.statistic
    print(f"Mean residual-vs-input r (lower=better): {np.nanmean(r_values):.3f}")
    print(f"Median residual-vs-input r (lower=better): {np.nanmedian(r_values):.3f}")
    print(f"Std residual-vs-input r: {np.nanstd(r_values):.3f}")
    print(f"% cells with residual-vs-input r > 0.5 (lower=better): {(r_values > 0.5).mean() * 100:.1f}%")

    return {
        'mean_ss': float(np.mean(ss_scores)),
        'mean_r': float(np.nanmean(r_values)),
        'median_r': float(np.nanmedian(r_values)),
        'std_r': float(np.nanstd(r_values)),
        'pct_r_gt_half': float((r_values > 0.5).mean() * 100),
    }


def print_stats(model, viz):
    total_params = sum(p.numel() for p in model.parameters())
    dict_params = sum(p.numel() for p in model.dictionary.parameters())
    print(f"params: dictionary(conv+readout)={dict_params}, total={total_params}")
    print(f"dictionary dim (N): {model.dictionary.n_dict}")
    return dict_params, total_params


# --------------------------------------------------------------------------
# rollout skill plots
# --------------------------------------------------------------------------
def eval_rollout(model, loader, rollout_steps, device, label):
    LB = model.lookback
    step_losses = [0.0] * rollout_steps
    step_counts = [0]   * rollout_steps
    rec_loss_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_seq in loader:
            x_seq = x_seq.to(device)
            z0    = model.lift(x_seq[:, 0:LB])
            x_rec = model.readout(z0)
            rec_loss_total += nn.functional.l1_loss(x_rec, x_seq[:, LB - 1]).item()
            z = z0
            for k in range(rollout_steps):
                tgt_idx = LB + k  # = (LB - 1) + (k + 1): k+1 steps past the current state
                if tgt_idx >= x_seq.shape[1]:
                    break
                z = model.advance(z)
                x_pred = model.readout(z)
                step_losses[k] += nn.functional.l1_loss(x_pred, x_seq[:, tgt_idx]).item()
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


def plot_eval_metrics_table(metrics, out_dir):
    """Single-figure summary of this run's headline eval metrics (params,
    ResDMD residual health, reconstruction skill, forecast skill), so two
    runs can be compared at a glance instead of grepping output_resdmd_viz.log.
    `metrics`: ordered {label: value_str} dict -- build it in __main__ from
    the return values of print_stats / koopman_modes / reconstruction_skill /
    eval_rollout, so this stays a pure formatter with no logic of its own.
    """
    rows = list(metrics.items())
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['metric', 'value'],
                      cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    table.scale(1, 1.4)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    ax.set_title('Eval metrics summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eval_metrics_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved eval_metrics_table.png')


def rollout_panel(model, ds, unnorm, apply_mask, rollout_steps, n_samples, device, out_dir, channel_idx=0, split_label='test'):
    LB = model.lookback
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
            z0    = model.lift(x_seq[:, 0:LB])
            x_rec = model.readout(z0)
            preds = []
            z = z0
            for _ in range(rollout_steps):
                z = model.advance(z)
                preds.append(model.readout(z))

        inp  = unnorm(x_seq[0, LB - 1, channel_idx].cpu().numpy(), channel_idx)
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
            # persistence check: does this rollout step's forecast actually
            # differ from just persisting the reconstruction forward, or is
            # K barely doing anything? -- independent of whether a target
            # frame exists for this step, unlike the accuracy metrics below
            persist_r, persist_mse = persistence_corr(rec, pred)
            axes[row, 2 + k * 2].set_xlabel(f'r(rec,fcst)={persist_r:.2f}\nmse={persist_mse:.2f}', fontsize=7)
            tgt_idx = LB + k  # = (LB - 1) + (k + 1)
            if tgt_idx < x_seq.shape[1]:
                tgt = unnorm(x_seq[0, tgt_idx, channel_idx].cpu().numpy(), channel_idx)
                show(axes[row, 2 + k * 2 + 1], tgt)
                tgt_flat = tgt.flatten()
                pred_flat = pred.flatten()
                mask = ~np.isnan(tgt_flat) & ~np.isnan(pred_flat)
                r = pearsonr(tgt_flat[mask], pred_flat[mask])
                print(f'rollout step {k}: var ratio={np.nanvar(tgt) / np.nanvar(pred):.3f}  '
                      f'r(tgt,fcst)={r.statistic:.3f}  persistence r(rec,fcst)={persist_r:.3f}  '
                      f'persistence mse(rec,fcst)={persist_mse:.3f}')
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

    dict_params, total_params = print_stats(model, viz)

    # reconstruction / eigenmode plots -> {out}/figs
    plot_dict_variance_vs_eigval(viz, val_ds, DEVICE, OUT_DIR)
    forecast(viz, val_ds, unnorm, apply_mask, OUT_DIR, N_SAMPLES, DEVICE, val_months)
    plot_loss_vs_anomaly_magnitude(viz, val_ds, OUT_DIR, DEVICE)
    n_spurious, n_eigs, resid_mean, resid_max, eigenvalues = koopman_modes(viz, arr, DEVICE, OUT_DIR, N_MODES, train_arr.shape[0])
    skill = reconstruction_skill(viz, val_ds, unnorm, apply_mask, tmask, OUT_DIR, N_SAMPLES, DEVICE)
    plot_physical_variance_vs_eigval(viz, val_ds, DEVICE, OUT_DIR)

    # rollout skill plots -> {out}/eval
    val_loader  = DataLoader(val_ds_roll,  batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds_roll, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    val_rec,  val_steps  = eval_rollout(viz, val_loader,  ROLLOUT_STEPS, DEVICE, 'val')
    test_rec, test_steps = eval_rollout(viz, test_loader, ROLLOUT_STEPS, DEVICE, 'test')
    plot_forecast_skill(val_steps, test_steps, val_rec, test_rec, ROLLOUT_STEPS, EVAL_DIR)

    for k in range(len(VARS)):
        rollout_panel(viz, val_ds_roll, unnorm, apply_mask, ROLLOUT_STEPS, N_SAMPLES, DEVICE, EVAL_DIR, k, split_label='val')

    # eval metrics summary table -> {out}/figs
    metrics = {
        'run': out,
        'n_dict': str(N_DICT),
        'dictionary params': f'{dict_params:,}',
        'total params': f'{total_params:,}',
        'ResDMD spurious eigenpairs': f'{n_spurious}/{n_eigs} (tol={RESID_TOL})',
        'ResDMD residual mean': f'{resid_mean:.4f}',
        'ResDMD residual max': f'{resid_max:.4f}',
        'recon mean SS (1 - MSE/var, higher=better)': f'{skill["mean_ss"]:.4f}',
        'residual-vs-input mean r (lower=better)': f'{skill["mean_r"]:.3f}',
        'residual-vs-input median r (lower=better)': f'{skill["median_r"]:.3f}',
        'residual-vs-input std r': f'{skill["std_r"]:.3f}',
        'residual-vs-input % cells r>0.5 (lower=better)': f'{skill["pct_r_gt_half"]:.1f}%',
        'val rec loss (L1)': f'{val_rec:.5f}',
        'test rec loss (L1)': f'{test_rec:.5f}',
    }
    for k, l in enumerate(val_steps):
        metrics[f'val rollout step {k + 1} loss (L1)'] = f'{l:.5f}'
    for k, l in enumerate(test_steps):
        metrics[f'test rollout step {k + 1} loss (L1)'] = f'{l:.5f}'
    plot_eval_metrics_table(metrics, OUT_DIR)
