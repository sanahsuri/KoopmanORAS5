import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.data import DataLoader

from models import KoopmanNet
from dataset import load_sst, remove_climatology, normalize, split, SSTDataset

import configparser
cfg = configparser.ConfigParser()
cfg.read('config.ini')

ZARR_PATH  = cfg['DATA']['zarr_path']
START      = cfg['DATA'].get('start', None)
END        = cfg['DATA'].get('end', None)
LATENT_DIM = cfg.getint('MODEL', 'latent_dim')
CHANNELS   = tuple(int(c) for c in cfg['MODEL']['channels'].split(','))
MODEL_PATH = 'runs/best_model.pt'
OUT_DIR    = 'figs'
N_MODES    = 9      # how many Koopman modes to plot
N_SAMPLES  = 4      # how many forecast panels to plot

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arr, months = load_sst(ZARR_PATH, start=START, end=END)
arr, _clim = remove_climatology(arr, months)
arr, sst_mean, sst_std = normalize(arr)
_, _, test_arr = split(arr)
test_ds = SSTDataset(test_arr)

Y, X = arr.shape[1], arr.shape[2]

TMASK_PATH = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/tmask_p.zarr' #ZARR_PATH.replace('opa0/sosstsst_na.zarr', 'tmask_crop.zarr')
try:
    tmask = xr.open_zarr(TMASK_PATH)['tmaskutil'].isel(t=0)
    tmask = tmask.sel(y=slice(0, 682), x=slice(0, 679)).values.astype(float)
    tmask[tmask == 0] = np.nan
except Exception:
    tmask = np.ones((Y, X))  # fallback if tmask not found

model = KoopmanNet(Y, X, latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()


def unnorm(x):
    return x * sst_std + sst_mean


def apply_mask(field):
    return field * tmask

# sst forecast 
sample_indices = np.linspace(0, len(test_ds) - 1, N_SAMPLES, dtype=int)

fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(18, N_SAMPLES * 3.5))
col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Forecast $x_{t_1}$', 'Target $x_{t_1}$']

for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=12)

for row, idx in enumerate(sample_indices):
    x_seq = test_ds[idx]                       # (2, 1, Y, X)
    x_t0, x_t1 = x_seq[0], x_seq[1]
    x_seq_b = x_seq.unsqueeze(0).to(DEVICE)    # (1, 2, 1, Y, X)

    with torch.no_grad():
        x_rec, x_preds, *_ = model(x_seq_b)
    x_pred = x_preds[0]

    fields = [
        unnorm(x_t0.numpy()[0]),
        unnorm(x_rec.cpu().numpy()[0, 0]),
        unnorm(x_pred.cpu().numpy()[0, 0]),
        unnorm(x_t1.numpy()[0]),
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

plt.suptitle('SST Reconstruction & One-Step Forecast (°C anomaly)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sst_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved sst_forecast.png')


# koopman modes 
K_matrix = model.K.weight.detach().cpu()  # (latent_dim, latent_dim)
eigenvalues, eigenvectors = torch.linalg.eig(K_matrix)

# encode the full latent trajectory and project onto K's eigenvectors to get
# each mode's time-coefficient b(t) = V^-1 z(t). Rank by the RMS amplitude of
# b(t) — i.e. how much each mode actually contributes to the observed
# variability — rather than by |eigenvalue| (persistence) alone.
full_ds = SSTDataset(arr, steps=0)
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
Z = []
with torch.no_grad():
    for x_seq in full_loader:
        z = model.encode(x_seq[:, 0].to(DEVICE))
        Z.append(z.cpu())
Z = torch.cat(Z, dim=0)  # (T, latent_dim)

V    = eigenvectors                          # (latent_dim, latent_dim) complex
Vinv = torch.linalg.inv(V)
b    = Vinv @ Z.T.to(torch.complex64)        # (latent_dim, T)
amplitude = b.abs().std(dim=1)               # RMS variability of each mode's coefficient

# sort by amplitude descending — most energetic modes first
order = amplitude.argsort(descending=True)
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]  # columns are eigenvectors
amplitude    = amplitude[order]
b            = b[order]                # (latent_dim, T), reordered to match

ncols = 3
nrows = (N_MODES + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for i in range(N_MODES):
    v_real = eigenvectors[:, i].real.unsqueeze(0).to(DEVICE)  # (1, latent_dim)

    with torch.no_grad():
        mode = model.decode(v_real).cpu().numpy()[0, 0]  # (Y, X)

    lam = eigenvalues[i]
    angle = lam.angle().abs().item()
    period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
    title = f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}  amp={amplitude[i]:.2f}'

    vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
    im = axes[i].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                        vmin=-vext, vmax=vext, aspect='auto')
    axes[i].set_title(title, fontsize=9)
    axes[i].set_xticks([]); axes[i].set_yticks([])
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

for j in range(N_MODES, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Koopman Modes (decoded eigenvectors of K)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/koopman_modes.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved koopman_modes.png')


# power spectrum of modes
T = b.shape[1]
freqs = np.fft.rfftfreq(T, d=1.0)               # cycles per month
periods_yr = np.full_like(freqs, np.nan)
periods_yr[1:] = 1.0 / (freqs[1:] * 12)         # skip DC (freq=0)

power = np.abs(np.fft.rfft(b.real.numpy(), axis=1)) ** 2  # (latent_dim, n_freq)

enso_band   = (periods_yr >= 2) & (periods_yr <= 7)
enso_power  = power[:, enso_band].sum(axis=1)
total_power = power[:, 1:].sum(axis=1)          # exclude DC
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
plt.savefig(f'{OUT_DIR}/mode_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved mode_spectrum.png')

# decode and plot the modes whose coefficient b(t) carries the most ENSO-band power
N_ENSO = 4
enso_order = np.argsort(enso_power)[::-1][:N_ENSO]

fig, axes = plt.subplots(1, N_ENSO, figsize=(N_ENSO * 5, 4))
for j, i in enumerate(enso_order):
    v_real = eigenvectors[:, i].real.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mode = model.decode(v_real).cpu().numpy()[0, 0]

    lam = eigenvalues[i]
    angle = lam.angle().abs().item()
    period_str = f'{2*np.pi/angle/12:.1f}yr' if angle > 1e-4 else '∞'
    title = (f'mode {i+1}  |λ|={lam.abs():.3f}  T={period_str}\n'
             f'amp={amplitude[i]:.2f}  ENSO frac={enso_frac[i]:.2f}')

    vext = np.nanpercentile(np.abs(apply_mask(mode)), 98)
    im = axes[j].imshow(apply_mask(mode), origin='lower', cmap='RdBu_r',
                        vmin=-vext, vmax=vext, aspect='auto')
    axes[j].set_title(title, fontsize=9)
    axes[j].set_xticks([]); axes[j].set_yticks([])
    plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

plt.suptitle('Koopman modes ranked by power in the ENSO band (2-7yr)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/enso_modes.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved enso_modes.png')


# eigenvalue spectrum
fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='unit circle')
eigs = eigenvalues.numpy()
ax.scatter(eigs.real, eigs.imag, c=np.abs(eigs), cmap='plasma', s=20, zorder=3)
ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
ax.set_aspect('equal')
ax.set_title('Koopman eigenvalue spectrum')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved eigenvalue_spectrum.png')
