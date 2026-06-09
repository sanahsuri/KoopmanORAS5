import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from models import KoopmanNet
from dataset import load_sst, remove_climatology, normalize, split, SSTDataset


# ── config ────────────────────────────────────────────────────────────────────
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

# ── load data ─────────────────────────────────────────────────────────────────
arr, months = load_sst(ZARR_PATH, start=START, end=END)
arr, _clim = remove_climatology(arr, months)
arr, sst_mean, sst_std = normalize(arr)
_, _, test_arr = split(arr)
test_ds = SSTDataset(test_arr)

Y, X = arr.shape[1], arr.shape[2]

# land mask — 0 where land, 1 where ocean
TMASK_PATH = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/tmask_p.zarr' #ZARR_PATH.replace('opa0/sosstsst_na.zarr', 'tmask_crop.zarr')
try:
    tmask = xr.open_zarr(TMASK_PATH)['tmaskutil'].isel(t=0)
    tmask = tmask.sel(y=slice(0, 682), x=slice(0, 679)).values.astype(float)
    tmask[tmask == 0] = np.nan
except Exception:
    tmask = np.ones((Y, X))  # fallback if tmask not found

# ── load model ────────────────────────────────────────────────────────────────
model = KoopmanNet(Y, X, latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()


def unnorm(x):
    return x * sst_std + sst_mean


def apply_mask(field):
    return field * tmask


# ── 1. SST reconstruction & forecast panels ───────────────────────────────────
sample_indices = np.linspace(0, len(test_ds) - 1, N_SAMPLES, dtype=int)

fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(18, N_SAMPLES * 3.5))
col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Forecast $x_{t_1}$', 'Target $x_{t_1}$']

for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=12)

for row, idx in enumerate(sample_indices):
    x_t0, x_t1 = test_ds[idx]
    x_t0_b = x_t0.unsqueeze(0).to(DEVICE)
    x_t1_b = x_t1.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_rec, x_pred, *_ = model(x_t0_b, x_t1_b)

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


# ── 2. Koopman modes ──────────────────────────────────────────────────────────
K_matrix = model.K.weight.detach().cpu()  # (latent_dim, latent_dim)
eigenvalues, eigenvectors = torch.linalg.eig(K_matrix)

# sort by |eigenvalue| descending — most persistent modes first
order = eigenvalues.abs().argsort(descending=True)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]  # columns are eigenvectors

ncols = 3
nrows = (N_MODES + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for i in range(N_MODES):
    v_real = eigenvectors[:, i].real.unsqueeze(0).to(DEVICE)  # (1, latent_dim)

    with torch.no_grad():
        mode = model.decode(v_real).cpu().numpy()[0, 0]  # (Y, X)

    lam = eigenvalues[i]
    title = f'mode {i+1}  |λ|={lam.abs():.3f}  ∠λ={lam.angle():.2f}'

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


# ── 3. Eigenvalue spectrum ────────────────────────────────────────────────────
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
