import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import xarray as xr
import sys
import ast

from torch.utils.data import DataLoader
from models import KoopmanNet
# from dataset import load_sst, remove_climatology, normalize, split, SSTDataset
from all_data import load_fields, remove_climatology, normalize, split, ORASDataset

import configparser
cfg = configparser.ConfigParser()
cfg.read('config.ini')

def try_cast(value):
    # Tries int, then float, then returns string
    for cast in (int, float, ast.literal_eval):
        try:
            return cast(value)
        except ValueError:
            continue
    return value

print("NEW")

ZARR_PATH  = cfg['DATA']['zarr_path']
VARS       = try_cast(cfg['DATA']['vars'])
START      = cfg['DATA'].get('start', None)
END        = cfg['DATA'].get('end', None)
LATENT_DIM = cfg.getint('MODEL', 'latent_dim')
CHANNELS   = tuple(int(c) for c in cfg['MODEL']['channels'].split(','))
out = cfg['OUTPUT']['dir']
log_path = os.path.join(out, 'output_viz.log')
sys.stdout = open(log_path, 'w', buffering=1)
sys.stderr = sys.stdout
OUT_DIR    = f'{out}/figs'
MODEL_PATH = f'{out}/runs/best_model.pt'
# MODEL_PATH = f'{out}/runs/ckpt_ep100.pt'
N_MODES    = 9      # how many Koopman modes to plot
N_SAMPLES  = 4      # how many forecast panels to plot

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arr, months = load_fields(ZARR_PATH, VARS, start=START, end=END)
arr, _clim = remove_climatology(arr, months)
plot_arr = arr
arr, mean, std = normalize(arr)
_, val_arr, _ = split(arr)
val_ds = ORASDataset(val_arr)

Y, X = arr.shape[1], arr.shape[2]

TMASK_PATH = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/tmask_p.zarr' #ZARR_PATH.replace('opa0/sosstsst_na.zarr', 'tmask_crop.zarr')
try:
    tmask = xr.open_zarr(TMASK_PATH)['tmaskutil'].isel(t=0)
    tmask = tmask.sel(y=slice(0, 682), x=slice(0, 679)).values.astype(float)
    tmask[tmask == 0] = np.nan
except Exception:
    tmask = np.ones((Y, X))  # fallback if tmask not found

Y, X = arr.shape[2], arr.shape[3]
C = arr.shape[1]
model = KoopmanNet(Y, X, latent_dim=LATENT_DIM, channels=CHANNELS, in_channels=C).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

print(torch.sigmoid(model.K.mu_raw))


def unnorm(x, channel_idx=0):
    return x * std.squeeze()[channel_idx] + mean.squeeze()[channel_idx]
 
 
def apply_mask(field):
    return field * tmask

# forecast
def forecast(): 
    channel_idx = 0  # opa0
    sample_indices = np.linspace(0, len(val_ds) - 1, N_SAMPLES, dtype=int)

    fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(18, N_SAMPLES * 3.5))
    col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Forecast $x_{t_1}$', 'Target $x_{t_1}$']

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    for row, idx in enumerate(sample_indices):
        x_seq = val_ds[idx]                       # (2, 1, Y, X)
        x_t0, x_t1 = x_seq[0], x_seq[1]
        x_seq_b = x_seq.unsqueeze(0).to(DEVICE)    # (1, 2, 1, Y, X)

        with torch.no_grad():
            x_rec, x_preds, *_ = model(x_seq_b)
        x_pred = x_preds[0]

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
    plt.savefig(f'{OUT_DIR}/forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved forecast.png')

def plot_loss_vs_anomaly_magnitude(out_dir=None): 
    model.eval()
    rec_losses = []
    fcst_losses = []
    anomaly_mags = [] 
    with torch.no_grad():
        for idx in range(len(val_ds)):
            x_seq = val_ds[idx]                      # (2, 1, Y, X)
            x_t0, x_t1 = x_seq[0], x_seq[1]
            x_seq_b = x_seq.unsqueeze(0).to(DEVICE)   # (1, 2, 1, Y, X)
 
            x_rec, x_preds, *_ = model(x_seq_b)
            x_pred = x_preds[0]
 
            rec_loss = torch.nn.functional.mse_loss(
                x_rec.cpu()[0, 0], x_t0[0]
            ).item()
            fcst_loss = torch.nn.functional.mse_loss(
                x_pred.cpu()[0, 0], x_t1[0]
            ).item()
 
            rec_losses.append(rec_loss)
            fcst_losses.append(fcst_loss)
 
            # anomaly magnitude of the input frame, in normalized units
            # (matches what the model/loss actually sees, pre-unnorm)
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
    fig.savefig(f"{OUT_DIR}/loss_vs_anomaly_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss_vs_anomaly_magnitude.png')

def plot_latent_variance_vs_mu(n_batches=4):
    """
    Uses globals already defined in the viz script:
        model, test_ds (or val_ds), DEVICE, OUT_DIR

    Compares per-latent-dimension variance of the encoded representation
    against K's eigenvalue magnitude (mu) for that same dimension's block,
    to check whether high-variance latent dims are landing on high- or
    low-persistence modes.
    """
    model.eval()
    z_list = []

    with torch.no_grad():
        # grab a handful of batches' worth of encoded z_seq[0] (the t0 latent)
        for idx in np.linspace(0, len(val_ds) - 1, min(len(val_ds), n_batches * 32), dtype=int):
            x_seq = val_ds[idx].unsqueeze(0).to(DEVICE)  # (1, S+1, C, Y, X)
            z0 = model.encode(x_seq[:, 0])                 # (1, latent_dim)
            z_list.append(z0.cpu().numpy()[0])

    z_arr = np.stack(z_list, axis=0)              # (N, latent_dim)
    latent_var = z_arr.var(axis=0)                # (latent_dim,)

    # mu is defined per 2x2 block (latent_dim // 2 values); each block
    # covers two consecutive latent dimensions, so repeat mu to align
    # with latent_dim for a direct per-dimension comparison.
    mu = torch.sigmoid(model.K.mu_raw).detach().cpu().numpy()   # (latent_dim // 2,)
    mu_per_dim = np.repeat(mu, 2)                                # (latent_dim,)

    assert latent_var.shape[0] == mu_per_dim.shape[0], (
        f"latent_dim mismatch: variance has {latent_var.shape[0]} dims, "
        f"mu_per_dim has {mu_per_dim.shape[0]} — check latent_dim consistency."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(mu_per_dim, latent_var, alpha=0.6, s=18, color="steelblue")
    axes[0].set_xlabel("K eigenvalue magnitude (mu) for this dim's block")
    axes[0].set_ylabel("latent dimension variance")
    axes[0].set_title("Latent variance vs. mode persistence")
    if len(latent_var) > 1:
        corr = np.corrcoef(mu_per_dim, latent_var)[0, 1]
        axes[0].annotate(f"corr = {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                          fontsize=10, va='top')

    # Sorted bar view: which dims carry the most variance, and what's their mu?
    order = np.argsort(latent_var)[::-1]
    axes[1].bar(range(len(order)), latent_var[order], color="steelblue", alpha=0.6, label="variance")
    ax2 = axes[1].twinx()
    ax2.plot(range(len(order)), mu_per_dim[order], color="darkorange", linewidth=1.2, label="mu")
    axes[1].set_xlabel("latent dim (sorted by variance, descending)")
    axes[1].set_ylabel("variance", color="steelblue")
    ax2.set_ylabel("mu", color="darkorange")
    axes[1].set_title("Variance-ranked dims: does high variance => high mu?")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/latent_variance_vs_mu.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved latent_variance_vs_mu.png')



# koopman modes 
def koopman_modes(not_flat=True):
    K_matrix = model.K.weight.detach().cpu()  # (latent_channels, latent_channels)
    eigenvalues, eigenvectors = torch.linalg.eig(K_matrix)

    full_ds = ORASDataset(arr, steps=0)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = model.encode(x_seq[:, 0].to(DEVICE))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)  # (T, latent_channels)

    V    = eigenvectors
    Vinv = torch.linalg.inv(V)
    b    = Vinv @ Z.T.to(torch.complex64)        # (latent_channels, T)
    amplitude = b.abs().std(dim=1)

    order = amplitude.argsort(descending=True)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    amplitude    = amplitude[order]
    b            = b[order]

    ncols = 3
    nrows = (N_MODES + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()
    for i in range(N_MODES):
        v_real = eigenvectors[:, i].real
        v_real = v_real.unsqueeze(0).to(DEVICE)  # (1, latent_dim)
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

def reconstruction_skill():
    from numpy.fft import fft2, fftshift
    sample_indices = np.linspace(0, len(val_ds) - 1, N_SAMPLES, dtype=int)
    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(18, N_SAMPLES * 3.5))
    col_titles = ['Input $x_{t_0}$', 'Reconstruction', 'Difference']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    fig_fft, axes_fft = plt.subplots(N_SAMPLES, 1, figsize=(6, N_SAMPLES * 3.5))
    if N_SAMPLES == 1:
        axes_fft = [axes_fft]

    samples_diff  = []
    samples_input = []
    ss_scores     = []

    for row, idx in enumerate(sample_indices):
        x_seq   = val_ds[idx]
        x_t0    = x_seq[0]
        x_seq_b = x_seq.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x_rec, _, *_ = model(x_seq_b)

        input_field = unnorm(x_t0.numpy()[0])
        recon_field = unnorm(x_rec.cpu().numpy()[0, 0])
        diff_field  = input_field - recon_field

        diff_field_filled = np.nan_to_num(diff_field, nan=0.0)
        F = fftshift(fft2(diff_field_filled))
        ax_fft = axes_fft[row]
        im_fft = ax_fft.imshow(np.log(np.abs(F) + 1), cmap='viridis')
        ax_fft.set_title(f'FFT of residual | sample {idx}', fontsize=10)
        ax_fft.set_xticks([]); ax_fft.set_yticks([])
        plt.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

        samples_diff.append(diff_field)
        samples_input.append(input_field)
        ss_scores.append(1 - (np.nanmean(diff_field**2) / np.nanvar(input_field)))

        vmin = np.nanpercentile(input_field, 2)
        vmax = np.nanpercentile(input_field, 98)
        abs_max = np.nanpercentile(np.abs(diff_field), 98)

        # Input
        ax = axes[row, 0]
        im = ax.imshow(apply_mask(input_field), origin='lower', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Reconstruction
        ax = axes[row, 1]
        im = ax.imshow(apply_mask(recon_field), origin='lower', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Difference
        ax = axes[row, 2]
        im = ax.imshow(apply_mask(diff_field), origin='lower', cmap='RdBu_r',
                       vmin=-abs_max, vmax=abs_max, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(f'sample {idx} | SS={ss_scores[-1]:.3f}', fontsize=9)

    plt.figure(fig.number)
    plt.suptitle('Reconstruction (°C anomaly)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/reconstruction_skill.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    plt.figure(fig_fft.number)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/residual_fft.png', dpi=150, bbox_inches='tight')
    plt.close(fig_fft)

    print(f'saved reconstruction_skill.png and residual_fft.png | mean SS={np.mean(ss_scores):.4f}')

    # Stack samples and compute spatial r_map
    diff_stack  = np.stack(samples_diff,  axis=0)  # (N_SAMPLES, nlat, nlon)
    input_stack = np.stack(samples_input, axis=0)  # (N_SAMPLES, nlat, nlon)

    nlat, nlon = input_field.shape[-2], input_field.shape[-1]
    ocean_mask = ~np.isnan(tmask)
    ocean_mask_flat = ocean_mask.flatten()

    diff_ocean  = diff_stack[:, ocean_mask]   # (N_SAMPLES, n_ocean)
    input_ocean = input_stack[:, ocean_mask]  # (N_SAMPLES, n_ocean)

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
    # fig, ax = plt.subplots(figsize=(18, 3.5))
    im = ax.imshow(apply_mask(r_spatial), origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    input_variance = np.nanvar(input_stack, axis=0)  # (nlat, nlon)
    ax = axes[1]
    # fig, ax = plt.subplots(figsize=(18, 3.5))
    im = ax.imshow(apply_mask(input_variance), origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ax.set_title('Pearson r (residual vs SST anomaly)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/r_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved r_map.png')

def print_stats():
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    k_params = sum(p.numel() for p in model.K.parameters())
    bottleneck_capacity = model.latent_dim * model.Yf * model.Xf
    print(f"params: enc={encoder_params}, dec={decoder_params}, K={k_params}")
    print(f"bottleneck capacity: {bottleneck_capacity}")


if __name__ == "__main__":
    print(OUT_DIR)
    plot_latent_variance_vs_mu(n_batches=4)
    forecast()
    plot_loss_vs_anomaly_magnitude()
    koopman_modes()
    reconstruction_skill()
    # print_stats()

