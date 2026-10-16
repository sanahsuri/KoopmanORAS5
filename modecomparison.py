"""
Compare discovered Koopman modes against known climate indices (ENSO, PDO,
SAM, NAO, AMO, ...), two independent ways, for every (discovered mode, known
index) pair listed in config.ini's [MODECOMPARISON] section:

  1. Temporal: Pearson r between the mode's own phase-pinned amplitude time
     series Re(b_i(t)) and the known index, checked over a small lag window
     (not just lag 0, since the model's phase convention needn't match the
     index's own).
  2. Spatial:
     (a) static -- correlate the mode's decoded spatial pattern (same
         snapshot koopman_modes.png plots) against an independently-built
         teleconnection map: at every pixel, the correlation between
         *observed* SST (not model output) and the known index. No model
         involvement on that side at all.
     (b) dynamic -- decode the mode's own time-varying contribution
         Re(b_i(t) * v_i) at every pixel and correlate each pixel's own
         time series against the index, producing a per-pixel correlation
         map. Unlike (a), a pixel's value here reflects both real
         teleconnection strength *and* that pixel's own phase lag within
         the mode (traveling-wave modes have different pixels peak at
         different phases) -- a low/negative value doesn't necessarily
         mean "unrelated", it can mean "related but phase-shifted".

Known indices are read from `{MODECOMPARISON.modes_dir}/{name}.zarr`, each a
1D zarr store with a `time_counter` (datetime64) coordinate and one data
variable named `name` -- the same format all of pdo/sam/oni/nao/amo/enso
already use.

Reuses resdmd_visualize.py's module-level setup (config, data loading,
model construction, and the globally-fit K/C_readout in `viz`) by importing
it -- run this the same way that script is run, e.g. from the run's own
directory:
    python3 modecomparison.py -f config.ini
"""
import ast
import os
import sys

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import resdmd_visualize as rv

MODES_DIR = rv.cfg.get('MODECOMPARISON', 'modes_dir',
                        fallback='/expanse/lustre/projects/ucd250/kringel/modes')
KNOWN_MODES = ast.literal_eval(rv.cfg.get('MODECOMPARISON', 'known_modes', fallback="['enso']"))
DISCOVERED_MODES = ast.literal_eval(rv.cfg.get('MODECOMPARISON', 'discovered_modes', fallback='[5]'))
MAX_LAG_MONTHS = rv.cfg.getint('MODECOMPARISON', 'max_lag_months', fallback=3)


def load_known_mode(name, modes_dir=MODES_DIR):
    """{(year, month): value} from {modes_dir}/{name}.zarr -- a 1D zarr
    store with a time_counter (datetime64) coord and a `name` data var."""
    ds = xr.open_zarr(os.path.join(modes_dir, f'{name}.zarr'))
    years = ds.time_counter.dt.year.values
    months = ds.time_counter.dt.month.values
    values = ds[name].values
    return {(int(y), int(m)): float(v) for y, m, v in zip(years, months, values) if not np.isnan(v)}


def calendar_months(start, n_months):
    """[(year, month), ...] of length n_months, starting at 'YYYY-MM'."""
    y, m = (int(p) for p in start.split('-'))
    out = []
    for _ in range(n_months):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def compute_all_modes(viz, arr, device, n_train, batch_size):
    """Same eigen-decomposition/ranking/phase-pin as koopman_modes(), minus
    the plotting -- the expensive shared part, computed once regardless of
    how many discovered_modes are requested. Returns amplitude-ranked
    eigenvalues/eigenvectors/amplitude/b plus z_mean and how many deduped
    modes exist."""
    K_matrix = viz.K.detach().cpu()
    eigenvalues, eigenvectors, _residuals = rv.resdmd_residuals(
        K_matrix, viz.G.cpu(), viz.A.cpu(), viz.L.cpu()
    )

    # LB: viz.lift() expects a window with the lookback dim intact (B, LB, C, H, W)
    # -- x_seq[:, 0] (dropping to (B, C, H, W)) and ORASDataset's default lookback=1
    # both silently assumed lookback=1 everywhere; match koopman_modes()/
    # mode_persistence.py's LB-aware pattern instead so this doesn't break for
    # models trained with lookback > 1.
    LB = viz.lookback
    full_ds = rv.ORASDataset(arr, steps=0, lookback=LB)
    full_loader = rv.DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = viz.lift(x_seq[:, 0:LB].to(device))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)

    V = eigenvectors
    Vinv = torch.linalg.inv(V)
    b = Vinv @ Z.T.to(torch.complex64)
    n_train_eff = n_train - LB + 1
    ref = rv._peak_loading_ref(V, viz, arr[LB - 1:n_train], device)
    eigenvectors, b = rv._pin_mode_phase(eigenvectors, b, ref, n_train_eff)
    amplitude = b.abs().std(dim=1)

    keep_mask = rv.dedupe_conjugate_pairs(eigenvalues)
    ranked = amplitude.clone()
    ranked[~keep_mask] = -float('inf')
    order = ranked.argsort(descending=True)

    return {
        'eigenvalues': eigenvalues[order],
        'eigenvectors': eigenvectors[:, order],
        'amplitude': amplitude[order],
        'b': b[order],
        'z_mean': Z.mean(dim=0).to(torch.float32).to(device),
        'n_kept': int(keep_mask.sum().item()),
    }


def pick_mode(all_modes, mode_rank):
    """1-indexed by amplitude rank -- cheap slice of compute_all_modes()'s output."""
    if mode_rank > all_modes['n_kept']:
        raise ValueError(f"mode_rank={mode_rank} exceeds {all_modes['n_kept']} deduped modes")
    idx = mode_rank - 1
    return (all_modes['eigenvalues'][idx], all_modes['eigenvectors'][:, idx],
            all_modes['amplitude'][idx], all_modes['b'][idx], all_modes['z_mean'])


N_PHASE_STEPS = 24  # 15-degree increments, matches resdmd_visualize.py's mode_phase_sweep convention


def best_rotation_against_map(pattern0, pattern90, target_flat, mask):
    """Search N_PHASE_STEPS phase offsets of the mode's pattern
    (cos(theta)*pattern0 + sin(theta)*pattern90) against a fixed target map
    (here, the known index's own observed teleconnection map), returning
    the (rotated pattern, signed r, theta_deg) at whichever phase maximizes
    |r| -- so the reported/plotted spatial pattern is whichever point in the
    mode's own oscillation cycle actually looks most like the real index,
    not whatever arbitrary phase _pin_mode_phase happened to pin it to.
    """
    best_r, best_theta, best_pattern = 0.0, 0.0, pattern0
    for step in range(N_PHASE_STEPS):
        theta_deg = step * 360 / N_PHASE_STEPS
        theta = np.deg2rad(theta_deg)
        pattern = np.cos(theta) * pattern0 + np.sin(theta) * pattern90
        p_flat = pattern.flatten()
        r, _ = pearsonr(p_flat[mask], target_flat[mask])
        if abs(r) > abs(best_r):
            best_r, best_theta, best_pattern = r, theta_deg, pattern
    # theta and theta+180 give identical |r| with opposite sign (the pattern
    # at theta+180 is the pixel-wise negative of the one at theta) -- the
    # search above can land on either member of that tied pair. The known
    # index's own sign is NOT arbitrary (positive = actually warms during
    # e.g. El Nino), unlike the mode's own _pin_mode_phase convention, so
    # always canonicalize to the positive-r member: the displayed pattern
    # should look directly like the real teleconnection map, not its
    # photo-negative, whenever an equally-valid same-|r| option exists.
    if best_r < 0:
        best_theta = (best_theta + 180) % 360
        best_pattern = -best_pattern
        best_r = -best_r
    return best_pattern, best_r, best_theta


def vectorized_pixel_corr(field_t, index_t):
    """field_t: (T, H, W) real. index_t: (T,) real, same T, same order.
    Returns (H, W) Pearson r per pixel via the closed-form correlation
    formula -- looping scipy.stats.pearsonr over ~460k pixels would take
    minutes; this is one vectorized pass."""
    T, H, W = field_t.shape
    x = field_t.reshape(T, -1)
    y = index_t
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = (xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    return r.reshape(H, W)


def temporal_comparison(b_row, dates, index, max_lag):
    """Pearson r between Re(b_row(t)) and the known index, at each lag in
    [-max_lag, +max_lag] months (b_row[i] paired against index at
    dates[i + lag]). Returns (lags, r_values, best_lag, best_r)."""
    b_series = b_row.real.numpy()
    lags = list(range(-max_lag, max_lag + 1))
    r_values = []
    for lag in lags:
        b_vals, n_vals = [], []
        for i in range(len(dates)):
            j = i + lag
            if 0 <= j < len(dates) and dates[j] in index:
                b_vals.append(b_series[i])
                n_vals.append(index[dates[j]])
        if len(b_vals) < 10:
            r_values.append(np.nan)
            continue
        r, _ = pearsonr(b_vals, n_vals)
        r_values.append(r)
    finite = [i for i, r in enumerate(r_values) if not np.isnan(r)]
    best_i = max(finite, key=lambda i: abs(r_values[i]))
    return lags, r_values, lags[best_i], r_values[best_i]


def save_plots(mode_rank, known_mode, mode_pattern, observed_map, dynamic_map, static_r,
               dynamic_r, b_row, dates, index_series, lags, r_values, best_lag, best_r):
    out_dir = rv.EVAL_DIR
    tag = f'mode{mode_rank}_{known_mode}'

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vext = np.nanpercentile(np.abs(rv.apply_mask(mode_pattern)), 98)
    im0 = axes[0].imshow(rv.apply_mask(mode_pattern), origin='lower', cmap='RdBu_r',
                          vmin=-vext, vmax=vext, aspect='auto')
    axes[0].set_title(f'Mode {mode_rank} decoded pattern')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='degC')
    # different diverging map from the degC mode pattern: this panel is a
    # correlation (r), not a temperature, and shouldn't read as one
    im1 = axes[1].imshow(rv.apply_mask(observed_map), origin='lower', cmap='PuOr_r',
                          vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title(f'Observed SST vs {known_mode.upper()} (r_pattern={static_r:.3f})')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='r')
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'Static spatial comparison -- mode {mode_rank} vs {known_mode.upper()}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{tag}_static_spatial.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # PuOr_r to match the observed r map -- this is a correlation too
    im = ax.imshow(rv.apply_mask(dynamic_map), origin='lower', cmap='PuOr_r',
                    vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Mode {mode_rank}: per-pixel corr(own contribution, {known_mode.upper()})\n'
                 f'(vs observed map: r={dynamic_r:.3f})', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='r')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{tag}_dynamic_spatial.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    b_norm = b_row.real.numpy()
    valid = ~np.isnan(index_series)
    b_norm = (b_norm - b_norm[valid].mean()) / b_norm[valid].std()
    n_norm = (index_series - np.nanmean(index_series)) / np.nanstd(index_series)
    years = [y + (m - 1) / 12 for y, m in dates]
    axes[0].plot(years, b_norm, label=f'mode {mode_rank} contribution coefficient Re(b_{mode_rank}(t)), standardized')
    axes[0].plot(years, n_norm, label=f'{known_mode.upper()}, standardized', alpha=0.8)
    axes[0].set_xlabel('year'); axes[0].legend()
    axes[0].set_title(f'Mode {mode_rank} vs {known_mode.upper()} (lag 0)')
    axes[1].plot(lags, r_values, marker='o')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_xlabel('lag (months, + = mode leads)'); axes[1].set_ylabel('r')
    axes[1].set_title(f'best: lag {best_lag:+d}, r={best_r:.3f}')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{tag}_temporal.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = rv.DEVICE
    n_train = rv.train_arr.shape[0]
    channel_idx = 0  # opa0 ensemble member, matches forecast()/rollout_panel()'s convention
    dates = calendar_months(rv.START, rv.arr.shape[0])

    print(f'discovered_modes={DISCOVERED_MODES}  known_modes={KNOWN_MODES}  '
          f'max_lag_months={MAX_LAG_MONTHS}')
    print('computing eigendecomposition...')
    all_modes = compute_all_modes(rv.viz, rv.arr, device, n_train, rv.BATCH_SIZE)
    print(f'{all_modes["n_kept"]} deduped modes available')

    for mode_rank in DISCOVERED_MODES:
        eigenvalue, eigenvector, amp, b_row, z_mean = pick_mode(all_modes, mode_rank)
        angle = eigenvalue.angle().abs().item()
        period_str = f'{2 * np.pi / angle / 12:.2f}yr' if angle > 1e-4 else 'inf'
        print(f'\n=== mode {mode_rank}: |lambda|={eigenvalue.abs():.3f}  '
              f'T={period_str}  amp={amp:.3f} ===')

        # decoded pattern at phase 0deg (Re(v)) and 90deg (Re(v*i) = -Im(v)) --
        # same closed-form trick as mode_persistence.py's rotated_pattern():
        # decode+unnorm_diff is linear in the eigenvector, so any other phase's
        # pattern is exactly cos(theta)*pattern0 + sin(theta)*pattern90. Used
        # below to align the mode against each known_mode's OWN observed
        # teleconnection map, rather than reusing the fixed theta=0 snapshot
        # _pin_mode_phase happens to have pinned it to -- that fixed phase has
        # no reason to be the one that best resembles a given real index.
        scale = amp.real.item()
        v_real = eigenvector.real
        v_imag = eigenvector.imag
        with torch.no_grad():
            mode_neg = rv.viz.readout(z_mean.unsqueeze(0)).cpu()
            v0 = (z_mean + scale * v_real.to(device)).unsqueeze(0).to(device)
            mode_pos0 = rv.viz.readout(v0).cpu()
            v90 = (z_mean + scale * (-v_imag).to(device)).unsqueeze(0).to(device)
            mode_pos90 = rv.viz.readout(v90).cpu()
        pattern0 = rv.unnorm_diff((mode_pos0 - mode_neg).numpy()[0, channel_idx], channel_idx)
        pattern90 = rv.unnorm_diff((mode_pos90 - mode_neg).numpy()[0, channel_idx], channel_idx)

        # mode's own time-varying contribution at every pixel -- also shared across known_modes
        C_readout_c = rv.viz.C_readout.to(torch.complex64).to(device)
        v_decoded = (eigenvector.to(torch.complex64).to(device) @ C_readout_c.T).cpu()
        v_decoded = v_decoded.view(rv.C, rv.Y, rv.X)[channel_idx]
        std_field = rv.std.squeeze(axis=0)[channel_idx]
        contribution = (b_row.view(-1, 1, 1) * v_decoded.view(1, rv.Y, rv.X)).real.numpy()
        contribution = contribution * std_field

        for known_mode in KNOWN_MODES:
            index = load_known_mode(known_mode)
            index_series = np.array([index.get(dm, np.nan) for dm in dates])
            valid = ~np.isnan(index_series)
            if valid.sum() < 24:
                print(f'  [{known_mode}] skipped -- only {valid.sum()} overlapping months')
                continue
            print(f'  [{known_mode}] {valid.sum()}/{len(dates)} months matched')

            lags, r_values, best_lag, best_r = temporal_comparison(b_row, dates, index, MAX_LAG_MONTHS)
            lag0_r = r_values[MAX_LAG_MONTHS]
            print(f'  [{known_mode}] temporal: best lag {best_lag:+d} r={best_r:.3f}  (lag 0: r={lag0_r:.3f})')

            observed_field = rv.plot_arr[valid, channel_idx]
            observed_map = vectorized_pixel_corr(observed_field, index_series[valid])
            om_flat = rv.apply_mask(observed_map).flatten()
            p0_flat = rv.apply_mask(pattern0).flatten()
            p90_flat = rv.apply_mask(pattern90).flatten()
            mask = ~np.isnan(p0_flat) & ~np.isnan(p90_flat) & ~np.isnan(om_flat)
            mode_pattern, static_r, static_theta = best_rotation_against_map(
                rv.apply_mask(pattern0), rv.apply_mask(pattern90), om_flat, mask)
            print(f'  [{known_mode}] static spatial: r={static_r:.3f}  '
                  f'(mode shown at its best-aligning phase, {static_theta:.0f} deg from pinned theta=0)')

            dynamic_map = vectorized_pixel_corr(contribution[valid], index_series[valid])
            dm_flat = rv.apply_mask(dynamic_map).flatten()
            mask2 = ~np.isnan(dm_flat) & ~np.isnan(om_flat)
            dynamic_r, _ = pearsonr(dm_flat[mask2], om_flat[mask2])
            print(f'  [{known_mode}] dynamic spatial: r={dynamic_r:.3f}')

            save_plots(mode_rank, known_mode, mode_pattern, observed_map, dynamic_map, static_r,
                       dynamic_r, b_row, dates, index_series, lags, r_values, best_lag, best_r)

    print('\ndone')


if __name__ == '__main__':
    log_path = os.path.join(rv.out, 'output_modecomparison.log')
    sys.stdout = open(log_path, 'w', buffering=1)
    sys.stderr = sys.stdout
    main()
