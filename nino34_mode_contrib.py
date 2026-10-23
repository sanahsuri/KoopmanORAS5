"""Month-by-month contribution of each Koopman mode group to SST in the Niño 3.4 box, through El Niño 2015-16.

Same runs, modes, groups and conventions as regional_mode_share.py (N=32 edmd_dl, 20261005 dict-size recheck; global
train-fit K and C_readout; amplitudes about their FULL-RECORD mean; groups = persistence clusters with a member in
>= 4 of the size's runs, plus 'other modes'):

  x_g(p, t) = sum_{k in g} c_k * Re(b_k(t) * u_k(p)) * sigma_p              [deg C]
  box mean  y_g(t) = sum_{p in box} w_p x_g(p, t) / sum_{p in box} w_p,  w_p = cos(lat_p)

so sum_g y_g(t) is the box mean of the model's readout of (Z(t) - mean Z) -- checked against a direct computation.
A group missing from a seed contributes 0 in that seed (as in the share plots). Lines = seed mean, band = seed min-max.
Reference: the user's downloaded NOAA ONI (3-month running Niño 3.4 anomaly, ERSST, NOAA's own climatology) --
not the same baseline as the model anomaly, so compare shapes and timing, not exact values. No index is computed here.

env: ONLY_N (32), METRIC (complex), GROUPS, LOSS (edmd_dl), WINDOW ('2015-05,2016-06'), SPAN ('2014-01,2017-12')
out: persistence/metric_{METRIC}/regional_share_N{N}/nino34_mode_contrib.{npz,txt,png}
"""
import sys, os, glob, json, gc
import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/kringel/posthocinterpretability/knn-resdmd')
import mode_persistence as mp

LOSS = os.environ.get('LOSS', 'edmd_dl')
A = f'/expanse/lustre/scratch/kringel/temp_project/20261005_dictsize_recheck/{LOSS}'
OUT = '/expanse/lustre/scratch/kringel/temp_project/20261005_dictsize_recheck/persistence'
if LOSS != 'edmd_dl':
    OUT = f'{OUT}/{LOSS}'
METRIC = os.environ.get('METRIC', 'complex')
CDIR = OUT if METRIC == 'pearson' else f'{OUT}/metric_{METRIC}'
SIZE = os.environ.get('ONLY_N', '32')
RES = f'{CDIR}/regional_share_N{SIZE}'
os.makedirs(RES, exist_ok=True)
GRID = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/opa0/sosstsst_p.zarr'
ONI = '/expanse/lustre/projects/ucd250/kringel/modes/oni.zarr'      # NOAA ONI as downloaded by the user
BOX = ((-5, 5), (190, 240))                                          # Niño 3.4, lon 0-360
WIN = [tuple(map(int, s.split('-'))) for s in os.environ.get('WINDOW', '2015-05,2016-06').split(',')]
SPAN = [tuple(map(int, s.split('-'))) for s in os.environ.get('SPAN', '2014-01,2017-12').split(',')]
clusters = json.load(open(f'{CDIR}/clusters.json'))
RUNS = sorted(glob.glob(f'{A}/dict{SIZE}/member*'))
LABELS = [f'N{SIZE}_{os.path.basename(d)}' for d in RUNS]


def n_runs_in(cl):
    return len({s.split(':')[0] for s in cl['members'] if s.split(':')[0] in LABELS})


if os.environ.get('GROUPS'):
    GROUPS = [int(g) for g in os.environ['GROUPS'].split(',')]
else:
    GROUPS = [k + 1 for k, cl in enumerate(clusters) if n_runs_in(cl) >= 4]
# physical reading of the N=32 complex clusters (index_comparison results), both ENSO groups labelled plain 'ENSO'
KNOWN = {'complex': {1: 'trend', 3: 'decadal', 4: 'ENSO', 6: 'ENSO'}}.get(METRIC, {}) if SIZE == '32' else {}


def gname(k):
    cl = clusters[k - 1]
    T = cl['period_yr']
    p = 'non-osc' if not np.isfinite(T) or T > 1e6 else f'{T:.2f} yr'
    flag = '' if cl.get('valid', True) else ', resid>0.5'
    return f"#{k} {KNOWN[k] + ' ' if k in KNOWN else ''}({p}{flag})"


GNAMES = [gname(k) for k in GROUPS] + ['other modes']
print('groups:', GNAMES, flush=True)


def calendar(start, n):
    y, m = map(int, start.split('-'))
    out = []
    for _ in range(n):
        out.append((y, m)); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def mode_amplitudes(rv):
    """same as regional_mode_share.mode_amplitudes (and index_comparison's): b, periods, eigenvectors, eigenvalues
    of the kept (deduped) modes in amplitude-ranked order, plus the lifted states Z"""
    eigenvalues, eigenvectors, residuals = rv.resdmd_residuals(
        rv.viz.K.detach().cpu(), rv.viz.G.cpu(), rv.viz.A.cpu(), rv.viz.L.cpu())
    LB = rv.viz.lookback
    loader = rv.DataLoader(rv.ORASDataset(rv.arr, steps=0, lookback=LB), batch_size=rv.BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        Z = torch.cat([rv.viz.lift(x[:, 0:LB].to(rv.DEVICE)).cpu() for x in loader])
    b = torch.linalg.inv(eigenvectors) @ Z.T.to(torch.complex64)
    n_train_eff = rv.train_arr.shape[0] - LB + 1
    ref = rv._peak_loading_ref(eigenvectors, rv.viz, rv.arr[LB - 1:rv.train_arr.shape[0]], rv.DEVICE)
    eigenvectors, b = rv._pin_mode_phase(eigenvectors, b, ref, n_train_eff)
    amplitude = b.abs().std(dim=1)
    keep = rv.dedupe_conjugate_pairs(eigenvalues)
    ranked = amplitude.clone(); ranked[~keep] = -float('inf')
    order = ranked.argsort(descending=True)[:int(keep.sum())]
    ang = eigenvalues[order].angle().abs().numpy()
    periods = np.where(ang > 1e-4, 2 * np.pi / np.maximum(ang, 1e-12) / 12, np.inf)
    return (b[order].numpy().astype(np.complex128), periods, eigenvectors[:, order].numpy().astype(np.complex128),
            eigenvalues[order].numpy(), Z.numpy().astype(np.float64))


# ---------------- per run ----------------
geo = None
Y_all, present = [], []
for mdir, label in zip(RUNS, LABELS):
    rv = mp.load_run(mdir)
    b, periods, V, lam, Z = mode_amplitudes(rv)
    cache = np.load(f'{OUT}/cache/{label}.npz')
    assert np.allclose(periods, cache['periods'], rtol=1e-3), f'{label}: mode order differs from cache'
    LB = rv.viz.lookback
    if geo is None:
        zg = xr.open_zarr(GRID)
        lat = zg['nav_lat'].values[:rv.Y, :rv.X]; lon = zg['nav_lon'].values[:rv.Y, :rv.X] % 360
        sig0 = np.asarray(rv.std).reshape(-1, rv.Y, rv.X)[0]                 # SST channel std, deg C
        ocean = np.isfinite(sig0) & np.isfinite(rv.tmask) & np.isfinite(lat)
        lat_o, lon_o = lat[ocean], lon[ocean]
        (la0, la1), (lo0, lo1) = BOX
        inbox = (lat_o >= la0) & (lat_o <= la1) & (lon_o >= lo0) & (lon_o <= lo1)
        wb = np.cos(np.deg2rad(lat_o)) * inbox
        wb = wb / wb.sum()
        print(f'Niño 3.4 box: {int(inbox.sum())} ocean pixels', flush=True)
        T = b.shape[1]
        dates = calendar(rv.START, T + LB - 1)[LB - 1:]
        geo = True
    else:
        assert b.shape[1] == T

    Cr = rv.viz.C_readout.detach().cpu().numpy().astype(np.float64)[:rv.Y * rv.X]
    sig_o = sig0[ocean]
    Cr_o = Cr[ocean.ravel()]                                                   # (P, Nz)
    ck = np.where(np.abs(lam.imag) > 1e-6, 2.0, 1.0)
    K = len(ck)
    m = (wb * sig_o) @ (Cr_o @ V)                                              # (K,) box mean of sigma * u_k
    bc = b - b.mean(1, keepdims=True)
    yk = ck[:, None] * (bc * m[:, None]).real                                  # (K, T) per-mode box-mean contribution
    # sanity: the modal sum reproduces the box mean of the readout of (Z - mean Z)
    direct = ((Z - Z.mean(0)) @ Cr_o.T * sig_o) @ wb
    print(f'{label}: K={K}; box-mean check max |modal sum - direct| = {np.abs(yk.sum(0) - direct).max():.2e} deg C '
          f'(direct std {direct.std():.3f})', flush=True)

    members = []
    for k in GROUPS:
        ms = sorted({int(s.split(':mode')[1]) - 1 for s in clusters[k - 1]['members'] if s.split(':')[0] == label})
        taken = {mm for prev in members for mm in prev}
        members.append([mm for mm in ms if mm not in taken])
    rest = sorted(set(range(K)) - {mm for ms in members for mm in ms})
    Yg = np.stack([yk[ms].sum(0) if ms else np.zeros(T) for ms in members] + [yk[rest].sum(0)])
    Y_all.append(Yg); present.append([len(ms) > 0 for ms in members] + [True])
    print(f'  {label} group modes (1-based):', {GNAMES[g]: [mm + 1 for mm in ms] for g, ms in enumerate(members)},
          flush=True)
    del rv; gc.collect(); torch.cuda.empty_cache()

Y_all = np.stack(Y_all)                                                   # (runs, G+1, T)
present = np.array(present)
tot = Y_all.sum(1)                                                        # (runs, T) model total
tdec = np.array([y + (mo - 0.5) / 12 for y, mo in dates])

ds = xr.open_zarr(ONI)
ot = ds.time_counter.dt.year.values + (ds.time_counter.dt.month.values - 0.5) / 12
ov = ds['oni'].values
o = np.argsort(ot, kind='stable'); ot, ov = ot[o], ov[o]                  # the store's time axis is not in order

np.savez(f'{RES}/nino34_mode_contrib.npz', groups=GNAMES, dates=np.array(dates), contrib=Y_all, total=tot,
         present=present, labels=LABELS, box=np.array(BOX))

# ---------------- text ----------------
iw = np.array([WIN[0] <= d <= WIN[1] for d in dates])
ws = (np.array([(y, mo) >= SPAN[0] and (y, mo) <= SPAN[1] for y, mo in dates]))
wm = Y_all[:, :, iw].mean(2)                                              # (runs, G+1) window-mean contribution
lines = [__doc__.split('\n\n')[0], '',
         f'N={SIZE} {LOSS}, {len(LABELS)} seeds; Niño 3.4 box {BOX}; window {WIN[0]}..{WIN[1]} ({int(iw.sum())} mo)',
         'seeds with the group: ' + ', '.join(f'{n} {int(c)}/{len(LABELS)}' for n, c in zip(GNAMES, present.sum(0))), '',
         f"{'group':<28} {'window mean (deg C)':>22} {'peak month (seed mean)':>24} {'corr w/ total, span':>20}"]
for g, n in enumerate(GNAMES):
    ym_ = Y_all[:, g].mean(0)
    pk = np.argmax(np.abs(ym_ * iw))
    c = np.corrcoef(ym_[ws], tot.mean(0)[ws])[0, 1] if ym_[ws].std() > 0 else np.nan
    lines.append(f'{n:<28} {wm[:, g].mean():+8.3f} ± {wm[:, g].std():.3f}        '
                 f'{dates[pk][0]}-{dates[pk][1]:02d} {ym_[pk]:+.2f}       {c:+6.2f}')
lines.append(f"{'model total':<28} {wm.sum(1).mean():+8.3f} ± {wm.sum(1).std():.3f}")
oi = np.array([WIN[0][0] + (WIN[0][1] - 0.5) / 12 - 1e-6 <= v <= WIN[1][0] + (WIN[1][1] - 0.5) / 12 + 1e-6 for v in ot])
lines.append(f"{'NOAA ONI (reference)':<28} {np.nanmean(ov[oi]):+8.3f}")
txt = '\n'.join(lines)
open(f'{RES}/nino34_mode_contrib.txt', 'w').write(txt + '\n')
print(txt, flush=True)

# ---------------- figure ----------------
INK, INK2, MUTED, GRIDC = '#0b0b0b', '#52514e', '#8a8983', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#b5b4ad'
SHADE = '#eda100'
cols = [OTHER if g == len(GNAMES) - 1 else CAT[g % len(CAT)] for g in range(len(GNAMES))]

fig, (al, ar) = plt.subplots(1, 2, figsize=(17, 6.4), facecolor='white', gridspec_kw=dict(width_ratios=[1.6, 1]))
fig.subplots_adjust(wspace=0.42)
x0, x1 = SPAN[0][0] + (SPAN[0][1] - 1) / 12, SPAN[1][0] + SPAN[1][1] / 12
al.axvspan(WIN[0][0] + (WIN[0][1] - 1) / 12, WIN[1][0] + WIN[1][1] / 12, color=SHADE, alpha=0.18, lw=0)
al.axhline(0, color=INK2, lw=0.8)
for g, n in enumerate(GNAMES):
    yg = Y_all[:, g]
    al.fill_between(tdec[ws], yg.min(0)[ws], yg.max(0)[ws], color=cols[g], alpha=0.18, lw=0)
    al.plot(tdec[ws], yg.mean(0)[ws], color=cols[g], lw=2, label=n)
al.plot(tdec[ws], tot.mean(0)[ws], color=INK, lw=2.4, label='model total (sum of all modes)')
so = (ot >= x0) & (ot <= x1)
al.plot(ot[so], ov[so], color=MUTED, lw=1.6, ls='--', label='NOAA ONI (downloaded, reference)')
al.set_xlim(x0, x1)
al.set_ylabel('contribution to Niño 3.4 box-mean SST (°C)', color=INK2)
al.tick_params(colors=INK2, labelsize=8.5, length=0)
al.grid(color=GRIDC, lw=0.6); al.set_axisbelow(True)
for sp in ('top', 'right'):
    al.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    al.spines[sp].set_color(GRIDC)
al.set_title('month by month: line = seed mean, band = seed range; shaded = El Niño window', fontsize=10, loc='left',
             color=INK)
al.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8.5)

order = np.arange(len(GNAMES))
v = wm.mean(0)
ar.barh(order, v, color=cols, height=0.62, edgecolor='white')
for i in order:
    ar.errorbar(v[i], i, xerr=wm[:, i].std(), color=INK2, lw=1, capsize=3)
    ar.scatter(wm[:, i], np.full(len(LABELS), i), s=12, color=INK, zorder=3)
lim = np.abs(wm).max() * 1.15
for i in order:
    ar.text(lim * 1.02, i, f'{v[i]:+.2f} °C', va='center', ha='left', fontsize=9, color=INK)
ar.axvline(0, color=INK2, lw=0.8)
nseed = present.sum(0)
ar.set_yticks(order); ar.invert_yaxis()
ar.set_yticklabels([n + (f' [{nseed[g]}/{len(LABELS)} seeds]' if nseed[g] < len(LABELS) else '')
                    for g, n in enumerate(GNAMES)], fontsize=9, color=INK2)
ar.set_xlim(-lim, lim)
ar.set_xlabel('mean contribution over the window (°C)', color=INK2)
ar.tick_params(colors=INK2, labelsize=8.5, length=0)
ar.grid(axis='x', color=GRIDC, lw=0.6); ar.set_axisbelow(True)
for sp in ar.spines.values():
    sp.set_visible(False)
ar.set_title(f'{WIN[0][0]}-{WIN[0][1]:02d}..{WIN[1][0]}-{WIN[1][1]:02d}: model total {wm.sum(1).mean():+.2f} °C; '
             f'dots = seeds', fontsize=10, loc='left', color=INK)
fig.suptitle(f'N={SIZE} {LOSS}: which mode groups make up the Niño 3.4 SST anomaly through El Niño 2015-16 '
             f'({len(LABELS)} seeds; amplitudes about the full-record mean)\nONI is NOAA\'s own index on its own '
             f'climatology -- compare shape and timing, not exact values', fontsize=10.5, color=INK, x=0.01, y=1.03,
             ha='left')
fig.savefig(f'{RES}/nino34_mode_contrib.png', dpi=130, bbox_inches='tight', facecolor='white')
print(f'saved {RES}/nino34_mode_contrib.npz, .txt, .png', flush=True)
