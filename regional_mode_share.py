"""Which Koopman modes matter more in one region / time window than they do for the whole model.

For each N=32 edmd_dl run of the 20261005 dict-size recheck (mode_persistence.load_run -> resdmd_visualize's
global train-fit K; amplitudes from opa0, same extraction and ordering as persistence/index_comparison.py,
checked against the cached periods so mode numbers line up with clusters.json):

  mode k's SST contribution  x_k(p, t) = c_k * Re(b_k(t) * u_k(p)) * sigma_p   [deg C]
      u_k = C_readout @ v_k, c_k = 2 for a complex mode (its conjugate partner adds the same term), 1 for a real one,
      b_k minus its FULL-RECORD mean (never a window mean -- a short window would otherwise delete slow modes).
  group g = one persistence cluster's member(s) in that run (clusters.json, complex metric), or "other" = every
      remaining mode. x_g = sum of its members.
  energy   E_g(R, T) = sum_{t in T} sum_{p in R} w_p x_g(p, t)^2,  w_p = cos(lat_p)  (area weight)
  share    E_g / sum_h E_h  -- the groups' relative importance in (R, T); sums to 1 over groups.
  ratio    share(R, T) / share(whole domain, whole record) -- > 1: the group matters more here than globally.
  cross    (E_model - sum_h E_h) / sum_h E_h, E_model = energy of the sum of all modes: the modes are not orthogonal,
      so in a small region they can reinforce (> 0) or cancel (< 0) each other; large |cross| = shares less meaningful.
  data     E_g / E_data, E_data = energy of the observed opa0 SST anomaly (full-record mean removed) in (R, T).
  cov      sum <x_g, d> / E_data -- covariance share; sums over groups to the model's projection on the data and can
      be negative. model R2 = 1 - |d - x_model|^2 / |d|^2 in (R, T).

All of these are exact trace products: with U = [Re u_k * sigma, Im u_k * sigma]_k (pixels x 2K) and
a(t) = [c_k Re b_k, -c_k Im b_k]_k, x(t) = U a(t), so E_g(R, T) = sum_{ij in g} G_R[ij] S_T[ij] with
G_R = U^T W_R U and S_T = sum_{t in T} a(t) a(t)^T. Checked against a direct pixel sum for the whole domain.

Null (is a local share actually unusual?): the same shares in N_NULL random boxes of the named region's lat/lon
extent (>= 50% of its ocean pixel count) and/or random windows of the same length (whole domain: time null only;
whole record: space null only). The same draws are used for every seed; the seed-mean share is ranked within the
seed-mean null -> pct (2-sided flag outside [2.5, 97.5]).

env: ONLY_N (32), METRIC (complex), GROUPS (1-based clusters.json numbers, default: every cluster with a member in
>= 4 of the size's runs), N_NULL (300), LOSS (edmd_dl)
out: persistence/metric_{METRIC}/regional_share_N{N}/{regional_share.txt, .json, .npz, share_heatmaps.png,
     share_bars.png, qb_lf_ratio.png}
"""
import sys, os, glob, json, gc
import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
N_NULL = int(os.environ.get('N_NULL', 300))
RES = f'{CDIR}/regional_share_N{SIZE}'
os.makedirs(RES, exist_ok=True)
GRID = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop/opa0/sosstsst_p.zarr'
clusters = json.load(open(f'{CDIR}/clusters.json'))
RUNS = sorted(glob.glob(f'{A}/dict{SIZE}/member*'))
LABELS = [f'N{SIZE}_{os.path.basename(d)}' for d in RUNS]


def n_runs_in(cl):
    return len({s.split(':')[0] for s in cl['members'] if s.split(':')[0] in LABELS})


if os.environ.get('GROUPS'):
    GROUPS = [int(g) for g in os.environ['GROUPS'].split(',')]
else:
    GROUPS = [k + 1 for k, cl in enumerate(clusters) if n_runs_in(cl) >= 4]
# physical reading of the N=32 complex clusters (index_comparison results); anything else is named by period
KNOWN = {'complex': {1: 'trend', 3: 'decadal', 4: 'QB-ENSO', 6: 'LF-ENSO'}}.get(METRIC, {}) if SIZE == '32' else {}


def gname(k):
    cl = clusters[k - 1]
    T = cl['period_yr']
    p = 'non-osc' if not np.isfinite(T) or T > 1e6 else f'{T:.2f} yr'
    flag = '' if cl.get('valid', True) else ', resid>0.5'
    return f"#{k} {KNOWN[k] + ' ' if k in KNOWN else ''}({p}{flag})"


GNAMES = [gname(k) for k in GROUPS] + ['other modes']
GPER = [clusters[k - 1]['period_yr'] for k in GROUPS] + [np.nan]
print('groups:', GNAMES, flush=True)

# lat (S..N), lon 0-360 (E); lon ranges are eastward from lo0 to lo1
REGIONS = {
    'whole domain':         None,
    'Niño 4':               ((-5, 5), (160, 210)),
    'Niño 3.4':             ((-5, 5), (190, 240)),
    'Niño 3':               ((-5, 5), (210, 270)),
    'Niño 1+2':             ((-10, 0), (270, 280)),
    'W. Pacific warm pool': ((-10, 10), (120, 160)),
    'Kuroshio Extension':   ((30, 45), (140, 180)),
    'N. Pacific (>20N)':    ((20, 60), (120, 250)),
    'California Current':   ((25, 45), (225, 245)),
    'S. Pacific subtropics': ((-40, -20), (180, 280)),
}
# (first month, last month) inclusive; 'test' is filled in from the split
WINDOWS = {
    'whole record':      None,
    'El Niño 1982-83':   ('1982-05', '1983-06'),
    'La Niña 1988-89':   ('1988-05', '1989-06'),
    'El Niño 1997-98':   ('1997-05', '1998-06'),
    'La Niña 1998-2001': ('1998-07', '2001-03'),
    'El Niño 2015-16':   ('2015-05', '2016-06'),
    '1979-1998':         ('1979-01', '1998-12'),
    '1999-2018':         ('1999-01', '2018-01'),
    'test months':       'test',
}


def calendar(start, n):
    y, m = map(int, start.split('-'))
    out = []
    for _ in range(n):
        out.append((y, m)); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def mode_amplitudes(rv):
    """same as index_comparison.mode_amplitudes: b, periods, eigenvectors, eigenvalues of the kept (deduped)
    modes in amplitude-ranked order, plus the lifted states Z"""
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


def box_mask(lat, lon, la, lo):
    (la0, la1), (lo0, lo1) = la, lo
    inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
    return (lat >= la0) & (lat <= la1) & inlon


def window_slice(win, dates, test0):
    if win is None:
        return (0, len(dates))
    if win == 'test':
        return (test0, len(dates))
    a, b = (tuple(map(int, s.split('-'))) for s in win)
    idx = [i for i, d in enumerate(dates) if a <= d <= b]
    return (idx[0], idx[-1] + 1)


def group_masks(n2k, members):
    """(G+1, 2K, 2K) 0/1 masks: block of each group's columns; last = every column in no group"""
    M = np.zeros((len(members) + 1, n2k, n2k))
    used = np.zeros(n2k, bool)
    for g, ms in enumerate(members):
        cols = np.array([c for m in ms for c in (2 * m, 2 * m + 1)], int)
        if len(cols):
            M[g][np.ix_(cols, cols)] = 1; used[cols] = True
    rest = np.where(~used)[0]
    M[-1][np.ix_(rest, rest)] = 1
    return M


# ---------------- per run ----------------
geo = None
per_run = []
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
        w_o = np.cos(np.deg2rad(lat_o))
        T = b.shape[1]
        dates = calendar(rv.START, T + LB - 1)[LB - 1:]
        test0 = rv.train_arr.shape[0] + rv.val_arr.shape[0] - (LB - 1)
        reg_px = {r: (np.ones(ocean.sum(), bool) if box is None else box_mask(lat_o, lon_o, *box))
                  for r, box in REGIONS.items()}
        for r in list(REGIONS):
            print(f'  region {r}: {int(reg_px[r].sum())} ocean pixels', flush=True)
            if reg_px[r].sum() == 0:
                print(f'  -> dropped (outside the domain)', flush=True)
                REGIONS.pop(r); reg_px.pop(r)
        win_sl = {w: window_slice(v, dates, test0) for w, v in WINDOWS.items()}
        for w, (t0, t1) in win_sl.items():
            print(f'  window {w}: {dates[t0]}..{dates[t1 - 1]} ({t1 - t0} mo)', flush=True)
        # null draws, shared by every run: random boxes of each region's extent, random windows of each length
        rng = np.random.default_rng(0)
        la_rng, lo_rng = (lat_o.min(), lat_o.max()), (lon_o.min(), lon_o.max())
        null_px, null_t = {}, {}
        for r, box in REGIONS.items():
            if box is None:
                continue
            (la0, la1), (lo0, lo1) = box
            dla, dlo, need = la1 - la0, (lo1 - lo0) % 360, 0.5 * reg_px[r].sum()
            draws, tries = [], 0
            while len(draws) < N_NULL and tries < 200 * N_NULL:
                tries += 1
                c_la = rng.uniform(la_rng[0], la_rng[1] - dla)
                c_lo = rng.uniform(lo_rng[0], lo_rng[1] - dlo)
                m = box_mask(lat_o, lon_o, (c_la, c_la + dla), (c_lo, c_lo + dlo))
                if m.sum() >= need:
                    draws.append(m)
            null_px[r] = draws
            print(f'  null boxes for {r}: {len(draws)} ({tries} tries)', flush=True)
        for w, (t0, t1) in win_sl.items():
            L = t1 - t0
            if L < T:
                st = rng.integers(0, T - L + 1, N_NULL)
                null_t[w] = [(s, s + L) for s in st]
        geo = True
    else:
        assert b.shape[1] == T

    # U: (pixels, 2K) physical-unit SST pattern columns; a(t): (T, 2K) amplitudes about the full-record mean
    Cr = rv.viz.C_readout.detach().cpu().numpy().astype(np.float64)[:rv.Y * rv.X]
    sig_o = sig0[ocean]
    Cr_o = Cr[ocean.ravel()]                                                   # (P, Nz)
    u = Cr_o @ V                                                               # (P, K) complex
    ck = np.where(np.abs(lam.imag) > 1e-6, 2.0, 1.0)
    K = len(ck)
    U = np.empty((u.shape[0], 2 * K)); U[:, 0::2] = u.real * sig_o[:, None]; U[:, 1::2] = u.imag * sig_o[:, None]
    bc = b - b.mean(1, keepdims=True)
    a = np.empty((T, 2 * K)); a[:, 0::2] = (ck[:, None] * bc.real).T; a[:, 1::2] = (-ck[:, None] * bc.imag).T
    d = np.nan_to_num(np.asarray(rv.arr[LB - 1:, 0], dtype=np.float64)[:, ocean]) * sig_o   # observed opa0 SST anomaly, deg C
    d = d - d.mean(0)

    # sanity: the modal sum reproduces the readout of (Z - mean Z) over the whole domain / record
    xhat = (Z - Z.mean(0)) @ Cr_o.T * sig_o
    E_direct = float((w_o * xhat ** 2).sum())
    G_dom = (U * w_o[:, None]).T @ U
    S_cum = np.concatenate([np.zeros((1, 2 * K, 2 * K)), np.cumsum(a[:, :, None] * a[:, None, :], 0)])
    E_trace = float((G_dom * S_cum[-1]).sum())
    print(f'{label}: K={K} modes; modal-sum check: trace {E_trace:.6g} vs direct {E_direct:.6g} '
          f'(rel diff {abs(E_trace - E_direct) / E_direct:.2e})', flush=True)
    del xhat

    members = []
    for k in GROUPS:
        ms = sorted({int(s.split(':mode')[1]) - 1 for s in clusters[k - 1]['members'] if s.split(':')[0] == label})
        taken = {m for prev in members for m in prev}
        if taken & set(ms):
            print(f'  WARNING {label}: cluster {k} shares modes {sorted(taken & set(ms))} with an earlier group; '
                  f'kept in the earlier one', flush=True)
        members.append([m for m in ms if m not in taken])
    M = group_masks(2 * K, members)
    print(f'  {label} group modes (1-based):', {GNAMES[g]: [m + 1 for m in ms] for g, ms in enumerate(members)},
          flush=True)

    def S_of(t0, t1):
        return S_cum[t1] - S_cum[t0]

    def energies(G, S):
        GS = G * S
        return np.einsum('gij,ij->g', M, GS), float(GS.sum())

    nR, nW, nG = len(REGIONS), len(WINDOWS), len(GNAMES)
    E = np.zeros((nR, nW, nG)); Emod = np.zeros((nR, nW)); Ed = np.zeros((nR, nW))
    cov = np.zeros((nR, nW, nG)); null_share = np.full((nR, nW, N_NULL, nG), np.nan)
    d2 = d ** 2
    G_box = {}
    for ri, (r, px) in enumerate(reg_px.items()):
        wr = w_o * px
        G = (U * wr[:, None]).T @ U
        G_box[r] = G
        H = (d * wr) @ U                                   # (T, 2K): U^T W_R d(t)
        Ht = a * H                                         # per-column <x, d> contributions at each t
        ed_t = d2 @ wr                                     # (T,)
        for wi, (w, (t0, t1)) in enumerate(win_sl.items()):
            E[ri, wi], Emod[ri, wi] = energies(G, S_of(t0, t1))
            Ed[ri, wi] = ed_t[t0:t1].sum()
            colsum = Ht[t0:t1].sum(0)                      # (2K,)
            cov[ri, wi] = [colsum[np.diag(Mg) > 0].sum() for Mg in M]
        # null Grams for this region's boxes
        if r in null_px:
            null_px[r + '|G'] = [(U[m] * w_o[m, None]).T @ U[m] for m in null_px[r]]
    for ri, r in enumerate(reg_px):
        for wi, w in enumerate(win_sl):
            if REGIONS[r] is None and w not in null_t:
                continue                                   # whole domain, whole record: nothing to compare against
            Gs = null_px[r + '|G'] if r in null_px else [G_box[r]] * N_NULL
            Ss = [S_of(*tt) for tt in null_t[w]] if w in null_t else [S_of(*win_sl[w])] * N_NULL
            for n, (G, S) in enumerate(zip(Gs, Ss)):
                e, _ = energies(G, S)
                null_share[ri, wi, n] = e / e.sum()
    tot = E.sum(-1)
    per_run.append(dict(label=label, members=[[m + 1 for m in ms] for ms in members], K=K,
                        share=E / tot[..., None], cross=(Emod - tot) / tot, data=E / Ed[..., None],
                        cov=cov / Ed[..., None], r2=1 - (Ed - 2 * cov.sum(-1) + Emod) / Ed,
                        null=null_share, present=np.array([len(ms) > 0 for ms in members] + [True])))
    for key in [k for k in null_px if k.endswith('|G')]:
        null_px.pop(key)
    del rv, U, d, d2; gc.collect(); torch.cuda.empty_cache()

# ---------------- combine seeds ----------------
RN, WN = list(REGIONS), list(WINDOWS)
share = np.stack([p['share'] for p in per_run])                   # (runs, R, W, G)
present = np.stack([p['present'] for p in per_run])               # (runs, G)
cross = np.stack([p['cross'] for p in per_run]); r2 = np.stack([p['r2'] for p in per_run])
data = np.stack([p['data'] for p in per_run]); covs = np.stack([p['cov'] for p in per_run])
null = np.stack([p['null'] for p in per_run])                     # (runs, R, W, n, G)
sh_m, sh_s = share.mean(0), share.std(0)
glob_m = sh_m[0, 0]                                               # whole domain, whole record
ratio = sh_m / glob_m
null_m = null.mean(0)                                             # seed-mean null (same draws every seed)
n_ok = np.isfinite(null_m).sum(2)                                 # draws actually made (box placement can run short)
with np.errstate(invalid='ignore', divide='ignore'):
    pct = np.where(n_ok > 0, 100 * (null_m <= sh_m[:, :, None, :]).sum(2) / n_ok, np.nan)
win_len = np.array([win_sl[w][1] - win_sl[w][0] for w in WN]) / 12      # years
short = np.array(GPER)[None, :] > win_len[:, None]                       # (W, G): window shorter than the period

np.savez(f'{RES}/regional_share.npz', regions=RN, windows=WN, groups=GNAMES, share=share, cross=cross, r2=r2,
         data=data, cov=covs, null_seed_mean=null_m, pct=pct, present=present, labels=LABELS)
json.dump(dict(regions={r: REGIONS[r] for r in RN}, windows={w: [str(dates[win_sl[w][0]]), str(dates[win_sl[w][1] - 1])]
                                                              for w in WN},
               groups=GNAMES, clusters=GROUPS, members={p['label']: p['members'] for p in per_run},
               share_mean=sh_m.tolist(), share_std=sh_s.tolist(), ratio_to_global=ratio.tolist(),
               null_pct=pct.tolist(), cross_mean=cross.mean(0).tolist(), model_r2_mean=r2.mean(0).tolist()),
          open(f'{RES}/regional_share.json', 'w'), indent=1, default=float)

# ---------------- text report ----------------
lines = [__doc__.split('\n\n')[0], '',
         f'N={SIZE} {LOSS}, {len(per_run)} seeds, clusters from {CDIR}/clusters.json; groups: ' + '; '.join(GNAMES),
         'members per run (1-based mode numbers): ' + json.dumps({p['label']: p['members'] for p in per_run}),
         'seeds with the group (absent = share 0 in that seed): ' +
         ', '.join(f'{n} {int(c)}/{len(per_run)}' for n, c in zip(GNAMES, present.sum(0))),
         'share = mean±std over seeds; ratio = seed-mean share / seed-mean global share; pct = percentile of the '
         'seed-mean share among the null (random boxes of the same extent and/or random windows of the same length); '
         '* = outside [2.5, 97.5]; † = window shorter than the group\'s period (share mostly an offset, read with care); '
         'data = E_g / E_obs; cov = covariance share; cross = (E_model - sum E_g) / sum E_g.', '']
gw = max(len(g) for g in GNAMES)
for ri, r in enumerate(RN):
    for wi, w in enumerate(WN):
        t0, t1 = win_sl[w]
        lines.append(f'=== {r} | {w} ({dates[t0][0]}-{dates[t0][1]:02d}..{dates[t1 - 1][0]}-{dates[t1 - 1][1]:02d}) '
                     f'  cross {cross[:, ri, wi].mean():+.2f}±{cross[:, ri, wi].std():.2f}  '
                     f'model R2 {r2[:, ri, wi].mean():.2f}±{r2[:, ri, wi].std():.2f} ===')
        lines.append(f"{'group':<{gw}} {'share':>13} {'global':>7} {'ratio':>6} {'pct':>6} {'data':>6} {'cov':>6}")
        for g, name in enumerate(GNAMES):
            p = pct[ri, wi, g]
            flag = ('*' if np.isfinite(p) and (p < 2.5 or p > 97.5) else ' ') + ('†' if short[wi, g] else ' ')
            lines.append(f'{name:<{gw}} {100 * sh_m[ri, wi, g]:6.1f}±{100 * sh_s[ri, wi, g]:4.1f}% '
                         f'{100 * glob_m[g]:6.1f}% {ratio[ri, wi, g]:6.2f} '
                         f"{(f'{p:5.1f}' if np.isfinite(p) else '    –'):>6}{flag} "
                         f'{data[:, ri, wi, g].mean():6.3f} {covs[:, ri, wi, g].mean():+6.3f}')
        lines.append('')

# biggest departures from the global mix that the null also calls unusual
lines.append('=== largest departures from the global share (|log2 ratio|, null-significant, window >= period) ===')
cand = []
for ri, r in enumerate(RN):
    for wi, w in enumerate(WN):
        for g, name in enumerate(GNAMES):
            p = pct[ri, wi, g]
            if np.isfinite(p) and (p < 2.5 or p > 97.5) and not short[wi, g] and glob_m[g] > 0.005:
                cand.append((abs(np.log2(ratio[ri, wi, g])), r, w, name, ratio[ri, wi, g], sh_m[ri, wi, g], p))
for _, r, w, name, rt, s, p in sorted(cand, reverse=True)[:30]:
    lines.append(f'  {name:<{gw}} {r:>22} | {w:<18} share {100 * s:5.1f}% (global {100 * glob_m[GNAMES.index(name)]:4.1f}%) '
                 f'x{rt:.2f}  pct {p:.1f}')

# the index_comparison hint: LF-ENSO vs QB-ENSO weight, region by region
qi = next((g for g, k in enumerate(GROUPS) if KNOWN.get(k) == 'QB-ENSO'), None)
li = next((g for g, k in enumerate(GROUPS) if KNOWN.get(k) == 'LF-ENSO'), None)
if qi is not None and li is not None:
    lr = share[..., li] / share[..., qi]                               # (runs, R, W)
    lines.append('\n=== LF-ENSO / QB-ENSO energy ratio (mean±std over seeds) -- global value in the first cell ===')
    lines.append(f"{'':>22} " + ' '.join(f'{w[:14]:>15}' for w in WN))
    for ri, r in enumerate(RN):
        lines.append(f'{r:>22} ' + ' '.join(f'{lr[:, ri, wi].mean():7.2f}±{lr[:, ri, wi].std():5.2f}  ' for wi in range(len(WN))))
txt = '\n'.join(lines)
open(f'{RES}/regional_share.txt', 'w').write(txt + '\n')
print(txt, flush=True)

# ---------------- figures ----------------
INK, INK2, MUTED, GRIDC, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e1e0d9', '#fcfcfb'
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#2a78d6', '#86b6ef', '#f0efec', '#f19b9a', '#e34948', '#a8302f'])
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#b5b4ad'

# 1) heatmaps: one panel per group, regions x windows, colour = log2(share / global share)
nG = len(GNAMES)
ncol = 2
nrow = (nG + 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 0.42 * len(RN) * nrow + 2.2 * nrow), squeeze=False,
                         facecolor='white')
fig.subplots_adjust(wspace=0.42, hspace=0.62, top=0.95)
for g, ax in enumerate(axes.flat):
    if g >= nG:
        ax.axis('off'); continue
    lr = np.log2(np.clip(ratio[:, :, g], 1e-3, None))
    im = ax.imshow(lr, cmap=DIV, vmin=-2, vmax=2, aspect='auto')
    for ri in range(len(RN)):
        for wi in range(len(WN)):
            p = pct[ri, wi, g]
            sig = np.isfinite(p) and (p < 2.5 or p > 97.5)
            txt_ = f'{100 * sh_m[ri, wi, g]:.0f}' if sh_m[ri, wi, g] >= 0.095 else f'{100 * sh_m[ri, wi, g]:.1f}'
            txt_ += ('*' if sig else '') + ('†' if short[wi, g] else '')
            ax.text(wi, ri, txt_, ha='center', va='center', fontsize=7.5,
                    color='white' if abs(lr[ri, wi]) > 1.3 else INK, fontweight='bold' if sig else 'normal')
    ax.set_xticks(range(len(WN))); ax.set_xticklabels(WN, rotation=35, ha='right', fontsize=8, color=INK2)
    ax.set_yticks(range(len(RN))); ax.set_yticklabels(RN, fontsize=8, color=INK2)
    ax.set_title(f'{GNAMES[g]} -- global share {100 * glob_m[g]:.1f}%', fontsize=10, loc='left', color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
cb.set_label('log2(local share / global share)', color=INK2); cb.ax.tick_params(colors=INK2, labelsize=8)
cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['÷4', '÷2', 'same', '×2', '×4'])
fig.suptitle(f'N={SIZE} {LOSS}: where each mode group matters more (red) or less (blue) than in the whole model\n'
             f'cell text = seed-mean share of mode energy (%); * = outside the random-box/window null 95% range; '
             f'† = window shorter than the period', fontsize=10.5, color=INK, x=0.06, y=0.985, ha='left')
fig.savefig(f'{RES}/share_heatmaps.png', dpi=130, bbox_inches='tight', facecolor='white')
plt.close(fig)


# 2) 100% stacked bars: regions over the whole record (left), whole domain by window (right)
def stacked(ax, rows, names):
    left = np.zeros(len(rows))
    for g in range(nG):
        col = OTHER if g == nG - 1 else CAT[g % len(CAT)]
        vals = np.array([r[g] for r in rows])
        ax.barh(range(len(rows)), vals, left=left, color=col, edgecolor='white', linewidth=1.5, height=0.72,
                label=GNAMES[g])
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 0.07:
                ax.text(l0 + v / 2, i, f'{100 * v:.0f}', ha='center', va='center', fontsize=7.5, color=INK)
        left += vals
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8.5, color=INK2); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(['0', '25', '50', '75', '100%'])
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis='x', color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.42 * max(len(RN), len(WN)) + 2.2), facecolor='white')
stacked(a1, [sh_m[ri, 0] for ri in range(len(RN))], RN)
a1.set_title('by region, whole record', fontsize=10, loc='left', color=INK)
stacked(a2, [sh_m[0, wi] for wi in range(len(WN))], WN)
a2.set_title('whole domain, by window', fontsize=10, loc='left', color=INK)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=min(4, nG), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(f'N={SIZE} {LOSS}: share of mode energy by group (seed mean; groups sum to 100%, cross-terms reported '
             f'separately in regional_share.txt)', fontsize=10.5, color=INK, x=0.01, ha='left')
fig.savefig(f'{RES}/share_bars.png', dpi=130, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 3) LF / QB ratio by Nino region over the whole record, every seed shown
if qi is not None and li is not None:
    lr = share[..., li] / share[..., qi]
    nino = [r for r in ('Niño 4', 'Niño 3.4', 'Niño 3', 'Niño 1+2') if r in RN]
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor='white')
    gl = lr[:, 0, 0]
    ax.axhline(gl.mean(), color=MUTED, lw=1, ls='--')
    ax.text(len(nino) - 0.55, gl.mean(), f' whole domain {gl.mean():.2f}', va='bottom', fontsize=8, color=INK2)
    for i, r in enumerate(nino):
        v = lr[:, RN.index(r), 0]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=40, color=CAT[0],
                   edgecolor='white', linewidth=1.5, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(nino))); ax.set_xticklabels([f'{r}' for r in nino], color=INK2)
    ax.set_ylabel('LF-ENSO / QB-ENSO energy', color=INK2)
    ax.set_title('west → east: LF-ENSO (4.3 yr) weight relative to QB-ENSO (2.7 yr); dots = seeds, bar = mean',
                 fontsize=9.5, loc='left', color=INK)
    ax.grid(axis='y', color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2)
    fig.savefig(f'{RES}/qb_lf_ratio.png', dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
print(f'saved {RES}/regional_share.txt, share_heatmaps.png, share_bars.png, qb_lf_ratio.png', flush=True)
