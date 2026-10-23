"""Companion to california_current_share.py for a time window instead of a region: when the window is (the user's
downloaded NOAA ONI with the window shaded, left) next to how the whole domain's mode energy splits between the
persistence-cluster groups over the whole record vs inside the window (stacked bars, right), read from
regional_mode_share.py's saved results.

Bar numbers are seed-mean shares (%); * = outside the random-window null 95% range (pct from the npz);
† = window shorter than the group's period -- mode amplitudes are taken about their full-record mean, so a slow mode's
share inside a short window is mostly its offset from that mean, not variability within the window.

env: ONLY_N (32), METRIC (complex), LOSS (edmd_dl), WINDOW (key in the npz, 'El Niño 2015-16')
out: persistence/metric_{METRIC}/regional_share_N{N}/window_share_{window}.png
"""
import os, json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOSS = os.environ.get('LOSS', 'edmd_dl')
OUT = '/expanse/lustre/scratch/kringel/temp_project/20261005_dictsize_recheck/persistence'
if LOSS != 'edmd_dl':
    OUT = f'{OUT}/{LOSS}'
METRIC = os.environ.get('METRIC', 'complex')
CDIR = OUT if METRIC == 'pearson' else f'{OUT}/metric_{METRIC}'
SIZE = os.environ.get('ONLY_N', '32')
WINDOW = os.environ.get('WINDOW', 'El Niño 2015-16')
RES = f'{CDIR}/regional_share_N{SIZE}'
ONI = '/expanse/lustre/projects/ucd250/kringel/modes/oni.zarr'      # NOAA ONI as downloaded by the user

z = np.load(f'{RES}/regional_share.npz')
meta = json.load(open(f'{RES}/regional_share.json'))
RN, WN, GNAMES = list(z['regions']), list(z['windows']), list(z['groups'])
share, pct = z['share'], z['pct']                                  # (runs, R, W, G), (R, W, G)
# both ENSO groups shown separately, labelled plain 'ENSO' (no QB/LF prefix)
GNAMES = [n.replace('QB-ENSO', 'ENSO').replace('LF-ENSO', 'ENSO') for n in GNAMES]
sh_m = share.mean(0)
ri, w0, wi = RN.index('whole domain'), WN.index('whole record'), WN.index(WINDOW)
nG = len(GNAMES)


def ym(s):
    return tuple(map(int, s.strip('()').split(',')))


(a0, b0), (a1, b1) = ym(meta['windows'][WINDOW][0]), ym(meta['windows'][WINDOW][1])
(r0y, r0m), (r1y, r1m) = ym(meta['windows']['whole record'][0]), ym(meta['windows']['whole record'][1])
win_yr = ((a1 - a0) * 12 + b1 - b0 + 1) / 12


def period_of(name):
    s = name.split('(')[-1].split(' yr')[0]
    try:
        return float(s)
    except ValueError:
        return np.nan                                              # non-osc / other modes


short = np.array([period_of(n) > win_yr or 'non-osc' in n for n in GNAMES])

ds = xr.open_zarr(ONI)
t = ds.time_counter.dt.year.values + (ds.time_counter.dt.month.values - 0.5) / 12
oni = ds['oni'].values
sel = (t >= r0y + (r0m - 1) / 12) & (t <= r1y + r1m / 12) & np.isfinite(oni)
t, oni = t[sel], oni[sel]
order = np.argsort(t, kind='stable')                               # the store's time axis is not in order
print(f'{WINDOW}: {a0}-{b0:02d}..{a1}-{b1:02d} ({win_yr:.2f} yr); ONI {len(t)} months, '
      f'{int((np.diff(t) <= 0).sum())} out-of-order steps, {len(t) - len(np.unique(t))} duplicate months', flush=True)
t, oni = t[order], oni[order]

INK, INK2, MUTED, GRIDC = '#0b0b0b', '#52514e', '#8a8983', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#b5b4ad'
WARM, COOL, SHADE = '#e34948', '#2a78d6', '#eda100'

fig, (at, ab) = plt.subplots(1, 2, figsize=(16, 5.6), facecolor='white', gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(wspace=0.32)

# ---- ONI timeline with the window shaded ----
at.axvspan(a0 + (b0 - 1) / 12, a1 + b1 / 12, color=SHADE, alpha=0.25, lw=0)
at.fill_between(t, oni, 0, where=oni >= 0, color=WARM, lw=0, interpolate=True)
at.fill_between(t, oni, 0, where=oni < 0, color=COOL, lw=0, interpolate=True)
at.axhline(0, color=INK2, lw=0.8)
for y in (0.5, -0.5):
    at.axhline(y, color=MUTED, lw=0.7, ls='--')
at.text(a0 + (b0 - 1) / 12, at.get_ylim()[1] * 0.92, f'{WINDOW}\n{a0}-{b0:02d}..{a1}-{b1:02d}', ha='right',
        va='top', fontsize=8.5, color=INK)
at.set_xlim(t.min(), t.max())
at.set_ylabel('ONI (°C)', color=INK2)
at.tick_params(colors=INK2, labelsize=8, length=0)
at.grid(axis='x', color=GRIDC, lw=0.6); at.set_axisbelow(True)
for sp in ('top', 'right'):
    at.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    at.spines[sp].set_color(GRIDC)
at.set_title('NOAA ONI (downloaded), window shaded; dashed = ±0.5 °C', fontsize=10, loc='left', color=INK)

# ---- bars ----
rows = [sh_m[ri, w0], sh_m[ri, wi]]
pcts = [np.full(nG, np.nan), pct[ri, wi]]
shorts = [np.zeros(nG, bool), short]
names = ['whole record\n(1979-2018)', WINDOW]
left = np.zeros(len(rows))
for g in range(nG):
    col = OTHER if g == nG - 1 else CAT[g % len(CAT)]
    vals = np.array([r[g] for r in rows])
    ab.barh(range(len(rows)), vals, left=left, color=col, edgecolor='white', linewidth=1.5, height=0.6,
            label=GNAMES[g])
    for i, (v, l0) in enumerate(zip(vals, left)):
        if v >= 0.07:
            p = pcts[i][g]
            s = f'{100 * v:.0f}' + ('*' if np.isfinite(p) and (p < 2.5 or p > 97.5) else '') + ('†' if shorts[i][g] else '')
            ab.text(l0 + v / 2, i, s, ha='center', va='center', fontsize=9, color=INK)
    left += vals
ab.set_yticks(range(len(rows))); ab.set_yticklabels(names, fontsize=10, color=INK2); ab.invert_yaxis()
ab.set_ylim(len(rows) - 0.4, -0.6)
ab.set_xlim(0, 1); ab.set_xticks([0, 0.25, 0.5, 0.75, 1]); ab.set_xticklabels(['0', '25', '50', '75', '100%'])
ab.tick_params(colors=INK2, labelsize=8, length=0)
ab.grid(axis='x', color=GRIDC, lw=0.7); ab.set_axisbelow(True)
for sp in ab.spines.values():
    sp.set_visible(False)
ab.set_title('share of mode energy by group, whole domain', fontsize=10, loc='left', color=INK)
h, l = ab.get_legend_handles_labels()
ab.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.45, -0.1))
fig.suptitle(f'N={SIZE} {LOSS}: whole domain, whole record vs {WINDOW} -- mode-group energy shares (seed mean, '
             f'{share.shape[0]} seeds)\n* = outside random-window null 95% range; † = window shorter than the period, '
             f'share mostly an offset from the full-record mean', fontsize=10.5, color=INK, x=0.01, y=1.02, ha='left')
tag = WINDOW.lower().replace('ñ', 'n').replace(' ', '_').replace('.', '')
fig.savefig(f'{RES}/window_share_{tag}.png', dpi=130, bbox_inches='tight', facecolor='white')
print(f'saved {RES}/window_share_{tag}.png', flush=True)
