"""One figure for the California Current: where the region is (map, left) next to how its mode energy splits
between the persistence-cluster groups (stacked bars, right), read from regional_mode_share.py's saved results.

Bars: top row = whole domain / whole record (the global mix, for reference); then the California Current in each
window. Numbers are seed-mean shares (%); * = outside the random-box/window null 95% range (pct from the npz);
† = window shorter than the group's period. Map: model ocean pixels, the region's ocean pixels filled, its box outlined.

env: ONLY_N (32), METRIC (complex), LOSS (edmd_dl), REGION ('California Current')
out: persistence/metric_{METRIC}/regional_share_N{N}/california_current_share.png
"""
import os, json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

LOSS = os.environ.get('LOSS', 'edmd_dl')
OUT = '/expanse/lustre/scratch/kringel/temp_project/20261005_dictsize_recheck/persistence'
if LOSS != 'edmd_dl':
    OUT = f'{OUT}/{LOSS}'
METRIC = os.environ.get('METRIC', 'complex')
CDIR = OUT if METRIC == 'pearson' else f'{OUT}/metric_{METRIC}'
SIZE = os.environ.get('ONLY_N', '32')
REGION = os.environ.get('REGION', 'California Current')
RES = f'{CDIR}/regional_share_N{SIZE}'
ZARR = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop'

z = np.load(f'{RES}/regional_share.npz')
meta = json.load(open(f'{RES}/regional_share.json'))
RN, WN, GNAMES = list(z['regions']), list(z['windows']), list(z['groups'])
share, pct = z['share'], z['pct']                                  # (runs, R, W, G), (R, W, G)
sh_m = share.mean(0)
ri = RN.index(REGION)
(la0, la1), (lo0, lo1) = meta['regions'][REGION]
nG = len(GNAMES)


def period_of(name):
    s = name.split('(')[-1].split(' yr')[0]
    try:
        return float(s)
    except ValueError:
        return np.nan                                              # non-osc / other modes


def months(w):
    a, b = (tuple(map(int, s.strip('()').split(','))) for s in meta['windows'][w])
    return (b[0] - a[0]) * 12 + b[1] - a[1] + 1


GPER = np.array([period_of(g) for g in GNAMES])

# grid: same crop/mask as resdmd_visualize (tmaskutil at t=0, cropped to the SST array's y/x)
sst = xr.open_zarr(f'{ZARR}/opa0/sosstsst_p.zarr')
Y, X = sst['sosstsst'].shape[-2:]
lat = sst['nav_lat'].values[:Y, :X]; lon = sst['nav_lon'].values[:Y, :X] % 360
tm = xr.open_zarr(f'{ZARR}/tmask_p.zarr')['tmaskutil'].isel(t=0).isel(y=slice(0, Y), x=slice(0, X)).values
ocean = (tm > 0) & np.isfinite(lat)
inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
inreg = ocean & (lat >= la0) & (lat <= la1) & inlon
print(f'{REGION}: {int(inreg.sum())} ocean pixels (regional_mode_share log: 6354 for the California Current)',
      flush=True)

INK, INK2, MUTED, GRIDC = '#0b0b0b', '#52514e', '#8a8983', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#b5b4ad'
LAND, SEA, REG = '#d9d8d2', '#f3f2ee', '#2a78d6'

fig, (am, ab) = plt.subplots(1, 2, figsize=(16, 6.2), facecolor='white', gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(wspace=0.32)

# ---- map ----
field = np.where(inreg, 2, np.where(ocean, 1, 0)).astype(float)
am.pcolormesh(lon, lat, field, cmap=ListedColormap([LAND, SEA, REG]), vmin=-0.5, vmax=2.5, shading='auto',
              rasterized=True)
am.add_patch(Rectangle((lo0, la0), (lo1 - lo0) % 360, la1 - la0, fill=False, ec=INK, lw=1.4))
am.text(lo0 + ((lo1 - lo0) % 360) / 2, la1 + 1.5, f'{REGION}\n{la0}–{la1}°N, {360 - lo1}–{360 - lo0}°W',
        ha='center', va='bottom', fontsize=8.5, color=INK)
am.set_xlim(np.nanmin(lon[ocean]), np.nanmax(lon[ocean])); am.set_ylim(np.nanmin(lat[ocean]), np.nanmax(lat[ocean]))
am.set_aspect('equal')
xt = np.arange(np.ceil(am.get_xlim()[0] / 30) * 30, am.get_xlim()[1] + 1, 30)
am.set_xticks(xt)
am.set_xticklabels([f'{int(v)}°E' if v < 180 else ('180°' if v == 180 else f'{int(360 - v)}°W') for v in xt])
yt = np.arange(np.ceil(am.get_ylim()[0] / 20) * 20, am.get_ylim()[1] + 1, 20)
am.set_yticks(yt); am.set_yticklabels([f'{int(abs(v))}°{"N" if v > 0 else "S" if v < 0 else ""}' for v in yt])
am.tick_params(colors=INK2, labelsize=8, length=0)
am.grid(color='white', lw=0.6)
for sp in am.spines.values():
    sp.set_color(GRIDC)
am.set_title(f'region: {int(inreg.sum())} model ocean pixels (filled) inside the box', fontsize=10, loc='left',
             color=INK)

# ---- bars ----
rows = [sh_m[0, 0]] + [sh_m[ri, wi] for wi in range(len(WN))]
pcts = [np.full(nG, np.nan)] + [pct[ri, wi] for wi in range(len(WN))]
shorts = [np.zeros(nG, bool)] + [GPER > months(w) / 12 for w in WN]
names = ['whole domain, whole record\n(global reference)'] + [f'{REGION}, {w}' if wi == 0 else w
                                                              for wi, w in enumerate(WN)]
left = np.zeros(len(rows))
for g in range(nG):
    col = OTHER if g == nG - 1 else CAT[g % len(CAT)]
    vals = np.array([r[g] for r in rows])
    ab.barh(range(len(rows)), vals, left=left, color=col, edgecolor='white', linewidth=1.5, height=0.72,
            label=GNAMES[g])
    for i, (v, l0) in enumerate(zip(vals, left)):
        if v >= 0.07:
            p = pcts[i][g]
            t = f'{100 * v:.0f}' + ('*' if np.isfinite(p) and (p < 2.5 or p > 97.5) else '') + ('†' if shorts[i][g] else '')
            ab.text(l0 + v / 2, i, t, ha='center', va='center', fontsize=7.5, color=INK)
    left += vals
ab.axhline(0.5, color=INK2, lw=0.8)
ab.set_yticks(range(len(rows))); ab.set_yticklabels(names, fontsize=8.5, color=INK2); ab.invert_yaxis()
ab.set_xlim(0, 1); ab.set_xticks([0, 0.25, 0.5, 0.75, 1]); ab.set_xticklabels(['0', '25', '50', '75', '100%'])
ab.tick_params(colors=INK2, labelsize=8, length=0)
ab.grid(axis='x', color=GRIDC, lw=0.7); ab.set_axisbelow(True)
for sp in ab.spines.values():
    sp.set_visible(False)
ab.set_title(f'{REGION}: share of mode energy by group, per window', fontsize=10, loc='left', color=INK)
h, l = ab.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=min(4, nG), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f'N={SIZE} {LOSS}: {REGION} -- location and mode-group energy shares (seed mean, {share.shape[0]} seeds; '
             f'* = outside random-box/window null 95% range; † = window shorter than the period)',
             fontsize=10.5, color=INK, x=0.01, ha='left')
tag = REGION.lower().replace(' ', '_').replace('.', '')
fig.savefig(f'{RES}/{tag}_share.png', dpi=130, bbox_inches='tight', facecolor='white')
print(f'saved {RES}/{tag}_share.png', flush=True)
