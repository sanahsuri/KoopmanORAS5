"""One figure for the California Current: where the region is (map, left) next to how its mode energy splits
between the persistence-cluster groups (stacked bars, right), read from regional_mode_share.py's saved results.

Bars: whole domain vs the region, both over the whole record. Numbers are seed-mean shares (%); * = outside the
random-box null 95% range (pct from the npz). Map: model ocean pixels, the region's ocean pixels filled, its box outlined.

env: ONLY_N (32), METRIC (complex), LOSS (edmd_dl), REGION (key in the npz, 'California Current'),
     LABEL (name shown on the figure, 'California coast')
out: persistence/metric_{METRIC}/regional_share_N{N}/california_coast_share.png
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
LABEL = os.environ.get('LABEL', 'California coast')
RES = f'{CDIR}/regional_share_N{SIZE}'
ZARR = '/expanse/lustre/projects/ucd245/ssuri/pacific_crop'

z = np.load(f'{RES}/regional_share.npz')
meta = json.load(open(f'{RES}/regional_share.json'))
RN, WN, GNAMES = list(z['regions']), list(z['windows']), list(z['groups'])
share, pct = z['share'], z['pct']                                  # (runs, R, W, G), (R, W, G)
# QB-ENSO + LF-ENSO -> one 'ENSO' group (in QB's slot): sum of the two groups' energies, so the bars still sum to
# 100%; their mutual cross term is not in the npz and stays out, as for every other pair. No null for the merged group.
qi = next(g for g, n in enumerate(GNAMES) if 'QB-ENSO' in n)
li = next(g for g, n in enumerate(GNAMES) if 'LF-ENSO' in n)
share = share.copy(); share[..., qi] += share[..., li]
pct = pct.copy(); pct[..., qi] = np.nan
keep = [g for g in range(len(GNAMES)) if g != li]
per = lambda n: n.split('(')[-1].split(' yr')[0]
GNAMES[qi] = f"{GNAMES[qi].split(' ')[0]}+{GNAMES[li].split(' ')[0]} ENSO ({per(GNAMES[qi])} + {per(GNAMES[li])} yr)"
GNAMES = [GNAMES[g] for g in keep]
share, pct = share[..., keep], pct[..., keep]
sh_m = share.mean(0)
ri = RN.index(REGION)
(la0, la1), (lo0, lo1) = meta['regions'][REGION]
nG = len(GNAMES)


# grid: same crop/mask as resdmd_visualize (tmaskutil at t=0, cropped to the SST array's y/x)
sst = xr.open_zarr(f'{ZARR}/opa0/sosstsst_p.zarr')
Y, X = sst['sosstsst'].shape[-2:]
lat = sst['nav_lat'].values[:Y, :X]; lon = sst['nav_lon'].values[:Y, :X] % 360
tm = xr.open_zarr(f'{ZARR}/tmask_p.zarr')['tmaskutil'].isel(t=0).isel(y=slice(0, Y), x=slice(0, X)).values
ocean = (tm > 0) & np.isfinite(lat)
inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
inreg = ocean & (lat >= la0) & (lat <= la1) & inlon
print(f'{LABEL} ({REGION} in the npz): {int(inreg.sum())} ocean pixels (regional_mode_share log: 6354)', flush=True)

INK, INK2, MUTED, GRIDC = '#0b0b0b', '#52514e', '#8a8983', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#b5b4ad'
LAND, SEA, REG = '#d9d8d2', '#f3f2ee', '#2a78d6'

fig, (am, ab) = plt.subplots(1, 2, figsize=(16, 5.6), facecolor='white', gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(wspace=0.32)

# ---- map ----
field = np.where(inreg, 2, np.where(ocean, 1, 0)).astype(float)
am.pcolormesh(lon, lat, field, cmap=ListedColormap([LAND, SEA, REG]), vmin=-0.5, vmax=2.5, shading='auto',
              rasterized=True)
am.add_patch(Rectangle((lo0, la0), (lo1 - lo0) % 360, la1 - la0, fill=False, ec=INK, lw=1.4))
am.text(lo0 + ((lo1 - lo0) % 360) / 2, la1 + 1.5, f'{LABEL}\n{la0}–{la1}°N, {360 - lo1}–{360 - lo0}°W',
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
wi = WN.index('whole record')
rows = [sh_m[0, wi], sh_m[ri, wi]]
pcts = [np.full(nG, np.nan), pct[ri, wi]]
names = ['whole domain', LABEL]
left = np.zeros(len(rows))
for g in range(nG):
    col = OTHER if g == nG - 1 else CAT[g % len(CAT)]
    vals = np.array([r[g] for r in rows])
    ab.barh(range(len(rows)), vals, left=left, color=col, edgecolor='white', linewidth=1.5, height=0.6,
            label=GNAMES[g])
    for i, (v, l0) in enumerate(zip(vals, left)):
        if v >= 0.07:
            p = pcts[i][g]
            t = f'{100 * v:.0f}' + ('*' if np.isfinite(p) and (p < 2.5 or p > 97.5) else '')
            ab.text(l0 + v / 2, i, t, ha='center', va='center', fontsize=9, color=INK)
    left += vals
ab.set_yticks(range(len(rows))); ab.set_yticklabels(names, fontsize=10, color=INK2); ab.invert_yaxis()
ab.set_ylim(len(rows) - 0.4, -0.6)
ab.set_xlim(0, 1); ab.set_xticks([0, 0.25, 0.5, 0.75, 1]); ab.set_xticklabels(['0', '25', '50', '75', '100%'])
ab.tick_params(colors=INK2, labelsize=8, length=0)
ab.grid(axis='x', color=GRIDC, lw=0.7); ab.set_axisbelow(True)
for sp in ab.spines.values():
    sp.set_visible(False)
ab.set_title('share of mode energy by group, whole record (1979-2018)', fontsize=10, loc='left', color=INK)
h, l = ab.get_legend_handles_labels()
ab.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.45, -0.1))
fig.suptitle(f'N={SIZE} {LOSS}: {LABEL} vs the whole domain -- location and mode-group energy shares '
             f'(seed mean, {share.shape[0]} seeds; * = outside random-box null 95% range)',
             fontsize=10.5, color=INK, x=0.01, ha='left')
tag = LABEL.lower().replace(' ', '_').replace('.', '')
fig.savefig(f'{RES}/{tag}_share.png', dpi=130, bbox_inches='tight', facecolor='white')
print(f'saved {RES}/{tag}_share.png', flush=True)
