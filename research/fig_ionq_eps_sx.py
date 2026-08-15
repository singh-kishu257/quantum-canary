import pathlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_PATH   = SCRIPT_DIR / 'data' / 'hardware' / \
              'ionq_canary_allq_0-19_20260810T213159.json'
FIG_DIR     = SCRIPT_DIR.parent / 'research' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


with open(DATA_PATH) as f:
    d = json.load(f)

N_QUBITS  = len(d['qubits'])                       
published = d['published_device_eps_1q']             
qubits    = np.arange(N_QUBITS)

eps   = np.array([d['canary'][str(q)]['eps_sx']       for q in qubits])
sigma = np.array([d['canary'][str(q)]['eps_sx_sigma'] for q in qubits])

ratio = eps / published
outlier_mask = ratio > 2.0          


C_NORMAL  = '#3A6B9E'    
C_OUTLIER = '#9E3A3A'    
C_SPEC    = '#1A3050'    
C_BAND    = '#C8D8E8'    
C_AXIS    = '#2A2A2A'


FW, FH   = 7.16, 3.60
fig, axes = plt.subplots(1, 2, figsize=(FW, FH),
                          gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})
fig.patch.set_facecolor('white')


ax = axes[0]
ax.set_facecolor('#F8F9FA')


ax.axhspan(published / 2, published * 2,
           color=C_BAND, alpha=0.55, zorder=1,
           label=r'$\times 2$ band around spec')


ax.axhline(published, color=C_SPEC, lw=1.2, ls='--', zorder=3,
           label=r'IonQ published $\varepsilon_{1q} = 2.0\times10^{-4}$')

colors = np.where(outlier_mask, C_OUTLIER, C_NORMAL)

for q in qubits:
    col = C_OUTLIER if outlier_mask[q] else C_NORMAL
    ax.errorbar(q, eps[q], yerr=sigma[q],
                fmt='o', color=col, mec=col, mfc=col,
                ecolor=col, elinewidth=0.9, capsize=2.5, capthick=0.9,
                ms=4.5, lw=0, zorder=4)


for q in qubits:
    if outlier_mask[q] or ratio[q] > 1.5:
        va   = 'bottom' if eps[q] > published else 'top'
        dy   = sigma[q] * 1.6 + 5e-6
        col  = C_OUTLIER if outlier_mask[q] else '#5A3A80'
        ax.text(q, eps[q] + dy, f'Q{q}',
                ha='center', va='bottom', fontsize=6.5,
                color=col, fontweight='bold', zorder=5)


ax.set_yscale('log')
ax.set_xlim(-0.8, N_QUBITS - 0.2)
ax.set_ylim(3e-5, 8e-4)
ax.set_xlabel('Qubit index', fontsize=8.5, labelpad=3)
ax.set_ylabel(r'$\varepsilon_{sx}$ (single-qubit gate error)', fontsize=8.5)
ax.set_xticks(qubits)
ax.set_xticklabels([str(q) for q in qubits], fontsize=6.5)
ax.tick_params(axis='y', labelsize=7.0)


ax.yaxis.set_minor_locator(mticker.LogLocator(
    base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: fr'$10^{{{int(round(np.log10(v)))}}}$'
                 if abs(np.log10(v) - round(np.log10(v))) < 0.01
                 else fr'${v*1e4:.0f}\!\times\!10^{{-4}}$'))

ax.grid(axis='y', which='major', color='#CCCCCC', lw=0.5, zorder=0)
ax.grid(axis='y', which='minor', color='#E8E8E8', lw=0.3, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor('#BBBBBB')
    spine.set_linewidth(0.6)


legend_elements = [
    Line2D([0], [0], color=C_SPEC,    lw=1.2, ls='--',
           label=r'Published $\varepsilon_{1q}=2.0\times10^{-4}$'),
    plt.Rectangle((0, 0), 1, 1, fc=C_BAND, ec='none', alpha=0.6,
                  label=r'Within $\times 2$ of spec'),
    Line2D([0], [0], marker='o', color=C_NORMAL, ms=5, lw=0,
           label='19/20 qubits (within band)'),
    Line2D([0], [0], marker='o', color=C_OUTLIER, ms=5, lw=0,
           label='Q3 (2.26× spec)'),
]
ax.legend(handles=legend_elements, fontsize=6.5, framealpha=0.92,
          edgecolor='#BBBBBB', loc='upper right', fancybox=False,
          handlelength=1.4, handletextpad=0.5)

ax.set_title(r'IonQ Forte-1 $\cdot$ Canary-recovered $\varepsilon_{sx}$, '
             r'all 20 qubits',
             fontsize=8.5, fontweight='bold', color='#141E2A', pad=4)


mean_v   = eps.mean()
median_v = np.median(eps)
ax.annotate(fr'mean $= {mean_v:.2e}$''\n'
            fr'median $= {median_v:.2e}$''\n'
            r'$N=20$ qubits',
            xy=(0.02, 0.05), xycoords='axes fraction',
            fontsize=6.3, color='#3A4A5A',
            bbox=dict(fc='white', ec='#BBBBBB', pad=3, lw=0.5))

ax2 = axes[1]
ax2.set_facecolor('#F8F9FA')

bins    = np.logspace(np.log10(0.3), np.log10(3.0), 12)
h_vals  = ratio
hcolors = [C_OUTLIER if r > 2 else C_NORMAL for r in h_vals]

counts, edges = np.histogram(h_vals, bins=bins)
for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    mask = (h_vals >= lo) & (h_vals < hi)
    col  = C_OUTLIER if any(outlier_mask[mask]) else C_NORMAL
    ax2.barh(0.5 * (lo + hi), mask.sum(),
             height=(hi - lo) * 0.85,
             color=col, edgecolor='white', lw=0.4, zorder=3)

ax2.axhline(1.0, color=C_SPEC, lw=1.2, ls='--', zorder=4)
ax2.axhspan(0.5, 2.0, color=C_BAND, alpha=0.55, zorder=1)

ax2.set_yscale('log')
ax2.set_ylim(3e-5 / published, 8e-4 / published)
ax2.set_xlabel('Count', fontsize=8.0, labelpad=3)
ax2.set_ylabel(r'$\varepsilon_{sx}\ /\ \varepsilon_\mathrm{spec}$',
               fontsize=8.0, labelpad=2)
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()
ax2.tick_params(axis='both', labelsize=6.5)
ax2.set_xticks([0, 5, 10])
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(1))

ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'{v:.1f}×'))

ax2.grid(axis='x', which='major', color='#CCCCCC', lw=0.5, zorder=0)
for spine in ax2.spines.values():
    spine.set_edgecolor('#BBBBBB')
    spine.set_linewidth(0.6)

ax2.text(0.97, 1.05, r'spec $\rightarrow$',
         ha='right', va='bottom', fontsize=5.8,
         color=C_SPEC, transform=ax2.get_yaxis_transform())

# save
for ext in ('pdf', 'png'):
    out = FIG_DIR / f'fig_ionq_eps_sx.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.03)
    print(f'Saved: {out}')