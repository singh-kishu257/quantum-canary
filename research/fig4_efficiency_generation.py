import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv
from collections import defaultdict
import pathlib

BASE = pathlib.Path(__file__).parent
data = defaultdict(lambda: defaultdict(dict))
with open(BASE / 'data' / 'fig7_benchmark.csv') as f:
    for row in csv.DictReader(f):
        if row['method'] != 'Canary':
            continue
        data[row['architecture']][row['parameter']][int(row['budget'])] = {
            'r2': float(row['r2']),
            'lo': float(row['r2_lo']),
            'hi': float(row['r2_hi']),
        }

ARCHS        = ['superconducting', 'trapped_ion', 'neutral_atom']
ARCH_LABELS  = ['Superconducting', 'Trapped-ion', 'Neutral atom']
PARAMS       = ['T1', 'T2', 'delta_omega', 'epsilon_sx']
PARAM_LABELS = [r'$T_1$', r'$T_2$', r'$\Delta\omega$', r'$\varepsilon_{sx}$']
NATIVE       = 9900

COLORS  = {'T1':'#3A6B9E', 'T2':'#4A8A60',
           'delta_omega':'#7050A0', 'epsilon_sx':'#A05030'}
MARKERS = {'T1':'o', 'T2':'s', 'delta_omega':'^', 'epsilon_sx':'D'}

YMIN, YMAX = -0.10, 1.010

FW, FH = 7.16, 2.90
fig, axes = plt.subplots(1, 3, figsize=(FW, FH), sharey=True)
fig.patch.set_facecolor('white')

for col, (arch, arch_label) in enumerate(zip(ARCHS, ARCH_LABELS)):
    ax = axes[col]
    ax.set_facecolor('#F8F9FA')
    ax.set_xscale('log')

    for param, plabel in zip(PARAMS, PARAM_LABELS):
        d = data[arch][param]
        budgets = sorted(d.keys())
        r2s = [max(d[b]['r2'], YMIN) for b in budgets]   # clip for display
        raw = [d[b]['r2'] for b in budgets]
        los = [max(d[b]['lo'], YMIN) for b in budgets]
        his = [min(d[b]['hi'], YMAX) for b in budgets]

        ax.plot(budgets, r2s, color=COLORS[param], lw=1.4,
                marker=MARKERS[param], ms=4.0, mfc=COLORS[param],
                mec=COLORS[param], label=plabel, zorder=4, clip_on=True)
        ax.fill_between(budgets, los, his,
                        color=COLORS[param], alpha=0.13, zorder=3)

        # Annotate clipped points (negative R²)
        for b, v in zip(budgets, raw):
            if v < YMIN:
                ax.text(b, YMIN + 0.005, f'{v:.0f}↓',
                        ha='center', va='bottom', fontsize=4.2,
                        color=COLORS[param], zorder=5,
                        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.7))

    # R²=0.95 line
    ax.axhline(0.95, color='#999999', lw=0.7, ls='--', zorder=2)
    # Native budget line
    ax.axvline(NATIVE, color='#3A4A5A', lw=0.8, ls=':', zorder=2, alpha=0.7)

    ax.set_title(arch_label, fontsize=8.5, fontweight='bold',
                 color='#1A2535', pad=3)
    ax.set_xlabel('Shot budget', fontsize=7.5, labelpad=3)
    if col == 0:
        ax.set_ylabel(r'$R^2$', fontsize=8.0)

    ax.set_ylim(YMIN, YMAX)
    ax.set_xlim(800, 62000)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x/1000):d}k' if x >= 1000 else str(int(x))))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis='y', which='major', color='#CCCCCC', lw=0.5, zorder=1)
    ax.grid(axis='y', which='minor', color='#EEEEEE', lw=0.3, zorder=1)
    ax.grid(axis='x', which='major', color='#DDDDDD', lw=0.4, zorder=1)

    # 0.95 label on first panel only
    if col == 0:
        ax.text(870, 0.957, r'$R^2\!=\!0.95$',
                ha='left', va='bottom', fontsize=5.0, color='#888888')
    # Native budget label
    ax.text(NATIVE*1.06, 0.02, '9,900\nshots', ha='left', va='bottom',
            fontsize=4.8, color='#3A4A5A', linespacing=1.2)

    for spine in ax.spines.values():
        spine.set_edgecolor('#BBBBBB')
        spine.set_linewidth(0.6)

# Legend
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           fontsize=7.2, framealpha=0.95, edgecolor='#AAAAAA',
           fancybox=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=1.6, handletextpad=0.4, columnspacing=1.0)

plt.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.20,
                    wspace=0.10)
fig.suptitle('Shot-budget efficiency  ·  Canary  ·  all parameters & architectures',
             fontsize=8.0, fontweight='bold', color='#1A2535', y=0.99)

plt.savefig(BASE / 'figures' / 'fig4_efficiency.png', dpi=300,
            bbox_inches='tight', facecolor='white', pad_inches=0.03)
plt.savefig(BASE / 'figures' / 'fig4_efficiency.pdf', dpi=300,
            bbox_inches='tight', facecolor='white', pad_inches=0.03)
print("Done.")