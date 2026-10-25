"""Fig. 1 -- Quantum Canary system architecture 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath

FONT = "Cambria"
mpl.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.fallback": None,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
})
W, H = 181.0, 90.5
PT = 25.4 / 72.0
BODY, TITLE = 8, 9
PITCH = 1.16 * BODY * PT

INK = "#1A1A1A"; GREY = "#4D4D4D"; OUTLINE = "#333333"; FRAME = "#404040"
ARROW_FC, ARROW_EC = "#A6A6A6", "#7A7A7A"
QPU_H, QPU_B = "#8DCFEF", "#E9F3FB"
QC_H, QC_B = "#F2A98A", "#FBE4D8"
DES_H, DES_B = "#FFD24D", "#FFF0C4"
EXE_H, EXE_B = "#8DCFEF", "#EAF3FB"
INV_H, INV_B = "#FF8080", "#FFD9D9"
OUT_H, OUT_B = "#6CCB52", "#E2F0D9"
F_QPU = "#F3DDF6"; F_YEL = "#FFFFC0"; F_LAV = "#F3DDF6"; F_PINK = "#FDE6EC"; F_ORG = "#F9CFB3"
F_GRN = "#9ED98C"; F_WHT = "#FFFFFF"; F_TILE = "#F8E6F6"
C_T1, C_RAM, C_ECHO, C_GATE = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
STAR = "#F2B705"
H_GREEN, H_YELLOW, H_RED = "#2CA02C", "#F2C14E", "#D62728"

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
OVERFLOW = []


# ----------------------------------------------------------------- helpers
def wmm(s, size=BODY, weight="normal"):
    fp = fm.FontProperties(family=FONT, size=size, weight=weight)
    return TextPath((0, 0), s, size=size, prop=fp).get_extents().width * PT


def _tokens(s):
    """Split on spaces, keeping $...$ math segments whole."""
    toks, cur, inmath = [], "", False
    for ch in s:
        if ch == "$":
            inmath = not inmath
        if ch == " " and not inmath:
            toks.append(cur); cur = ""
        else:
            cur += ch
    toks.append(cur)
    return [t for t in toks if t]


def wrap(s, width, size=BODY, weight="normal"):
    out, cur = [], ""
    for w in _tokens(s):
        t = (cur + " " + w).strip()
        if wmm(t, size, weight) <= width or not cur:
            cur = t
        else:
            out.append(cur); cur = w
    out.append(cur)
    return out


def text(x, y, s, size=BODY, weight="normal", ha="left", va="baseline", color=INK, z=10, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, zorder=z, **kw)


def rbox(x0, y0, x1, y1, fc, ec=OUTLINE, lw=0.75, r=1.6, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))


def section(x0, y0, x1, y1, hh, title, hc, bc, z=0, size=TITLE):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=bc, ec="none", zorder=z))
    ax.add_patch(Rectangle((x0, y1 - hh), x1 - x0, hh, fc=hc, ec=GREY, lw=0.5, zorder=z + 0.1))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=FRAME, lw=0.8,
                           ls=(0, (3.2, 1.8)), zorder=z + 0.2))
    text((x0 + x1) / 2, y1 - hh / 2, title, size=size, weight="bold", ha="center", va="center_baseline", z=z + 1)


def star(x, y, r=1.25, z=12):
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    rad = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
    ax.add_patch(Polygon(np.c_[x + rad * np.cos(ang), y + rad * np.sin(ang)], closed=True,
                         fc=STAR, ec="#7A5A00", lw=0.4, joinstyle="miter", zorder=z))


def block_arrow(pts, shaft=2.0, head_w=4.2, head_l=2.6, z=6):
    pts = [np.array(p, float) for p in pts]
    s = shaft / 2
    tip, prev = pts[-1], pts[-2]
    d = (tip - prev) / np.linalg.norm(tip - prev); n = np.array([-d[1], d[0]])
    base = tip - d * head_l
    if len(pts) == 2:
        p0 = pts[0]
        poly = [p0 + n * s, base + n * s, base + n * head_w / 2, tip, base - n * head_w / 2, base - n * s, p0 - n * s]
    else:
        p0, c = pts[0], pts[1]
        d0 = (c - p0) / np.linalg.norm(c - p0); n0 = np.array([-d0[1], d0[0]])

        def corner(sg):
            a = p0 + n0 * s * sg; b = c + n * s * sg
            return np.array([a[0], b[1]]) if abs(d0[0]) < 1e-9 else np.array([b[0], a[1]])
        poly = [p0 + n0 * s, corner(+1), base + n * s, base + n * head_w / 2, tip, base - n * head_w / 2,
                base - n * s, corner(-1), p0 - n0 * s]
    ax.add_patch(Polygon(poly, closed=True, fc=ARROW_FC, ec=ARROW_EC, lw=0.5, joinstyle="miter", zorder=z))


def badge(x, y, n):
    ax.add_patch(Circle((x, y), 1.7, fc="#262626", ec="white", lw=0.5, zorder=12))
    text(x, y - 0.05, str(n), weight="bold", ha="center", va="center", color="white", z=13)


def items(x0, x1, y_top, entries, y_floor, name):
    """Marker + wrapped text rows. Marker: None | 'dot' | 'star' | ('sw', colour) | ('hl', colour)."""
    y = y_top
    for mk, s in entries:
        indent = 0.0 if mk is None else 2.6
        lines = wrap(s, x1 - x0 - indent)
        base = y - 0.70 * BODY * PT
        if mk == "dot":
            ax.add_patch(Circle((x0 + 0.8, base + 1.0), 0.45, fc=INK, ec="none", zorder=11))
        elif mk == "star":
            star(x0 + 0.95, base + 1.05)
        elif isinstance(mk, tuple) and mk[0] == "sw":
            ax.add_patch(Rectangle((x0 + 0.2, base - 0.45), 1.2, 2.5, fc=mk[1], ec="none", zorder=11))
        elif isinstance(mk, tuple) and mk[0] == "hl":
            ax.add_patch(Circle((x0 + 0.9, base + 1.0), 0.95, fc=mk[1], ec=OUTLINE, lw=0.4, zorder=11))
        for ln in lines:
            text(x0 + indent, base, ln)
            base -= PITCH
        y = base + PITCH - 0.26 * BODY * PT - 0.6
    if y < y_floor:
        OVERFLOW.append((name, round(y_floor - y, 2)))
    return y


def step_box(x0, y0, x1, y1, n, title, fc, entries=(), ls="-", pad=1.6, top=5.05):
    rbox(x0, y0, x1, y1, fc=fc, ls=ls)
    if n is not None:
        badge(x0 + 0.6, y1 - 0.6, n)
    text((x0 + x1) / 2, y1 - 2.55, title, size=TITLE, weight="bold", ha="center", va="center_baseline")
    return items(x0 + pad, x1 - pad, y1 - top, entries, y0 + 0.4, title)


# ----------------------------------------------------------------- sections
HDR, SUBH, TOPY = 5.2, 3.9, 89.5
A = (1.0, 25.0); B = (27.0, 134.0); C = (136.0, 180.0)
section(A[0], 1.0, A[1], TOPY, HDR, "Cloud QPU", QPU_H, QPU_B)
section(B[0], 1.0, B[1], TOPY, HDR, "Quantum Canary", QC_H, QC_B)
section(C[0], 1.0, C[1], TOPY, HDR, "To the user", OUT_H, OUT_B)
# key-feature legend inside the Quantum Canary header bar
lx = B[1] - 1.6 - wmm("Key feature")
star(lx - 1.9, TOPY - HDR / 2 + 0.05)
text(lx, TOPY - HDR / 2, "Key feature", va="center_baseline", z=2)

BX0, BX1 = B[0] + 1.5, B[1] - 1.5                  # sub-panel extent
P1 = (55.9, 83.1); P2 = (32.0, 50.2); P3 = (2.2, 30.6)  # (bottom, top) of the three sub-panels
section(BX0, P1[0], BX1, P1[1], SUBH, "Probe design and submission", DES_H, DES_B, z=1, size=BODY)
section(BX0, P2[0], BX1, P2[1], SUBH, "Execution paths", EXE_H, EXE_B, z=1, size=BODY)
section(BX0, P3[0], BX1, P3[1], SUBH, "Inversion and validation", INV_H, INV_B, z=1, size=BODY)
IX0, IX1 = BX0 + 1.5, BX1 - 1.5
G = 3.6


def row(panel):
    return panel[0] + 1.1, panel[1] - SUBH - 1.1


# ----------------------------------------------------------------- QPU
QB = (2.4, 2.2, 23.6, TOPY - HDR - 1.2)
rbox(*QB, fc=F_QPU, r=2.0)
qc = (QB[0] + QB[2]) / 2
CHIP = "#463F70"; cs, cy = 9.0, 76.8
ax.add_patch(FancyBboxPatch((qc - cs / 2, cy - cs / 2), cs, cs, boxstyle="round,pad=0,rounding_size=0.9",
                            fc=CHIP, ec="none", zorder=4))
for k in np.linspace(-cs / 2 + 1.8, cs / 2 - 1.8, 4):
    for (a0, b0, a1, b1) in [(k, cs / 2, k, cs / 2 + 1.1), (k, -cs / 2, k, -cs / 2 - 1.1),
                             (cs / 2, k, cs / 2 + 1.1, k), (-cs / 2, k, -cs / 2 - 1.1, k)]:
        ax.plot([qc + a0, qc + a1], [cy + b0, cy + b1], color=CHIP, lw=0.8, solid_capstyle="butt", zorder=4)
qpos = [(-2.0, 2.0), (2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)]
for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    ax.plot([qc + qpos[a][0], qc + qpos[b][0]], [cy + qpos[a][1], cy + qpos[b][1]], color="#C9C3E6", lw=0.8, zorder=5)
for (dx, dy) in qpos:
    ax.add_patch(Circle((qc + dx, cy + dy), 0.95, fc="#FFE27A", ec="white", lw=0.4, zorder=6))
text(qc, cy - cs / 2 - 3.9, "Qubits", ha="center", color=GREY)
yy = 46.0
for name, kind in [("IBM", "superconducting"), ("IonQ", "trapped ion")]:
    text(qc, yy, name, weight="bold", ha="center"); yy -= PITCH
    text(qc, yy, kind, ha="center", color=GREY); yy -= PITCH + 0.5

# ----------------------------------------------------------------- row 1: design + submission
r0, r1 = row(P1)
w1 = [31.0, 31.5]; w1.append(IX1 - IX0 - sum(w1) - 2 * G)
X1 = [IX0, IX0 + w1[0] + G, IX0 + w1[0] + w1[1] + 2 * G]
step_box(X1[0], r0, X1[0] + w1[0], r1, 1, "Priors and design", F_YEL, [
    (None, r"Live $T_1$, $T_2$, $\varepsilon$, readout error (else defaults)"),
    ("star", "Delays where data are most informative"),
])
step_box(X1[1], r0, X1[1] + w1[1], r1, 2, "74 probe circuits", F_LAV, [
    (("sw", C_T1), "Relaxation \u00d73"), (("sw", C_RAM), "Ramsey X/Y \u00d78"),
    (("sw", C_ECHO), "Hahn echo \u00d73"), (("sw", C_GATE), "Clifford RB \u00d760"),
])
step_box(X1[2], r0, X1[2] + w1[2], r1, 3, "One short job", F_YEL, [
    ("star", "Parallel groups: shots scale with groups, not qubits"),
    ("dot", "10 interleaved blocks"),
])

# ----------------------------------------------------------------- row 2: execution paths (options)
r0, r1 = row(P2)
w2 = (IX1 - IX0 - G) / 2
step_box(IX0, r0, IX0 + w2, r1, None, "IonQ trapped ions", F_WHT, [
    ("dot", "Gate-error probe only, with an optional cost cap"),
], ls=(0, (3, 1.6)))
step_box(IX0 + w2 + G, r0, IX1, r1, None, "In-job monitoring", F_WHT, [
    ("star", "Experimental: probes woven into your job; results untouched"),
], ls=(0, (3, 1.6)))

# ----------------------------------------------------------------- row 3: inversion + health
r0, r1 = row(P3)
w3 = [58.0]; w3.append(IX1 - IX0 - w3[0] - G)
X3 = [IX0, IX0 + w3[0] + G]
x0, x1 = X3[0], X3[0] + w3[0]
rbox(x0, r0, x1, r1, fc=F_PINK)
badge(x0 + 0.6, r1 - 0.6, 4)
text((x0 + x1) / 2, r1 - 2.55, "Fit noise models", size=TITLE, weight="bold", ha="center", va="center_baseline")
yb0 = r1 - 5.05 - 0.70 * BODY * PT
fam = [("Relaxation", C_T1, "$T_1$"), ("Ramsey", C_RAM, r"$T_2^*,\ \Delta\omega$"),
       ("Hahn echo", C_ECHO, "$T_2$"), ("Clifford RB", C_GATE, r"$\varepsilon$")]
col_x = [x0 + 1.6]
for c in (0, 1):
    names = [fam[2 * c][0], fam[2 * c + 1][0]]
    ar = 1.9 + max(wmm(n) for n in names) + 1.0
    for j in (0, 1):
        name, col, sym = fam[2 * c + j]
        xo = col_x[c]; yb = yb0 - j * (PITCH + 0.3)
        ax.add_patch(Rectangle((xo, yb - 0.45), 1.2, 2.5, fc=col, ec="none", zorder=11))
        text(xo + 1.9, yb, name)
        text(xo + ar, yb, "\u2192", color=GREY)
        text(xo + ar + wmm("\u2192") + 0.7, yb, sym)
    end = col_x[c] + ar + wmm("\u2192") + 0.7 + max(wmm(fam[2 * c][2]), wmm(fam[2 * c + 1][2]))
    col_x.append(end + 1.8)
assert col_x[-1] - 1.8 <= x1 - 1.4, col_x
items(x0 + 1.6, x1 - 1.6, yb0 - 2 * (PITCH + 0.3) + 0.70 * BODY * PT - 0.3, [
    ("star", r"Readout error in the $T_1$, Ramsey and echo models (RB cancels it); $T_2 \leq 2T_1$; $\sigma$ for each"),
], r0 + 0.4, "fit")
step_box(X3[1], r0, IX1, r1, 5, "Health", F_ORG, [
    (("hl", H_GREEN), "GREEN: all fits pass"),
    (("hl", H_YELLOW), r"YELLOW: fit over limit, or $T_2^*$, $T_2$ differ > 20%"),
    (("hl", H_RED), "RED: a fit failed"),
    ("star", r"$\chi^2$ limits: 5% false alarms"),
])

# ----------------------------------------------------------------- To the user
CX0, CX1 = C[0] + 1.5, C[1] - 1.5
IF = (67.6, TOPY - HDR - 1.2)
DE = (46.4, 65.7)
DR = (32.4, 44.6)
RP = (2.2, 30.4)
step_box(CX0, IF[0], CX1, IF[1], None, "Interfaces", F_LAV, [
    ("dot", "Python: check, compare, Monitor"),
    ("dot", "CLI: quantum-canary"),
])
step_box(CX0, DE[0], CX1, DE[1], None, "Decisions", F_WHT, pad=1.1, entries=[
    ("star", "RED qubit \u2192 re-run the check"),
    ("dot", r"Decoupling if $T_2^*/T_2 < 0.8$"),
    ("dot", r"Frame update if $|\Delta\omega|\,T_2^* > 1\,\mathrm{rad}$"),
])
step_box(CX0, DR[0], CX1, DR[1], None, "Drift between checks", F_WHT, [
    ("dot", r"Holm-corrected $z$-tests; stricter for repeats"),
])
# report box with result tiles
x0, x1 = CX0, CX1
rbox(x0, RP[0], x1, RP[1], fc=F_GRN)
badge(x0 + 0.6, RP[1] - 0.6, 6)
text(x0 + 3.0, RP[1] - 2.55, "Report", size=TITLE, weight="bold", va="center_baseline")
ex, ey, ew_, eh = x1 - 5.0, RP[1] - 2.9, 5.6, 2.4
verts = [(ex - ew_ / 2, ey), (ex - ew_ / 4, ey + eh), (ex + ew_ / 4, ey + eh), (ex + ew_ / 2, ey),
         (ex + ew_ / 4, ey - eh), (ex - ew_ / 4, ey - eh), (ex - ew_ / 2, ey)]
ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.CURVE4] * 6), fc="white", ec=INK, lw=0.7, zorder=8))
ax.add_patch(Circle((ex, ey), 0.95, fc=INK, ec="none", zorder=9))
tiles = [(r"$T_1 \pm \sigma$", C_T1), (r"$T_2 \pm \sigma$", C_ECHO), (r"$T_2^* \pm \sigma$", C_RAM),
         (r"$\Delta\omega \pm \sigma$", C_RAM), (r"$\varepsilon \pm \sigma$", C_GATE), ("health", None)]
tw, th = (x1 - x0 - 2.4 - 1.2) / 2, 4.3
for k, (s, col) in enumerate(tiles):
    tx = x0 + 1.2 + (k % 2) * (tw + 1.2)
    ty = RP[1] - 6.0 - (k // 2) * (th + 0.9)
    rbox(tx, ty - th, tx + tw, ty, fc=F_TILE, lw=0.6, r=0.9, z=8)
    if col:
        ax.add_patch(Rectangle((tx + 0.8, ty - th + 0.8), 0.9, th - 1.6, fc=col, ec="none", zorder=9))
        text(tx + 2.4, ty - th / 2, s, va="center_baseline", z=10)
    else:
        for j, hc in enumerate((H_GREEN, H_YELLOW, H_RED)):
            ax.add_patch(Circle((tx + 1.6 + j * 2.2, ty - th / 2), 0.85, fc=hc, ec=OUTLINE, lw=0.4, zorder=9))
        text(tx + 8.2, ty - th / 2, "health", va="center_baseline", z=10)
ty_end = RP[1] - 6.0 - 3 * (th + 0.9)
items(x0 + 1.4, x1 - 1.4, ty_end + 0.2, [
    (None, "+ reasons, prior sources; JSON"),
], RP[0] + 0.4, "report")

# ----------------------------------------------------------------- block arrows
r1lo, r1hi = row(P1); ymid1 = r1lo + 6.0
block_arrow([(QB[2], ymid1), (X1[0], ymid1)])                                  # calibration data
block_arrow([(X1[0] + w1[0], ymid1), (X1[1], ymid1)])
block_arrow([(X1[1] + w1[1], ymid1), (X1[2], ymid1)])
block_arrow([(CX0, IF[0] + 4.0), (X1[2] + w1[2], IF[0] + 4.0)])  # request
BUS = (P2[1] + P1[0]) / 2
bx = X1[2] + w1[2] / 2
block_arrow([(bx, r1lo), (bx, BUS), (QB[2], BUS)], shaft=3.4, head_w=5.6, head_l=2.8)  # probe job
text((QB[2] + bx) / 2 + 2.0, BUS, "Probe job", ha="center", va="center_baseline", z=11)
r3lo, r3hi = row(P3); ymid3 = (r3lo + r3hi) / 2
block_arrow([(QB[2], ymid3), (X3[0], ymid3)])                                  # counts
block_arrow([(X3[0] + w3[0], ymid3), (X3[1], ymid3)])
block_arrow([(IX1, ymid3), (CX0, ymid3)])
# QPU port labels
QR = QB[2] - 0.9
text(QR, ymid1 + 2.0, "Calibration data", ha="right")
text(QR, BUS + 2.6, "Runs probes", ha="right")
text(QR, ymid3 + 2.0, "Counts", ha="right")
print("overflow:", OVERFLOW)

OUT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(exist_ok=True)
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(OUT_DIR / f"fig1.{ext}", **kw)
print("saved:", OUT_DIR / "fig1.pdf", OUT_DIR / "fig1.png")
