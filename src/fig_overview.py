"""Paper figure (page 1): the prediction task and measured B-H trajectories that make it hard.
(a) one test window: B and H, warm-up with known H, prediction part; (b) deep saturation at low frequency;
(c) DC-biased minor loops with fast ripple; (d) temperature; (e) frequency (rate dependence)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mc2lib import ROOT, windows, load_material

INK, MUTED, SHADE = "#0b0b0b", "#8a8984", "#efeeea"
C = ["#2a78d6", "#eb6834", "#1baf7a"]                  # categorical slots 1-3 (validated all-pairs)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "xtick.major.size": 2, "ytick.major.size": 2})


def by_f(mat, f):
    """The file of a material measured at the switching frequency f."""
    return next(d for d in load_material(mat) if abs(d["f"] - f) < 1)


def span(d, T, lo, hi, W=1000):
    """1000-sample window of a row at temperature T whose flux range best matches [lo, hi]."""
    rows = np.where(np.round(d["T"]) == T)[0]; best = (np.inf, 0, 0)
    for r in rows:
        for s0 in range(0, d["B"].shape[1] - W, 250):
            b = d["B"][r, s0:s0 + W]; sc = abs(b.min() - lo) + abs(b.max() - hi)
            if sc < best[0]: best = (sc, r, s0)
    _, r, s0 = best
    return d["B"][r, s0:s0 + W], d["H"][r, s0:s0 + W]


def peak(d, W=1000):
    """1000-sample window around the largest |H| of a file (deep saturation)."""
    r = int(np.argmax(np.abs(d["H"]).max(1))); c = int(np.argmax(np.abs(d["H"][r])))
    s0 = min(max(c - W // 2, 0), d["B"].shape[1] - W)
    return d["B"][r, s0:s0 + W], d["H"][r, s0:s0 + W], d["T"][r]


fig = plt.figure(figsize=(3.5, 4.3), constrained_layout=True)
gs = fig.add_gridspec(4, 2, height_ratios=[0.55, 0.85, 1, 1])
# (a) the task
w = windows("test", "3C94"); k = 500
nrev = (np.diff(np.sign(np.diff(w["B"], axis=1)), axis=1) != 0).sum(1)
q = np.argsort(nrev)[int(0.9 * len(nrev))]                    # a window with many reversals
t = np.arange(1000) / 16
axb, axh = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
for ax in (axb, axh):
    ax.axvspan(0, k / 16, color=SHADE, lw=0); ax.set_xlim(0, t[-1])
axb.plot(t, 1e3 * w["B"][q], color=C[0], lw=0.7); axb.set_ylabel("B in mT"); axb.tick_params(labelbottom=False)
axh.plot(t[:k], w["H"][q, :k], color=C[1], lw=0.7)
axh.plot(t[k:], w["H"][q, k:], color=C[1], lw=0.7, ls=(0, (2, 1.5)))
axh.set_ylabel("H in A/m"); axh.set_xlabel("time in µs")
yl = axh.get_ylim(); ytop = yl[1] + 0.12 * (yl[1] - yl[0]); axh.set_ylim(yl[0], ytop)
axh.text(k / 32, ytop, "warm-up: B, H, T known", ha="center", va="top", fontsize=6.5, color=INK)
axh.text((k + 1000) / 32, ytop, "predict H from B, T", ha="center", va="top", fontsize=6.5, color=INK)
axb.set_title(f"(a) MC2 task, 3C94, {w['T'][q]:.0f} °C", fontsize=7, loc="left", color=INK)

def loop(ax, title):
    """Title and axis labels of a B-H loop panel."""
    ax.set_title(title, fontsize=7, loc="left", color=INK); ax.set_xlabel("H in A/m"); ax.set_ylabel("B in mT")

# (b) deep saturation, 3C90 at 50 kHz and 70 °C
B, H, T = peak(by_f("3C90", 50e3)); ax = fig.add_subplot(gs[2, 0]); ax.plot(H, 1e3 * B, color=C[0], lw=0.6)
loop(ax, "(b) saturation, 3C90")
# (c) biased minor loops with fast ripple, FEC014 at 800 kHz
d = by_f("FEC014", 800e3); kern = np.ones(20) / 20
drift = np.array([np.ptp(np.convolve(b[-1000:], kern, "valid")) for b in d["B"]]); i = int(np.argmax(drift))
ax = fig.add_subplot(gs[2, 1]); ax.plot(d["H"][i, -1000:], 1e3 * d["B"][i, -1000:], color=C[0], lw=0.5)
loop(ax, "(c) minor loops, FEC014")
# (d) temperature, 3C94 at 200 kHz, windows spanning the same flux range
d = by_f("3C94", 200e3); ax = fig.add_subplot(gs[3, 0])
for c, T in zip(C, (25, 50, 70)):
    B, H = span(d, T, -0.15, 0.15); ax.plot(H, 1e3 * B, color=c, lw=0.5, label=f"{T} °C")
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
ax.legend(frameon=False, fontsize=6, handlelength=1, loc="upper left"); loop(ax, "(d) temperature, 3C94")
# (e) frequency, 3C94 at 25 °C, windows spanning the same flux range
ax = fig.add_subplot(gs[3, 1])
for c, f in zip(C, (50e3, 200e3, 800e3)):
    B, H = span(by_f("3C94", f), 25, -0.15, 0.15); ax.plot(H, 1e3 * B, color=c, lw=0.5, label=f"{f / 1e3:.0f} kHz")
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
ax.legend(frameon=False, fontsize=6, handlelength=1, loc="upper left"); loop(ax, "(e) frequency, 3C94")
fig.savefig(ROOT / "paper/fig/overview.pdf", metadata={"CreationDate": None}); fig.savefig(ROOT / "paper/fig/overview.png", dpi=200)
