"""README animation: one test window with minor loops (python fig_gif.py 3E6 100 514 for the README), traced in time and in the B-H plane. During the warm-up
(first k samples) H is measured and the circuit only follows; afterwards it predicts H from B and the temperature.
Uses the classical circuit (32 parameters per material) on the CPU reference scan.
Usage: python fig_gif.py [material] [k] [window] [out]   -> paper/fig/bh.gif"""
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import compact_m, scan
compact_m.scan = scan.scan_ref; compact_m.DEV = torch.device("cpu")
from compact_m import CompactM
from mc2lib import ROOT, windows, metrics

MAT = sys.argv[1] if len(sys.argv) > 1 else "N87"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
Q = int(sys.argv[3]) if len(sys.argv) > 3 else None                                # window index (else automatic)
OUT = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / "paper/fig/bh.gif")
INK, MUTED, GRID, WARM, MEAS, CIRC = "#0b0b0b", "#6f6e69", "#e4e3de", "#efeeea", "#2a78d6", "#eb6834"

ck = torch.load(ROOT / "models/nere.pt", map_location="cpu")
m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]); m.load_state_dict(ck["sd"], strict=False); m.eval()
e, mid = ck["names"].index("final_e1"), ck["mats"].index(MAT)
w = windows("test", MAT)
if "k" in w:
    w = {n: v[w["k"] == K] for n, v in w.items()}
with torch.no_grad():
    Hp = np.concatenate([m(torch.tensor(w["B"][j:j + 256]), torch.tensor(w["H"][j:j + 256]), torch.tensor(w["T"][j:j + 256]),
                           torch.full((len(w["B"][j:j + 256]),), mid), K)[e].numpy() for j in range(0, len(w["B"]), 256)])
sre, _ = metrics(Hp, w["H"], w["B"], K)
# a window with minor loops: many reversals of dB/dt, then the median SRE among those
rev = (np.diff(np.sign(np.diff(w["B"], axis=1)), axis=1) != 0).sum(1)
amp = np.ptp(w["B"], axis=1)
cand = np.where((rev >= 6) & (rev <= 16) & (amp >= np.percentile(amp, 70)))[0]
q = cand[np.argsort(sre[cand])[len(cand) // 2]] if Q is None else Q
B, H, P, T = 1e3 * w["B"][q], w["H"][q], Hp[q], w["T"][q]
print(f"{MAT} window {q}: {rev[q]} reversals, {T:.0f} °C, SRE {100 * sre[q]:.1f} %")
t = np.arange(1000) / 16                                                              # µs (16 MHz sampling)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig = plt.figure(figsize=(8.0, 3.3), dpi=100, facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=(1.55, 1), left=0.09, right=0.985, bottom=0.14, top=0.86, wspace=0.28, hspace=0.12)
aB, aH, aL = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
for a in (aB, aH):
    a.axvspan(0, K / 16, color=WARM, lw=0); a.set_xlim(0, t[-1])
aB.plot(t, B, color=INK, lw=1.0); aB.set_ylabel("B in mT"); aB.tick_params(labelbottom=False)
aH.set_xlabel("time in µs"); aH.set_ylabel("H in A/m")
pad = 0.08 * (max(H.max(), P.max()) - min(H.min(), P.min()))
lim = (min(H.min(), P.min()) - pad, max(H.max(), P.max()) + pad)
aH.set_ylim(*lim); aL.set_xlim(*lim); aL.set_ylim(B.min() - 0.08 * np.ptp(B), B.max() + 0.08 * np.ptp(B))
aL.set_xlabel("H in A/m"); aL.set_ylabel("B in mT"); aL.grid(color=GRID, lw=0.5); aL.set_axisbelow(True)
lHm, = aH.plot([], [], color=MEAS, lw=1.6, label="measured")
lHp, = aH.plot([], [], color=CIRC, lw=1.4, label="classical circuit (32 parameters)")
lLm, = aL.plot([], [], color=MEAS, lw=1.6); lLp, = aL.plot([], [], color=CIRC, lw=1.4)
dm, = aL.plot([], [], "o", ms=5, color=MEAS, mec="white", mew=1); dp, = aL.plot([], [], "o", ms=5, color=CIRC, mec="white", mew=1)
fig.legend(handles=[lHm, lHp], loc="upper left", bbox_to_anchor=(0.08, 0.995), ncol=2, frameon=False, fontsize=9)
fig.text(0.985, 0.955, f"ferrite {MAT}, {T:.0f} °C, test window", ha="right", fontsize=8.5, color=MUTED)
stat = fig.text(0.985, 0.905, "", ha="right", fontsize=8.5, color=INK)

STEP, HOLD = 10, 25
frames = list(range(STEP, 1001, STEP)) + [1000] * HOLD


def draw(n):
    lHm.set_data(t[:n], H[:n]); lLm.set_data(H[:n], B[:n]); dm.set_data([H[n - 1]], [B[n - 1]])
    if n > K:
        lHp.set_data(t[K - 1:n], P[K - 1:n]); lLp.set_data(P[K - 1:n], B[K - 1:n]); dp.set_data([P[n - 1]], [B[n - 1]])
        s = np.sqrt(((P[K:n] - H[K:n]) ** 2).sum() / (H[K:n] ** 2).sum())
        stat.set_text(f"prediction from B and temperature, SRE {100 * s:.1f} %")
    else:
        lHp.set_data([], []); lLp.set_data([], []); dp.set_data([], [])
        stat.set_text("warm-up: H measured, the circuit state is estimated")
    return lHm, lHp, lLm, lLp, dm, dp, stat


anim = FuncAnimation(fig, draw, frames=frames, blit=False)
anim.save(OUT, writer=PillowWriter(fps=25))

# shrink: one shared 16-color palette from the last frame, no dithering, frames stored as differences (~170 kB)
from PIL import Image
src = Image.open(OUT); fr = []
for i in range(src.n_frames):
    src.seek(i); fr.append(src.convert("RGB"))
pal = fr[-1].quantize(16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
fr = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in fr]
fr[0].save(OUT, save_all=True, append_images=fr[1:], duration=[40] * (len(fr) - 1) + [40 * HOLD], loop=0, optimize=True, disposal=1)
