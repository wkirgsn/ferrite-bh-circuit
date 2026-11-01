"""Paper figure: the classical and the gain-scheduled circuits among the best half (by each metric) of the 32 MC2 final
contenders (official results, results/mc2_official_results.xlsx): parameters per material over the average
95th-percentile sequence error and energy error on the five final-evaluation materials, as in the challenge's ranking
plot. Classical circuit: final model; MLP16- / MLP64-scheduled circuit: hyper_m.py, option AB, joint training. All:
official_eval.py on the official test set, mean and range over training seeds (until the seeds of a scheduled circuit
are complete, the available ones; NSCHED limits them to the first seeds, e.g. 5 for arXiv v1).
RHINO-MAG GRU8 (team SiegenPaderborn) is shown at its official position. Contestants above NERE_MAX are not shown.
Usage: python fig_ranking.py   -> paper/fig/ranking.pdf"""
import glob, json, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adjustText import adjust_text
from mc2lib import ROOT

INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"
OURS, RHINO = "#eb6834", "#2a78d6"                    # categorical slots as in the other figures (this work orange)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK})
TOP = 16                                              # best half of the 32 contenders per metric (official ranking)
NERE_MAX = 10.0                                       # energy-error panel: contestants above 10 % not shown

rk = pd.read_excel(ROOT / "results/mc2_official_results.xlsx", "Ranking")
avg = lambda fs: [json.load(open(f))["average"] for f in fs]
R = lambda pat: sorted(glob.glob(str(ROOT / "results" / pat)))
NSCHED = 5                                            # seeds of the scheduled circuits used (0 .. NSCHED-1)
S = lambda base, first, var: R(f"official_{first}_{var}.json") + sum((R(f"official_{base}_s{i}_{var}.json") for i in range(1, NSCHED)), [])
ours = [  # (label, parameters per material, official averages of all seeds, marker)
    ("classical circuit", 32, avg(R("official_nere_final_e1.json") + R("official_ne_s*_final_e1.json")), "o"),
    ("MLP16-scheduled circuit", 656, avg(S("hyAB16j", "hyAB16j_long", "final_e1+hyperAB16j")), "D"),
    ("MLP64-scheduled circuit", 5552, avg(S("hyAB64j", "hyAB64j_s0", "final_e1+hyperAB64j")), "D")]
for lab, n, a, _ in ours:
    print(f"{lab}: {len(a)} seed(s)")

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
for ax, col, rcol, key, xlab in ((axs[0], "Average 95% Sequence Error", "Sequence Error Ranking", "sre_p95", "average 95th-percentile SRE in %"),
                                 (axs[1], "Average 95% Energy Error", "Energy Error Ranking", "nere_p95", "average 95th-percentile NERE in %")):
    texts = []
    for _, r in rk[(rk[rcol] <= TOP) | (rk["Team"] == "SiegenPaderborn")].iterrows():
        x, y = float(r[col]), float(r["Average Parameters"])
        if key == "nere_p95" and x > NERE_MAX:
            continue
        if r["Team"] == "SiegenPaderborn":
            ax.scatter(x, y, s=22, color=RHINO, zorder=4)
            texts.append(ax.text(x, y, "SiegenPaderborn (RHINO-MAG)", fontsize=6, color=INK))
            continue
        ax.scatter(x, y, s=10, color=MUTED, zorder=2, lw=0)
        texts.append(ax.text(x, y, r["Team"], fontsize=6, color=MUTED))
    xs = [np.mean([o[key] for o in a]) for _, _, a, _ in ours]
    ax.plot(xs, [n for _, n, _, _ in ours], ls="--", lw=0.8, color=OURS, alpha=0.5, zorder=3)   # classical -> MLP16 -> MLP64
    for (lab, npar, a, mk), x in zip(ours, xs):
        v = np.array([o[key] for o in a])
        ax.errorbar(x, npar, xerr=[[x - v.min()], [v.max() - x]], fmt=mk, ms=5 if mk == "o" else 4.5, color=OURS, ecolor=OURS,
                    elinewidth=0.8, capsize=2, zorder=5)
        off, ha = ((6, 3) if npar == 656 else (6, -3)), "left"        # fixed, right of the marker (free area)
        ax.annotate(lab, (x, npar), xytext=off, textcoords="offset points", ha=ha, va="bottom" if off[1] > 0 else "top",
                    fontsize=6.5, color=INK, fontweight="bold")
        print(f"{lab} {key} {x:.2f}: rank {int((rk[col] < x).sum()) + 1} of {len(rk) + 1}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(15, 6e6)
    ticks = [10, 15, 20, 30, 40] if key == "sre_p95" else [1.5, 2, 3, 5, 7, 10]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.set_xlabel(xlab); ax.set_ylabel("parameters per material")
    adjust_text(texts, ax=ax, expand=(1.15, 1.4), arrowprops=dict(arrowstyle="-", color=GRID, lw=0.4))
fig.savefig(ROOT / "paper/fig/ranking.pdf", metadata={"CreationDate": None}); fig.savefig(ROOT / "paper/fig/ranking.png", dpi=200)
