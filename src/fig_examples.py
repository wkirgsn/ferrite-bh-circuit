"""Paper figure: representative test windows (median SRE of the classical circuit, 50/50 scenario) for several materials."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mc2lib import ROOT, windows, metrics
import compact_m, scan
from compact_m import CompactM, DEV

if DEV.type == "cpu":   # GPU busy or absent: the pure-PyTorch reference scan gives the same result
    compact_m.scan = scan.scan_ref

SEL = [("nere", "final_e1", m) for m in ("3C94", "N49", "77", "3F4", "N30", "FEC007")]
INK, MUTED, MEAS, LAW = "#0b0b0b", "#8a8984", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 3, figsize=(7.1, 3.0), constrained_layout=True)
k = 500
for ax, (ckn, var, mt) in zip(axs.ravel(), SEL):
    ck = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV)
    m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
    w = windows("test", mt); e, mid = ck["names"].index(var), ck["mats"].index(mt)
    if "k" in w:
        w = {n: v[w["k"] == k] for n, v in w.items()}
    with torch.no_grad():                                                            # chunks of 512 windows (memory)
        Hp = np.concatenate([m(torch.tensor(w["B"][j:j + 512], device=DEV), torch.tensor(w["H"][j:j + 512], device=DEV),
                               torch.tensor(w["T"][j:j + 512], device=DEV), torch.full((len(w["B"][j:j + 512]),), mid, device=DEV),
                               k)[e].cpu().numpy() for j in range(0, len(w["B"]), 512)])
    sre, _ = metrics(Hp, w["H"], w["B"], k); q = np.argsort(sre)[len(sre) // 2]
    t = np.arange(1000) / 16                                                         # µs
    ax.axvspan(0, k / 16, color="#efeeea", lw=0)
    ax.plot(t, w["H"][q], color=MEAS, lw=0.8, label="measured")
    ax.plot(t[k:], Hp[q, k:], color=LAW, lw=0.8, label="classical circuit")
    ax.set_title(f"{mt}, {w['T'][q]:.0f} °C, SRE {100*sre[q]:.1f} %", fontsize=7, color=INK)
    ax.set_xlim(0, t[-1]); ax.tick_params(colors=MUTED, labelcolor=INK, length=2)
for ax in axs[1]: ax.set_xlabel("time in µs")
for ax in axs[:, 0]: ax.set_ylabel("$H$ in A/m")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False, fontsize=7)
fig.savefig(ROOT / "paper/fig/examples.pdf", metadata={"CreationDate": None}); fig.savefig(ROOT / "paper/fig/examples.png", dpi=150)
