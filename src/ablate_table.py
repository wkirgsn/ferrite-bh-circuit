"""Ablation of the final model (paper table tab:ablation) from results/eval_ablate.txt (eval_m.py of the run with
configs/ablate.json): each row removes one classical element from the 32-parameter model, all variants trained jointly.
Usage: python ablate_table.py   -> paper/tables/ablation.typ (a placeholder while the run is missing)"""
import statistics as st
from mc2lib import ROOT
from path_table import parse, G

ROWS = [("final", "classical circuit", 32), ("no_knee", "without saturation knee", 29), ("no_cubic", "without cubic term", 30),
        ("rate_indep", "rate-independent plays ($alpha = 1$)", 30), ("no_relax", "without relaxations", 24),
        ("no_feedback", "without mean-field feedback", 27), ("no_meas_corr", "without measurement correction of $B$", 28),
        ("no_lag", "without lag of $H$", 31)]
p = ROOT / "results/eval_ablate.txt"
d = parse(p) if p.exists() else {}
lines = []
if all(v in d and len(d[v]) == 15 for v, _, _ in ROWS):
    g = d[G]; mats = list(d["final"]); ref = st.mean(d["final"][m][0] for m in mats)
    for v, desc, npar in ROWS:
        x = d[v]; mean = [st.mean(x[m][j] for m in mats) for j in range(4)]; wins = sum(x[m][0] < g[m][0] for m in mats)
        delta = "" if v == "final" else f"{mean[0] - ref:+.2f}"
        lines.append(f"    [{desc}], [{npar}], [{mean[0]:.2f}], [{delta}], " + ", ".join(f"[{y:.2f}]" for y in mean[1:]) + f", [{wins}],")
else:
    lines = [f"    [{desc}], [{npar}], [–], [–], [–], [–], [–], [–],  // PENDING: ablate run" for _, desc, npar in ROWS]
tab = f"""#figure(placement: top, scope: "parent", kind: table,
  text(size: 8pt, table(
    columns: 8, align: (left,) + (right,) * 7, stroke: none,
    table.hline(),
    table.header([*Model*], [*\\#par*], [*SRE*], [*$Delta$*], [*10/90*], [*50/50*], [*90/10*], [*W*]),
    table.hline(),
{chr(10).join(lines)}
    table.hline(),
  )),
  caption: [Ablation of the classical circuit: each row removes one classical element, all variants trained jointly with the
  same data and seed. Mean test SRE in % over the 15 materials, overall and per scenario; $Delta$: change against the
  classical circuit; W: materials (of 15) on which the variant is more accurate than GRU8.],
) <tab:ablation>
"""
open(ROOT / "paper/tables/ablation.typ", "w").write(tab)
print(tab)
