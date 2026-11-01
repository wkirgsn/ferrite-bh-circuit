"""Main results table of the paper (tab:main) from results/stats_<ckpt>_<variant>.json (main_stats.py): per material the
mean and 95th-percentile SRE and NERE of the published RHINO-MAG GRU8 and of the classical circuit, and its mean SRE per
scenario. Usage: python main_table.py <ckpt> <variant>   -> paper/tables/main_results.typ"""
import sys, json, numpy as np
from mc2lib import ROOT, EVAL_MATS

ckn, var = sys.argv[1:3]
st = json.load(open(ROOT / f"results/stats_{ckn}_{var}.json"))
order = [m for m in ["3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87"] + list(EVAL_MATS) if m in st]
KEYS = ["sre", "sre_p95", "nere", "nere_p95"]
b = lambda a, ref: f"*{a:.2f}*" if a < ref else f"{a:.2f}"
rows = []
for mt in order:
    g, o = st[mt]["gru"], st[mt]["model"]
    rows.append(f"    [{mt}], " + ", ".join(f"[{g[k]:.2f}], [{b(o[k], g[k])}]" for k in KEYS) + ", "
                + ", ".join(f"[{b(x, y)}]" for x, y in zip(o["scen"], g["scen"])) + ",")
mg = {k: np.mean([st[m]["gru"][k] for m in order]) for k in KEYS}; mo = {k: np.mean([st[m]["model"][k] for m in order]) for k in KEYS}
sg = np.mean([st[m]["gru"]["scen"] for m in order], 0); so = np.mean([st[m]["model"]["scen"] for m in order], 0)
wins = {k: sum(st[m]["model"][k] < st[m]["gru"][k] for m in order) for k in KEYS}
H = lambda s: f"table.cell(colspan: 2, align: center)[*{s}*]"
tab = f"""#figure(placement: top, scope: "parent", kind: table,
  text(size: 8pt, table(
    columns: 12, align: (left,) + (right,) * 11, stroke: none, column-gutter: (0pt, 0pt, 6pt, 0pt, 6pt, 0pt, 6pt, 0pt, 6pt),
    table.hline(),
    table.header([], {H("SRE")}, {H("SRE, 95th perc.")}, {H("NERE")}, {H("NERE, 95th perc.")}, table.cell(colspan: 3, align: center)[*SRE per scenario*],
      [*Material*], [GRU8], [classical], [GRU8], [classical], [GRU8], [classical], [GRU8], [classical], [10/90], [50/50], [90/10]),
    table.hline(),
{chr(10).join(rows)}
    table.hline(),
    [*mean*], """ + ", ".join(f"[{mg[k]:.2f}], [{b(mo[k], mg[k])}]" for k in KEYS) + ", " + ", ".join(f"[{b(x, y)}]" for x, y in zip(so, sg)) + f""",
    [*better*], """ + ", ".join(f"[], [{wins[k]}/15]" for k in KEYS) + """, [], [], [],
    table.hline(),
  )),
  caption: [Test results in % of the published RHINO-MAG GRU8 (336 parameters) and of the classical circuit (32 parameters) on
  identical windows: mean and 95th percentile of the sequence relative error (SRE) and of the normalized energy error
  (NERE), each per scenario and averaged over the three scenarios; the classical circuit's mean SRE per scenario. Bold: better than
  GRU8. The final-evaluation materials 3C92, 3C95, FEC007, FEC014 and T37 are evaluated on the official MC2 test set.],
) <tab:main>
"""
open(ROOT / "paper/tables/main_results.typ", "w").write(tab)
print(tab); print("better:", wins)
