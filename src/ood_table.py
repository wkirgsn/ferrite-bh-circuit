"""Extrapolation / data-efficiency table (paper tab:ood) of the final model and the GRU8 replica from
results/ood_<law ckpt>_gru_ood.json (ood_eval.py): test SRE on the windows outside the training restriction, restricted
and full-data model, mean over 3C94, N87, 3F4, 77 and the three scenarios.
Usage: python ood_table.py [law ckpt=ood_final]   -> paper/tables/ood.typ (a placeholder while the run is missing)"""
import sys, json, numpy as np
from mc2lib import ROOT

ck = sys.argv[1] if len(sys.argv) > 1 else "ood_final"
SPECS = [("T25_50", "25, 50 °C $arrow$ test at 70 °C"), ("T25_70", "25, 70 °C $arrow$ test at 50 °C"),
         ("fmax320", "$f <= 320$ kHz $arrow$ test at 500, 800 kHz"), ("frac0.1", "10 % of the sequences"),
         ("frac0.02", "2 % of the sequences")]
p = ROOT / f"results/ood_{ck}_gru_ood.json"
d = json.load(open(p)) if p.exists() else {}
lines = []
for spec, desc in SPECS:
    r = [v for k, v in d.items() if k.split("#")[1] == spec]
    if len(r) == 4:
        m = {k: np.mean([x[k] for x in r]) for k in r[0]}
        lines.append(f"    [{desc}], [{m['law_restricted']:.1f} / {m['law_full']:.1f}], [{m['gru_restricted']:.1f} / {m['gru_full']:.1f}],")
    else:
        lines.append(f"    [{desc}], [–], [–],  // PENDING: ood_final run")
tab = f"""#figure(
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    table.header([*Training data*], [*Classical (32)*], [*GRU8 (336)*]),
    table.hline(),
{chr(10).join(lines)}
    table.hline(),
  ),
  caption: [Test SRE (%) on the windows outside the training restriction, restricted / full-data model,
    mean over 3C94, N87, 3F4 and 77 and the three scenarios. GRU8: the RHINO-MAG architecture trained with the same
    pipeline.],
) <tab:ood>
"""
open(ROOT / "paper/tables/ood.typ", "w").write(tab)
print(tab)
