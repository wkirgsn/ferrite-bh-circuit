"""Extrapolation / data-efficiency table (paper tab:ood) of the classical circuit, the MLP64-scheduled circuit and the GRU8
replica from results/ood_<ckpt>_gru_ood.json (ood_eval.py): test SRE on the windows outside the training restriction,
restricted and full-data model, mean over 3C94, N87, 3F4, 77 and the three scenarios.
Usage: python ood_table.py [classical ckpt=ood_final] [scheduled ckpt=hyAB64j_ood]   -> paper/tables/ood.typ
("–" for a missing run)"""
import sys, json, numpy as np
from mc2lib import ROOT

ck = sys.argv[1] if len(sys.argv) > 1 else "ood_final"
cks = sys.argv[2] if len(sys.argv) > 2 else "hyAB64j_ood"
SPECS = [("T25_50", "25, 50 °C $arrow$ 70 °C"), ("T25_70", "25, 70 °C $arrow$ 50 °C"),
         ("fmax320", "$<= 320$ kHz $arrow$ \\ 500, 800 kHz"), ("frac0.1", "10 % of the data"), ("frac0.02", "2 % of the data")]
rd = lambda c: (lambda p: json.load(open(p)) if p.exists() else {})(ROOT / f"results/ood_{c}_gru_ood.json")
d, ds = rd(ck), rd(cks)


def cell(d, spec, model):
    """Restricted / full-data SRE of a model (law or gru) for one restriction, mean over the four materials."""
    r = [v for k, v in d.items() if k.split("#")[1] == spec]
    if len(r) != 4:
        return "[–]"
    return f"[{np.mean([x[model + '_restricted'] for x in r]):.1f} / {np.mean([x[model + '_full'] for x in r]):.1f}]"


lines = [f"    [{desc}], {cell(d, spec, 'law')}, {cell(ds, spec, 'law')}, {cell(d, spec, 'gru')}," for spec, desc in SPECS]
tab = f"""#figure(
  table(
    columns: 4, align: (left, right, right, right), stroke: none,
    table.hline(),
    table.header([*Training $arrow$ test*], [*Classical (32)*], [*MLP64 (5552)*], [*GRU8 (336)*]),
    table.hline(),
{chr(10).join(lines)}
    table.hline(),
  ),
  caption: [Test SRE (%) on the windows outside the training restriction, restricted / full-data model,
    mean over 3C94, N87, 3F4 and 77 and the three scenarios. MLP64: the MLP64-scheduled circuit (one seed). GRU8: the
    RHINO-MAG architecture trained with the same pipeline.],
) <tab:ood>
"""
open(ROOT / "paper/tables/ood.typ", "w").write(tab)
print(tab)
