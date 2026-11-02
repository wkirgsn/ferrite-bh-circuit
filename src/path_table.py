"""Final model and its smaller variants (paper table tab:path) from the eval_*.txt files of eval_m.py: mean test SRE over the 15
materials, per scenario, and the number of materials on which the model beats the published GRU8.
Usage: python path_table.py   -> paper/tables/path.typ"""
import re, statistics as st
from mc2lib import ROOT, RAW_LABEL

LETTER = {v: k for k, v in RAW_LABEL.items()}                     # old result files use the masked letters


def parse(f):
    """Test SRE per variant and material from an eval_m.py output: {variant: {material: [mean, 10/90, 50/50, 90/10]}}."""
    d = {}
    for l in open(f):
        m = re.match(r"TEST (\S+) (.+?)\s+n=\s*\d+\s+SRE avg\s+([\d.]+)(?: p95 [\d.]+)? \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", l)
        if m:
            d.setdefault(m[2].strip(), {})[LETTER.get(m[1], m[1])] = [float(m[i]) for i in (3, 4, 5, 6)]
    return d


G = "RHINO GRU8 (published)"
# (description, [(eval file, variant), ...] -- several entries: seeds, mean +- sd), parameters
ROWS = [
    ("*classical circuit*", [("nere", "final_e1")] + [(f"ne_s{s}", "final_e1") for s in range(1, 10)], 32),
    ("without lag of $H$", [("small_ne", "g31_pas_e1")], 31),
    ("without lag of $H$, not passive", [("small_ne", "g31_e1")] + [(f"sne_s{s}", "g31_e1") for s in range(1, 10)], 31),
    ("$theta.alt$-independent measurement correction", [("small_ne", "g31_pas_hl3_tauc_e1")], 30),
    ("common feedback gain", [("small_ne", "g26_pas_hl3_fbop_e1")], 29),
    ("common feedback gain, $theta.alt$-independent measurement correction", [("small_ne", "g26_pas_hl3_e1")] + [(f"sne_s{s}", "g26_pas_hl3_e1") for s in range(1, 10)], 27),
    ("\\+ $theta.alt$-independent feedback gain, cubic term, density slope", [("small_ne", "p24_gamc_a3c_dwc_hl3i_e1")], 24),
]
if __name__ == "__main__":
    lines, gru = [], None
    for desc, runs, npar in ROWS:
        vals = []
        for f, v in runs:
            p = ROOT / f"results/eval_{f}.txt"
            if not p.exists():
                continue
            d = parse(p); gru = gru or d.get(G)
            if v in d and len(d[v]) == 15:
                vals.append(d[v])
        if not vals:
            continue
        mats = list(vals[0]); g = gru
        assert g is not None, f"no {G} results in results/eval_*.txt"
        mean = [st.mean(st.mean(x[m][j] for m in mats) for x in vals) for j in range(4)]
        wins = st.mean(sum(x[m][0] < g[m][0] for m in mats) for x in vals)
        sd = st.stdev([st.mean(x[m][0] for m in mats) for x in vals]) if len(vals) > 1 else None
        tot = f"{mean[0]:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")
        lines.append(f"    [{desc}], [{npar}], [{tot}], " + ", ".join(f"[{x:.2f}]" for x in mean[1:]) + f", [{wins:.0f}]," +
                     (f"  // {len(vals)} runs" if len(vals) > 1 else ""))
    assert gru is not None, f"no {G} results in results/eval_*.txt"
    gm = [st.mean(gru[m][j] for m in gru) for j in range(4)]
    tab = f"""#figure(placement: top, scope: "parent", kind: table,
      text(size: 8pt, table(
        columns: 7, align: (left,) + (right,) * 6, stroke: none,
        table.hline(),
        table.header([*Model*], [*\\#par*], [*SRE*], [*10/90*], [*50/50*], [*90/10*], [*W*]),
        table.hline(),
    {chr(10).join(lines)}
        table.hline(),
        [RHINO-MAG GRU8 @vater2026rhino], [336], [{gm[0]:.2f}], {", ".join(f"[{x:.2f}]" for x in gm[1:])}, [],
        table.hline(),
      )),
      caption: [The classical circuit and smaller variants: mean test SRE in % over the 15 materials, overall and per
      scenario. W: materials (of 15) on which the model is more accurate than GRU8; ± is the standard deviation over
      training seeds where several were run. All models are trained on SRE + NERE.],
    ) <tab:path>
    """
    open(ROOT / "paper/tables/path.typ", "w").write(tab)
    print(tab)
