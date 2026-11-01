"""Physical parameters of fitted laws at T = 50 °C (t = 0), next to the specimen data from Test_Info.xlsx.
The coefficients are taken from a forward pass (capture), so constraints such as option passive are applied.
Usage: python param_table.py ckpt:variant [ckpt:variant ...]   -> also writes results/params_<first ckpt>.csv"""
import sys, numpy as np, pandas as pd, torch
from mc2lib import ROOT
from compact_m import CompactM, DEV


def specimen(mat):
    p = ROOT / f"data/info/{mat}_Test_Info.xlsx"
    if not p.exists():
        return {}
    x = pd.read_excel(p, header=None).set_index(0)[1]
    return dict(core=x["Shape"], N=int(x["N1"]), Le=float(x["Le (m)"]), Ae=float(x["Ae (m2)"]))


rows = []
for arg in sys.argv[1:]:
    ck_name, var = arg.split(":")
    ck = torch.load(ROOT / f"models/{ck_name}.pt", map_location=DEV); e = ck["names"].index(var)
    m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
    m.capture = True; B = torch.zeros(1, 200, device=DEV); B[0] = 0.1 * torch.sin(torch.arange(200, device=DEV) / 20.0)
    for i, mt in enumerate(ck["mats"]):
        with torch.no_grad():                                # one dummy window at 50 °C: coefficients as the law uses them
            m(B, torch.zeros_like(B), torch.tensor([50.0], device=DEV), torch.tensor([i], device=DEV), 100)
        c = {n: v[e, 0].detach().cpu().numpy().ravel().astype(np.float64) for n, v in m.cap.items() if n not in ("u", "static", "hsat", "mmod")}
        tau = m.p["tau"][e, i].detach().cpu().numpy(); beta = c["beta"]; r = np.sort(c["r"]); M = len(r)
        rows.append(dict(mat=mt, **specimen(mt), tau1=tau[0, 0], tau2=tau[1, 0], tau1_t=tau[0, 1], tau2_t=tau[1, 1],
                         a1=c["a1"][0], Bs=c["Bs"][0], bw=c["bw"][0], alpha=c["alpha"][0], r_min=r[0], r_max=r[-1],
                         Trelax1=-1 / np.log(1 - beta[0]), Trelax2=-1 / np.log(1 - beta[1]),
                         gamma=c["gamma"][:M].mean(), hlag=c["hla"][0], ckpt=ck_name))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
print(df.drop(columns=["ckpt"]).to_string(index=False))
(ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / f"results/params_{sys.argv[1].split(':')[0]}.csv", index=False)
if True:  # typst table for the paper
    srt = lambda x, y: (min(x, y), max(x, y))
    lines = []
    fz = lambda T: "—" if T > 1e4 else (f"{T:.1f}" if T < 10 else f"{T:.0f}")      # > 1e4 samples: inactive
    for r in rows:
        t1, t2 = srt(r["Trelax1"], r["Trelax2"])
        lines.append(f"  [{r['mat']}], [{r.get('core', '')}], [{r.get('N', '')}], [{r['tau1']:.2f}], [{r['tau2']:.2f}], "
                     f"[{r['a1']:.0f}], [{r['Bs']:.2f}], [{r['alpha']:.2f}], [{fz(t1)}], [{fz(t2)}], [{1e3 * r['gamma']:.2f}], [{r['hlag']:.2f}],")
    tab = ("#figure(placement: top, scope: \"parent\", kind: table,\n  text(size: 8pt, table(\n    columns: 12, align: (left, left) + (right,) * 10, stroke: none,\n    table.hline(),\n"
           "    table.header([*Mat.*], [*Core*], [$N$], [$tau_1$], [$tau_2$], [$a_1$], [$B_s$], [$alpha$], [$T_(z 1)$], [$T_(z 2)$], [$gamma$], [$eta$]),\n"
           "    table.hline(),\n" + "\n".join(lines) + "\n    table.hline(),\n  )),\n"
           "  caption: [Fitted parameters at 50 °C. $tau$ and relaxation time constants $T_z$ in samples (62.5 ns; 1.4 is the bound $beta_j <= 1\/2$, — inactive), $a_1$ in A/m/T, "
           "$B_s$ in T, $gamma$ (mean over the plays) in mT/(A/m), $eta$: pole of the $H$-measurement lag. $N$: turns of the specimen.],\n) <tab:params_fit>\n")
    open(ROOT / "paper/tables/params_fit.typ", "w").write(tab)
