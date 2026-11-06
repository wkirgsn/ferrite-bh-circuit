"""Export the final model (nere:final_e1) to plain JSON per material for the NumPy reference implementation
(ferrite_bh/model.py) and the SPICE subcircuits (ferrite_bh/spice.py): the 32 fitted parameters by name, the warm-up
estimator parameters, the validity range and the test errors.
Usage: python export_params.py [ckpt:variant=nere:final_e1]   -> ferrite_bh/params/<material>.json"""
import sys, json, re, numpy as np, torch
from mc2lib import ROOT, windows
from compact_m import CompactM
from export_hub import validity as training_range

spec = sys.argv[1] if len(sys.argv) > 1 else "nere:final_e1"
ckn, var = spec.split(":")
ck = torch.load(ROOT / f"models/{ckn}.pt", map_location="cpu"); e = ck["names"].index(var)
opts = {k: v for k, v in ck["variants"][e].items() if k not in ("name", "shared", "energy_w")}
assert opts == {"fbop": 1, "k0": 1, "cubic": 1, "wsmooth": 1, "qest": 1, "lqk": 1, "qest2": 1, "rgeo": 1, "gsmooth": 1,
                "tlin": 1, "passive": 1, "hlag": 3}, f"ferrite_bh implements the final model only, got {opts}"
assert CompactM([ck["variants"][e]], 1).n_params() == [32]

test = {}
for l in open(ROOT / f"results/eval_{ckn}.txt"):
    m = re.match(r"TEST (\S+) (\S+)\s+n=\s*\d+\s+SRE avg\s+([\d.]+) p95 ([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+NERE ([\d.]+)", l)
    if m and m[2] == var:
        test[m[1]] = dict(sre_mean=float(m[3]), sre_p95=float(m[4]), sre_k100=float(m[5]), sre_k500=float(m[6]),
                          sre_k900=float(m[7]), nere_mean=float(m[8]))

DESC = {   # name: (description, unit)
    "tau1_0": ("measurement correction, coefficient of dB/dn at 50 degC", "samples"),
    "tau1_1": ("measurement correction, slope of tau1 in t = (T - 50)/25", "samples"),
    "tau2_0": ("measurement correction, coefficient of d2B/dn2 at 50 degC", "samples^2"),
    "tau2_1": ("measurement correction, slope of tau2 in t", "samples^2"),
    "a1_0": ("initial slope of the reversible curve at 50 degC", "A/m/T"), "a1_1": ("slope of a1 in t", "A/m/T"),
    "a3_0": ("cubic coefficient / 100 at 50 degC", "100 A/m/T^3"), "a3_1": ("slope of a3 / 100 in t", "100 A/m/T^3"),
    "Bs_0": ("knee position at 50 degC (floored at 0.15 T)", "T"), "Bs_1": ("slope of Bs in t", "T"),
    "log_bw": ("log of the knee width b", "log T"),
    "log_r_first": ("log width of the narrowest play at 50 degC (widths log-equidistant)", "log T"),
    "log_r_last": ("log width of the widest play at 50 degC", "log T"), "rT": ("slope of all log widths in t", "-"),
    "alpha_0": ("logit of the play rate alpha at 50 degC", "-"), "alpha_1": ("slope of logit alpha in t", "-"),
    "dw0_0": ("play weight density at the mean log width, 50 degC (before w <= 0)", "A/m/T"),
    "dw0_1": ("slope of dw0 in t", "A/m/T"), "dw1_0": ("slope of the weight density over the standardized log width", "A/m/T"),
    "dw1_1": ("slope of dw1 in t", "A/m/T"),
    "beta_1": ("logit of relaxation rate 1 (beta = sigmoid / 2)", "-"), "beta_2": ("logit of relaxation rate 2", "-"),
    "c1_0": ("relaxation 1 weight before c = -softplus, 50 degC", "A/m/T"), "c1_1": ("slope of c1 in t", "A/m/T"),
    "c2_0": ("relaxation 2 weight before c = -softplus, 50 degC", "A/m/T"), "c2_1": ("slope of c2 in t", "A/m/T"),
    "gam_0": ("common feedback gain at 50 degC", "mT per A/m"), "gam_1": ("slope of the feedback gain in t", "mT per A/m"),
    "gop_play": ("feedback offset of the plays per unit standardized log width", "mT per A/m"),
    "gop_rel1": ("feedback offset of relaxation 1", "mT per A/m"), "gop_rel2": ("feedback offset of relaxation 2", "mT per A/m"),
    "hl": ("logit of the H-measurement lag pole (eta = 0.6 sigmoid)", "-"),
}

out = ROOT / "ferrite_bh/params"; out.mkdir(parents=True, exist_ok=True)
for i, mt in enumerate(ck["mats"]):
    g = lambda n: ck["sd"][f"p.{n}"][e, i].double().numpy()
    tau, cz, dw = g("tau"), g("cz"), g("dw")
    val = dict(tau1_0=tau[0, 0], tau1_1=tau[0, 1], tau2_0=tau[1, 0], tau2_1=tau[1, 1], a1_0=g("a1")[0], a1_1=g("a1")[1],
               a3_0=g("a3")[0], a3_1=g("a3")[1], Bs_0=g("bs")[0], Bs_1=g("bs")[1], log_bw=g("log_bw")[0],
               log_r_first=g("log_r")[0], log_r_last=g("log_r")[-1], rT=g("rT")[0], alpha_0=g("alpha")[0], alpha_1=g("aT")[0],
               dw0_0=dw[0, 0], dw0_1=dw[0, 1], dw1_0=dw[1, 0], dw1_1=dw[1, 1], beta_1=g("beta")[0], beta_2=g("beta")[1],
               c1_0=cz[0, 0], c1_1=cz[0, 1], c2_0=cz[1, 0], c2_1=cz[1, 1], gam_0=g("gam")[0], gam_1=g("gam")[1],
               gop_play=g("gop")[0], gop_rel1=g("gop")[6], gop_rel2=g("gop")[7], hl=g("hl")[0])
    assert len(val) == 32 and set(val) == set(DESC)
    w = windows("test", mt)
    rec = dict(
        material=mt, model="32-parameter B-H model (Kirchgaessner 2026), final model, trained on MagNet Challenge 2 data",
        source=f"models/{ckn}.pt, variant {var}", sample_period_s=6.25e-8,
        units=dict(B="T", H="A/m", T="degC", time="samples of 62.5 ns"),
        validity=dict(temperatures_degC=sorted(float(x) for x in np.unique(np.round(w["T"]))),
                      B_peak_max_T=round(float(np.abs(w["B"]).max()), 3),
                      training_range=training_range(mt),
                      note="trained on the MC2 measurements of this material (B_peak_max_T: largest |B| in its test windows); "
                           "temperature dependence at most linear in t = (T - 50)/25; training_range: T and the maxima of |B| and of the "
                           "per-sample |dB| and |d2B| over the training sequences, checked by the range warning"),
        test_errors_percent=test.get(mt, {}),
        parameters={n: dict(value=float(val[n]), description=DESC[n][0], unit=DESC[n][1]) for n in DESC},
        estimator=dict(description="warm-up estimator (not counted as model parameters): prior initial play states "
                                   "(g, g2), initial-mismatch gains K0 and offset prior cpr, observer gains L (x1e-3 for "
                                   "the 8 states, last: offset), log prior strength of the amplitude factors (lq, lqk)",
                       g=g("g").tolist(), g2=g("g2").tolist(), K0=g("K0").tolist(), cpr=float(g("cpr")[0]),
                       L=g("L").tolist(), lq=float(g("lq")[0]), lqk=float(g("lqk")[0])))
    (out / f"{mt}.json").write_text(json.dumps(rec, indent=1) + "\n")
print(f"{len(ck['mats'])} materials -> {out}")
