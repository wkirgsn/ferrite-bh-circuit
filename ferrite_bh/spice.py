"""SPICE subcircuits of the material model (the equivalent circuit of the paper), one .lib per material.

    .subckt <MAT>_core p n params: Tc=25 N=1 Ae=1 le=1
    V(p, n) = N Ae dB/dt (winding voltage), current into p = H le / N; Tc in degC, Ae in m^2, le in m.

Material only: driven by B itself (no measurement correction tau1, tau2 and no H lag of the MC2 test bench), started
from the demagnetized state (all currents zero at t = 0, so B must start at 0). The temperature polynomials are
evaluated by .param expressions, so Tc can be swept. Elements: the reversible curve as a nonlinear inductor (B source),
per play an inductor 1/|w_i| in series with a viscous Coulomb slider (dead-zone resistor, threshold |w_i| r_i), per
relaxation an inductor 1/|c_j| in series with a resistor, each behind a feedback source gamma_k dH/dt.
Written for ngspice (B sources, .param functions exp, ln, max, min); LTspice accepts the same syntax.
Usage: python -m ferrite_bh spice [material ...]   -> ferrite_bh/spice/<material>.lib"""
from pathlib import Path
import numpy as np
from .model import load, materials, DT

OUT = Path(__file__).parent / "spice"


def subckt(material):
    """SPICE subcircuit of a material as text (temperature, turns and core geometry as parameters)."""
    rec = load(material); P = {k: v["value"] for k, v in rec["parameters"].items()}
    M, J = 6, 2
    lr = P["log_r_first"] + (P["log_r_last"] - P["log_r_first"]) * np.linspace(0, 1, M)
    x = (lr - lr.mean()) / (lr.std(ddof=1) + 1e-6)                    # standardized log widths (independent of T)
    f = lambda v: f"{v:.12g}"
    name = f"{material}_core"
    L = [f"* {material}: 32-parameter B-H model (Kirchgaessner 2026), material equivalent circuit",
         f"* fitted on MagNet Challenge 2 data; trained temperatures {rec['validity']['temperatures_degC']} degC",
         "* V(p,n) = N*Ae*dB/dt, current into p = H*le/N. Start from B = 0 (demagnetized).",
         f".subckt {name} p n params: Tc=25 N=1 Ae=1 le=1",
         ".param tt={(Tc-50)/25}",
         f".param a1={{{f(P['a1_0'])}+({f(P['a1_1'])})*tt}} a3={{100*({f(P['a3_0'])}+({f(P['a3_1'])})*tt)}}",
         f".param Bs={{max({f(P['Bs_0'])}+({f(P['Bs_1'])})*tt,0.15)}} bw={{exp({f(P['log_bw'])})}}",
         f".param alpha={{1/(1+exp(-({f(P['alpha_0'])}+({f(P['alpha_1'])})*tt)))}} kapa={{-ln(1-alpha)/{DT:g}}}",
         f".param gc={{1e-3*({f(P['gam_0'])}+({f(P['gam_1'])})*tt)}}"]
    for i in range(M):
        L += [f".param r{i}={{exp({f(lr[i])}+({f(P['rT'])})*tt)}}"
              f" w{i}={{-ln(1+exp(-({f(P['dw0_0'])}+({f(P['dw0_1'])})*tt+({f(P['dw1_0'])}+({f(P['dw1_1'])})*tt)*({f(x[i])}))))}}"
              f" g{i}={{gc+1e-3*({f(P['gop_play'])})*({f(x[i])})}}"]
    for j in range(J):
        L += [f".param c{j}={{-ln(1+exp({f(P[f'c{j + 1}_0'])}+({f(P[f'c{j + 1}_1'])})*tt))}}"
              f" kapb{j}={{-ln(1-0.5/(1+exp(-({f(P[f'beta_{j + 1}'])}))))/{DT:g}}} g{M + j}={{gc+1e-3*({f(P[f'gop_rel{j + 1}'])})}}"]
    L += ["Eu t 0 p n {1/(N*Ae)}", "Vsense t c 0", "Fport p n Vsense {le/N}",         # unit core inside: V = dB/dt, I = H
          "Gint 0 fl c 0 1", "Cint fl 0 1", "Rint fl 0 1e15",                         # V(fl) = B
          "Fx 0 x Vsense 1", "Lx x 0 1",                                              # V(x) = dH/dt
          "Bmain c 0 I={a1}*V(fl)+{a3}*V(fl)*V(fl)*V(fl)+10*(exp(min((V(fl)-{Bs})/{bw},15))-exp(min((-V(fl)-{Bs})/{bw},15)))"]
    for k in range(M + J):
        wk = f"w{k}" if k < M else f"c{k - M}"
        L += [f"Eg{k} c d{k} x 0 {{-g{k}}}", f"Gf{k} 0 f{k} d{k} 0 1", f"Cf{k} f{k} 0 1", f"Rf{k} f{k} 0 1e15",
              f"Blin{k} d{k} 0 I={{{wk}}}*V(f{k})", f"Eb{k} e{k} 0 d{k} 0 1",
              f"Lp{k} e{k} m{k} {{-1/{wk}}}", f"Vs{k} m{k} a{k} 0"]
        if k < M:                                                                    # play: viscous Coulomb slider
            L += [f"Bsl{k} a{k} 0 V={{kapa/(-{wk})}}*(max(I(Vs{k})-{{-{wk}*r{k}}},0)+min(I(Vs{k})+{{-{wk}*r{k}}},0))"]
        else:                                                                        # relaxation: resistor
            L += [f"Rp{k} a{k} 0 {{kapb{k - M}/(-{wk})}}"]
        L += [f"Fm{k} d{k} 0 Vs{k} 1"]
    L += [f".ends {name}"]
    return "\n".join(L) + "\n"


def write(mats=None):
    """Write ferrite_bh/spice/<material>.lib for the given materials (default: all); returns the directory."""
    OUT.mkdir(exist_ok=True)
    for mt in mats or materials():
        (OUT / f"{mt}.lib").write_text(subckt(mt))
    return OUT
