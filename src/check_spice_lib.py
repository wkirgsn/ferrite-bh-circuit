"""Check the SPICE subcircuits (ferrite_bh/spice/<material>.lib) against the NumPy model (ferrite_bh.simulate, material
only, from the demagnetized state) on B waveforms of test windows, at 25 and 70 degC, with a scaled winding and core.
Usage: python check_spice_lib.py [materials=all] [n windows=2]"""
import sys, os, subprocess, tempfile, numpy as np
from mc2lib import ROOT, windows
from spice_check import NGSPICE
sys.path.insert(0, str(ROOT)); import ferrite_bh as fb
from ferrite_bh import spice

DT = fb.DT; N, AE, LE = 8, 1.2e-4, 0.06
mats = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else fb.materials()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
spice.write(mats); rng = np.random.default_rng(0); allr = []
for mt in mats:
    w = windows("test", mt)
    for q in rng.choice(len(w["B"]), n, replace=False):
        for T in (25.0, 70.0):
            B = np.r_[w["B"][q][0] * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2, w["B"][q][1:]].astype(np.float64)
            Hn = fb.simulate(mt, B, T)
            v = N * AE * np.diff(B) / DT; pts = ["0 0"]
            for i, x in enumerate(v):
                pts += [f"{i * DT + 1e-11:.12e} {x:.9e}", f"{(i + 1) * DT:.12e} {x:.9e}"]
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "t")
                open(p + ".cir", "w").write("\n".join([
                    f"* check {mt}", f".include {spice.OUT / (mt + '.lib')}", "Vsrc p 0 PWL(" + " ".join(pts) + ")",
                    f"X1 p 0 {mt}_core params: Tc={T} N={N} Ae={AE} le={LE}",
                    ".options reltol=1e-5 abstol=1e-9 method=gear", f".tran {DT / 4:.6e} {(len(B) - 1) * DT:.6e} 0 {DT / 4:.6e}",
                    ".control", "run", "set wr_singlescale", f"wrdata {p}.out -i(Vsrc)", "quit", ".endc", ".end"]) + "\n")
                r = subprocess.run([NGSPICE, "-b", p + ".cir"], capture_output=True, text=True)
                if not os.path.exists(p + ".out"):
                    print(mt, q, T, "ngspice failed:", (r.stdout + r.stderr)[-800:]); continue
                t, i = np.loadtxt(p + ".out", unpack=True)
            Hs = np.interp(np.arange(len(B)) * DT, t, i) * N / LE
            e = 100 * np.sqrt(((Hs - Hn)[299:] ** 2).mean() / (Hn[299:] ** 2).mean()); allr.append(e)
            print(f"{mt:6s} win {q:5d} T {T:.0f}  SPICE vs NumPy {e:6.3f} %", flush=True)
print(f"{len(allr)} runs: median {np.median(allr):.3f} %, max {np.max(allr):.3f} %")
