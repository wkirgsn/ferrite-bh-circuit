"""SPICE subcircuits (ferrite_bh/spice/<material>.lib) against the NumPy circuit, open loop from the demagnetized state
(skipped without ngspice)."""
import os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pytest
import ferrite_bh as fb

N, AE, LE = 8, 1.2e-4, 0.06


@pytest.mark.parametrize("mat", ["N87", "3C90", "77"])
def test_spice_matches_numpy(mat, ngspice):
    per = 160; n = np.arange(6 * per)
    B = 0.15 * np.minimum(n / (2 * per), 1) * np.sin(2 * np.pi * n / per); T = 25.0
    Hn = fb.simulate(mat, B, T)
    v = N * AE * np.diff(B) / fb.DT; pts = ["0 0"]
    for i, x in enumerate(v):
        pts += [f"{i * fb.DT + 1e-11:.12e} {x:.9e}", f"{(i + 1) * fb.DT:.12e} {x:.9e}"]
    lib = Path(fb.__file__).parent / "spice" / f"{mat}.lib"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t")
        open(p + ".cir", "w").write("\n".join([
            f"* test {mat}", f".include {lib}", "Vsrc p 0 PWL(" + " ".join(pts) + ")",
            f"X1 p 0 {mat}_core params: Tc={T} N={N} Ae={AE} le={LE}",
            ".options reltol=1e-5 abstol=1e-9 method=gear", f".tran {fb.DT / 4:.6e} {(len(B) - 1) * fb.DT:.6e} 0 {fb.DT / 4:.6e}",
            ".control", "run", "set wr_singlescale", f"wrdata {p}.out -i(Vsrc)", "quit", ".endc", ".end"]) + "\n")
        subprocess.run([ngspice, "-b", p + ".cir"], capture_output=True, text=True)
        t, i = np.loadtxt(p + ".out", unpack=True)
    Hs = np.interp(np.arange(len(B)) * fb.DT, t, i) * N / LE
    e = np.sqrt(((Hs - Hn)[2 * per:] ** 2).mean() / (Hn[2 * per:] ** 2).mean())
    assert e < 0.02, e
