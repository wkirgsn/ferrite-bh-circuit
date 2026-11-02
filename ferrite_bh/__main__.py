"""Command line of ferrite_bh (NumPy only).

  python -m ferrite_bh list                                         materials with their test errors
  python -m ferrite_bh coefficients N87 --T 25                      model coefficients at 25 degC (JSON)
  python -m ferrite_bh simulate N87 --T 25 --B b.csv [--out h.csv] [--measured]
        open-loop from the demagnetized state; b.csv: one column of B in T at 16 MHz starting at 0;
        default the material alone (true field), --measured: as the MC2 test bench measures it
  python -m ferrite_bh predict N87 --T 25 --B b.csv --H h_warmup.csv [--out h.csv]
        MC2 setting: the measured H of the first k samples (k = length of h_warmup.csv) sets the state
  python -m ferrite_bh spice [N87 ...]                              (re)write ferrite_bh/spice/<material>.lib
Output: one column of H in A/m (stdout if --out is not given)."""
import argparse, json, sys
import numpy as np
from . import model, spice


def main(argv=None):
    """Command line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m ferrite_bh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("coefficients"); c.add_argument("material"); c.add_argument("--T", type=float, required=True)
    for name in ("simulate", "predict"):
        s = sub.add_parser(name); s.add_argument("material"); s.add_argument("--T", type=float, required=True)
        s.add_argument("--B", required=True); s.add_argument("--out")
        if name == "predict":
            s.add_argument("--H", required=True, help="measured H over the warm-up (one column, A/m)")
        else:
            s.add_argument("--measured", action="store_true", help="include measurement correction and lag")
    sp = sub.add_parser("spice"); sp.add_argument("materials", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for mt in model.materials():
            r = model.load(mt); e = r["test_errors_percent"]
            print(f"{mt:7s} test SRE {e.get('sre_mean', float('nan')):5.2f} %  (k=100/500/900: {e.get('sre_k100')}/{e.get('sre_k500')}/"
                  f"{e.get('sre_k900')})  temperatures {r['validity']['temperatures_degC']} degC")
    elif a.cmd == "coefficients":
        co = model.coefficients(a.material, a.T)
        print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in co.items()}, indent=1))
    elif a.cmd in ("simulate", "predict"):
        B = np.loadtxt(a.B, ndmin=1)
        H = model.simulate(a.material, B, a.T, material_only=not a.measured) if a.cmd == "simulate" else \
            model.predict(a.material, B, np.loadtxt(a.H, ndmin=1), a.T)
        np.savetxt(a.out or sys.stdout, H, fmt="%.6g")
    elif a.cmd == "spice":
        print("wrote", spice.write(a.materials or None))


if __name__ == "__main__":
    main()
