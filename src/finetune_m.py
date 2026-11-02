"""Refit selected parameter groups of a fitted law on another data set (other specimen / material), the rest frozen.
Usage: python finetune_m.py --init <ckpt>:<variant>:<material slot> --mat <target material> --groups '<json>' --name <out>
groups: {"variant name": ["param", ...] or "all"}; estimator parameters (g, g2, L, K0, cpr) are always refitted.
Each group becomes one variant of a single-material CompactM, initialized from the source law."""
import argparse, json, time
import numpy as np, torch
from mc2lib import ROOT, split_mat, windows
from compact_m import CompactM, LAW, evaluate_M, DEV
from common import GPUSampler, sre_E, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--mat", required=True); ap.add_argument("--groups", required=True)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=2048)
ap.add_argument("--win", type=int, default=600); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--ks", default="100,200,300,450"); ap.add_argument("--name", required=True)
ap.add_argument("--qreg", type=float, default=0.0, help="option qest: penalty weight of q^2, as in compact_m")
args = ap.parse_args(); torch.manual_seed(0)
ckn, var, slot = args.init.split(":")
src = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV); e0, s0 = src["names"].index(var), src["mats"].index(slot)
groups = json.loads(args.groups); names = list(groups)
base = {k: v for k, v in src["variants"][e0].items() if k not in ("name", "shared")}
variants = [dict(base, name=n, shared=[]) for n in names]
m = CompactM(variants, 1, src["M"], src["J"]).to(DEV)
with torch.no_grad():
    for k, p in m.p.items():
        p[:] = src["sd"][f"p.{k}"][e0, s0].to(DEV)
mask = {k: torch.tensor([float(groups[n] == "all" or k in groups[n] or k not in LAW) for n in names], device=DEV) for k in m.p}
for k, p in m.p.items():
    p.register_hook(lambda g, k=k: g * mask[k].view(-1, *[1] * (g.dim() - 1)))
part = split_mat(args.mat)["train"]; smp = GPUSampler(part, DEV)
vw = {k: v[np.random.default_rng(0).permutation(len(v))][:500] for k, v in windows("val", args.mat).items()}
summarize(evaluate_M(m, vw, 0), names, m.n_params(), f"val {args.mat} (before)")
opt = torch.optim.Adam(m.parameters(), args.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
ew = float(base.get("energy_w", 0)); gen = torch.Generator(DEV).manual_seed(1); rng = np.random.default_rng(1); t0 = time.time()
for it in range(args.steps):
    B, H, T = smp(args.bs, args.win, gen); k = int(rng.choice([int(x) for x in args.ks.split(",")]))
    Hp = m(B, H, T, torch.zeros(len(B), device=DEV), k); sre = sre_E(Hp, H, k)
    if ew > 0:   # NERE term of the source model's loss (compact_m energy_w)
        dB = torch.gradient(B, dim=1)[0][None]
        sre = sre + ew * ((Hp[..., k:] - H[None, :, k:]) * dB[..., k:]).sum(-1).abs() / (H[None] * dB).abs().sum(-1).clamp(min=1e-9)
    q = m.last_q
    if args.qreg > 0 and q is not None:
        sre = sre + args.qreg * (q ** 2).mean(-1); m.last_q = None
    lossE = sre.mean(1)
    opt.zero_grad(); lossE.sum().backward(); opt.step(); sched.step()
    if it % 500 == 0:
        print(f"it {it} loss " + " ".join(f"{100*x:.2f}" for x in lossE.tolist()) + f"  t {time.time()-t0:.0f}s", flush=True)
summarize(evaluate_M(m, vw, 0), names, m.n_params(), f"val {args.mat} (after)")
torch.save({"variants": variants, "names": names, "mats": [args.mat], "M": src["M"], "J": src["J"], "sd": m.state_dict(),
            "init": args.init, "groups": groups}, ROOT / f"models/{args.name}.pt")
