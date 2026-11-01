"""Helpers shared by the training and evaluation scripts: random training windows drawn on the device, the sequence
relative error of a batch, and the one-line validation/test summary."""
import numpy as np, torch


class GPUSampler:
    """Random training windows drawn on the device (files weighted by their number of sequences)."""
    def __init__(s, parts, dev):
        s.d = [{k: torch.tensor(p[k], device=dev) for k in ("B", "H", "T", "dt") if k in p} for p in parts]
        s.sizes = torch.tensor([float(p["B"].shape[0]) for p in parts], device=dev); s.dev = dev
        s.last_dt = None                     # per-row step size of the last batch (data at varying sampling rates)

    def __call__(s, bs, win, gen):
        fi = torch.multinomial(s.sizes, bs, replacement=True, generator=gen)
        Bs, Hs, Ts, Ds = [], [], [], []
        for f, d in enumerate(s.d):
            n = int((fi == f).sum())
            if n == 0: continue
            q = torch.randint(0, d["B"].shape[0], (n,), device=s.dev, generator=gen)
            st = torch.randint(0, d["B"].shape[1] - win + 1, (n,), device=s.dev, generator=gen)
            ii = st[:, None] + torch.arange(win, device=s.dev)
            Bs.append(d["B"][q[:, None], ii]); Hs.append(d["H"][q[:, None], ii]); Ts.append(d["T"][q])
            if "dt" in d: Ds.append(d["dt"][q])
        s.last_dt = torch.cat(Ds) if Ds else None
        return torch.cat(Bs), torch.cat(Hs), torch.cat(Ts)


def sre_E(Hp, H, k):
    """Hp (E,N,W), H (N,W) -> (E,N)."""
    return torch.sqrt(((Hp[..., k:] - H[None, :, k:]) ** 2).mean(-1)) / torch.sqrt((H[None, :, k:] ** 2).mean(-1))


def summarize(res, names, nparams, tag):
    for e, nm in enumerate(names):
        sre = [res[sc][0][e] for sc in res]; nere = [res[sc][1][e] for sc in res]
        print(f"{tag} {nm:24s} n={nparams[e]:3d}  SRE avg {100*np.mean([x.mean() for x in sre]):5.2f} p95 {100*np.mean([np.percentile(x, 95) for x in sre]):5.2f}"
              f" | " + " ".join(f"{100*x.mean():5.2f}" for x in sre) + "  NERE " + " ".join(f"{100*x.mean():4.2f}" for x in nere), flush=True)
