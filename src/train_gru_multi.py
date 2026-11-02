"""E independent GRU8 models (RHINO-MAG 'reduced features' replica, 336 params each, as train_gru.py) trained in one
batched pass, one per material slot (slots may carry a training restriction "<mat>#<spec>", see mc2lib.restrict).
Usage: python train_gru_multi.py --mats 3C94,3C94#T25_50,... --name <out> [--steps 20000 --bs 512]"""
import argparse, time, numpy as np, torch
from mc2lib import split_mat, ROOT
from common import GPUSampler

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiGRU(torch.nn.Module):
    """E independent GRU8 models (one per material slot) evaluated together: the RHINO-MAG GRU8 replica."""
    norm: torch.Tensor; fe: torch.Tensor                                        # input/output scales, derivative scales
    def __init__(s, E, hid=8, nin=4):
        """E models with hid hidden units and nin inputs, initialized uniformly as torch.nn.GRU (fixed seed)."""
        super().__init__()
        g = torch.Generator().manual_seed(0); a = hid ** -0.5
        u = lambda *sh: torch.nn.Parameter((torch.rand(*sh, generator=g) * 2 - 1) * a)
        s.Wi, s.Wh, s.bi, s.bh = u(E, 3 * hid, nin), u(E, 3 * hid, hid), u(E, 3 * hid), u(E, 3 * hid)
        s.hid = hid
        s.register_buffer("norm", torch.ones(E, 3)); s.register_buffer("fe", torch.ones(E, 2))   # Bm, Hm, Tm; dB, d2B max

    def cell(s, x, h):                                            # x (E,N,nin), h (E,N,hid), as torch.nn.GRUCell
        """One GRU step of all models: x (E, N, nin), h (E, N, hid) -> new h, as torch.nn.GRUCell."""
        gi = torch.einsum("egi,eni->eng", s.Wi, x) + s.bi[:, None]; gh = torch.einsum("egh,enh->eng", s.Wh, h) + s.bh[:, None]
        ir, iz, inn = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        return (1 - z) * torch.tanh(inn + r * hn) + z * h

    def forward(s, B, H, T, k):  # ty: ignore[invalid-mutable-override]  (torch declares forward as an attribute)
        """B, H (E,N,W), T (E,N); first k samples of H known (teacher forced into hidden state 0). Returns (E,N,W)."""
        Bm, Hm, Tm = (s.norm[:, i, None, None] for i in range(3))
        b = B / Bm; db = torch.gradient(b, dim=-1)[0]; d2b = torch.gradient(db, dim=-1)[0]
        x = torch.stack([b, (T[..., None] / Tm).expand_as(b), db / s.fe[:, 0, None, None], d2b / s.fe[:, 1, None, None]], -1)
        hn = H[..., :k] / Hm
        h = torch.cat([hn[..., :1], torch.zeros(*B.shape[:2], s.hid - 1, device=B.device)], -1); out = [hn]
        for t in range(1, k):
            h = s.cell(x[:, :, t], h); h = torch.cat([hn[..., t:t + 1], h[..., 1:]], -1)
        for t in range(k, B.shape[-1]):
            h = s.cell(x[:, :, t], h); out.append(h[..., :1])
        return torch.cat(out, -1) * Hm


def gru_predict(m, e, w, k, chunk=1024):
    """Predictions of model e on windows w with warm-up k: (N, W) numpy."""
    out = []
    with torch.no_grad():
        for i0 in range(0, len(w["B"]), chunk):
            B = torch.tensor(w["B"][i0:i0 + chunk], device=DEV); H = torch.tensor(w["H"][i0:i0 + chunk], device=DEV)
            T = torch.tensor(w["T"][i0:i0 + chunk], device=DEV)
            sub = MultiGRU(1).to(DEV)
            sub.load_state_dict({kk: v[e:e + 1] for kk, v in m.state_dict().items()})
            out.append(sub(B[None], H[None], T[None], k)[0].cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mats", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--P", type=int, default=28); ap.add_argument("--F", type=int, default=156)
    args = ap.parse_args(); torch.manual_seed(0)
    mats = args.mats.split(","); E = len(mats)
    parts = [split_mat(mt)["train"] for mt in mats]
    m = MultiGRU(E).to(DEV)
    with torch.no_grad():
        for e, pa in enumerate(parts):
            Bm = max(np.abs(d["B"]).max() for d in pa); Hm = max(np.abs(d["H"]).max() for d in pa)
            m.norm[e] = torch.tensor([Bm, Hm, 70.0])
            fe = [[np.abs(db := np.gradient(d["B"][:200] / Bm, axis=1)).max(), np.abs(np.gradient(db, axis=1)).max()] for d in pa]
            m.fe[e] = torch.tensor(np.max(fe, 0))
    samplers = [GPUSampler(pa, DEV) for pa in parts]
    opt = torch.optim.Adam(m.parameters(), args.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator(DEV).manual_seed(0); P, F = args.P, args.F; t0 = time.time()
    for it in range(args.steps):
        B, H, T = (torch.stack(x) for x in zip(*[smp(args.bs, P + F, gen) for smp in samplers]))   # (E,bs,W)
        Hp = m(B, H, T, P); Bm, Hm = m.norm[:, 0, None, None], m.norm[:, 1, None, None]
        wgt = torch.abs(torch.diff(B[..., P - 1:], dim=-1)) / Bm
        err = torch.sqrt((((Hp[..., P:] - H[..., P:]) / Hm) ** 2 * wgt).mean(-1)); rms = torch.sqrt((H ** 2).mean(-1)) / Hm[..., 0]
        loss = (err / rms).mean(-1)                                                               # (E,)
        opt.zero_grad(); loss.sum().backward()
        with torch.no_grad():                                                                     # per-model clipping at 1
            gs = [p.grad for p in m.parameters() if p.grad is not None]
            gn = torch.sqrt(sum(((g ** 2).flatten(1).sum(1) for g in gs), torch.zeros((), device=gs[0].device)))
            for g in gs: g *= (1.0 / gn.clamp(min=1.0)).view(-1, *[1] * (g.dim() - 1))
        opt.step(); sched.step()
        if it % 1000 == 0 or it == args.steps - 1:
            print(f"it {it} loss " + "/".join(f"{100*x:.2f}" for x in loss.tolist()) + f" t {time.time()-t0:.0f}s", flush=True)
            torch.save({"mats": mats, "sd": m.state_dict()}, ROOT / f"models/{args.name}.pt")
