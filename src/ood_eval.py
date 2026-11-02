"""Extrapolation / data-efficiency table: law (compact_m checkpoint, or a scheduled circuit of hyper_m.py) and GRU8
replica (train_gru_multi checkpoint, or "-" for the law only), both trained on the same slots "<mat>" (full data) and
"<mat>#<spec>" (restricted training).
Per restricted slot: test SRE (mean over the three scenarios) on the test windows outside the training restriction
(mc2lib.ood_mask; all windows for frac specs), next to the full-data model of the same material on the same windows.
Usage: python ood_eval.py <law ckpt> <law variant> <gru ckpt|->   -> results/ood_<law>_<gru>.json"""
import sys, json, numpy as np, torch
from mc2lib import windows, select_k, SCENARIOS, ood_mask, ROOT
from compact_m import DEV
from loader import load
from train_gru_multi import MultiGRU, gru_predict

lck, lvar, gck = sys.argv[1:4]
law, names, mats = load(lck); law.eval()                               # classical circuit or scheduled (hyper_m) checkpoint
ck = {"names": names, "mats": mats}
e = ck["names"].index(lvar)
models = {"law": ck["mats"]}
if gck != "-":
    g = torch.load(ROOT / f"models/{gck}.pt", map_location=DEV); gru = MultiGRU(len(g["mats"])).to(DEV); gru.load_state_dict(g["sd"])
    models["gru"] = g["mats"]


def law_predict(mid, w, k, chunk=256):
    """Predictions of the circuit of slot index mid on windows w with warm-up k (in chunks, no gradients)."""
    out = []
    with torch.no_grad():
        for i0 in range(0, len(w["B"]), chunk):
            B = torch.tensor(w["B"][i0:i0 + chunk], device=DEV); H = torch.tensor(w["H"][i0:i0 + chunk], device=DEV)
            T = torch.tensor(w["T"][i0:i0 + chunk], device=DEV)
            out.append(law(B, H, T, torch.full((len(B),), mid, device=DEV), k)[e].cpu().numpy())
    return np.concatenate(out)


def sre_windows(nm, slot):
    """Per-window SRE (3 scenarios concatenated in a fixed order) of model nm trained on slot."""
    w = windows("test", slot.split("#")[0]); out = []
    for sc, k in SCENARIOS.items():
        ws = select_k(w, k)
        Hp = law_predict(ck["mats"].index(slot), ws, k) if nm == "law" else gru_predict(gru, g["mats"].index(slot), ws, k)
        out.append(np.sqrt(((Hp[:, k:] - ws["H"][:, k:]) ** 2).mean(1) / (ws["H"][:, k:] ** 2).mean(1)))
    return out


res = {}
for slot in ck["mats"]:
    if "#" not in slot or not all(slot in v for v in models.values()): continue
    base, spec = slot.split("#"); w = windows("test", base)
    masks = [ood_mask(select_k(w, k), spec, base) for k in SCENARIOS.values()]
    r = {}
    for nm in models:
        for tag, sl in (("restricted", slot), ("full", base)):
            s = sre_windows(nm, sl); r[f"{nm}_{tag}"] = 100 * float(np.mean([x[m].mean() for x, m in zip(s, masks)]))
    res[slot] = r
    print(f"{slot:16s} " + "  ".join(f"{k} {v:6.2f}" for k, v in r.items()), flush=True)
json.dump(res, open(ROOT / f"results/ood_{lck}_{gck}.json", "w"), indent=1)
