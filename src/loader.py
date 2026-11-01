"""Load a checkpoint of models/: the compact model (CompactM, all variants of a run) or a hybrid of it with a learned
residual (kind "hybrid": GRU, kind "state_mlp": MLP on the states) or a hyper-network on its coefficients (kind
"hyper"). Returns (model, variant names, materials)."""
import torch
from mc2lib import ROOT
from compact_m import CompactM, DEV


def load(ckn):
    ck = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV)
    kind = ck.get("kind")
    if kind == "hyper":
        from hyper_m import HyperM
        m = HyperM(ck["variant"], ck["mats"], ck["M"], ck["J"], ck["hidden"], ck["opt"], ck.get("wenc", 0)).to(DEV); m.load_state_dict({k: v for k, v in ck["sd"].items() if k != "sg"})
        if ck.get("qjoint"): m.qjoint = dict(lam=ck["qjoint"])
        return m, ck["names"], ck["mats"]
    if kind in ("hybrid", "state_mlp"):
        if kind == "hybrid":
            from hybrid import Hybrid as Cls
        else:
            from state_mlp import StateMLP as Cls
        m = Cls(ck["variant"], ck["mats"], ck["M"], ck["J"], ck["hidden"]).to(DEV); m.load_state_dict(ck["sd"])
        return m, ck["names"], ck["mats"]
    m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
    return m, ck["names"], ck["mats"]
