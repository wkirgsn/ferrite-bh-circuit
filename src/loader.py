"""Load a checkpoint of models/: the classical circuit (CompactM, all variants of a run) or a gain-scheduled circuit
(kind "hyper", hyper_m.HyperM). Returns (model, variant names, materials). Entries of checkpoints trained with earlier
exploration options (parameters and switches no longer in the model, all inactive) are skipped."""
import torch
from mc2lib import ROOT
from compact_m import CompactM, DEV


def _load(m, sd):
    """Load a state dict, skipping entries the model does not have; all its parameters must be present."""
    own = m.state_dict()
    missing = {n for n, _ in m.named_parameters()} - sd.keys()
    assert not missing, f"checkpoint lacks parameters {sorted(missing)}"
    m.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False)
    return m


def load(ckn):
    """Model, variant names and materials of the checkpoint models/<ckn>.pt."""
    ck = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV)
    if ck.get("kind") == "hyper":
        from hyper_m import HyperM
        m = _load(HyperM(ck["variant"], ck["mats"], ck["M"], ck["J"], ck["hidden"], ck["opt"]).to(DEV), ck["sd"])
    else:
        m = _load(CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV), ck["sd"])
    return m, ck["names"], ck["mats"]
