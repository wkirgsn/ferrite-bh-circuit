"""Save the published RHINO GRU8 predictions on our test windows (run in the RHINO-MAG env, JAX_PLATFORMS=cpu; RHINO_MAG: path of the RHINO-MAG
repository, default ../RHINO-MAG).
Writes data/pred/<mat>_gru.npz with one (N, W) array per scenario. Usage: python rhino_pred.py <mat> [<mat> ...]"""
import os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax.numpy as jnp
from rhmag.utils.model_evaluation import reconstruct_model_from_file
from mc2lib import windows, select_k, SCENARIOS, ROOT, EVAL_MATS, RAW_LABEL
for mat in sys.argv[1:]:
    pat = f"{RAW_LABEL.get(mat, mat)}_GRU8_final-reduced-features-f32_*_seed12.eqx" if mat in EVAL_MATS else f"{mat}_GRU8_reduced-features-f32_*_seed12.eqx"
    m = reconstruct_model_from_file(sorted((Path(os.environ.get("RHINO_MAG", Path(__file__).resolve().parents[2] / "RHINO-MAG")) / "data/models").glob(pat))[0].name)
    w = windows("test", mat); out = {}
    for sc, k in SCENARIOS.items():
        ws = select_k(w, k)
        Hp = m(B_past=jnp.array(ws["B"][:, :k]), H_past=jnp.array(ws["H"][:, :k]), B_future=jnp.array(ws["B"][:, k:]), T=jnp.array(ws["T"]))
        out[sc] = np.concatenate([ws["H"][:, :k], np.array(Hp)], 1)
    np.savez(ROOT / f"data/pred/{mat}_gru.npz", **out); print(mat, "done", flush=True)
