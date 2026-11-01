"""Run the published RHINO-MAG GRU8 of a material on our test windows (run inside the RHINO-MAG env;
RHINO_MAG: path of the RHINO-MAG repository, default ../RHINO-MAG). Writes data/<mat>_rhino_res.npy.
Usage: python eval_rhino.py [material]   (default 3C94)"""
import os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax.numpy as jnp
from rhmag.utils.model_evaluation import reconstruct_model_from_file
from mc2lib import report, windows, ROOT
mat = sys.argv[1] if len(sys.argv) > 1 else "3C94"
from mc2lib import EVAL_MATS, RAW_LABEL
pat = f"{RAW_LABEL.get(mat, mat)}_GRU8_final-reduced-features-f32_*_seed12.eqx" if mat in EVAL_MATS else f"{mat}_GRU8_reduced-features-f32_*_seed12.eqx"
f = sorted((Path(os.environ.get("RHINO_MAG", Path(__file__).resolve().parents[2] / "RHINO-MAG")) / "data/models").glob(pat))[0]
m = reconstruct_model_from_file(f.name)
w = windows("test", mat)
def pred(B, H, T, k):
    Hp = m(B_past=jnp.array(B[:, :k]), H_past=jnp.array(H[:, :k]), B_future=jnp.array(B[:, k:]), T=jnp.array(T))
    return np.concatenate([H[:, :k], np.array(Hp)], 1)
r = report(pred, w, f"RHINO GRU8 {mat} (published)")
np.save(ROOT / ("data/rhino_res.npy" if mat == "3C94" else f"data/{mat}_rhino_res.npy"), r, allow_pickle=True)
