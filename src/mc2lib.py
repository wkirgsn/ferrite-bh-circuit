"""Shared data handling and evaluation for the MagNet Challenge 2 experiments (3C94, 3C90, ...).

Data: 7 files (switching frequency 50k..800k Hz), each row one B/H sequence sampled at 16 MHz
containing 100 switching periods, temperature per row.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FS = 16e6
FREQS = [50e3, 80e3, 125e3, 200e3, 320e3, 500e3, 800e3]
SCENARIOS = {"10/90": 100, "50/50": 500, "90/10": 900}
WIN = 1000
# Final-evaluation materials of MC2: few training sequences, official labelled test set (one warm-up length per window)
EVAL_MATS = ["3C92", "3C95", "FEC007", "FEC014", "T37"]
# The raw competition folders and the published RHINO-MAG model files carry the labels under which these materials
# were masked during the competition; used only to locate those external files.
RAW_LABEL = dict(zip(EVAL_MATS, "ABCDE"))


def load_material(mat="3C94"):
    """Returns list over files of dicts with B, H (n, L) float32, T (n,), switching frequency f and file number.
    Materials have 5-7 files; f follows from the row length (100 switching periods at FS)."""
    if mat.startswith("mc1:"):
        return load_mc1(mat[4:])
    d = ROOT / "data" / "npy" / mat
    out = []
    for i in range(1, 8):
        if not (d / f"{mat}_{i}_B.npy").exists():
            break
        B = np.load(d / f"{mat}_{i}_B.npy")
        H = np.load(d / f"{mat}_{i}_H.npy")
        T = np.load(d / f"{mat}_{i}_T.npy").ravel()
        f = min(FREQS, key=lambda x: abs(100 * FS / x - B.shape[1]))
        out.append(dict(B=B, H=H, T=T, f=f, file=i))
    return out


def load_mc1(mat):
    """MC1 (MagNet raw) records converted by mc1_convert.py, without DC bias, grouped by frequency into "files";
    rows are cropped to the shortest whole-period length of the group. gid: record index in data/mc1/<mat>.npz.
    "<mat>@70" keeps only temperatures <= 70 °C (the MC2 range)."""
    mat, tmax = (mat.split("@")[0], float(mat.split("@")[1])) if "@" in mat else (mat, 1e9)
    d = np.load(ROOT / f"data/mc1/{mat}.npz"); out = []
    ok = (d["Hdc"] == 0) & (d["T"] <= tmax)
    for i, f in enumerate(np.unique(d["f"][ok])):
        g = np.where(ok & (d["f"] == f))[0]; L = d["n16"][g].min()
        out.append(dict(B=d["B"][g, :L], H=d["H"][g, :L], T=d["T"][g], f=f, file=i + 1, gid=g))
    return out


def split(data, seed=0, frac=(0.7, 0.15, 0.15)):
    """Split each file's sequences into train/val/test, stratified by temperature."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for d in data:
        idx = {"train": [], "val": [], "test": []}
        for t in np.unique(d["T"]):
            ii = rng.permutation(np.where(d["T"] == t)[0])
            n1, n2 = int(frac[0] * len(ii)), int((frac[0] + frac[1]) * len(ii))
            idx["train"] += list(ii[:n1]); idx["val"] += list(ii[n1:n2]); idx["test"] += list(ii[n2:])
        for k, v in idx.items():
            v = np.sort(np.array(v, dtype=int))
            parts[k].append(dict(B=d["B"][v], H=d["H"][v], T=d["T"][v], f=d["f"], file=d["file"], rows=v,
                                 **({"gid": d["gid"][v]} if "gid" in d else {})))
    return parts


def restrict(parts, spec):
    """Training restriction for extrapolation / data-efficiency studies, applied to train and val (test untouched):
    "T25_50" keeps those temperatures, "fmax320" files up to 320 kHz, "frac0.1" a random 10 % of each file's sequences."""
    rng = np.random.default_rng(7); out = {}
    for kind, pl in parts.items():
        if kind == "test": out[kind] = pl; continue
        new = []
        for d in pl:
            if spec.startswith("fmax") and float(d["f"]) > float(spec[4:]) * 1e3: continue
            if spec.startswith("T"): keep = np.isin(np.round(d["T"]), [float(x) for x in spec[1:].split("_")])
            elif spec.startswith("frac"): keep = rng.random(len(d["T"])) < float(spec[4:])
            else: keep = np.ones(len(d["T"]), bool)
            if keep.sum() == 0: continue
            new.append({k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v) for k, v in d.items()})
        out[kind] = new
    return out


def ood_mask(w, spec, mat):
    """Test windows outside the training restriction `spec` (True = extrapolation)."""
    if spec.startswith("T"): return ~np.isin(np.round(w["T"]), [float(x) for x in spec[1:].split("_")])
    if spec.startswith("fmax"):
        fs = {int(d["file"]): float(d["f"]) for d in load_material(mat)}                  # file id -> frequency
        return np.array([fs[int(i)] for i in w["fid"]]) > float(spec[4:]) * 1e3
    return np.ones(len(w["T"]), bool)


def split_mat(mat, seed=0):
    """Train/val/test split of a material. For the final-evaluation materials all data is train/val (85/15); their
    test set is the official one. "<mat>#<spec>" restricts train/val (see restrict)."""
    if "#" in mat:
        base, spec = mat.split("#"); return restrict(split_mat(base, seed), spec)
    official = mat in EVAL_MATS                                           # these have an official, separate test set
    return split(load_material(mat), seed, (0.85, 0.15, 0.0) if official else (0.7, 0.15, 0.15))


def official_test(mat):
    """Official MC2 test windows of a final-evaluation material: B, H, T (N,), k (N,) warm-up length per window."""
    import pandas as pd
    d = ROOT / f"data/raw/mc2/evaltest/Material {RAW_LABEL[mat]} - {mat}/{mat}_Testing_"
    rd = lambda q: pd.read_csv(str(d) + q + ".csv", header=None).to_numpy(np.float32)
    P = rd("Padded_H_seq"); k = np.isnan(P).argmax(1)
    w = dict(B=rd("True_B_seq"), H=rd("True_H_seq"), T=rd("True_T").ravel(), k=k, fid=np.zeros(len(k), int))
    if not set(np.unique(k)) <= set(SCENARIOS.values()):   # material C: the padded file is a placeholder (only H[0])
        del w["k"]                                          # -> every window is evaluated in all three scenarios
    return w


def make_windows(part, n_per_file=300, seed=1, win=WIN):
    """Draw challenge-like evaluation windows. Returns dict of arrays (N, win) plus T, file, and the
    full-history prefix length `start` (useful for oracle experiments)."""
    rng = np.random.default_rng(seed)
    Bs, Hs, Ts, Fs, st, rows = [], [], [], [], [], []
    for d in part:
        n, L = d["B"].shape
        seq = rng.integers(0, n, n_per_file)
        s0 = rng.integers(0, L - win, n_per_file)
        for q, s in zip(seq, s0):
            Bs.append(d["B"][q, s:s + win]); Hs.append(d["H"][q, s:s + win])
            Ts.append(d["T"][q]); Fs.append(d["file"]); st.append(s); rows.append(q)
    return dict(B=np.stack(Bs), H=np.stack(Hs), T=np.array(Ts, np.float32), fid=np.array(Fs),
                start=np.array(st), row=np.array(rows))


def metrics(H_pred, H_true, B, k):
    """SRE and NERE per window. H_pred/H_true/B: (N, WIN), prediction starts at index k.
    NERE: energy error of the predicted part relative to sum |H dB| over the window (windows are not
    periodic, so the signed sum is no loss and can be ~0)."""
    e = H_pred[:, k:] - H_true[:, k:]
    sre = np.sqrt((e ** 2).mean(1)) / np.sqrt((H_true[:, k:] ** 2).mean(1))
    dB = np.gradient(B, axis=1)
    nere = np.abs((dB[:, k:] * e).sum(1)) / np.abs(dB * H_true).sum(1)
    return sre, nere


def select_k(w, k):
    """Windows evaluated with warm-up k: all of them, or for official test sets only those with that k."""
    return w if "k" not in w else {n: v[w["k"] == k] for n, v in w.items()}


def report(predict, windows, name="", verbose=True):
    """predict(B (N,W), H (N,W) with H[:, k:] unknown, T (N,), k) -> H_pred (N,W)."""
    res = {}
    for sc, k in SCENARIOS.items():
        w = select_k(windows, k)
        Hin = w["H"].copy(); Hin[:, k:] = np.nan
        Hp = predict(w["B"], Hin, w["T"], k)
        sre, nere = metrics(Hp, w["H"], w["B"], k)
        res[sc] = dict(sre=sre, nere=nere)
    if verbose:
        s = " | ".join(f"{sc}: SRE {100*r['sre'].mean():5.2f}/{100*np.percentile(r['sre'],95):5.2f}  "
                       f"NERE {100*r['nere'].mean():5.2f}/{100*np.percentile(r['nere'],95):5.2f}"
                       for sc, r in res.items())
        avg = np.mean([r["sre"].mean() for r in res.values()]) * 100
        avg95 = np.mean([np.percentile(r["sre"], 95) for r in res.values()]) * 100
        print(f"{name:28s} SRE avg {avg:5.2f} p95 {avg95:5.2f} || {s}", flush=True)
    return res


def windows(kind, mat="3C94"):
    """Fixed evaluation windows of a material: kind "test" (300 per file, seed 1) or "val" (150 per file, seed 5).
    Created on first use; 3C94 keeps its original file names."""
    if kind == "test" and "#" in mat: mat = mat.split("#")[0]          # restricted training, full test set
    p = ROOT / "data" / (f"{kind}_windows.npz" if mat == "3C94" else f"{mat.replace(':', '_').replace('@', '_T')}_{kind}_windows.npz")
    if not p.exists():
        if kind == "test" and mat in EVAL_MATS:
            np.savez(p, **official_test(mat))
        else:
            n, seed = {"test": (300, 1), "val": (150, 5)}[kind]
            np.savez(p, **make_windows(split_mat(mat)[kind], n, seed=seed))
    return dict(np.load(p))
