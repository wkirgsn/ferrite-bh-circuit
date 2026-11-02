"""Score predictions on the official MC2 final-evaluation set (3C92, 3C95, FEC007, FEC014, T37) with the challenge's
metrics (Test/Error_Distribution_Plot_Template.m of the MC2 repository): per window the sequence relative error
SRE = rms(H_pred - H_meas) / rms(H_meas) and the normalized energy error NERE = sum((H_mid,pred - H_mid,meas) dB) / |Loss|
over the predicted part (H_mid: trapezoidal midpoints, Loss from <mat>_Testing_True_Loss.csv), both in %. The rows of
each file are ordered in thirds with warm-up k = 100, 500, 900. The ranking uses the 95th percentile of |SRE| and
|NERE| per scenario (MATLAB prctile), averaged over the scenarios and the five materials. Checked against the official
per-material results of all teams (Result_Final.xlsx): for the published RHINO-MAG GRU8 models this reproduces the
official values of that team (SiegenPaderborn) on 3C92, 3C95, FEC014 and T37 to within 0.3 percentage points; FEC007
differs (26.1 % / 2.8 % vs 27.0 % / 4.85 %), because its local padded warm-up file holds only H[0] and the warm-up
lengths used officially are unknown (thirds assumed here).
Usage: python official_eval.py gru | <ckpt>:<variant> [...]   -> results/official_<name>.json, a line in results/runs.jsonl"""
import sys, json, numpy as np, pandas as pd, torch
from mc2lib import ROOT, RAW_LABEL, official_test

MATS = ["3C92", "3C95", "FEC007", "FEC014", "T37"]
KS = [100, 500, 900]


def official(mat):
    """Official MC2 test set of a final-evaluation material, with the warm-up length k and the reference loss per window."""
    w = official_test(mat); n = len(w["B"])
    w["k"] = np.repeat(KS, n // 3)                                       # thirds (FEC007's padded file is a placeholder)
    d = ROOT / f"data/raw/mc2/evaltest/Material {RAW_LABEL[mat]} - {mat}/{mat}_Testing_True_Loss.csv"
    w["loss"] = pd.read_csv(d, header=None).to_numpy(float).ravel()
    return w


def prctile(x, p):
    """MATLAB prctile (as in the official template): sorted values at the positions 100 (i - 0.5) / n."""
    x = np.sort(x); n = len(x); return float(np.interp(p, 100 * (np.arange(1, n + 1) - 0.5) / n, x))


def summary(w, s, n):
    """Mean and 95th percentile per warm-up scenario (the template scores each segment separately), averaged."""
    f = lambda x, g: float(np.mean([g(x[w["k"] == k]) for k in KS]))
    return dict(sre_mean=f(s, np.mean), sre_p95=f(s, lambda x: prctile(x, 95)),
                nere_mean=f(n, np.mean), nere_p95=f(n, lambda x: prctile(x, 95)))


def score(w, Hp):
    """Per-window SRE and energy error in % as in the official MC2 evaluation (predicted part only)."""
    sre, nere = [], []
    for i in range(len(Hp)):
        k = w["k"][i]; p, m, B = Hp[i, k:].astype(float), w["H"][i, k:].astype(float), w["B"][i, k:].astype(float)
        sre.append(100 * np.sqrt(np.mean((p - m) ** 2)) / np.sqrt(np.mean(m ** 2)))
        dB = np.diff(B); e = ((p[1:] + p[:-1]) / 2 - (m[1:] + m[:-1]) / 2) * dB
        nere.append(100 * e.sum() / abs(w["loss"][i]))
    return np.abs(sre), np.abs(nere)


def predict_gru(mat, w):
    """Predictions of the GRU8 replica (data/pred/<mat>_gru.npz) in the row order of the official test set."""
    d = np.load(ROOT / f"data/pred/{mat}_gru.npz"); n = len(w["k"]) // 3; sc = ["10/90", "50/50", "90/10"]
    if len(d[sc[0]]) == n:                                                # one third of the rows per scenario
        return np.concatenate([d[s] for s in sc])
    return np.concatenate([d[s][i * n:(i + 1) * n] for i, s in enumerate(sc)])    # all rows in every scenario


def predict_model(ckn, var, mat, w):
    """Predictions of variant var of checkpoint ckn on the official test set, each window with its own k."""
    from compact_m import DEV
    import compact_m, scan
    if DEV.type == "cpu":
        compact_m.scan = scan.scan_ref
    from loader import load
    m, names, mats = load(ckn)                                            # classical or scheduled circuit
    e, mid = names.index(var), mats.index(mat); out = np.zeros_like(w["H"])
    with torch.no_grad():
        for k in KS:
            idx = np.where(w["k"] == k)[0]
            for j in range(0, len(idx), 512):
                ii = idx[j:j + 512]; t = lambda x: torch.tensor(x[ii], device=DEV)
                out[ii] = m(t(w["B"]), t(w["H"]), t(w["T"]), torch.full((len(ii),), mid, device=DEV), k)[e].cpu().numpy()
    return out


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        res = {}
        for mat in MATS:
            w = official(mat)
            Hp = predict_gru(mat, w) if spec == "gru" else predict_model(*spec.split(":"), mat, w)
            s, n = score(w, Hp)
            res[mat] = summary(w, s, n)
            print(f"{spec} {mat:7s} " + "  ".join(f"{k} {v:6.2f}" for k, v in res[mat].items()), flush=True)
        avg = {k: float(np.mean([res[m][k] for m in MATS])) for k in res[MATS[0]]}
        print(f"{spec} AVERAGE " + "  ".join(f"{k} {v:6.2f}" for k, v in avg.items()), flush=True)
        res["average"] = avg
        name = spec.replace(":", "_")
        json.dump({m: {k: float(v) for k, v in r.items()} for m, r in res.items()}, open(ROOT / f"results/official_{name}.json", "w"), indent=1)
        import runlog                                                     # one record in results/runs.jsonl
        ckn, var = spec.split(":") if spec != "gru" else ("gru", "RHINO GRU8")
        cfg = runlog.config(torch.load(ROOT / f"models/{ckn}.pt", map_location="cpu", weights_only=False)) if spec != "gru" else {}
        runlog.append(runlog.official_record(ckn, var, res, cfg))
