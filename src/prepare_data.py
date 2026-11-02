"""Convert the downloaded MagNet Challenge data into the arrays the code reads (see README, "Data").

    data/raw/mc2/Training Data/<mat>.zip        10 MC2 training materials (CSV)       -> data/npy/<mat>/, data/info/
    data/raw/mc2/evaltrain/Material <X>/        training data of the 5 final-evaluation materials (CSV, labels A-E)
                                                                                      -> data/npy/<mat>/
    data/raw/mc2/test-set-5-eval-materials.zip  official labelled test set of these 5 -> data/raw/mc2/evaltest/ (read as is)
    data/raw/mc1/Raw Data.zip                   MagNet (MC1) raw records (.mat), transfer study -> data/mc1/<mat>.npz

Usage: python prepare_data.py [mc2] [eval] [mc1]   (default: all; existing outputs are skipped)"""
import subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from mc2lib import ROOT, EVAL_MATS, RAW_LABEL

RAW = ROOT / "data/raw"
MC2_MATS = ["3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87"]
MC1_MATS = MC2_MATS


def csv_to_npy(src, prefix, dst, mat):
    """<src>/<prefix>_<i>_{B,H,T}.csv -> <dst>/<mat>_<i>_{B,H,T}.npy (float32)."""
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(1, 8):
        for q in "BHT":
            f = src / f"{prefix}_{i}_{q}.csv"
            if f.exists():
                a = pd.read_csv(f, header=None, dtype=np.float32, engine="c").to_numpy()
                np.save(dst / f"{mat}_{i}_{q}.npy", a); print(f"{mat}: {f.name} {a.shape}", flush=True)


def mc2():
    """Convert the MC2 training data of the 10 materials (zipped CSV files) to data/npy/<material>/."""
    for mat in MC2_MATS:
        if (ROOT / f"data/npy/{mat}/{mat}_1_B.npy").exists():
            print("skip", mat); continue
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(RAW / f"mc2/Training Data/{mat}.zip") as z:
            z.extractall(tmp)
            (ROOT / "data/info").mkdir(parents=True, exist_ok=True)
            (ROOT / f"data/info/{mat}_Test_Info.xlsx").write_bytes((Path(tmp) / mat / "Test_Info.xlsx").read_bytes())
            csv_to_npy(Path(tmp) / mat, mat, ROOT / f"data/npy/{mat}", mat)


def evalmats():
    """Convert the training data and the labeled test set of the 5 final-evaluation materials."""
    for mat in EVAL_MATS:
        if not (ROOT / f"data/npy/{mat}/{mat}_1_B.npy").exists():
            lab = RAW_LABEL[mat]
            csv_to_npy(RAW / f"mc2/evaltrain/Material {lab}", lab, ROOT / f"data/npy/{mat}", mat)
    if not (RAW / "mc2/evaltest").exists():
        with zipfile.ZipFile(RAW / "mc2/test-set-5-eval-materials.zip") as z:
            z.extractall(RAW / "mc2/evaltest")


def mc1():
    """Convert the MagNet (MC1) raw data of the transfer materials to data/mc1/<material>.npz."""
    (ROOT / "data/mc1").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW / "mc1/Raw Data.zip") as z:
        for mat in MC1_MATS:
            if (ROOT / f"data/mc1/{mat}.npz").exists():
                print("skip mc1", mat); continue
            with tempfile.TemporaryDirectory() as tmp:
                for n in z.namelist():
                    if n.startswith(f"{mat}/") and n.endswith(".mat"):
                        z.extract(n, tmp)
                        subprocess.run([sys.executable, str(Path(__file__).parent / "mc1_convert.py"), str(Path(tmp) / n),
                                        str(ROOT / f"data/mc1/{mat}.npz")], check=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["mc2", "eval", "mc1"]
    for s in steps:
        {"mc2": mc2, "eval": evalmats, "mc1": mc1}[s]()
