"""Structured run records and run status.

results/runs.jsonl: one JSON line per evaluation, appended by eval_m.py (kind "test": test SRE / NERE per material on the
15 materials) and official_eval.py (kind "official": MC2 metrics on the five final-evaluation materials). Each line
carries the checkpoint name, variant, training configuration from the checkpoint, git commit and date; later lines for
the same (kind, name, variant) supersede earlier ones. logs/<name>.status: running | evaluating | done | failed: <reason>,
written by the queue scripts (scripts/lib.sh).

Usage: python runlog.py [regex]                 table of all runs (test and official merged), names matching regex
       python runlog.py --status <name> <text>  write logs/<name>.status
       python runlog.py --backfill             records from the existing results/*.json (one-off)"""
import sys, json, re, subprocess, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/runs.jsonl"


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def config(ck):
    """Training configuration stored in a checkpoint: scalar entries and the saved command-line arguments."""
    c = {k: v for k, v in ck.items() if isinstance(v, (int, float, str, bool)) and k not in ("names", "mats")}
    c.update(ck.get("args", {}))
    return c


def append(rec):
    """Append one record (dict) to results/runs.jsonl with commit and date."""
    rec = {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
           "date": datetime.datetime.now().isoformat(timespec="seconds"), **rec}
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def status(name, text):
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / f"logs/{name}.status").write_text(text.strip() + "\n")


def test_record(name, variant, res, cfg):
    """Record of kind "test" from eval_m.py's per-material result {mat: {variant: {sre, scen, nere, nparams, val}}}."""
    mats = [m for m in res if variant in res[m]]
    r = {m: res[m][variant] for m in mats}
    mean = lambda xs: sum(xs) / len(xs)
    return {"kind": "test", "name": name, "variant": variant, "params_per_mat": r[mats[0]]["nparams"], "config": cfg,
            "test_sre": {m: round(r[m]["sre"], 4) for m in mats},
            "test_nere": {m: round(mean(r[m]["nere"]), 4) for m in mats},
            "test_sre_mean": round(mean([r[m]["sre"] for m in mats]), 4),
            "test_nere_mean": round(mean([mean(r[m]["nere"]) for m in mats]), 4),
            "val_sre_mean": round(mean([r[m]["val"] for m in mats]), 4) if all("val" in r[m] for m in mats) else None,
            "n_mats": len(mats)}


def official_record(name, variant, res, cfg):
    """Record of kind "official" from official_eval.py's {mat: {sre_mean, sre_p95, nere_mean, nere_p95}, "average": ...}."""
    return {"kind": "official", "name": name, "variant": variant, "config": cfg,
            **{f"off_{k}": round(v, 4) for k, v in res["average"].items()},
            "off_per_mat": {m: {k: round(v, 4) for k, v in r.items()} for m, r in res.items() if m != "average"}}


def load():
    """All records, the latest per (kind, name, variant)."""
    import pandas as pd
    recs = {}
    for line in RUNS.read_text().splitlines():
        r = json.loads(line); recs[(r["kind"], r["name"], r["variant"])] = r
    return pd.DataFrame(list(recs.values()))


def table(pattern=None):
    """Test and official records merged per (name, variant)."""
    df = load()
    t = df[df.kind == "test"][["name", "variant", "params_per_mat", "test_sre_mean", "test_nere_mean", "commit"]]
    o = df[df.kind == "official"][["name", "variant", "off_sre_p95", "off_nere_p95"]] if "off_sre_p95" in df else None
    m = t.merge(o, on=["name", "variant"], how="outer") if o is not None else t
    if pattern:
        m = m[m.name.str.contains(pattern) | m.variant.str.contains(pattern)]
    return m.sort_values(["name", "variant"])


def backfill():
    """Records from the result files already in results/ (date and commit of the file's last commit)."""
    n = 0
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = lambda: {"commit": _git("log", "-1", "--format=%h", "--", str(f)), "date": _git("log", "-1", "--format=%cI", "--", str(f))
                        or datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"), "backfilled": True}
        if f.name.startswith("official_") and "average" in d:
            spec = f.stem[len("official_"):]
            if spec == "gru":
                name, var = "gru", "RHINO GRU8"
            else:                                                       # <ckpt>_<variant>: split where models/<ckpt>.pt exists
                cut = [i for i, ch in enumerate(spec) if ch == "_" and (ROOT / f"models/{spec[:i]}.pt").exists()]
                if not cut:
                    continue
                name, var = spec[:cut[0]], spec[cut[0] + 1:]
            rec = official_record(name, var, d, {})
        elif isinstance(d, dict) and d and all(isinstance(v, dict) for v in d.values()) and \
                all("sre" in x and "nere" in x for v in d.values() for k, x in v.items() if k != "GRU8" and isinstance(x, dict)):
            vs = {k for v in d.values() for k in v if k != "GRU8"}
            for var in sorted(vs):
                try:
                    rec = test_record(f.stem, var, d, {})
                except (KeyError, IndexError, ZeroDivisionError, TypeError):
                    continue
                with open(RUNS, "a") as fh:
                    fh.write(json.dumps({**meta(), **rec}) + "\n"); n += 1
            continue
        else:
            continue
        with open(RUNS, "a") as fh:
            fh.write(json.dumps({**meta(), **rec}) + "\n"); n += 1
    print(f"{n} records backfilled into {RUNS}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--status"]:
        status(a[1], " ".join(a[2:]))
    elif a[:1] == ["--backfill"]:
        backfill()
    else:
        import pandas as pd
        pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 40)
        print(table(a[0] if a else None).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
