"""
External multimodal baseline: does late fusion change the equivalence or the
"simplest model wins" conclusion?

We implement soft-voting late fusion in the style of Kiflay et al. (2024) and
MIND (2021): one classifier per modality, predictions combined by averaging the
per-class probabilities.

Design choice: the base learner is held CONSTANT across arms
(HistGradientBoostingClassifier for every model), so the only thing that varies
between early fusion and late fusion is the fusion strategy itself. Using
XGBoost for one arm and something else for the other would confound learner and
fusion.

Arms
  early-fusion   : one model on the concatenated 92-dim vector (the Trad-B setup)
  soft-voting    : IAM / flow / TI models, unweighted mean of probabilities
  weighted-vote  : same, weights proportional to validation macro F1

Run from anywhere: python3 scripts/soft_voting_baseline.py
"""
from __future__ import annotations
import json, pickle, statistics as st, sys, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parent.parent   # repo root
sys.path.insert(0, str(ROOT))                  # the pickles reference src.benchmark.*
OUT = ROOT / "results/soft_voting_results.json"
SEEDS = [42, 43, 44, 45, 46]
IAM_BINS, N_CLASS, CRITICAL = 64, 4, 3


# ----------------------------------------------------------------- features
def iam_histogram(seq: np.ndarray, n: int) -> np.ndarray:
    """Normalised action-type histogram, identical to src/baselines/traditional.py."""
    h = np.zeros(IAM_BINS, dtype=np.float32)
    for idx in seq[:n, 0]:
        h[min(int(idx), IAM_BINS - 1)] += 1.0
    return h / n if n > 0 else h


def load(split_dir: Path, split: str):
    clusters = pickle.load(open(split_dir / f"benchmark_{split}.pkl", "rb"))
    iam = np.stack([iam_histogram(c.iam_seq, int(c.iam_seq_len)) for c in clusters])
    flow = np.stack([c.flow_vec for c in clusters]).astype(np.float32)
    ti = np.stack([c.ti_vec for c in clusters]).astype(np.float32)
    y = np.array([int(c.severity_label) for c in clusters], dtype=np.int64)
    return {"iam": iam, "flow": flow, "ti": ti}, y


def learner(seed: int):
    # max_features<1 gives per-split column subsampling, mirroring the
    # subsample=0.9 stochasticity of the XGBoost baseline in the paper.
    # Without it HistGB is deterministic and every seed returns the same
    # number, which would make the across-seed tests meaningless.
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1, max_features=0.8,
        class_weight="balanced", random_state=seed)


def proba(model, X):
    """Expand to a full 4-class probability matrix (a class may be absent)."""
    p = np.zeros((X.shape[0], N_CLASS), dtype=np.float64)
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return p


def scores(y, pred):
    return {
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "critical_recall": recall_score(y, pred, labels=[CRITICAL],
                                        average="macro", zero_division=0),
    }


# ----------------------------------------------------------------- one regime
def run_regime(name: str, split_dir: Path):
    Xtr, ytr = load(split_dir, "train")
    Xva, yva = load(split_dir, "val")
    Xte, yte = load(split_dir, "test")
    print(f"\n### {name}: train {len(ytr)}  val {len(yva)}  test {len(yte)}")

    per_arm: dict[str, list[dict]] = {"early_fusion": [], "soft_voting": [],
                                      "weighted_voting": []}
    per_modality: dict[str, list[float]] = {"iam": [], "flow": [], "ti": []}

    for seed in SEEDS:
        t0 = time.time()

        # --- arm 1: early fusion on the concatenated 92-dim vector -----------
        cat = lambda X: np.hstack([X["iam"], X["flow"], X["ti"]])
        m = learner(seed).fit(cat(Xtr), ytr)
        per_arm["early_fusion"].append(scores(yte, m.predict(cat(Xte))))

        # --- arms 2 and 3: one model per modality, combine probabilities -----
        te_p, va_macro = {}, {}
        for mod in ("iam", "flow", "ti"):
            mm = learner(seed).fit(Xtr[mod], ytr)
            te_p[mod] = proba(mm, Xte[mod])
            va_macro[mod] = f1_score(yva, mm.predict(Xva[mod]),
                                     average="macro", zero_division=0)
            per_modality[mod].append(
                f1_score(yte, mm.predict(Xte[mod]), average="weighted",
                         zero_division=0))

        stacked = np.stack([te_p[m_] for m_ in ("iam", "flow", "ti")])
        per_arm["soft_voting"].append(scores(yte, stacked.mean(0).argmax(1)))

        w = np.array([va_macro[m_] for m_ in ("iam", "flow", "ti")], dtype=np.float64)
        w = w / w.sum()
        per_arm["weighted_voting"].append(
            scores(yte, (stacked * w[:, None, None]).sum(0).argmax(1)))

        print(f"  seed {seed}  ({time.time()-t0:.1f}s)  "
              f"early={per_arm['early_fusion'][-1]['weighted_f1']:.4f}  "
              f"soft={per_arm['soft_voting'][-1]['weighted_f1']:.4f}  "
              f"wtd={per_arm['weighted_voting'][-1]['weighted_f1']:.4f}")

    return per_arm, per_modality


# ----------------------------------------------------------------- stats
def agg(runs, key):
    v = [r[key] for r in runs]
    return st.mean(v), st.pstdev(v)


def welch(a, b):
    import math
    ma, mb, va, vb, na, nb = st.mean(a), st.mean(b), st.variance(a), st.variance(b), len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, float("nan"), 1.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # two-sided p via regularised incomplete beta
    def betacf(x, y, z):
        c, d = 1.0, 1.0 - (x + y) * z / (x + 1.0)
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30); h = d
        for mI in range(1, 200):
            m2 = 2 * mI
            aa = mI * (y - mI) * z / ((x - 1.0 + m2) * (x + m2))
            d = 1.0 / max(abs(1.0 + aa * d), 1e-30) * (1 if 1.0 + aa * d > 0 else -1)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30); h *= d * c
            aa = -(x + mI) * (x + y + mI) * z / ((x + m2) * (x + 1.0 + m2))
            d = 1.0 / max(abs(1.0 + aa * d), 1e-30) * (1 if 1.0 + aa * d > 0 else -1)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c; h *= de
            if abs(de - 1.0) < 3e-12: break
        return h
    def binc(x, y, z):
        if z <= 0: return 0.0
        if z >= 1: return 1.0
        lb = (math.lgamma(x + y) - math.lgamma(x) - math.lgamma(y)
              + x * math.log(z) + y * math.log1p(-z))
        if z < (x + 1.0) / (x + y + 2.0):
            return math.exp(lb) * betacf(x, y, z) / x
        return 1.0 - math.exp(lb) * betacf(y, x, 1.0 - z) / y
    p = binc(df / 2.0, 0.5, df / (df + t * t))
    return t, df, p


CKPT = ROOT / "results/.sv_ckpt.json"
DIRS = {"iid": ROOT / "data/processed", "temporal": ROOT / "data/processed_temporal"}


def run_one(regime: str, seed: int, cache: dict):
    """One (regime, seed) cell. Splits are cached across calls in `cache`."""
    if regime not in cache:
        d = DIRS[regime]
        cache[regime] = (load(d, "train"), load(d, "val"), load(d, "test"))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = cache[regime]

    cat = lambda X: np.hstack([X["iam"], X["flow"], X["ti"]])
    out = {"early_fusion": scores(yte, learner(seed).fit(cat(Xtr), ytr).predict(cat(Xte)))}

    te_p, va_macro, mod_f1 = {}, {}, {}
    for mod in ("iam", "flow", "ti"):
        mm = learner(seed).fit(Xtr[mod], ytr)
        te_p[mod] = proba(mm, Xte[mod])
        va_macro[mod] = f1_score(yva, mm.predict(Xva[mod]), average="macro", zero_division=0)
        mod_f1[mod] = f1_score(yte, mm.predict(Xte[mod]), average="weighted", zero_division=0)

    stacked = np.stack([te_p[m_] for m_ in ("iam", "flow", "ti")])
    out["soft_voting"] = scores(yte, stacked.mean(0).argmax(1))
    w = np.array([va_macro[m_] for m_ in ("iam", "flow", "ti")]); w = w / w.sum()
    out["weighted_voting"] = scores(yte, (stacked * w[:, None, None]).sum(0).argmax(1))
    out["per_modality"] = mod_f1
    out["voting_weights"] = dict(zip(("iam", "flow", "ti"), w.round(4).tolist()))
    return out


def drive(budget=32.0):
    """Resumable driver: does as many cells as fit in `budget` seconds."""
    done = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    todo = [(r, s) for r in ("iid", "temporal") for s in SEEDS
            if f"{r}|{s}" not in done]
    t0, cache = time.time(), {}
    for regime, seed in todo:
        if time.time() - t0 > budget:
            break
        done[f"{regime}|{seed}"] = run_one(regime, seed, cache)
        CKPT.write_text(json.dumps(done))
        print(f"  {regime} seed {seed} done  "
              f"(early {done[f'{regime}|{seed}']['early_fusion']['weighted_f1']:.4f}, "
              f"soft {done[f'{regime}|{seed}']['soft_voting']['weighted_f1']:.4f})",
              flush=True)
    left = 2 * len(SEEDS) - len(done)
    print(f"progress {len(done)}/{2*len(SEEDS)}  remaining {left}")
    return left


if __name__ == "__main__":
    if "--drive" in sys.argv:
        sys.exit(0 if drive() == 0 else 7)

    results = {}
    for regime, d in (("iid", ROOT / "data/processed"),
                      ("temporal", ROOT / "data/processed_temporal")):
        arms, mods = run_regime(regime, d)
        results[regime] = {"arms": arms, "per_modality_weighted_f1": mods}

    print("\n" + "=" * 78)
    print("SOFT-VOTING LATE FUSION vs EARLY FUSION  (base learner held constant)")
    print("=" * 78)
    for regime in ("iid", "temporal"):
        print(f"\n--- {regime} ---")
        arms = results[regime]["arms"]
        for arm in ("early_fusion", "soft_voting", "weighted_voting"):
            w = agg(arms[arm], "weighted_f1")
            m = agg(arms[arm], "macro_f1")
            c = agg(arms[arm], "critical_recall")
            print(f"  {arm:<16} W-F1 {w[0]:.3f}±{w[1]:.3f}   "
                  f"M-F1 {m[0]:.3f}±{m[1]:.3f}   crit-R {c[0]:.3f}±{c[1]:.3f}")
        a = [r["weighted_f1"] for r in arms["early_fusion"]]
        b = [r["weighted_f1"] for r in arms["soft_voting"]]
        t, df, p = welch(a, b)
        print(f"  early vs soft-voting: diff={st.mean(a)-st.mean(b):+.4f}  "
              f"t={t:.2f}  df={df:.1f}  p={p:.4f}  "
              f"{'SIGNIFICANT' if p < .05 else 'not significant'}")
        ca = [r["critical_recall"] for r in arms["early_fusion"]]
        cb = [r["critical_recall"] for r in arms["soft_voting"]]
        t2, df2, p2 = welch(ca, cb)
        print(f"  critical recall:      diff={st.mean(ca)-st.mean(cb):+.4f}  "
              f"t={t2:.2f}  p={p2:.4f}")
        print("  per-modality W-F1: " + "  ".join(
            f"{k}={st.mean(v):.3f}" for k, v in
            results[regime]["per_modality_weighted_f1"].items()))

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwritten -> {OUT}")
