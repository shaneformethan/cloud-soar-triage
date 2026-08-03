"""
Severity-mapping sensitivity check on REAL CSE-CIC-IDS2018 flows.

Trains a gradient-boosted tree directly on real per-flow records under a
*temporal* split (earliest 70% of each class trains, latest 30% tests) using two
defensible attack->severity mappings, and reports whether the qualitative
findings (flow signal is real; DoS/DDoS critical class is hardest under temporal
shift; high weighted F1 is carried by easy classes) survive the relabeling.

  ORIGINAL (paper): benign->Info; web+infiltration->Medium;
                    SSH/FTP brute-force + botnet->High; DoS/DDoS->Critical.
  ALT (impact-weighted): brute-force->Medium (attempt, not compromise);
                    infiltration->High; botnet->Critical (C2/exfil).

Flow-only path, CPU-only (HistGradientBoosting). This grounds the
severity-mapping robustness claim in real data without the multimodal pipeline.
"""
from __future__ import annotations
import sys, glob, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

FEATURES = [
    "Dst Port","Protocol","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
    "TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Mean",
    "Bwd Pkt Len Max","Bwd Pkt Len Mean","Flow Byts/s","Flow Pkts/s",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd Header Len","Bwd Header Len","Pkt Len Mean","Pkt Len Std",
    "SYN Flag Cnt","ACK Flag Cnt","Init Fwd Win Byts","Init Bwd Win Byts",
    "Down/Up Ratio",
]
SEV = ["informational","medium","high","critical"]

def norm(lbl):
    return str(lbl).strip().lower()

def map_orig(k):
    if k=="benign": return 0
    if k in ("brute force -web","brute force -xss","sql injection","infilteration","infiltration"): return 1
    if k in ("ftp-bruteforce","ssh-bruteforce","bot","heartbleed"): return 2
    if "ddos" in k or "dos" in k: return 3
    if "bot" in k or "bruteforce" in k or "heartbleed" in k: return 2
    if "infilt" in k or "sql" in k or "xss" in k or "web" in k: return 1
    return None

def map_alt(k):
    if k=="benign": return 0
    if k in ("brute force -web","brute force -xss","sql injection","ftp-bruteforce","ssh-bruteforce"): return 1
    if k in ("infilteration","infiltration","heartbleed"): return 2
    if k=="bot": return 3
    if "ddos" in k or "dos" in k: return 3
    if "bot" in k: return 3
    if "infilt" in k or "heartbleed" in k: return 2
    if "sql" in k or "xss" in k or "web" in k or "bruteforce" in k: return 1
    return None

CAP = 3000            # max rows per raw label
NROWS = 350_000       # rows read per day-file (fast inline mode)
want = set(FEATURES) | {"Label","Timestamp"}

def load(seed=42):
    rng = np.random.default_rng(seed)
    rows_feat, rows_lbl, rows_t = [], [], []
    counts = {}
    for f in sorted(glob.glob("data/raw/cic_ids2018/*.csv")):
        print(f"[load] {f}", flush=True)
        chunk = pd.read_csv(f, usecols=lambda c: c in want, nrows=NROWS, low_memory=False)
        chunk = chunk.dropna(subset=["Label"])
        for lab, grp in chunk.groupby("Label"):
            k = norm(lab)
            if k in ("","label","nan"): continue
            have = counts.get(k,0)
            if have >= CAP: continue
            take = min(CAP-have, len(grp))
            g = grp.sample(take, random_state=int(rng.integers(1e9))) if len(grp)>take else grp
            rows_feat.append(g[FEATURES]); rows_lbl += [k]*len(g)
            rows_t.append(g["Timestamp"])
            counts[k] = have+len(g)
    X = pd.concat(rows_feat, ignore_index=True).replace([np.inf,-np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    t = pd.to_datetime(pd.concat(rows_t, ignore_index=True), errors="coerce")
    labs = np.array(rows_lbl)
    print("[load] label counts:", counts, flush=True)
    return X, labs, t.to_numpy()

def temporal_split(t, y, frac=0.70):
    tr = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y==c)[0]
        order = idx[np.argsort(t[idx])]
        k = int(len(order)*frac)
        tr[order[:k]] = True
    return tr

def run(X, labs, t, mapper, name):
    y = np.array([mapper(k) for k in labs])
    keep = y!=None
    X2, y2, t2 = X[keep], y[keep].astype(int), t[keep]
    tr = temporal_split(t2, y2)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)
    clf.fit(X2[tr], y2[tr])
    pred = clf.predict(X2[~tr])
    yt = y2[~tr]
    wf1 = f1_score(yt, pred, average="weighted")
    perc = f1_score(yt, pred, average=None, labels=[0,1,2,3])
    crit_rec = recall_score(yt, pred, labels=[3], average="macro", zero_division=0)
    print(f"\n=== {name} mapping (temporal) ===", flush=True)
    print(f"weighted F1 = {wf1:.3f}", flush=True)
    for i,s in enumerate(SEV):
        print(f"  {s:13s} F1={perc[i]:.3f}", flush=True)
    print(f"  critical recall = {crit_rec:.3f}", flush=True)
    return wf1, perc, crit_rec

if __name__ == "__main__":
    t0=time.time()
    X, labs, t = load()
    print(f"[load] done in {time.time()-t0:.0f}s, X={X.shape}", flush=True)
    run(X, labs, t, map_orig, "ORIGINAL")
    run(X, labs, t, map_alt, "ALT (impact-weighted)")
    print(f"\n[all done in {time.time()-t0:.0f}s]", flush=True)
