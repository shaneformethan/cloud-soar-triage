"""
Build a REAL threat-intelligence pool from AbuseIPDB + AlienVault OTX.

The CSE-CIC-IDS2018 CSVs ship without IP columns, so to use *live* threat
intelligence (draft Section III-E) we curate a pool of real IPs and enrich them
through the user's API keys:

  * Malicious IPs come from the AbuseIPDB blacklist, binned by abuse-confidence
    into three severity tiers (medium / high / critical).
  * Benign IPs are a fixed list of well-known reputable hosts (severity 0).

Each unique IP is enriched once (real AbuseIPDB /check + OTX lookup, cached on
disk by ThreatIntelEnricher) into the 16-dim vector of Section III-B.  IPs are
partitioned into disjoint train/val/test subsets so no real IP's vector leaks
across splits.  Output: ``data/processed/ti_pool.pkl``.

Usage:
    python scripts/fetch_ti_pool.py --per_tier 60
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.threat_intel import ThreatIntelEnricher

# Well-known reputable IPs (public DNS resolvers, major CDNs / clouds).
_BENIGN_IPS = [
    "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9", "149.112.112.112",
    "208.67.222.222", "208.67.220.220", "4.2.2.1", "4.2.2.2",
    "13.107.21.200", "13.107.42.14", "20.190.160.0", "23.0.0.0",
    "104.16.0.1", "104.16.1.1", "104.17.0.1", "104.18.0.1",
    "151.101.0.1", "151.101.64.1", "172.217.0.0", "142.250.0.0",
    "34.0.0.1", "35.190.0.1", "52.84.0.1", "54.230.0.1",
    "157.240.0.1", "31.13.64.1", "199.232.0.1", "192.0.66.1",
    "140.82.112.3", "140.82.113.3", "185.199.108.153", "185.199.109.153",
    "17.253.144.10", "23.215.0.1", "96.16.0.1", "2.16.0.1",
    "198.41.0.4", "199.9.14.201", "192.33.4.12", "199.7.91.13",
]


def _load_env() -> None:
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def fetch_blacklist(api_key: str, limit: int = 10000, conf_min: int = 25):
    """Return list of (ip, confidence) from the AbuseIPDB blacklist."""
    r = requests.get(
        "https://api.abuseipdb.com/api/v2/blacklist",
        headers={"Key": api_key, "Accept": "application/json"},
        params={"confidenceMinimum": conf_min, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    return [(d["ipAddress"], int(d["abuseConfidenceScore"])) for d in data]


def _tier_for_conf(conf: int) -> int | None:
    """Map abuse-confidence to a severity tier (1=med, 2=high, 3=crit)."""
    if conf >= 90:
        return 3
    if conf >= 70:
        return 2
    if conf >= 25:
        return 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_tier", type=int, default=60,
                    help="max IPs to enrich per severity tier")
    ap.add_argument("--out", default="data/processed/ti_pool.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.chdir(Path(__file__).parent.parent)
    _load_env()
    ab = os.environ.get("ABUSEIPDB_API_KEY") or os.environ.get("ABUSEIPDB_KEY", "")
    if not ab:
        raise SystemExit("No ABUSEIPDB_API_KEY in .env")

    rng = np.random.default_rng(args.seed)
    enricher = ThreatIntelEnricher(cache_dir="data/processed/ti_cache")

    print("[ti] Fetching AbuseIPDB blacklist …")
    bl = fetch_blacklist(ab, limit=10000, conf_min=25)
    print(f"[ti] blacklist returned {len(bl)} IPs")

    # Bin malicious IPs into tiers.
    tiers: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for ip, conf in bl:
        t = _tier_for_conf(conf)
        if t:
            tiers[t].append(ip)
    for t in tiers:
        rng.shuffle(tiers[t])
        tiers[t] = tiers[t][: args.per_tier]
    benign = list(_BENIGN_IPS)
    rng.shuffle(benign)
    print(f"[ti] tier counts: med={len(tiers[1])} high={len(tiers[2])} "
          f"crit={len(tiers[3])} benign={len(benign)}")

    # Enrich every unique IP (real API, cached) -> 16-dim vector.
    pool_by_sev: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    sev_ips = {0: benign, 1: tiers[1], 2: tiers[2], 3: tiers[3]}
    for sev, ips in sev_ips.items():
        for i, ip in enumerate(ips):
            vec = enricher.enrich_bucket([ip])
            pool_by_sev[sev].append(vec.astype(np.float32))
            if (i + 1) % 20 == 0:
                print(f"[ti]   sev{sev}: enriched {i+1}/{len(ips)}")
            time.sleep(0.05)  # be gentle with the API

    # Report mean reputation per tier (sanity: should rise with severity).
    for sev in (0, 1, 2, 3):
        arr = np.array(pool_by_sev[sev]) if pool_by_sev[sev] else np.zeros((1, 16))
        print(f"[ti] sev{sev}: n={len(pool_by_sev[sev])} "
              f"mean_rep={arr[:,0].mean():.3f} mean_abuse={arr[:,2].mean():.3f}")

    # Real IP reputation is effectively BINARY (the AbuseIPDB blacklist is
    # dominated by confidence-100 IPs), so we collapse all malicious tiers into
    # a single "malicious" pool used for any attack severity (1/2/3); the
    # severity tier itself is carried by the flow / IAM modalities.  Benign IPs
    # form the severity-0 pool.
    benign_vecs = list(pool_by_sev[0])
    mal_vecs = list(pool_by_sev[1]) + list(pool_by_sev[2]) + list(pool_by_sev[3])
    rng.shuffle(benign_vecs)
    rng.shuffle(mal_vecs)

    def _part(v: list) -> tuple[list, list, list]:
        n = len(v)
        if n == 0:
            return [], [], []
        n_tr = max(int(round(n * 0.7)), 1)
        n_va = max(int(round(n * 0.1)), 1) if n >= 3 else 0
        return v[:n_tr], v[n_tr:n_tr + n_va], (v[n_tr + n_va:] or v[:1])

    b_tr, b_va, b_te = _part(benign_vecs)
    m_tr, m_va, m_te = _part(mal_vecs)
    splits = {
        "train": {0: b_tr, 1: m_tr, 2: m_tr, 3: m_tr},
        "val":   {0: b_va, 1: m_va, 2: m_va, 3: m_va},
        "test":  {0: b_te, 1: m_te, 2: m_te, 3: m_te},
    }
    print(f"[ti] split sizes: benign(tr/va/te)={len(b_tr)}/{len(b_va)}/{len(b_te)} "
          f"malicious(tr/va/te)={len(m_tr)}/{len(m_va)}/{len(m_te)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(splits, f)
    print(f"[ti] Saved real TI pool -> {args.out}")


if __name__ == "__main__":
    main()
