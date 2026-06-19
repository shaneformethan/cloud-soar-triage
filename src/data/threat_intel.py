"""
Threat Intelligence enrichment → 16-dim feature vector (Section III-B).

Matches IP addresses from a flow bucket against:
  - AbuseIPDB  (IP reputation, abuse confidence, report count)
  - AlienVault OTX  (pulse count, adversary, malware family, threat score)

16-dim vector layout:
  0   ip_reputation_score      continuous [0,1]  (AbuseIPDB confidence / 100)
  1   ip_is_abusive            binary            (confidence > 25)
  2   ip_abuse_confidence      continuous [0,1]
  3   ip_country_risk          continuous [0,1]  (high-risk country heuristic)
  4   domain_age_days_norm     continuous [0,1]  (OTX domain age, clipped at 3650d)
  5   domain_is_new            binary            (age < 30 days or unknown)
  6   hash_match_count_norm    continuous [0,1]  (OTX hash matches, clipped at 10)
  7   hash_has_match           binary
  8   campaign_alignment_score continuous [0,1]  (OTX pulse score)
  9   is_in_active_campaign    binary
  10  otx_pulse_count_norm     continuous [0,1]  (pulse count / 50, clipped)
  11  otx_adversary_known      binary
  12  otx_malware_family_known binary
  13  otx_targeted_country     binary
  14  abuseipdb_reports_norm   continuous [0,1]  (total reports / 1000, clipped)
  15  otx_threat_score         continuous [0,1]  (composite)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

TI_DIM = 16

# High-risk country codes (DESIGN DECISION: based on common threat actor
# attribution in cybersecurity literature).
_HIGH_RISK_COUNTRIES: frozenset[str] = frozenset(
    {"CN", "RU", "KP", "IR", "NG", "RO", "UA", "BR", "IN", "VN"}
)

_ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
_OTX_URL_IP = "https://otx.alienvault.com/api/v1/indicators/IPv4/{ip}/general"


class ThreatIntelEnricher:
    """
    Looks up IPs against AbuseIPDB and OTX, caches results on disk,
    and returns a 16-dim float32 vector per bucket.
    """

    def __init__(
        self,
        abuseipdb_key: str = "",
        otx_key: str = "",
        cache_dir: str | Path = "data/processed/ti_cache",
        cache_ttl_hours: int = 24,
    ):
        self.abuseipdb_key = (
            abuseipdb_key
            or os.getenv("ABUSEIPDB_API_KEY", "")
            or os.getenv("ABUSEIPDB_KEY", "")
        )
        self.otx_key = (
            otx_key
            or os.getenv("OTX_API_KEY", "")
            or os.getenv("OTX_KEY", "")
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_hours * 3600
        self._dummy = not self.abuseipdb_key and not self.otx_key

    def enrich_bucket(self, ip_list: list[str]) -> np.ndarray:
        """
        Aggregate TI signals for all IPs in a 5-min bucket.
        Returns shape (16,) float32, element-wise max across IPs.
        """
        if not ip_list:
            return np.zeros(TI_DIM, dtype=np.float32)

        vecs = [self._enrich_ip(ip) for ip in set(ip_list)]
        return np.stack(vecs, axis=0).max(axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Per-IP enrichment                                                    #
    # ------------------------------------------------------------------ #

    def _enrich_ip(self, ip: str) -> np.ndarray:
        if self._dummy or ip in ("0.0.0.0", "127.0.0.1", ""):
            return self._dummy_vec(ip)

        cache_path = self.cache_dir / f"{_hash(ip)}.json"
        cached = _load_cache(cache_path, self.cache_ttl)
        if cached is not None:
            return self._parse_cached(cached)

        abuse_data = self._query_abuseipdb(ip)
        otx_data = self._query_otx(ip)
        payload = {"abuse": abuse_data, "otx": otx_data, "ts": time.time()}
        _save_cache(cache_path, payload)
        return self._build_vec(abuse_data, otx_data)

    def _query_abuseipdb(self, ip: str) -> dict:
        try:
            resp = requests.get(
                _ABUSEIPDB_URL,
                headers={"Key": self.abuseipdb_key, "Accept": "application/json"},
                params={"ipAddress": ip, "maxAgeInDays": 90},
                timeout=5,
            )
            return resp.json().get("data", {})
        except Exception:
            return {}

    def _query_otx(self, ip: str) -> dict:
        try:
            resp = requests.get(
                _OTX_URL_IP.format(ip=ip),
                headers={"X-OTX-API-KEY": self.otx_key},
                timeout=5,
            )
            return resp.json()
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Vector builders                                                      #
    # ------------------------------------------------------------------ #

    def _build_vec(self, abuse: dict, otx: dict) -> np.ndarray:
        v = np.zeros(TI_DIM, dtype=np.float64)

        # AbuseIPDB fields
        conf = float(abuse.get("abuseConfidenceScore", 0))
        country = str(abuse.get("countryCode", ""))
        total_reports = float(abuse.get("totalReports", 0))

        v[0] = conf / 100.0
        v[1] = float(conf > 25)
        v[2] = conf / 100.0
        v[3] = float(country in _HIGH_RISK_COUNTRIES)
        v[14] = min(total_reports / 1000.0, 1.0)

        # OTX fields
        pulse_info = otx.get("pulse_info", {})
        pulses = pulse_info.get("count", 0)
        adversary = pulse_info.get("pulses", [{}])[0].get("adversary", "") if pulse_info.get("pulses") else ""
        malware = bool(pulse_info.get("pulses", [{}])[0].get("malware_families", [])) if pulse_info.get("pulses") else False
        targeted = bool(pulse_info.get("pulses", [{}])[0].get("targeted_countries", [])) if pulse_info.get("pulses") else False

        # Domain / whois age (OTX doesn't always return; default unknown = new)
        whois = otx.get("whois", "")
        domain_age = _parse_domain_age_days(whois)
        v[4] = min(domain_age / 3650.0, 1.0) if domain_age >= 0 else 0.0
        v[5] = float(domain_age < 30) if domain_age >= 0 else 1.0

        v[6] = 0.0   # hash match count (requires separate hash lookups)
        v[7] = 0.0
        v[8] = min(pulses / 50.0, 1.0)
        v[9] = float(pulses > 0)
        v[10] = min(pulses / 50.0, 1.0)
        v[11] = float(bool(adversary))
        v[12] = float(malware)
        v[13] = float(targeted)
        v[15] = min(conf / 100.0 * 0.5 + min(pulses / 50.0, 1.0) * 0.5, 1.0)

        return v.astype(np.float32)

    def _parse_cached(self, cached: dict) -> np.ndarray:
        return self._build_vec(cached.get("abuse", {}), cached.get("otx", {}))

    def _dummy_vec(self, ip: str) -> np.ndarray:
        """
        Deterministic synthetic TI vector for dummy mode.
        Uses IP hash so the same IP always gets the same vector.
        """
        seed = int(_hash(ip)[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.random(TI_DIM).astype(np.float32)
        # Binarise the binary fields
        for i in (1, 5, 7, 9, 11, 12, 13):
            v[i] = float(v[i] > 0.7)
        return v


# ------------------------------------------------------------------ #
# Cache utilities                                                      #
# ------------------------------------------------------------------ #

def _hash(ip: str) -> str:
    return hashlib.md5(ip.encode()).hexdigest()


def _load_cache(path: Path, ttl: float) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) < ttl:
            return data
    except Exception:
        pass
    return None


def _save_cache(path: Path, data: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _parse_domain_age_days(whois_str: str) -> int:
    """Return domain age in days from OTX whois string, or -1 if unknown."""
    import re
    from datetime import datetime
    match = re.search(r"Created Date:\s*(\d{4}-\d{2}-\d{2})", whois_str)
    if not match:
        return -1
    try:
        created = datetime.strptime(match.group(1), "%Y-%m-%d")
        return (datetime.utcnow() - created).days
    except Exception:
        return -1
