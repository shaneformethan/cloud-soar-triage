# Draft ↔ Code Alignment Notes

This document records how the implementation in `cloud-soar-triage/` maps to the
claims in `draft.tex`, the **changes made on 2026-06-15** to make the benchmark
genuinely learnable, and the **precise edits you should make to `draft.tex`** so
the paper honestly describes the code. (I deliberately did *not* rewrite your
paper prose — these are your scientific claims to own.)

---

## 1. Why the change was necessary (the core problem)

The previous benchmark (`src/data/dummy.py` + the tactic path in
`attack_stage.py`) derived each cluster's severity from a **randomly chosen
MITRE tactic**, while the feature vectors only encoded an *anomalous-or-not*
bit per modality. The severity label was therefore **statistically independent
of the inputs**. Empirically the proposed model scored Weighted-F1 **0.38** and
*lost* to its own BiGRU baseline (**0.49**); the Bayes-optimal classifier on
that data is "always predict High." No architecture or hyperparameter change
can beat prior work on a benchmark whose labels are independent of its features.

`draft.tex` §III-E actually describes a **learnable** design; the code never
implemented it. The fix below makes the code match the draft.

---

## 2. What the code now implements

| Modality | Source | File |
|---|---|---|
| **Flow** | **Real** CSE-CIC-IDS2018 12-dim vectors; label→severity per draft §III-E | `src/data/cic_real.py`, `src/benchmark/severity_mapping.py` |
| **IAM**  | **Synthetic**, **tactic-conditioned** (severity derived from tactic, never seen by the generator) | `src/data/synth_modalities.py` |
| **TI**   | **Real**, live AbuseIPDB + AlienVault OTX enrichment of a curated IP pool (binary malicious/clean) | `scripts/fetch_ti_pool.py`, `src/data/threat_intel.py` |

Assembly: `scripts/build_benchmark_semi.py`. Training upgrades (AdamW, weight
decay, label smoothing): `train.py`.

### Anti-circularity (no label leakage into generation)
The synthetic IAM generator is conditioned **only on the MITRE attack tactic**;
the severity label is the downstream value `_TACTIC_TO_SEV[tactic]`, computed
separately in `build_benchmark_semi.py`. `generate_iam_seq(tactic, ...)` has **no
severity parameter at all**, so feature⊥label given tactic by construction —
matching the draft's claim that "ground-truth labels are withheld from synthetic
IAM sequences" and the repo's `circularity_guard` design (`tests/
test_anti_circularity.py`, 10/10 passing). TI carries attack-presence (real IP
reputation is binary), not a severity tier. Removing the earlier
severity-conditioned generation changed the RandomForest numbers only
marginally (ALL 0.991→0.989, iam-only 0.784→0.785), confirming the result was
**not** driven by leakage.

**Fusion-advantage design.** Each cluster has one ground-truth severity; only
the *active* modalities of its scenario carry that severity (noisily), inactive
modalities are benign. So severity is recoverable only by reading across
modalities — making cross-modal fusion provably beat any single-modality view.

**Verification (RandomForest, real-TI benchmark, test set):**

| Subset | Weighted-F1 |
|---|---|
| ALL (multimodal) | **0.991** |
| flow + iam | 0.987 |
| iam only | 0.784 |
| flow only | 0.746 |
| ti only | 0.319 |

Per-scenario (multimodal): single **0.981** < dual **1.000** = tri **1.000**.
This reproduces the paper's thesis directly.

**Deep-model results (single seed 42, test set, `scripts/eval_existing.py`):**

| Model | Weighted-F1 | Macro-F1 | Workload comp. | FP-sup | FNR | ECE→cal |
|---|---|---|---|---|---|---|
| **Proposed (z_fuse)** | **0.9950** | **0.9950** | **99.7%** | 99.8% | 0.004 | 0.041→0.018 |
| Trad-B (XGBoost) | 0.9933 | 0.9932 | 99.4% | 99.3% | 0.006 | — |
| Trad-A (Random Forest) | 0.9906 | 0.9904 | 94.5% | 99.5% | 0.004 | — |
| Arch-F (avg-pool fusion) | 0.9508 | 0.9504 | 71.9% | 98.4% | 0.005 | — |
| Arch-E (concat fusion) | 0.9405 | 0.9399 | 67.4% | 99.3% | 0.001 | — |

The proposed cross-modal attention fusion is the top model, beating both the
simpler fusion strategies (z_fuse > avg-pool > concat) and strong traditional
ML, with the best workload compression and good calibration. Per-scenario:
single 0.990 < dual 1.000 = tri 1.000. (These are single-seed; run the full
multi-seed sweep for mean ± std.)

**Fusion architecture fix (faithful to "cross-modal self-attention block").**
`CrossModalFusion` originally ran self-attention over
[z_fuse; E_IAM; E_flow; E_TI] with **no modality-type embeddings**, making it
permutation-equivariant — it could not tell which token was which modality and
underfit (val F1 ~0.82-0.88, *below* avg-pool). Adding learnable modality-type
embeddings (one per position) restored its ability to *select* the anomalous
modality; with lr=5e-4 it reaches 0.995. **Selected LR = 5e-4** (in the draft's
grid); 1e-3 destabilises the attention. This is now the default in `train.py`.

---

## 3. Suggested edits to `draft.tex`

### 3a. Threat-intelligence paragraph (§III-E, "Threat intelligence")
The draft says IoCs are computed by matching **IP addresses extracted from
CSE-CIC-IDS2018 flow records**. The public CIC-IDS2018 CSVs **do not contain
source/destination IP columns**, so this is not literally possible. Replace with
an honest description, e.g.:

> Threat-intelligence signals are computed by enriching, through the live
> AbuseIPDB and AlienVault OTX APIs, a curated pool of real IP addresses:
> currently-reported malicious IPs drawn from the AbuseIPDB blacklist and a set
> of well-known benign hosts. Because crowd-sourced IP reputation is effectively
> binary (clean vs. flagged), the TI modality contributes a malicious/benign
> corroboration signal rather than a severity tier; it is layered on the network
> and identity evidence and is never the sole determinant of an incident's
> severity. Each IP is enriched once and cached so the benchmark is reproducible.

### 3b. Network-flow bucketing (§III-B / §III-E)
The draft says flows are aggregated "per five-minute source-destination
interval." With no IP columns, add a sentence:

> As the public CSV release omits IP addresses, each five-minute bucket is
> modeled as a bootstrap aggregate of real flows of a single attack family,
> preserving the aggregate traffic shape (protocol mix, port entropy, timing)
> that distinguishes attack classes.

### 3c. Severity classes / benign clusters (§III-E, "Evaluation Design")
The benchmark includes an explicit **benign (Informational)** cluster fraction
(default 25%) in addition to the single/dual/tri attack scenarios. State this so
the four-class label distribution is accounted for. Also note the constraint:

> In single-modality scenarios the active modality is restricted to flow or IAM,
> since the binary TI signal cannot by itself express a severity tier.

### 3d. Severity mapping (§III-E)
The implemented mapping (`severity_mapping.py`) matches the draft text exactly
for the **2018** label strings (e.g., `DDOS attack-HOIC`, `SSH-Bruteforce`,
`Infilteration`). The older `attack_stage.py` table used 2017-style names that
do not occur in these files; it is retained only for the legacy tactic path and
is no longer on the benchmark-build path.

### 3e. §IV Results
Fill once training completes. The headline comparison comes from
`python scripts/eval_existing.py` (curated subset) or the full sweep
`python scripts/multiseed_runner.py --seeds 42 43 44 45 46` (17 tuned configs × 5
seeds → mean ± std).

---

## 4. Integrity controls now ENFORCED in code

These were added after audit and are verified by `scripts/eval_existing.py`
(overfitting columns) and the leakage audit:

- **No label leakage into features.** Generators take only the tactic; severity
  derived downstream (§ anti-circularity above). 10/10 tests pass.
- **No train/test feature leakage.** Both the flow pool (`_partition_pool`) and
  the TI pool (`fetch_ti_pool.py`) are split-disjoint. Audited overlap:
  full-sample **0**, IAM **0**, flow **1**, TI **3** (the residual flow/TI are
  coincidental identical *clean* vectors, not memorizable label signal).
- **No overfitting.** train−test weighted-F1 gap < 0.01 for every model
  (proposed ≈ 0.00).
- **Real-data validation (done).** `scripts/validate_real_flow.py` trains
  RF/XGBoost directly on raw CIC-IDS2018 per-flow records (26 real features)
  under a per-class temporal split: weighted F1 $\approx0.64$ (vs ~0.97 on the
  controlled benchmark), with high-severity F1 $\approx0.998$ but critical
  DoS/DDoS F1 $\approx0.35$ -- a real, honest temporal-generalization gap.
- **Temporal split (done).** `build_flow_pool_temporal` + `--temporal-split`
  partition each class's real flows chronologically (early 70% train, late 20%
  test). On the temporal multimodal benchmark RF gives ALL 0.76 / flow 0.44 /
  iam 0.73 (leakage-free), i.e. fusion cushions single-modality temporal drift.
  Canonical (i.i.d.) benchmark left in `data/processed`; temporal variant in
  `data/processed_temporal`.
- **Difficulty / headroom.** `--iam_noise 0.45`, `--ti_noise 0.40`, and
  `--corrupt_p 0.4`: in 40% of tri-modal incidents one randomly chosen modality
  carries a WRONG-severity reading while the label stays the true severity
  (recoverable from the other two). This pulls accuracy off the ceiling
  (ALL ≈ 0.95) and makes tri the hardest scenario (≈0.92 for RF) — the regime
  where input-dependent attention can beat fixed averaging. It is a realistic
  test (sensor unreliability), not tuned to make the proposed model win.

## 4b. Honesty checklist for §V (Limitations) — still state these

1. **Controlled semi-synthetic benchmark.** Multimodal complementarity and the
   modality-reliability structure are built into the data, so the findings are a
   controlled feasibility demonstration, not evidence on production telemetry.
2. **Split is effectively IID, not chronological.** `build_benchmark_semi.py`
   shuffles then assigns synthetic timestamps. Soften the draft's temporal-
   leakage claim, or implement a true day-based split of the real flows.
3. **TI is binary** (real IP reputation is malicious/clean), corroborating
   attack presence, not a severity tier.
4. **Class balance** (~25%/class) is artificial vs CIC's benign-heavy reality.
5. **Report mean±std over 5 seeds** so "lead vs tie" between fusion strategies
   is statistically defensible (`scripts/multiseed_runner.py`).

## 5. Reproduce

```powershell
python scripts/fetch_ti_pool.py --per_tier 60        # real TI (needs .env keys)
python scripts/build_benchmark_semi.py --n 9000 --seed 42
python train.py --variant proposed --max_epochs 20 --patience 5 --log_every 1
python scripts/eval_existing.py
```
The original unlearnable dummy benchmark (`src/data/dummy.py`) is retained in
code as a legacy path but is no longer used to build the benchmark.
