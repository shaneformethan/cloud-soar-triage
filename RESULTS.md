# Evaluation Results

Multi-modal cloud security incident triage benchmark. All numbers are **mean ± std over 5 seeds** on the held-out test set.

Two regimes: **i.i.d.** (controlled benchmark) and **temporal** (per-class chronological split of the real flows: earliest 70% of each class trains, latest 20% tests).

## Main comparison

Weighted F1 (W-F1), macro F1 (M-F1) and critical-class recall (Crit-R) for both regimes. WC = analyst workload compression, FP-sup = false-positive suppression, both measured on the i.i.d. benchmark only, at the 3% FNR ceiling.

| Configuration | W-F1 (iid) | M-F1 (iid) | Crit-R (iid) | W-F1 (temp) | M-F1 (temp) | Crit-R (temp) | WC | FP-sup |
|---|---|---|---|---|---|---|---|---|
| **Tri-modal, deep fusion** | | | | | | | | |
| Arch-D (Transformer, no PE) | 0.973±0.002 | 0.972 | 0.989 | 0.822±0.005 | 0.822 | 0.680 | 96.0 | 99.3 |
| Arch-A (BiLSTM) | 0.973±0.002 | 0.972 | 0.977 | 0.792±0.032 | 0.792 | 0.606 | 96.4 | 99.1 |
| IAM-priority fusion | 0.971±0.003 | 0.970 | 0.980 | 0.789±0.023 | 0.789 | 0.622 | 95.9 | 99.1 |
| Arch-B (BiGRU) | 0.970±0.002 | 0.969 | 0.973 | 0.796±0.012 | 0.796 | 0.612 | 96.7 | 98.3 |
| Arch-C (1D-CNN) | 0.970±0.002 | 0.970 | 0.975 | 0.751±0.022 | 0.750 | 0.476 | 98.2 | 98.9 |
| Arch-F (avg-pool) | 0.970±0.001 | 0.970 | 0.984 | 0.786±0.028 | 0.786 | 0.605 | 95.6 | 98.9 |
| Arch-E (concat) | 0.970±0.002 | 0.969 | 0.985 | 0.759±0.026 | 0.758 | 0.499 | 96.0 | 99.2 |
| Cross-modal (z_fuse) | 0.969±0.001 | 0.968 | 0.981 | 0.805±0.009 | 0.805 | 0.661 | 96.0 | 99.1 |
| **Tri-modal, tree ensembles** | | | | | | | | |
| XGBoost | 0.967±0.001 | 0.967 | 0.988 | 0.848±0.005 | 0.847 | 0.541 | 97.4 | 98.6 |
| Random Forest | 0.955±0.002 | 0.955 | 0.973 | 0.757±0.003 | 0.755 | 0.331 | 80.9 | 98.1 |
| **Reduced modality subsets** | | | | | | | | |
| IAM + Flow | 0.967±0.003 | 0.966 | 0.988 | 0.796±0.006 | 0.797 | 0.681 | 95.3 | 98.2 |
| Flow + TI | 0.781±0.003 | 0.780 | 0.694 | 0.414±0.004 | 0.414 | 0.028 | 50.8 | 99.7 |
| IAM + TI | 0.775±0.010 | 0.774 | 0.652 | 0.772±0.001 | 0.773 | 0.730 | 9.4 | 98.8 |
| Flow only | 0.764±0.002 | 0.766 | 0.669 | 0.365±0.007 | 0.365 | 0.015 | 50.5 | 98.8 |
| IAM only | 0.749±0.004 | 0.750 | 0.639 | 0.752±0.006 | 0.752 | 0.694 | 50.6 | 98.7 |
| TI only | 0.307±0.016 | 0.296 | 0.176 | 0.280±0.010 | 0.281 | 0.209 | 0.0 | 97.4 |
| **Late fusion (matched base learner)** | | | | | | | | |
| Early fusion (control) | 0.965±0.002 | 0.965 | 0.983 | 0.830±0.004 | 0.829 | 0.585 | – | – |
| Soft voting | 0.932±0.003 | 0.932 | 0.928 | 0.771±0.008 | 0.769 | 0.470 | – | – |
| Weighted voting | 0.947±0.003 | 0.947 | 0.963 | 0.784±0.012 | 0.783 | 0.497 | – | – |
| **External baseline** | | | | | | | | |
| DeepCASE (flow only) | 0.302 | 0.306 | 0.697 | 0.057 | 0.055 | 0.072 | – | – |

DeepCASE is a sequential log model run outside its intended regime — a sanity reference, not a competing fusion method. The late-fusion arms use fixed hyperparameters and were not run through the routing protocol.

## Key findings

**1. Modality coverage dominates the fusion mechanism.** Every early-fusion
configuration holding both Flow and IAM reaches 0.955–0.973 W-F1 (i.i.d.);
single-modality models reach 0.307–0.764. Among the eight deep tri-modal
variants the spread is 0.969–0.973: ANOVA F(7,32)=1.71, p=0.14, and TOST
confirms equivalence within ±0.01 (least-equivalent pair p=0.002).

Including the tree ensembles the ANOVA does reject — F(8,36)=3.19, p=0.008 with
XGBoost, F(9,40)=29.54, p<0.001 with Random Forest as well. The difference is
real but small: XGBoost still falls inside the ±0.01 margin against every deep
variant (largest TOST p=0.006), while Random Forest sits 0.018 below the best
deep variant and is the single configuration driving the rejection.

**2. Late fusion is worse, not better.** Soft voting reaches 0.932 W-F1 against
0.965 for the early-fusion control on the same learner (t=19.9, p<0.001), and
0.771 against 0.830 under drift (t=13.1, p<0.001). Weighting by validation macro
F1 recovers about half the loss. TI alone reaches only 0.321 W-F1, so averaging
its probabilities dilutes the other two; the learned weights settle near 0.18 for
TI against 0.41 each for IAM and flow.

**3. Under drift the ranking depends on the metric.** XGBoost has the best
temporal W-F1 at 0.848 vs 0.822 for Arch-D (Welch t(8.0)=7.52, p=0.0001,
Hedges g=4.30). On critical-class recall the order inverts: 0.541 vs 0.680
(t(6.0)=-16.6, p<0.0001, g=-9.49). Six of the eight deep variants recover more
critical incidents than XGBoost does. No configuration wins on both metrics.

**4. Calibration holds i.i.d. and fails under drift.** Temperature scaling cuts
ECE from 0.036 to 0.021 (cross-modal, i.i.d.). Under the temporal split it stops
working: 0.093 → 0.095. A scalar temperature fitted before the shift cannot
correct a model whose errors have changed shape.

## Real-data validation (raw CIC-IDS2018 flows, temporal split)

Classifiers trained directly on raw CSE-CIC-IDS2018 per-flow records (26 real
features, real labels), per-class temporal split:

| Model | Weighted F1 | high-sev F1 | critical (DoS/DDoS) F1 |
|---|---|---|---|
| XGBoost | 0.64 | ~0.998 | ~0.35 (recall ~0.21) |
| Random Forest | 0.62 | ~0.998 | ~0.33 |

Both the flow signal and the temporal collapse are present in the unmodified
data, so neither is an artefact of benchmark construction.
Reproduce with `python scripts/validate_real_flow.py`.

## Reproduce

```
python scripts/fetch_ti_pool.py            # real TI pool (needs .env keys)
python scripts/build_benchmark_semi.py     # i.i.d. benchmark
python scripts/build_benchmark_semi.py --temporal-split \
       --processed_dir data/processed_temporal   # temporal benchmark
python scripts/multiseed_runner.py --seeds 42 43 44 45 46   # 17 tuned configs (GPU)
python scripts/soft_voting_baseline.py     # 3 fusion-stage arms (CPU, ~2 min)
python scripts/make_results.py             # regenerate this file
```
