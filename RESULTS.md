# Evaluation Results

Multi-modal cloud security incident triage benchmark. All numbers are **mean ± std over 5 seeds** on the held-out test set, from `scripts/multiseed_runner.py` (17 configurations × 5 seeds = 85 runs).

Two evaluation regimes: **i.i.d.** (controlled benchmark) and **temporal** (per-class chronological split of the real flows: earliest 70% of each class trains, latest 20% tests).

## Main comparison

| Configuration | W-F1 (i.i.d.) | W-F1 (temporal) | Workload comp. % | FP-supp. % | FNR |
|---|---|---|---|---|---|
| **Tri-modal (Flow + IAM + TI)** | | | | | |
| Arch-A (BiLSTM) | 0.973±0.002 | 0.792±0.036 | 96.4 | 99.1 | 0.002±0.001 |
| Arch-D (Transformer, no PE) | 0.973±0.002 | 0.822±0.006 | 96.0 | 99.3 | 0.002±0.001 |
| IAM-priority fusion | 0.971±0.004 | 0.789±0.025 | 95.9 | 99.1 | 0.002±0.001 |
| Arch-C (1D-CNN) | 0.970±0.002 | 0.751±0.025 | 98.2 | 98.9 | 0.004±0.003 |
| Arch-F (avg-pool) | 0.970±0.001 | 0.786±0.031 | 95.6 | 98.9 | 0.003±0.002 |
| Arch-E (concat) | 0.970±0.002 | 0.759±0.029 | 96.0 | 99.2 | 0.003±0.002 |
| Arch-B (BiGRU) | 0.970±0.002 | 0.796±0.013 | 96.7 | 98.3 | 0.003±0.002 |
| Cross-modal (z_fuse) | 0.969±0.001 | 0.805±0.011 | 96.0 | 99.1 | 0.003±0.003 |
| XGBoost | 0.967±0.001 | 0.848±0.005 | 97.4 | 98.6 | 0.007±0.000 |
| Random Forest | 0.955±0.002 | 0.757±0.003 | 80.9 | 98.1 | 0.002±0.000 |
| **Reduced modality subsets** | | | | | |
| IAM + Flow | 0.967±0.003 | 0.796±0.006 | 95.3 | 98.2 | 0.002±0.001 |
| Flow + TI | 0.781±0.003 | 0.414±0.004 | 50.8 | 99.7 | 0.008±0.012 |
| IAM + TI | 0.775±0.012 | 0.772±0.002 | 9.4 | 98.8 | 0.036±0.023 |
| Flow only | 0.764±0.002 | 0.365±0.008 | 50.5 | 98.8 | 0.000±0.000 |
| IAM only | 0.749±0.004 | 0.752±0.007 | 50.6 | 98.7 | 0.000±0.000 |
| TI only | 0.307±0.018 | 0.280±0.012 | 0.0 | 97.4 | 0.000±0.000 |
| **External baseline** | | | | | |
| DeepCASE (flow) | 0.302±0.006 | 0.057±0.006 | 100.0 | 0.0 | 0.000±0.000 |

*Workload compression, FP-suppression, and FNR are on the i.i.d. (deployment) benchmark, at the 3% FNR ceiling.*

## Key findings

1. **Modality coverage dominates architecture.** Any configuration with both Flow and IAM reaches ~0.97 W-F1 (i.i.d.); single-modality models are far weaker (0.31-0.76). Among tri-modal models the spread is only 0.967-0.973 — cross-modal attention, pooling, concatenation, recurrent/convolutional encoders, and gradient-boosted trees are statistically tied.

2. **Temporal shift is the real challenge.** Under the temporal split every model drops to ~0.75-0.85. The collapse is in the real flow stream (Flow-only 0.76→0.37; DeepCASE 0.30→0.06) because attack tooling evolves over the two-week capture; the synthetic IAM stream does not shift (IAM-only ~0.75 both regimes). XGBoost is the strongest temporal model.

3. **Calibration enables SOAR auto-dispatch.** Temperature scaling brings the multimodal model's ECE to ~0.02 (i.i.d.); the multimodal models automate ~96% of incidents below a 1% false-negative rate, suppressing >98% of non-actionable alerts.

## Real-data validation (raw CIC-IDS2018 flows, temporal split)

Classifiers trained directly on raw CSE-CIC-IDS2018 per-flow records (26 real features, real labels), per-class temporal split:

| Model | Weighted F1 | high-sev F1 | critical (DoS/DDoS) F1 |
|---|---|---|---|
| XGBoost | 0.64 | ~0.998 | ~0.35 (recall ~0.21) |
| Random Forest | 0.62 | ~0.998 | ~0.33 |

This confirms the flow signal is real (not a benchmark artifact) and that temporal generalization is hard for evolving DoS/DDoS tooling. Reproduce with `python scripts/validate_real_flow.py`.

## Reproduce

```
python scripts/fetch_ti_pool.py            # real TI pool (needs .env keys)
python scripts/build_benchmark_semi.py     # i.i.d. benchmark
python scripts/build_benchmark_semi.py --temporal-split \
       --processed_dir data/processed_temporal   # temporal benchmark
python scripts/multiseed_runner.py --seeds 42 43 44 45 46   # 17x5 (GPU)
python scripts/make_results.py             # regenerate this file
```
