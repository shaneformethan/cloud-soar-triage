# Which Design Choices Matter in Multi-Modal Cloud Incident Triage?

Code and benchmark for the paper *"Which Design Choices Matter in Multi-Modal
Cloud Incident Triage? A Reproducible Benchmark and Controlled Comparison"*
(under review).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is an **empirical/comparative study**, not a single proposed model. It
provides (1) a semi-synthetic multimodal benchmark, (2) a simulated
confidence-routing protocol, and (3) a controlled comparison of 20
configurations under two regimes.

Two findings carry the paper:

- **Modality coverage dominates the fusion mechanism.** Identity and network
  evidence together are necessary and close to sufficient. The three
  early-fusion mechanisms are equivalent within ±0.01 weighted F1 (ANOVA
  p=0.14; TOST p=0.002), and moving fusion *after* the classifier (soft voting)
  costs 0.034 rather than helping.
- **Under temporal drift the ranking depends on the metric.** Gradient boosting
  has the best weighted F1 (0.848 vs 0.822, p=0.0001) but loses 0.139 of
  critical-class recall to the best deep variant (p<0.0001). No configuration
  wins on both. Selecting a model on weighted F1 alone ships the one that misses
  roughly one critical incident in two.

See **[RESULTS.md](RESULTS.md)** for the full result tables and
**[DRAFT_ALIGNMENT.md](DRAFT_ALIGNMENT.md)** for how the code maps to the paper.

---

## The benchmark (semi-synthetic)

Each sample is a 5-minute *event cluster* aligning three modalities:

| Modality | Source | Shape |
|----------|--------|-------|
| Network flow | **Real** CSE-CIC-IDS2018, label→severity, bootstrap-aggregated 5-min buckets | (12,) float |
| IAM events | **Synthetic**, conditioned on the MITRE ATT&CK tactic (never on the severity label) | (64, 5) int |
| Threat intel | **Real** AbuseIPDB + AlienVault OTX reputation of curated IPs, queried once and cached (binary) | (16,) float |

Severity classes: 0 informational, 1 medium, 2 high, 3 critical.
Integrity controls (see DRAFT_ALIGNMENT.md): no label leakage into generation
(anti-circularity), split-disjoint flow/TI pools (no train/test feature leakage),
and an optional per-class **temporal** split of the real flows.

---

## Quick start

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Secrets — copy the template and add your own API keys
cp .env.example .env             # then edit .env
```

`.env` is git-ignored, so your keys are never committed. Free AbuseIPDB and OTX
tiers are sufficient. To run without any API keys or the dataset, set
`DATA_MODE=dummy` in `.env` and use the smoke test below.

## Build the benchmark

```bash
python scripts/fetch_ti_pool.py                 # real TI pool (needs API keys)
python scripts/build_benchmark_semi.py --n 9000 --seed 42        # i.i.d. -> data/processed/
python scripts/build_benchmark_semi.py --temporal-split \
       --processed_dir data/processed_temporal                   # temporal split
```

## Run the comparison (17 tuned configs × 5 seeds)

```bash
python scripts/multiseed_runner.py --seeds 42 43 44 45 46 --device cuda
# GPU ~2–4 h, CPU ~40 h, resumable. Colab: notebooks/colab_multiseed.ipynb (see COLAB_GUIDE.md)

python scripts/make_results.py     # regenerate RESULTS.md from the JSONs
python scripts/make_table.py       # LaTeX results table for the paper

# External late-fusion baseline (soft voting), CPU-only, ~2 min
python scripts/soft_voting_baseline.py
```

## Real-data validation (raw CIC flows, temporal split)

```bash
python scripts/validate_real_flow.py
```

## Tests

```bash
pytest tests/ -v        # dimensions, splits, anti-circularity
python main.py --smoke  # 2-epoch end-to-end smoke test
```

---

## Project structure

```
src/
  data/        cic_real.py (real flow pool), synth_modalities.py (tactic-cond. IAM/TI),
               threat_intel.py, dataset.py
  benchmark/   severity_mapping.py (CIC-2018 label->severity), aligner.py, splitter.py,
               scenario_builder.py, attack_stage.py
  models/      encoders.py, fusion.py (CrossModalFusion + ablations), model.py
  baselines/   arch_variants.py (BiLSTM/BiGRU/CNN/fusion), traditional.py (RF/XGB), deepcase.py
  calibration/ temperature.py    soar/ integration.py    evaluation/ metrics.py
  utils/       circularity_guard.py
scripts/       fetch_ti_pool.py  build_benchmark_semi.py  multiseed_runner.py
               make_results.py  make_table.py  validate_real_flow.py
results/       multiseed_iid.json  multiseed_temporal.json
train.py  main.py
```

## Configurations compared (20)

**17 tuned configurations** (`scripts/multiseed_runner.py`): cross-modal
attention (z_fuse), IAM-priority fusion, Arch-A/B/C/D (BiLSTM, BiGRU, 1D-CNN,
Transformer w/o positional encoding), Arch-E/F (concat, average pooling), 6
modality subsets, Random Forest, XGBoost, and DeepCASE (external, out-of-regime
sanity reference).

**3 fusion-stage arms** (`scripts/soft_voting_baseline.py`): soft voting,
weighted voting, and an early-fusion control. These use fixed hyperparameters
rather than the search above. The base learner is held identical across all
three so that the only quantity varying between them is where the modalities
meet; the control is untuned on exactly the same terms, which keeps the
early-versus-late comparison internally fair.

## Selected hyperparameters

AdamW (lr 5e-4, weight decay 1e-4), label smoothing 0.05, batch 64, max 100
epochs with early stopping (patience 10), ReduceLROnPlateau, Xavier init, d=256,
8 heads. Confidence calibrated by temperature scaling.

---

## Citation

```bibtex
@misc{formethan2026cloudtriage,
  title  = {Which Design Choices Matter in Multi-Modal Cloud Incident Triage?
            A Reproducible Benchmark and Controlled Comparison},
  author = {Formethan, Shane M. R. and Wilson and Chowanda, Andry and
            Junior, Franz Adeta},
  year   = {2026},
  note   = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).
Network-flow features derive from the public CSE-CIC-IDS2018 dataset; threat
intelligence from the public AbuseIPDB and AlienVault OTX services. Please
respect those providers' terms when redistributing derived data.
