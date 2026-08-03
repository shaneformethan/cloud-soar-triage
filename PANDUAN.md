# Panduan Menjalankan & Evaluasi

Semua perintah dijalankan dari dalam folder `cloud-soar-triage\` menggunakan **Windows PowerShell**.

---

## 0. Benchmark Semi-Sintetik (Draft §III-E) — WAJIB BACA

Benchmark utama paper dibangun oleh `scripts/build_benchmark_semi.py`, sesuai
deskripsi draft §III-E ("Dataset Sources and Characteristics"):

- **Flow** : fitur 12-dim **nyata** dari CSE-CIC-IDS2018. Label serangan
  dipetakan langsung ke severity (`src/benchmark/severity_mapping.py`):
  benign→Informational, web/infiltration→Medium, brute-force SSH/FTP+bot→High,
  DoS/DDoS→Critical. Tiap bucket 5-menit dimodelkan sebagai agregat acak
  (bootstrap) dari flow nyata satu kelas serangan (`src/data/cic_real.py`).
- **IAM** : sesi sintetik **berkondisi-severity** (4 kelas sesi, satu per
  severity) dengan noise terkontrol (`src/data/synth_modalities.py`).
- **TI**  : vektor 16-dim sintetik berkondisi-severity.

**Desain keunggulan fusi:** tiap insiden punya satu severity ground-truth;
hanya modalitas *aktif* yang membawa sinyal severity tersebut (dengan noise),
modalitas non-aktif bersifat benign. Akibatnya model harus membaca lintas
modalitas → **fusi lintas-modal mengungguli modalitas tunggal** (klaim inti
paper). Verifikasi cepat (RandomForest) menunjukkan W-F1 multimodal ≈ 0.97 vs
flow-only ≈ 0.69, iam-only ≈ 0.75, ti-only ≈ 0.67; dan tri > dual > single.

```powershell
# Bangun benchmark (parse CIC sekali, lalu di-cache ke cic_flow_pool.pkl)
python scripts/build_benchmark_semi.py --n 9000 --seed 42

# Paksa parse ulang CSV CIC mentah
python scripts/build_benchmark_semi.py --n 9000 --seed 42 --rebuild-pool
```

> Benchmark lama (dummy, label acak/non-learnable) sudah tidak dipakai;
> generator-nya (`src/data/dummy.py`) hanya disimpan sebagai jejak legacy.

Setelah benchmark dibangun, lanjut ke training/evaluasi seperti biasa
(Bagian 4–5). Untuk evaluasi cepat hanya dari checkpoint yang sudah ada
(tanpa melatih ulang 17 konfigurasi): `python scripts/eval_existing.py`.

---

## 1. Setup Awal (Satu Kali)

### Buat dan aktifkan virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

> Jika muncul error "execution of scripts is disabled", jalankan dulu:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Lalu ulangi `Activate.ps1`.

### Install semua dependency

```powershell
pip install -r requirements.txt
```

---

## 2. Cek Instalasi (Opsional tapi Disarankan)

```powershell
pytest tests/ -v
```

Tiga test yang dijalankan:
- `test_dimensions` — verifikasi dimensi IAM (64×5), flow (12,), TI (16,)
- `test_splits` — verifikasi split 70/10/20 dan skenario 30/40/30
- `test_anti_circularity` — verifikasi label severity tidak bocor ke IAM generator

Semua harus **PASSED**.

---

## 3. Smoke Test (< 1 Menit)

Cek bahwa semua komponen berjalan tanpa error sebelum training penuh.

```powershell
python main.py --smoke
```

Output yang diharapkan:
```
=== SMOKE TEST (2 epochs) ===
  Logits shape: torch.Size([32, 4]) ✓
  Params: xxx,xxx
  Smoke test PASSED.
```

---

## 4. Training

### Training proposed model (lintas modal, z_fuse fusion)

```powershell
python train.py
```

Checkpoint disimpan otomatis ke `checkpoints/proposed_best.pt` saat val F1 membaik.
Training berhenti otomatis jika tidak ada peningkatan selama 10 epoch.

### Training ablation variants (untuk hasil lengkap di paper)

```powershell
python train.py --variant arch_a   # BiLSTM IAM encoder
python train.py --variant arch_b   # BiGRU IAM encoder
python train.py --variant arch_c   # 1D-CNN IAM encoder
python train.py --variant arch_d   # Transformer tanpa positional encoding
python train.py --variant arch_e   # Concatenation + MLP fusion
python train.py --variant arch_f   # Average pooling fusion
```

Setiap checkpoint disimpan ke `checkpoints/<variant>_best.pt`.

### Modality ablation (6 subset + IAM-prioritized fusion, §III-F/IV)

```powershell
python train.py --variant iam_only
python train.py --variant flow_only
python train.py --variant ti_only
python train.py --variant iam_flow
python train.py --variant iam_ti
python train.py --variant flow_ti
python train.py --variant iam_priority   # fusion z_fuse diganti query IAM tetap
```

### Opsi training lanjutan

```powershell
# Ganti learning rate dan batch size
python train.py --lr 5e-4 --batch_size 32

# Gunakan GPU jika tersedia
python train.py --device cuda

# Ganti seed (Section IV: 5 seed per konfigurasi)
python train.py --variant arch_a --seed 43
```

Untuk seed selain 42, nama checkpoint otomatis jadi `<variant>_seed<seed>_best.pt`.

---

## 5. Evaluasi

### Full evaluation (proposed + semua baseline + tabel perbandingan)

```powershell
python main.py
```

Jika checkpoint belum ada, `main.py` akan training otomatis terlebih dahulu.

### Evaluasi satu variant saja

```powershell
python main.py --variant arch_a
```

### Output evaluasi mencakup (sesuai §III-F draft):

| Metrik | Keterangan |
|--------|-----------|
| Analyst workload compression | % insiden tanpa intervensi analis, dilaporkan pada FNR ≤ 3% |
| FP suppression rate | % alert informational yang benar diklasifikasikan (target ≥ 70%) |
| Mean time-to-triage | Latensi pipeline per sampel dalam milidetik |
| ECE sebelum/sesudah | Expected Calibration Error sebelum dan sesudah temperature scaling |
| Weighted F1 | Per model dan per skenario (single/dual/tri) |
| Confusion matrix | 4×4, kelas informational/medium/high/critical |

Di akhir, tercetak **tabel ringkasan** semua model:
```
══════════════════════════════════════════════════════════════════════
  SUMMARY: Proposed vs Baselines (test set)
  Model                W-F1      WC     FPR     FNR    MTTD ms
  ──────────────────────────────────────────────────────────────────
  proposed           0.xxxx   xx.x%   xx.x%   0.xxx      x.xxx
  arch_a .. arch_f   0.xxxx   xx.x%   xx.x%   0.xxx      x.xxx
  iam_only .. flow_ti 0.xxxx  xx.x%   xx.x%   0.xxx      x.xxx
  iam_priority       0.xxxx   xx.x%   xx.x%   0.xxx      x.xxx
  trad_rf            0.xxxx   xx.x%   xx.x%   0.xxx        nan
  trad_xgb           0.xxxx   xx.x%   xx.x%   0.xxx        nan
  deepcase           0.xxxx   xx.x%   xx.x%   0.xxx        nan
══════════════════════════════════════════════════════════════════════
```

`python main.py` sekarang mengevaluasi **17 konfigurasi**: proposed, 6 arsitektur
ablation (arch_a–arch_f), 6 modality-subset ablation, iam_priority, trad_rf,
trad_xgb, dan deepcase — sesuai klaim "17 tuned configurations" di draft §III (ditambah 3 lengan fusion-stage di scripts/soft_voting_baseline.py).
Checkpoint yang belum ada akan otomatis di-training.

---

## 5b. Random Search (60 trial, Table I)

Mencari hyperparameter arsitektur terbaik untuk model proposed (60-trial random
search, Bergstra & Bengio):

```powershell
python scripts/random_search.py --trials 60 --seed 42
```

Setiap trial training cepat (15 epoch, patience 3) lalu diranking berdasarkan
val weighted F1. Hasil terbaik disimpan ke `config/best_hp.json`. Catatan:
IAM sequence length (32/64/128) TIDAK termasuk pencarian ini karena memerlukan
rebuild benchmark — dimensi ini ditangani terpisah sebagai sensitivity
analysis §IV-C.

---

## 5c. Multi-Seed Runner (17 konfigurasi × 5 seed = 85 run)

```powershell
python scripts/multiseed_runner.py --seeds 42 43 44 45 46
```

Menjalankan training + evaluasi untuk semua 17 konfigurasi pada 5 seed,
lalu mencetak tabel ringkasan **mean ± std** dan menyimpan hasil lengkap ke
`results/multiseed_results.json`. Untuk uji cepat, kurangi epoch dan/atau seed:

```powershell
python scripts/multiseed_runner.py --seeds 42 --max_epochs 5 --variants proposed arch_a
```

---

## 6. Menggunakan Data Real (Untuk Paper Final)

Secara default, benchmark sudah tersedia di `data/processed/` dengan 2.000 sampel sintetis (1.400 train / 200 val / 400 test) yang cukup untuk eksplorasi awal.

Untuk reproduksi penuh dengan data asli:

### Download CIC-IDS2018

```powershell
# Lihat instruksi lengkap
python scripts/download_data.py --cic-instructions

# Opsi paling mudah via Kaggle
pip install kaggle
kaggle datasets download solarmainframe/ids-intrusion-csv
Expand-Archive ids-intrusion-csv.zip -DestinationPath data\raw\cic_ids2018\
```

### Download CloudTrail corpus (flaws.cloud)

```powershell
python scripts/download_data.py --cloudtrail
```

### Tambah API key threat intelligence

Edit `config/default.yaml`:
```yaml
threat_intel:
  abuseipdb_key: "ISI_KEY_ABUSEIPDB_DISINI"
  otx_key: "ISI_KEY_ALIENVAULT_OTX_DISINI"
```

> API key gratis: daftar di https://www.abuseipdb.com dan https://otx.alienvault.com

### Rebuild benchmark dengan data real

```powershell
python scripts/build_benchmark.py --config config/default.yaml
```

### Lanjutkan training dan evaluasi seperti biasa

```powershell
python train.py
python main.py
```

---

## 7. Memindahkan Project

Copy seluruh folder `cloud-soar-triage\` ke lokasi baru, lalu ulangi langkah setup:

```powershell
cd <lokasi-baru>\cloud-soar-triage
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --smoke
```

File `data/processed/*.pkl` (benchmark) ikut terbawa sehingga tidak perlu build ulang.

---

## 8. Ringkasan Urutan Perintah

```powershell
# Setup (satu kali)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verifikasi
pytest tests/ -v
python main.py --smoke

# Training
python train.py
python train.py --variant arch_a
python train.py --variant arch_b
python train.py --variant arch_c
python train.py --variant arch_d
python train.py --variant arch_e
python train.py --variant arch_f

# Evaluasi
python main.py
```
