# Menjalankan 17 konfigurasi × 5 seed di Google Colab (GPU)

Panduan untuk menjalankan tabel §IV penuh (`scripts/multiseed_runner.py`) di
Colab dengan GPU. **Resumable**: hasil disimpan ke Google Drive tiap selesai
satu run, jadi kalau Colab terputus tinggal jalankan ulang sel **Run**.

Estimasi waktu di GPU Colab: **~2–4 jam** untuk 85 run (vs ~40 jam di CPU).

---

## Langkah 1 — Siapkan folder project (sekali saja)

Yang perlu diunggah ke Drive hanyalah kode + benchmark yang sudah jadi
(tidak perlu CSV CIC mentah / API key — datanya sudah di-build):

```
cloud-soar-triage/
├── src/  scripts/  config/
├── train.py  main.py  requirements.txt
└── data/processed/benchmark_train.pkl  benchmark_val.pkl  benchmark_test.pkl
```

**Cara unggah (pilih salah satu):**

- **A. Zip lalu unggah** (paling ringkas): zip folder `cloud-soar-triage`
  (boleh sertakan `data/processed/*.pkl`; abaikan `.venv/`, `data/raw/`,
  `checkpoints/` lama). Unggah `cloud-soar-triage.zip` ke **My Drive** (root).
  Notebook akan otomatis meng-unzip.

  PowerShell untuk membuat zip ramping:
  ```powershell
  cd C:\Users\forme\Documents\BCA\Cawu5\research-methodology\Code
  Compress-Archive -Path cloud-soar-triage\src,cloud-soar-triage\scripts,`
    cloud-soar-triage\config,cloud-soar-triage\train.py,cloud-soar-triage\main.py,`
    cloud-soar-triage\requirements.txt,cloud-soar-triage\data\processed `
    -DestinationPath cloud-soar-triage.zip
  ```
  > Catatan: struktur di dalam zip harus menaruh `train.py` di root project.
  > Jika hasil zip menempatkan file langsung di root, cukup pastikan saat
  > di-extract ada folder dengan `train.py` di dalamnya — sel notebook sudah
  > menangani folder bertingkat.

- **B. Salin folder langsung** ke `My Drive/cloud-soar-triage` lewat
  drive.google.com (drag-and-drop). Pastikan `data/processed/*.pkl` ikut.

---

## Langkah 2 — Buka notebook di Colab

1. Buka <https://colab.research.google.com> → **File → Upload notebook** →
   pilih `notebooks/colab_multiseed.ipynb` (dari repo ini), **atau**
   taruh notebook itu di Drive dan buka dari sana.
2. **Runtime → Change runtime type → Hardware accelerator = GPU → Save.**

---

## Langkah 3 — Jalankan sel berurutan

1. **Mount Drive** — izinkan akses.
2. **Locate project** — otomatis menemukan / meng-unzip folder, dan memverifikasi
   `benchmark_train.pkl` ada.
3. **Install deps** — memverifikasi GPU aktif.
4. **RUN** — menjalankan 17×5. Biarkan berjalan. Hasil tersimpan ke
   `My Drive/cloud-soar-triage/results/multiseed_results.json`.
5. **Inspect** — tampilkan progres & mean ± std kapan saja.

### Kalau Colab terputus (disconnect / timeout)
Cukup buka kembali, jalankan sel 1–3 lagi, lalu **jalankan ulang sel RUN**.
Skrip membaca JSON di Drive dan **melewati run yang sudah selesai** — lanjut
dari yang belum. Tidak ada progres yang hilang.

---

## Langkah 4 — Ambil hasilnya

Setelah selesai, unduh `results/multiseed_results.json` dari Drive. Isinya:
- `summary`: mean/std/n per konfigurasi untuk W-F1, workload compression,
  FP-suppression, FNR, MTTD.
- `raw`: laporan lengkap per (konfigurasi, seed).

Pakai angka `summary` untuk mengisi Tabel §IV (mean ± std).

---

## Tips

- **Colab Free** bisa terputus ~setelah beberapa jam / saat idle. Karena
  resumable, cukup jalankan ulang. **Colab Pro** memberi sesi lebih panjang.
- **Kaggle** alternatif (GPU gratis ~30 jam/minggu): unggah folder sebagai
  Dataset, lalu jalankan perintah `multiseed_runner.py` yang sama dengan
  `--device cuda` dan `--out` ke `/kaggle/working/`.
- Untuk uji cepat dulu: tambahkan `--max_epochs 8 --seeds 42` agar selesai
  dalam menit, memastikan semuanya jalan, lalu jalankan penuh.
- Checkpoint ditulis ke `/content/ms_ckpts` (lokal, cepat). Itu hilang saat
  sesi berakhir — tidak masalah: run yang belum tercatat di JSON akan dilatih
  ulang dari awal saat resume.
