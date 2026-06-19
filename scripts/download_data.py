"""
Data downloader for flaws.cloud CloudTrail corpus.

Usage:
    python scripts/download_data.py --cloudtrail
    python scripts/download_data.py --all

CIC-IDS2018 must be downloaded manually from Kaggle or the CIC website
(requires terms acceptance). See README for instructions.
"""

import argparse
import os
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# flaws.cloud CloudTrail corpus download URLs.
# Source: https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/
CLOUDTRAIL_FILES = [
    (
        "https://summitroute.com/downloads/flaws_cloudtrail_logs.tar",
        "data/raw/cloudtrail/flaws_cloudtrail_logs.tar",
    ),
]


def download_file(url: str, dest: str, desc: str = "") -> None:
    if not HAS_DEPS:
        raise RuntimeError("Install requests and tqdm: pip install requests tqdm")
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return
    print(f"  Downloading {desc or url} → {dest_path}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved: {dest_path}")


def download_cloudtrail() -> None:
    print("=== Downloading flaws.cloud CloudTrail corpus ===")
    for url, dest in CLOUDTRAIL_FILES:
        download_file(url, dest, desc="CloudTrail logs")
    print("Done. Extract with: tar -xf data/raw/cloudtrail/flaws_cloudtrail_logs.tar"
          " -C data/raw/cloudtrail/")


def print_cic_instructions() -> None:
    print("""
=== CIC-IDS2018 download instructions ===

Option A (Kaggle — easiest):
  1. Install kaggle CLI: pip install kaggle
  2. Configure API key: https://www.kaggle.com/docs/api
  3. Run:
       kaggle datasets download solarmainframe/ids-intrusion-csv
       unzip ids-intrusion-csv.zip -d data/raw/cic_ids2018/

Option B (Official CIC):
  1. Visit: https://www.unb.ca/cic/datasets/ids-2018.html
  2. Fill out the access request form
  3. Download via the provided AWS S3 link:
       aws s3 sync s3://<provided-bucket>/CIC-IDS-2018/ data/raw/cic_ids2018/

Expected result: data/raw/cic_ids2018/*.csv (≈7 GB total)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--cloudtrail", action="store_true",
                        help="Download flaws.cloud CloudTrail corpus")
    parser.add_argument("--cic-instructions", action="store_true",
                        help="Print CIC-IDS2018 download instructions")
    parser.add_argument("--all", action="store_true",
                        help="Download everything (CloudTrail + print CIC instructions)")
    args = parser.parse_args()

    if args.cloudtrail or args.all:
        download_cloudtrail()
    if args.cic_instructions or args.all:
        print_cic_instructions()
    if not any(vars(args).values()):
        parser.print_help()
