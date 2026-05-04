"""Download Citi Bike trip data CSVs from S3."""

import zipfile
from pathlib import Path

import requests

from src.config import CITIBIKE_S3_BASE, DATA_RAW, MONTHS


def download_month(month: str, output_dir: Path = DATA_RAW) -> Path:
    """
    Download and extract a single month of Citi Bike data.
    """
    filename = f"{month}-citibike-tripdata.zip"
    url = f"{CITIBIKE_S3_BASE}/{filename}"
    zip_path = output_dir / filename
    extract_dir = output_dir / month

    if extract_dir.exists() and any(extract_dir.glob("*.csv")):
        print(f"Already extracted: {month}")
        return extract_dir

    print(f"Downloading {url}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting {month}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    zip_path.unlink()
    print(f"Done: {month}")
    return extract_dir


def download_all():
    """Download all configured months."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        download_month(month)


if __name__ == "__main__":
    download_all()
