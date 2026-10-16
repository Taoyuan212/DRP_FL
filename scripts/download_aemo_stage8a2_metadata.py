"""Download immutable official AEMO MMS registration metadata for Stage 8A-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "DISPATCHABLEUNIT", "DUDETAIL", "DUDETAILSUMMARY", "GENUNITS", "GENUNITS_UNIT",
    "STADUALLOC", "STATION", "STATIONOWNER", "STATIONOWNERTRK",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=6)
    args = parser.parse_args()
    year, month = args.year, args.month
    target = ROOT / f"data/aemo/raw/metadata/mmsdm_{year}{month:02d}"
    listing_url = (
        f"https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
        f"MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
    )
    target.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "ENG-Stage8A2-research-audit/1.0"
    listing = session.get(listing_url, timeout=(15, 30))
    listing.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', listing.text, flags=re.I)
    selected = []
    for href in hrefs:
        upper = href.upper()
        match = re.search(r"PUBLIC_ARCHIVE%23([^%]+)%23FILE01%23\d+\.ZIP", upper)
        if match and match.group(1) in TABLES:
            selected.append((match.group(1), href))
    found = {table for table, _ in selected}
    missing = sorted(TABLES - found)
    if missing:
        raise RuntimeError(f"MMS listing missing required tables: {missing}")
    records = []
    for table, href in sorted(selected):
        url = urljoin(listing_url, href)
        target_file = target / Path(href).name.replace("%23", "#")
        if not target_file.exists():
            partial = target_file.with_suffix(target_file.suffix + ".part")
            with session.get(url, timeout=(15, 60), stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as f:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            partial.replace(target_file)
        records.append({"table": table, "url": url, "file": str(target_file.relative_to(ROOT)),
                        "size_bytes": target_file.stat().st_size, "sha256": sha256(target_file)})
        print(json.dumps(records[-1]), flush=True)
    (target / "metadata_download_manifest.json").write_text(
        json.dumps({"listing_url": listing_url, "listing_status": listing.status_code,
                    "listing_sha256": hashlib.sha256(listing.content).hexdigest(), "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
