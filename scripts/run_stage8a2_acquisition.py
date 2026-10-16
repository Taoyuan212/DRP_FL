"""Stage 8A-2 official AEMO acquisition and SCADA smoke/long-window audit.

This script never trains a model and never disables TLS verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import certifi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/aemo/raw"
ZIP_DIR = RAW / "nemweb"
PROCESSED = ROOT / "data/aemo/processed_raw"
OUT = ROOT / "results/stage8a2"
ARCHIVE_URL = "https://www.nemweb.com.au/REPORTS/ARCHIVE/Dispatch_SCADA/"
FILE_RE = re.compile(r'href=["\']([^"\']*PUBLIC_DISPATCHSCADA_(\d{8})\.zip)["\']', re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_download(session: requests.Session, url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with session.get(url, timeout=(15, 90), stream=True) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    partial.replace(target)


def parse_aemo_csv(blob: bytes, source_file: str, wind_duids: set[str]) -> tuple[pd.DataFrame, dict]:
    rows = list(csv.reader(io.StringIO(blob.decode("utf-8-sig", errors="strict"))))
    header_index = next((i for i, row in enumerate(rows) if "DUID" in row and "SETTLEMENTDATE" in row), None)
    if header_index is None:
        raise ValueError(f"No DUID/SETTLEMENTDATE header in {source_file}")
    header = rows[header_index]
    width = len(header)
    records = [row for row in rows[header_index + 1:] if len(row) == width and row and row[0] == "D"]
    frame = pd.DataFrame(records, columns=header)
    required = {"DUID", "SETTLEMENTDATE", "SCADAVALUE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns {sorted(required - set(frame.columns))} in {source_file}")
    frame = frame[["SETTLEMENTDATE", "DUID", "SCADAVALUE"]].copy()
    frame = frame[frame["DUID"].isin(wind_duids)].copy()
    frame["timestamp_raw"] = frame["SETTLEMENTDATE"]
    frame["timestamp_canonical"] = pd.to_datetime(frame["SETTLEMENTDATE"], errors="coerce")
    frame["SCADAVALUE_MW"] = pd.to_numeric(frame["SCADAVALUE"], errors="coerce")
    frame["source_file"] = source_file
    frame = frame[["timestamp_raw", "timestamp_canonical", "DUID", "SCADAVALUE_MW", "source_file"]]
    duplicate_count = int(frame.duplicated(["timestamp_canonical", "DUID"]).sum())
    audit = {
        "csv_member": source_file,
        "header_columns": header,
        "wind_rows": len(frame),
        "wind_duids_matched": int(frame["DUID"].nunique()),
        "timestamp_parse_failures": int(frame["timestamp_canonical"].isna().sum()),
        "scada_numeric_failures": int(frame["SCADAVALUE_MW"].isna().sum()),
        "duplicate_duid_timestamps": duplicate_count,
        "negative_scada_rows": int((frame["SCADAVALUE_MW"] < 0).sum()),
        "zero_scada_rows": int((frame["SCADAVALUE_MW"] == 0).sum()),
        "min_scada_mw": float(frame["SCADAVALUE_MW"].min()) if len(frame) else None,
        "max_scada_mw": float(frame["SCADAVALUE_MW"].max()) if len(frame) else None,
    }
    return frame, audit


def audit_zip(path: Path, wind_duids: set[str]) -> tuple[pd.DataFrame, dict]:
    if not zipfile.is_zipfile(path):
        raise zipfile.BadZipFile(path)
    with zipfile.ZipFile(path) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise zipfile.BadZipFile(f"CRC failure: {bad_member}")
        outer_csv = [x for x in zf.namelist() if x.lower().endswith(".csv")]
        nested_zips = [x for x in zf.namelist() if x.lower().endswith(".zip")]
        frames: list[pd.DataFrame] = []
        member_audits: list[dict] = []
        for member in outer_csv:
            frame, member_audit = parse_aemo_csv(zf.read(member), member, wind_duids)
            frames.append(frame)
            member_audits.append(member_audit)
        for nested_name in nested_zips:
            nested_blob = zf.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_blob)) as nested:
                nested_bad = nested.testzip()
                if nested_bad:
                    raise zipfile.BadZipFile(f"Nested CRC failure: {nested_name}/{nested_bad}")
                csv_members = [x for x in nested.namelist() if x.lower().endswith(".csv")]
                if len(csv_members) != 1:
                    raise ValueError(f"Expected one CSV in nested {nested_name}, got {csv_members}")
                csv_name = csv_members[0]
                frame, member_audit = parse_aemo_csv(nested.read(csv_name), f"{nested_name}/{csv_name}", wind_duids)
                frames.append(frame)
                member_audits.append(member_audit)
        if not frames:
            raise ValueError(f"No CSV or nested ZIP members in {path.name}")
        frame = pd.concat(frames, ignore_index=True)
        audit = {
            "outer_csv_member_count": len(outer_csv),
            "nested_zip_member_count": len(nested_zips),
            "csv_member_count": len(member_audits),
            "wind_rows": len(frame),
            "wind_duids_matched": int(frame["DUID"].nunique()),
            "timestamp_parse_failures": int(frame["timestamp_canonical"].isna().sum()),
            "scada_numeric_failures": int(frame["SCADAVALUE_MW"].isna().sum()),
            "duplicate_duid_timestamps": int(frame.duplicated(["timestamp_canonical", "DUID"]).sum()),
            "negative_scada_rows": int((frame["SCADAVALUE_MW"] < 0).sum()),
            "zero_scada_rows": int((frame["SCADAVALUE_MW"] == 0).sum()),
            "min_scada_mw": float(frame["SCADAVALUE_MW"].min()) if len(frame) else None,
            "max_scada_mw": float(frame["SCADAVALUE_MW"].max()) if len(frame) else None,
            "schema_variants": sorted({tuple(x["header_columns"]) for x in member_audits}),
        }
    audit.update({
        "zip_file": path.name,
        "zip_size_bytes": path.stat().st_size,
        "zip_sha256": sha256(path),
        "zip_valid": True,
    })
    return frame, audit


def read_wind_duids() -> set[str]:
    path = ROOT / "results/stage8a1/aemo_wind_duid_inventory.csv"
    frame = pd.read_csv(path)
    return set(frame["DUID"].astype(str))


def interval_audit(frame: pd.DataFrame) -> dict:
    diffs = []
    for _, group in frame.drop_duplicates(["DUID", "timestamp_canonical"]).groupby("DUID"):
        values = group["timestamp_canonical"].sort_values().diff().dropna().dt.total_seconds().div(60)
        diffs.extend(values.tolist())
    counts = Counter(diffs)
    modal = counts.most_common(1)[0][0] if counts else None
    return {
        "modal_interval_minutes": modal,
        "interval_distribution_minutes": {str(k): v for k, v in sorted(counts.items())},
        "five_minute_empirically_verified": modal == 5.0,
    }


def update_stream_stats(frame: pd.DataFrame, stats: dict, last_by_duid: dict[str, pd.Timestamp]) -> None:
    stats["row_count"] += len(frame)
    stats["wind_duids"].update(frame["DUID"].dropna().astype(str))
    stats["duplicate_duid_timestamps"] += int(frame.duplicated(["timestamp_canonical", "DUID"]).sum())
    stats["timestamp_parse_failures"] += int(frame["timestamp_canonical"].isna().sum())
    stats["scada_numeric_failures"] += int(frame["SCADAVALUE_MW"].isna().sum())
    stats["negative_scada_rows"] += int((frame["SCADAVALUE_MW"] < 0).sum())
    values = frame["SCADAVALUE_MW"].dropna()
    if len(values):
        stats["min_scada_mw"] = min(stats["min_scada_mw"], float(values.min()))
        stats["max_scada_mw"] = max(stats["max_scada_mw"], float(values.max()))
    timestamps = frame["timestamp_canonical"].dropna()
    if len(timestamps):
        stats["first_timestamp"] = min(stats["first_timestamp"], timestamps.min())
        stats["last_timestamp"] = max(stats["last_timestamp"], timestamps.max())
    for duid, group in frame.dropna(subset=["timestamp_canonical"]).groupby("DUID"):
        ordered = group["timestamp_canonical"].drop_duplicates().sort_values()
        if not len(ordered):
            continue
        prior = last_by_duid.get(str(duid))
        if prior is not None:
            stats["interval_counts"][(ordered.iloc[0] - prior).total_seconds() / 60] += 1
        stats["interval_counts"].update(ordered.diff().dropna().dt.total_seconds().div(60).tolist())
        last_by_duid[str(duid)] = ordered.iloc[-1]


def finish_stream_stats(stats: dict) -> dict:
    counts = stats["interval_counts"]
    modal = counts.most_common(1)[0][0] if counts else None
    return {
        "first_timestamp_raw": str(stats["first_timestamp"]),
        "last_timestamp_raw": str(stats["last_timestamp"]),
        "row_count": stats["row_count"],
        "wind_duids_matched": len(stats["wind_duids"]),
        "duplicate_duid_timestamps": stats["duplicate_duid_timestamps"],
        "timestamp_parse_failures": stats["timestamp_parse_failures"],
        "scada_numeric_failures": stats["scada_numeric_failures"],
        "negative_scada_rows": stats["negative_scada_rows"],
        "min_scada_mw": stats["min_scada_mw"],
        "max_scada_mw": stats["max_scada_mw"],
        "modal_interval_minutes": modal,
        "interval_distribution_minutes": {str(k): v for k, v in sorted(counts.items())},
        "five_minute_empirically_verified": modal == 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "long"], required=True)
    parser.add_argument("--smoke-start", default="20260101")
    parser.add_argument("--smoke-days", type=int, default=7)
    parser.add_argument("--long-start", default="20250811")
    parser.add_argument("--long-end", default="20260820")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "ENG-Stage8A2-research-audit/1.0"

    response = session.get(ARCHIVE_URL, timeout=(15, 30))
    response.raise_for_status()
    listing_hash = hashlib.sha256(response.content).hexdigest()
    files = sorted({date: name for name, date in FILE_RE.findall(response.text)}.items())
    if not files:
        raise RuntimeError("Official archive listing yielded no DISPATCHSCADA files")

    if args.mode == "smoke":
        start = pd.Timestamp(args.smoke_start)
        wanted = {(start + pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(args.smoke_days)}
    else:
        wanted = {date for date, _ in files if args.long_start <= date <= args.long_end}
    selected = [(date, name) for date, name in files if date in wanted]
    if len(selected) != len(wanted):
        missing = sorted(wanted - {date for date, _ in selected})
        raise RuntimeError(f"Archive listing missing requested dates: {missing}")

    wind_duids = read_wind_duids()
    frames: list[pd.DataFrame] = []
    audits = []
    writer = None
    long_target = PROCESSED / "aemo_dispatch_scada_wind_long.parquet"
    long_partial = long_target.with_suffix(".parquet.part")
    if args.mode == "long" and long_partial.exists():
        long_partial.unlink()
    stream_stats = {
        "row_count": 0, "wind_duids": set(), "duplicate_duid_timestamps": 0,
        "timestamp_parse_failures": 0, "scada_numeric_failures": 0, "negative_scada_rows": 0,
        "min_scada_mw": float("inf"), "max_scada_mw": float("-inf"),
        "first_timestamp": pd.Timestamp.max, "last_timestamp": pd.Timestamp.min,
        "interval_counts": Counter(),
    }
    last_by_duid: dict[str, pd.Timestamp] = {}
    download_started = time.time()
    try:
        for index, (date, name) in enumerate(selected, 1):
            url = urljoin(ARCHIVE_URL, name)
            target = ZIP_DIR / Path(name).name
            atomic_download(session, url, target)
            frame, audit = audit_zip(target, wind_duids)
            audit.update({"file_date": date, "source_url": url})
            audits.append(audit)
            if args.mode == "long":
                update_stream_stats(frame, stream_stats, last_by_duid)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(long_partial, table.schema, compression="zstd")
                writer.write_table(table)
            else:
                frames.append(frame)
            print(json.dumps({"mode": args.mode, "file": Path(name).name, "progress": f"{index}/{len(selected)}", "wind_rows": len(frame)}, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.close()

    if args.mode == "long":
        coverage = finish_stream_stats(stream_stats)
        long_partial.replace(long_target)
    else:
        combined = pd.concat(frames, ignore_index=True)
        interval = interval_audit(combined)
        coverage = {
            "first_timestamp_raw": str(combined["timestamp_canonical"].min()),
            "last_timestamp_raw": str(combined["timestamp_canonical"].max()),
            "row_count": len(combined),
            "wind_duids_matched": int(combined["DUID"].nunique()),
            "duplicate_duid_timestamps": int(combined.duplicated(["timestamp_canonical", "DUID"]).sum()),
            "timestamp_parse_failures": int(combined["timestamp_canonical"].isna().sum()),
            "scada_numeric_failures": int(combined["SCADAVALUE_MW"].isna().sum()),
            "negative_scada_rows": int((combined["SCADAVALUE_MW"] < 0).sum()),
            "min_scada_mw": float(combined["SCADAVALUE_MW"].min()),
            "max_scada_mw": float(combined["SCADAVALUE_MW"].max()),
            **interval,
        }
    audit_payload = {
        "mode": args.mode,
        "archive_url": ARCHIVE_URL,
        "archive_listing_status": response.status_code,
        "archive_listing_sha256": listing_hash,
        "archive_available_start": files[0][0],
        "archive_available_end": files[-1][0],
        "archive_file_count": len(files),
        "selected_file_count": len(selected),
        "requests_version": requests.__version__,
        "certifi_version": certifi.__version__,
        "tls_verify": True,
        "elapsed_seconds": round(time.time() - download_started, 3),
        "files": audits,
        "combined": coverage,
    }
    suffix = "smoke" if args.mode == "smoke" else "long"
    (OUT / f"aemo_scada_{suffix}_audit.json").write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
    if args.mode == "smoke":
        combined.to_parquet(PROCESSED / "aemo_dispatch_scada_wind_smoke.parquet", index=False)
    print(json.dumps({"status": "COMPLETE", "mode": args.mode, **coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
