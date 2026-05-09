from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import wfdb


BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"
USER_AGENT = "clinical-counterfactual-memory/1.0"


def urlopen_with_retry(url: str, timeout: int, retries: int):
    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(req, timeout=timeout)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"failed to open {url}: {last_error}")


def download_file(url: str, dest: Path, timeout: int, retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with urlopen_with_retry(url, timeout, retries) as response, part.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    part.replace(dest)


def fetch_record_list_preview(base_url: str, preview_rows: int, timeout: int, retries: int) -> list[dict[str, str]]:
    url = base_url.rstrip("/") + "/record_list.csv"
    with urlopen_with_retry(url, timeout, retries) as response:
        wrapper = io.TextIOWrapper(response, encoding="utf-8", newline="")
        reader = csv.DictReader(wrapper)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= preview_rows:
                break
    if not rows:
        raise RuntimeError("record_list preview is empty")
    return rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_full_record_list(base_url: str, out_path: Path, timeout: int, retries: int) -> None:
    url = base_url.rstrip("/") + "/record_list.csv"
    download_file(url, out_path, timeout, retries)


def normalize_record_stem(path_value: str) -> str:
    stem = path_value.strip().replace("\\", "/").lstrip("/")
    if stem.endswith(".hea") or stem.endswith(".dat"):
        stem = stem.rsplit(".", 1)[0]
    return stem


def download_record_pair(base_url: str, out_root: Path, record_stem: str, timeout: int, retries: int) -> dict[str, str]:
    stem = normalize_record_stem(record_stem)
    rel_stem = Path(stem)
    for suffix in (".hea", ".dat"):
        url = base_url.rstrip("/") + "/" + stem + suffix
        dest = out_root / (stem + suffix)
        download_file(url, dest, timeout, retries)
    return {
        "record_stem": stem,
        "local_stem": str((out_root / rel_stem).resolve()),
    }


def inspect_record(local_stem: Path, context: dict[str, str]) -> dict[str, str | int | float]:
    record = wfdb.rdrecord(str(local_stem))
    age = ""
    sex = ""
    for comment in getattr(record, "comments", []) or []:
        if comment.startswith("age:"):
            age = comment.split(":", 1)[1].strip()
        if comment.startswith("sex:"):
            sex = comment.split(":", 1)[1].strip()
    return {
        "subject_id": context.get("subject_id", ""),
        "study_id": context.get("study_id", ""),
        "ecg_time": context.get("ecg_time", ""),
        "record_stem": context["record_stem"],
        "fs": float(record.fs),
        "n_samples": int(record.p_signal.shape[0]),
        "n_leads": int(record.p_signal.shape[1]),
        "sig_names": "|".join(record.sig_name),
        "age": age,
        "sex": sex,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a tiny local MIMIC-IV-ECG sample and verify WFDB loading.")
    parser.add_argument("--out-root", type=Path, default=Path("Data") / "MIMIC_IV_ECG_bootstrap")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--n-records", type=int, default=3)
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--download-full-record-list", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    preview_rows = fetch_record_list_preview(args.base_url, max(args.preview_rows, args.n_records), args.timeout, args.retries)
    preview_path = args.out_root / "record_list_preview.csv"
    write_csv(preview_rows, preview_path)

    if args.download_full_record_list:
        download_full_record_list(args.base_url, args.out_root / "record_list.csv", args.timeout, args.retries)

    selected_rows = preview_rows[: args.n_records]
    inspections: list[dict[str, str | int | float]] = []
    downloaded: list[dict[str, str]] = []
    for row in selected_rows:
        record_stem = row["path"]
        download_info = download_record_pair(args.base_url, args.out_root, record_stem, args.timeout, args.retries)
        downloaded.append(download_info)
        inspections.append(inspect_record(args.out_root / record_stem, {**row, **download_info}))

    inspection_csv = args.out_root / "record_inspection.csv"
    write_csv(inspections, inspection_csv)
    summary = {
        "base_url": args.base_url,
        "out_root": str(args.out_root),
        "n_records_requested": args.n_records,
        "n_records_downloaded": len(downloaded),
        "preview_path": str(preview_path),
        "inspection_csv": str(inspection_csv),
        "downloaded_records": downloaded,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_root / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
