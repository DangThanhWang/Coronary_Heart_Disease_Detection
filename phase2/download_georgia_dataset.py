from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/georgia"
DIR_RE = re.compile(r'href="(g\d+/)"')
FILE_RE = re.compile(r'href="([^"]+\.(?:hea|mat))"[^<]*</a>\s+\S+\s+\S+\s+(\d+)')


@dataclass(frozen=True)
class RemoteFile:
    url: str
    rel_path: Path
    size: int


def read_text(url: str, timeout: int, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "clinical-counterfactual-memory/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"failed to read index {url}: {last_error}")


def discover_files(base_url: str, timeout: int) -> list[RemoteFile]:
    root_html = read_text(base_url.rstrip("/") + "/", timeout)
    dirs = sorted(set(DIR_RE.findall(root_html)), key=lambda s: (len(s), s))
    files: list[RemoteFile] = []
    for folder in dirs:
        folder_url = base_url.rstrip("/") + "/" + folder
        html = read_text(folder_url, timeout)
        for name, size in FILE_RE.findall(html):
            files.append(RemoteFile(url=folder_url + name, rel_path=Path(folder) / name, size=int(size)))
    return files


def is_complete(path: Path, size: int) -> bool:
    return path.exists() and path.stat().st_size == size


def download_one(item: RemoteFile, out_root: Path, timeout: int, retries: int) -> tuple[str, int]:
    dest = out_root / item.rel_path
    if is_complete(dest, item.size):
        return "skipped", item.size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(item.url, headers={"User-Agent": "clinical-counterfactual-memory/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, part.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if part.stat().st_size != item.size:
                raise IOError(f"size mismatch: got {part.stat().st_size}, expected {item.size}")
            part.replace(dest)
            return "downloaded", item.size
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"failed {item.url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Georgia subset of PhysioNet/CinC Challenge 2020.")
    parser.add_argument("--out-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.base_url, args.timeout)
    manifest = {
        "base_url": args.base_url,
        "out_root": str(args.out_root),
        "n_files": len(files),
        "total_bytes": int(sum(f.size for f in files)),
        "files": [{"url": f.url, "rel_path": str(f.rel_path), "size": f.size} for f in files],
    }
    (args.out_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"discovered files={len(files)} total_gb={manifest['total_bytes'] / 1e9:.3f}")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    bytes_done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item, args.out_root, args.timeout, args.retries) for item in files]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                status, size = future.result()
                counts[status] += 1
                bytes_done += size
            except Exception as exc:
                counts["failed"] += 1
                print(f"[failed] {exc}")
            if i % 250 == 0 or i == len(files):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"progress {i}/{len(files)} downloaded={counts['downloaded']} "
                    f"skipped={counts['skipped']} failed={counts['failed']} "
                    f"done_gb={bytes_done / 1e9:.3f} speed_mb_s={bytes_done / elapsed / 1e6:.2f}"
                )
    summary = {**manifest, "counts": counts, "elapsed_seconds": time.time() - started}
    (args.out_root / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if counts["failed"]:
        raise SystemExit(f"Download completed with failures: {counts['failed']}")
    print(json.dumps({"counts": counts, "out_root": str(args.out_root)}, indent=2))


if __name__ == "__main__":
    main()
