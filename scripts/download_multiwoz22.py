#!/usr/bin/env python3
"""Download and combine MultiWOZ 2.2 dialogue shards from GitHub."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/budzianowski/multiwoz/master/"
    "data/MultiWOZ_2.2"
)
DEFAULT_SPLITS = "train:17,dev:2,test:2"


def parse_splits(value: str) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid split format '{item}'. Use name:count (for example train:17)."
            )
        name, count_text = item.split(":", 1)
        name = name.strip()
        count = int(count_text.strip())
        if not name or count <= 0:
            raise ValueError(f"Invalid split entry '{item}'.")
        pairs.append((name, count))
    if not pairs:
        raise ValueError("No valid splits provided.")
    return pairs


def shard_names(count: int) -> List[str]:
    return [f"dialogues_{i:03d}.json" for i in range(1, count + 1)]


def download_file(url: str, dest: Path, timeout: float, overwrite: bool) -> bool:
    if dest.exists() and not overwrite:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return True


def load_dialogues(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Fallback for alternate JSON structures.
        if "dialogues" in payload and isinstance(payload["dialogues"], list):
            return payload["dialogues"]
        return [payload]
    raise ValueError(f"Unexpected JSON structure in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MultiWOZ 2.2 shards and create a combined local JSON file."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Root URL containing split folders train/dev/test.",
    )
    parser.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help="Comma-separated split:count list, e.g. train:17,dev:2,test:2.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/multiwoz22",
        help="Directory for downloaded shard files.",
    )
    parser.add_argument(
        "--combined-out",
        default="data/raw/multiwoz22.json",
        help="Combined JSON output path.",
    )
    parser.add_argument(
        "--manifest-out",
        default="data/raw/multiwoz22_manifest.json",
        help="Download manifest JSON output path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between downloads to reduce request burst.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload shard files even if local files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = parse_splits(args.splits)
    raw_dir = Path(args.raw_dir)
    combined_out = Path(args.combined_out)
    manifest_out = Path(args.manifest_out)

    plan: List[Tuple[str, Path, str]] = []
    for split, count in splits:
        for shard in shard_names(count):
            url = f"{args.base_url}/{split}/{shard}"
            dest = raw_dir / f"{split}_{shard}"
            plan.append((url, dest, split))

    print("Planned downloads:")
    for url, dest, _ in plan:
        print(f"- {url} -> {dest}")

    if args.dry_run:
        print("Dry run only; no files were downloaded.")
        return 0

    downloaded = 0
    reused = 0
    combined: List[Dict] = []
    split_counts: Dict[str, int] = {}

    for idx, (url, dest, split) in enumerate(plan, start=1):
        try:
            did_download = download_file(
                url=url, dest=dest, timeout=args.timeout, overwrite=args.overwrite
            )
            if did_download:
                downloaded += 1
                print(f"[{idx}/{len(plan)}] downloaded {dest}")
            else:
                reused += 1
                print(f"[{idx}/{len(plan)}] reused {dest}")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

        shard_dialogues = load_dialogues(dest)
        combined.extend(shard_dialogues)
        split_counts[split] = split_counts.get(split, 0) + len(shard_dialogues)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    combined_out.parent.mkdir(parents=True, exist_ok=True)
    with combined_out.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)

    manifest = {
        "base_url": args.base_url,
        "splits": [{"name": name, "count": count} for name, count in splits],
        "downloaded_shards": downloaded,
        "reused_shards": reused,
        "total_shards": len(plan),
        "dialogues_total": len(combined),
        "dialogues_by_split": split_counts,
        "combined_out": str(combined_out),
        "raw_dir": str(raw_dir),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Download complete")
    print(f"Shards downloaded: {downloaded}")
    print(f"Shards reused:     {reused}")
    print(f"Dialogues total:   {len(combined)}")
    print(f"Combined file:     {combined_out}")
    print(f"Manifest file:     {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
