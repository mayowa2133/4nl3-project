#!/usr/bin/env python3
"""Create initial and reannotation assignment CSV files from annotation pool."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

FIELDNAMES = [
    "instance_id",
    "dialogue_id",
    "turn_id",
    "system_context",
    "user_utterance",
    "label",
    "annotator_id",
    "annotation_pass",
    "notes",
]


def read_pool(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: row.get(k, "") for k in FIELDNAMES})
    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 assignment files.")
    parser.add_argument(
        "--pool",
        default="data/processed/annotation_pool.csv",
        help="Input pool CSV path.",
    )
    parser.add_argument(
        "--annotators",
        default="adesanym,zajkeskn,dhiraajd",
        help="Comma-separated annotator IDs in assignment order.",
    )
    parser.add_argument(
        "--per-annotator",
        type=int,
        default=350,
        help="Unique initial instances per annotator.",
    )
    parser.add_argument(
        "--overlap-total",
        type=int,
        default=135,
        help="Total number of duplicate reannotation assignments.",
    )
    parser.add_argument(
        "--initial-out",
        default="data/processed/assignments_initial.csv",
        help="Output CSV for initial assignments.",
    )
    parser.add_argument(
        "--reannotation-out",
        default="data/processed/assignments_reannotation.csv",
        help="Output CSV for reannotation assignments.",
    )
    parser.add_argument(
        "--combined-out",
        default="data/processed/annotations_combined_template.csv",
        help="Output CSV with both initial and reannotation rows.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_path = Path(args.pool)
    initial_out = Path(args.initial_out)
    reannotation_out = Path(args.reannotation_out)
    combined_out = Path(args.combined_out)
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]

    if len(annotators) < 2:
        raise ValueError("Need at least two annotators.")
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    rows = read_pool(pool_path)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    required_unique = args.per_annotator * len(annotators)
    if len(rows) < required_unique:
        raise ValueError(
            f"Pool has {len(rows)} rows, but need {required_unique} for initial split."
        )

    selected = rows[:required_unique]

    # Initial split
    initial_rows: List[Dict[str, str]] = []
    per_annotator_rows: Dict[str, List[Dict[str, str]]] = {}
    start = 0
    for annotator in annotators:
        end = start + args.per_annotator
        chunk = selected[start:end]
        start = end
        per_annotator_rows[annotator] = chunk
        for base_row in chunk:
            row = dict(base_row)
            row["annotator_id"] = annotator
            row["annotation_pass"] = "initial"
            row["label"] = ""
            row["notes"] = ""
            initial_rows.append(row)

    # Reannotation split in a ring: each annotator gets rows from previous annotator.
    base_overlap = args.overlap_total // len(annotators)
    remainder = args.overlap_total % len(annotators)
    overlap_counts = [base_overlap + (1 if i < remainder else 0) for i in range(len(annotators))]

    reannotation_rows: List[Dict[str, str]] = []
    for i, target_annotator in enumerate(annotators):
        source_annotator = annotators[(i - 1) % len(annotators)]
        source_rows = per_annotator_rows[source_annotator]
        take_n = overlap_counts[i]
        if take_n > len(source_rows):
            raise ValueError(
                f"Overlap request {take_n} exceeds source rows {len(source_rows)} for {source_annotator}."
            )
        chosen = rng.sample(source_rows, k=take_n)
        for base_row in chosen:
            row = dict(base_row)
            row["annotator_id"] = target_annotator
            row["annotation_pass"] = "reannotation"
            row["label"] = ""
            row["notes"] = ""
            reannotation_rows.append(row)

    combined_rows = initial_rows + reannotation_rows

    write_rows(initial_out, initial_rows)
    write_rows(reannotation_out, reannotation_rows)
    write_rows(combined_out, combined_rows)

    print("Assignment creation complete")
    print(f"Pool rows available: {len(rows)}")
    print(f"Initial rows written: {len(initial_rows)} -> {initial_out}")
    print(f"Reannotation rows written: {len(reannotation_rows)} -> {reannotation_out}")
    print(f"Combined rows written: {len(combined_rows)} -> {combined_out}")
    print("Ring overlap mapping:")
    for i, target_annotator in enumerate(annotators):
        source_annotator = annotators[(i - 1) % len(annotators)]
        print(f"  {target_annotator} reannotates from {source_annotator}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
