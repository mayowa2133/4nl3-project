#!/usr/bin/env python3
"""Build an annotation pool CSV from a local MultiWOZ 2.2 JSON export."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def infer_speaker(turn: Dict) -> Optional[str]:
    for key in ("speaker", "role"):
        value = turn.get(key)
        if isinstance(value, str):
            v = value.lower()
            if any(x in v for x in ("user", "usr", "human")):
                return "user"
            if any(x in v for x in ("system", "sys", "assistant", "bot")):
                return "system"

    metadata = turn.get("metadata")
    if isinstance(metadata, dict):
        # In canonical MultiWOZ logs, user turns usually have empty metadata,
        # system turns contain non-empty metadata with dialogue state updates.
        return "user" if len(metadata) == 0 else "system"
    return None


def extract_text(turn: Dict) -> str:
    for key in ("text", "utterance", "transcript"):
        value = turn.get(key)
        if isinstance(value, str):
            return normalize_whitespace(value)
    return ""


def iter_dialogues(payload) -> Iterator[Tuple[str, Dict]]:
    if isinstance(payload, list):
        for idx, dialogue in enumerate(payload):
            if not isinstance(dialogue, dict):
                continue
            dialogue_id = (
                dialogue.get("dialogue_id")
                or dialogue.get("dialogue_idx")
                or dialogue.get("id")
                or f"dialogue_{idx:05d}"
            )
            yield str(dialogue_id), dialogue
        return

    if isinstance(payload, dict):
        if isinstance(payload.get("dialogues"), list):
            for idx, dialogue in enumerate(payload["dialogues"]):
                if not isinstance(dialogue, dict):
                    continue
                dialogue_id = (
                    dialogue.get("dialogue_id")
                    or dialogue.get("dialogue_idx")
                    or dialogue.get("id")
                    or f"dialogue_{idx:05d}"
                )
                yield str(dialogue_id), dialogue
            return

        # Canonical MultiWOZ `data.json` style: {dialogue_id: dialogue_dict}
        for dialogue_id, dialogue in payload.items():
            if isinstance(dialogue, dict):
                yield str(dialogue_id), dialogue


def iter_turn_pairs(dialogue: Dict) -> Iterator[Tuple[int, str, str]]:
    turns = dialogue.get("turns")
    if isinstance(turns, list):
        for idx in range(1, len(turns)):
            prev_turn = turns[idx - 1]
            cur_turn = turns[idx]
            if not isinstance(prev_turn, dict) or not isinstance(cur_turn, dict):
                continue
            if infer_speaker(prev_turn) == "system" and infer_speaker(cur_turn) == "user":
                system_text = extract_text(prev_turn)
                user_text = extract_text(cur_turn)
                yield idx, system_text, user_text
        return

    log = dialogue.get("log")
    if isinstance(log, list):
        for idx in range(1, len(log)):
            prev_turn = log[idx - 1]
            cur_turn = log[idx]
            if not isinstance(prev_turn, dict) or not isinstance(cur_turn, dict):
                continue
            if infer_speaker(prev_turn) == "system" and infer_speaker(cur_turn) == "user":
                system_text = extract_text(prev_turn)
                user_text = extract_text(cur_turn)
                yield idx, system_text, user_text


def build_rows(
    input_path: Path,
    min_tokens: int,
    max_tokens: int,
    max_instances: Optional[int],
    seed: int,
    shuffle: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    collected: List[Dict[str, str]] = []
    counters = {
        "dialogues_seen": 0,
        "pairs_seen": 0,
        "filtered_empty": 0,
        "filtered_length": 0,
    }

    for dialogue_id, dialogue in iter_dialogues(payload):
        counters["dialogues_seen"] += 1
        for turn_id, system_context, user_utterance in iter_turn_pairs(dialogue):
            counters["pairs_seen"] += 1
            if not system_context or not user_utterance:
                counters["filtered_empty"] += 1
                continue

            n_tokens = token_count(user_utterance)
            if n_tokens < min_tokens or n_tokens > max_tokens:
                counters["filtered_length"] += 1
                continue

            instance_id = f"mw22_{dialogue_id}_{turn_id}"
            collected.append(
                {
                    "instance_id": instance_id,
                    "dialogue_id": dialogue_id,
                    "turn_id": str(turn_id),
                    "system_context": system_context,
                    "user_utterance": user_utterance,
                    "label": "",
                    "annotator_id": "",
                    "annotation_pass": "initial",
                    "notes": "",
                }
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(collected)

    if max_instances is not None and max_instances > 0:
        collected = collected[:max_instances]

    counters["written"] = len(collected)
    return collected, counters


def write_csv(rows: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract (previous system turn, current user turn) pairs from MultiWOZ 2.2 JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to local MultiWOZ JSON file (for example data/raw/multiwoz22.json).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/annotation_pool.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--min-tokens", type=int, default=1, help="Minimum user token count.")
    parser.add_argument("--max-tokens", type=int, default=40, help="Maximum user token count.")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=1200,
        help="Maximum rows to keep after filtering and optional shuffle.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for shuffle.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before truncating max instances.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows, counters = build_rows(
        input_path=input_path,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_instances=args.max_instances,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    write_csv(rows, output_path)

    print("Extraction complete")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Dialogues seen:   {counters['dialogues_seen']}")
    print(f"Pairs seen:       {counters['pairs_seen']}")
    print(f"Filtered empty:   {counters['filtered_empty']}")
    print(f"Filtered length:  {counters['filtered_length']}")
    print(f"Rows written:     {counters['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
