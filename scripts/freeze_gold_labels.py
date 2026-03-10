#!/usr/bin/env python3
"""Freeze one gold label per instance and emit Person A handoff artifacts."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/annotations_labeled_long.csv"
AGREEMENT_PATH = ROOT / "data/processed/agreement_summary.json"
OUTPUT_DIR = ROOT / "data/processed"
DOCS_DIR = ROOT / "docs/competition_launch"

VALID_LABELS = [
    "REQUEST",
    "INFORM_CONSTRAINT",
    "CONFIRM_ACCEPT",
    "CORRECT_CLARIFY",
    "SOCIAL",
]

ADJUDICATIONS = {
    "mw22_MUL0125.json_16": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Provides the hotel destination detail needed for the taxi request rather than accepting a proposal.",
    },
    "mw22_MUL1119.json_4": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Explicitly accepts the booking proposal before adding stay details, so acceptance remains dominant.",
    },
    "mw22_PMUL3163.json_8": {
        "final_label": "REQUEST",
        "rationale": "Starts a new taxi task and asks the system to arrange it; the arrival time is supporting detail.",
    },
    "mw22_MUL0121.json_6": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Answers the cuisine-choice question by supplying a preference rather than ending the exchange socially.",
    },
    "mw22_SSNG0074.json_4": {
        "final_label": "REQUEST",
        "rationale": "The user declines a sub-question about area and asks for a recommendation plus a booking.",
    },
    "mw22_MUL2418.json_2": {
        "final_label": "REQUEST",
        "rationale": "Asks the system to find a specific restaurant by name, which is a direct search request.",
    },
    "mw22_MUL1284.json_4": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Corrects the offered lodging type from guesthouse to hotel, and correction takes precedence.",
    },
    "mw22_PMUL0719.json_12": {
        "final_label": "REQUEST",
        "rationale": "Introduces a new train search task; the day and route are details attached to that request.",
    },
    "mw22_PMUL3246.json_6": {
        "final_label": "REQUEST",
        "rationale": "Starts a new restaurant-finding request while also specifying cheap and central constraints.",
    },
    "mw22_MUL0870.json_2": {
        "final_label": "REQUEST",
        "rationale": "The core act is asking the system to find museums, not merely confirming the previous turn.",
    },
    "mw22_MUL1252.json_8": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Supplies star-rating and price-range preferences in response to the system's question.",
    },
    "mw22_MUL1728.json_12": {
        "final_label": "SOCIAL",
        "rationale": "Closes the exchange with no task-directed content after receiving the requested information.",
    },
    "mw22_PMUL0048.json_6": {
        "final_label": "REQUEST",
        "rationale": "Defers booking and asks to see the available options first, making the turn an information request.",
    },
    "mw22_PMUL2750.json_14": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Accepts the hotel choice and proceeds with booking details, which fits the acceptance rule.",
    },
    "mw22_MUL1083.json_10": {
        "final_label": "REQUEST",
        "rationale": "Begins a new taxi task and asks the system to arrange transport between locations.",
    },
    "mw22_MUL1543.json_6": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Uses 'instead' to revise the failed train option, so correction outranks the new search request.",
    },
    "mw22_PMUL0373.json_20": {
        "final_label": "REQUEST",
        "rationale": "Requests a taxi booking and asks for the booking information in the same turn.",
    },
    "mw22_MUL2397.json_12": {
        "final_label": "REQUEST",
        "rationale": "Accepts the attraction information but then launches a new taxi-booking request.",
    },
    "mw22_PMUL2446.json_8": {
        "final_label": "REQUEST",
        "rationale": "Starts a new attraction-search request for the east side of town.",
    },
    "mw22_SSNG0174.json_2": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Answers the cuisine question by specifying a desired food type rather than correcting anything.",
    },
    "mw22_PMUL2993.json_2": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Commits to the offered hotel with an explicit booking instruction for the selected place.",
    },
    "mw22_MUL0067.json_12": {
        "final_label": "REQUEST",
        "rationale": "Asks the system to pick the matching restaurant closest to a reference location.",
    },
    "mw22_SSNG0258.json_2": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Provides side-of-town and lodging-type preferences in direct response to the system's prompt.",
    },
    "mw22_PMUL4392.json_2": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Explicitly rejects the hotel suggestion before adding a train request, so correction dominates.",
    },
    "mw22_PMUL1974.json_10": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Corrects the system's assumption about needing tickets before moving to a hotel search.",
    },
    "mw22_WOZ20545.json_4": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Responds to the fallback-food prompt by naming a new cuisine preference.",
    },
    "mw22_PMUL4625.json_4": {
        "final_label": "REQUEST",
        "rationale": "Moves on from the attraction and asks for help with a new train task.",
    },
    "mw22_SNG01399.json_4": {
        "final_label": "SOCIAL",
        "rationale": "Closes the conversation with thanks and no remaining task content.",
    },
    "mw22_PMUL2475.json_14": {
        "final_label": "REQUEST",
        "rationale": "Asks for a taxi while also specifying a departure-time constraint for that taxi.",
    },
    "mw22_MUL0200.json_4": {
        "final_label": "INFORM_CONSTRAINT",
        "rationale": "Provides a revised cuisine preference in response to the system's request to broaden the search.",
    },
    "mw22_PMUL3820.json_4": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Calls out the missing phone-number request and asks the system to fix that omission.",
    },
    "mw22_MUL0418.json_4": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Explicitly says yes to the booking proposal before providing ticket-count details.",
    },
    "mw22_PMUL1357.json_12": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Accepts the train offer and then supplies the passenger count needed to complete the booking.",
    },
    "mw22_PMUL3494.json_10": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Rejects the area-change option and redirects the search to a different cuisine type instead.",
    },
    "mw22_MUL0538.json_12": {
        "final_label": "SOCIAL",
        "rationale": "Only acknowledges the attraction information and thanks the system for the help.",
    },
    "mw22_SSNG0022.json_8": {
        "final_label": "REQUEST",
        "rationale": "Asks for another restaurant option that preserves the same area and price range.",
    },
    "mw22_PMUL4111.json_10": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Accepts the recommendation and asks the system to carry out the reservation.",
    },
    "mw22_SSNG0097.json_6": {
        "final_label": "REQUEST",
        "rationale": "Agrees to try another restaurant and requests one with the same area and price range.",
    },
    "mw22_MUL0011.json_18": {
        "final_label": "CORRECT_CLARIFY",
        "rationale": "Revises the stay length from two nights to one night, which is a direct correction.",
    },
    "mw22_SNG0673.json_2": {
        "final_label": "CONFIRM_ACCEPT",
        "rationale": "Explicitly accepts the reservation offer and supplies party size and time details.",
    },
}


def load_annotations() -> list[dict[str, str]]:
    with INPUT_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def choose_representative(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row["annotation_pass"] == "initial":
            return row
    return rows[0]


def build_outputs(rows: list[dict[str, str]]):
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)

    gold_rows: list[dict[str, str]] = []
    adjudication_rows: list[dict[str, str]] = []
    seen_order: list[str] = []
    for row in rows:
        if row["instance_id"] not in grouped:
            continue
        if row["instance_id"] not in seen_order:
            seen_order.append(row["instance_id"])

    for instance_id in seen_order:
        group = grouped[instance_id]
        representative = choose_representative(group)
        labels = {row["label"] for row in group}
        if len(group) == 1:
            final_label = representative["label"]
        elif len(labels) == 1:
            final_label = next(iter(labels))
        else:
            adjudication = ADJUDICATIONS[instance_id]
            final_label = adjudication["final_label"]
            adjudication_rows.append(
                {
                    "instance_id": instance_id,
                    "dialogue_id": representative["dialogue_id"],
                    "turn_id": representative["turn_id"],
                    "system_context": representative["system_context"],
                    "user_utterance": representative["user_utterance"],
                    "label_option_a": group[0]["label"],
                    "annotator_a": group[0]["annotator_id"],
                    "annotation_pass_a": group[0]["annotation_pass"],
                    "label_option_b": group[1]["label"],
                    "annotator_b": group[1]["annotator_id"],
                    "annotation_pass_b": group[1]["annotation_pass"],
                    "recommended_label": final_label,
                    "rationale": adjudication["rationale"],
                    "status": "owner_adjudicated_pending_team_ratification",
                }
            )

        gold_rows.append(
            {
                "instance_id": representative["instance_id"],
                "dialogue_id": representative["dialogue_id"],
                "turn_id": representative["turn_id"],
                "system_context": representative["system_context"],
                "user_utterance": representative["user_utterance"],
                "label": final_label,
            }
        )

    return gold_rows, adjudication_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_distribution_svg(path: Path, counts: Counter[str], total: int) -> None:
    width = 900
    height = 520
    margin_left = 170
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(counts.values())
    scale = plot_height / max_count
    labels = VALID_LABELS
    bar_width = 95
    gap = 35
    chart_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7"/>',
        '<text x="40" y="42" font-family="Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">Final Gold Label Distribution</text>',
        '<text x="40" y="64" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#52606d">Frozen dataset size: '
        + str(total)
        + "</text>",
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#9aa5b1" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#9aa5b1" stroke-width="1"/>',
    ]
    tick_count = 5
    for tick in range(tick_count + 1):
        tick_value = round(max_count * tick / tick_count)
        y = margin_top + plot_height - (tick_value * scale)
        chart_parts.append(
            f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#7b8794" stroke-width="1"/>'
        )
        chart_parts.append(
            f'<text x="{margin_left - 14}" y="{y + 4:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#52606d">{tick_value}</text>'
        )

    x = margin_left + 18
    colors = ["#0b6e4f", "#dd6b20", "#1f77b4", "#8b1e3f", "#6b46c1"]
    for label, color in zip(labels, colors, strict=True):
        count = counts[label]
        bar_height = count * scale
        y = margin_top + plot_height - bar_height
        chart_parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" rx="6"/>'
        )
        chart_parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#102a43">{count}</text>'
        )
        chart_parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#102a43">{label}</text>'
        )
        pct = count / total * 100
        chart_parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 42}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#52606d">{pct:.1f}%</text>'
        )
        x += bar_width + gap

    chart_parts.append("</svg>")
    path.write_text("\n".join(chart_parts))


def render_distribution_png(path: Path, counts: Counter[str], total: int) -> bool:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/4nl3-mplconfig")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    labels = VALID_LABELS
    values = [counts[label] for label in labels]
    colors = ["#0b6e4f", "#dd6b20", "#1f77b4", "#8b1e3f", "#6b46c1"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Final Gold Label Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#d9e2ec")
    ax.set_facecolor("#fffdf7")
    fig.patch.set_facecolor("#fffdf7")

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value}\n({value / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def build_markdown(
    gold_rows: list[dict[str, str]],
    adjudication_rows: list[dict[str, str]],
    counts: Counter[str],
) -> str:
    agreement = json.loads(AGREEMENT_PATH.read_text())
    total = len(gold_rows)
    top_pairs = sorted(
        (
            (count, f"{source} -> {target}")
            for source, inner in agreement["confusion_matrix"].items()
            for target, count in inner.items()
            if source != target and count > 0
        ),
        reverse=True,
    )[:3]
    lines = [
        "# Data Preparation Handoff",
        "",
        "## Summary",
        "",
        "This dataset is derived from MultiWOZ 2.2 and uses one `(previous system turn, current user turn)` pair as the annotation unit.",
        f"The frozen gold dataset contains **{total}** unique instances with one final intent label per `instance_id`.",
        "",
        "## Ground-Truth Policy",
        "",
        "Ground truth was frozen with a single-label policy:",
        "",
        "1. Non-overlap items keep their original initial annotation.",
        "2. Overlap items with matching labels keep the agreed label.",
        "3. Overlap items with disagreement receive an owner adjudication label based on the published annotation guidelines.",
        "",
        f"Overlap agreement before adjudication was **{agreement['percent_agreement'] * 100:.2f}%** with nominal Krippendorff's alpha **{agreement['krippendorff_alpha_nominal']:.4f}**.",
        f"There were **{len(adjudication_rows)}** overlap disagreements requiring adjudication.",
        "",
        "## Final Label Distribution",
        "",
        "| Label | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]
    for label in VALID_LABELS:
        count = counts[label]
        lines.append(f"| {label} | {count} | {count / total * 100:.1f}% |")

    lines.extend(
        [
            "",
            "The class distribution is **imbalanced**: `REQUEST` is the largest class and `CORRECT_CLARIFY` remains the smallest.",
            "",
            "## Disagreement Hotspots",
            "",
            "The most common pre-adjudication disagreement patterns were:",
        ]
    )
    for count, pair in top_pairs:
        lines.append(f"- `{pair}`: {count} cases")

    lines.extend(
        [
            "",
            "These disagreements cluster around turns that mix acceptance, correction, and a new downstream request in the same utterance.",
            "",
            "## Handoff Notes",
            "",
            "- Person B should use `data/processed/final_gold_labels.csv` as the only source of truth for splits and baselines.",
            "- Person C should use the counts in `data/processed/final_label_distribution.csv` and the chart in `docs/competition_launch/final_label_distribution.png` (fallback: `final_label_distribution.svg`).",
            "- The adjudication log in `data/processed/adjudication_log.csv` marks each disputed overlap case as `owner_adjudicated_pending_team_ratification`; do not describe this as group adjudication until teammates sign off.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_annotations()
    gold_rows, adjudication_rows = build_outputs(rows)

    if len(ADJUDICATIONS) != len(adjudication_rows):
        raise ValueError(
            f"Expected {len(ADJUDICATIONS)} adjudicated rows, found {len(adjudication_rows)}."
        )

    counts = Counter(row["label"] for row in gold_rows)
    invalid = [label for label in counts if label not in VALID_LABELS]
    if invalid:
        raise ValueError(f"Unexpected gold labels found: {invalid}")

    if len(gold_rows) != 1050:
        raise ValueError(f"Expected 1050 gold rows, found {len(gold_rows)}")
    if len({row["instance_id"] for row in gold_rows}) != 1050:
        raise ValueError("Gold rows do not have unique instance_id values.")

    write_csv(
        OUTPUT_DIR / "final_gold_labels.csv",
        gold_rows,
        [
            "instance_id",
            "dialogue_id",
            "turn_id",
            "system_context",
            "user_utterance",
            "label",
        ],
    )
    write_csv(
        OUTPUT_DIR / "adjudication_log.csv",
        adjudication_rows,
        [
            "instance_id",
            "dialogue_id",
            "turn_id",
            "system_context",
            "user_utterance",
            "label_option_a",
            "annotator_a",
            "annotation_pass_a",
            "label_option_b",
            "annotator_b",
            "annotation_pass_b",
            "recommended_label",
            "rationale",
            "status",
        ],
    )

    distribution_rows = [
        {
            "label": label,
            "count": str(counts[label]),
            "percentage": f"{counts[label] / len(gold_rows) * 100:.4f}",
        }
        for label in VALID_LABELS
    ]
    write_csv(
        OUTPUT_DIR / "final_label_distribution.csv",
        distribution_rows,
        ["label", "count", "percentage"],
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    render_distribution_svg(DOCS_DIR / "final_label_distribution.svg", counts, len(gold_rows))
    render_distribution_png(DOCS_DIR / "final_label_distribution.png", counts, len(gold_rows))
    (DOCS_DIR / "data_preparation.md").write_text(
        build_markdown(gold_rows, adjudication_rows, counts)
    )


if __name__ == "__main__":
    main()
