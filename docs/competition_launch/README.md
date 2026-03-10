# Competition Launch Handoff

This folder is the canonical handoff point for the Competition Launch deliverables produced by Person A.

## What Was Generated

- `data/processed/final_gold_labels.csv`
  - Frozen gold dataset with exactly one final label per `instance_id`.
- `data/processed/adjudication_log.csv`
  - The 40 disagreement cases from overlap review, including the recommended final label and rationale.
- `data/processed/final_label_distribution.csv`
  - Final counts and percentages for each label in the frozen dataset.
- `docs/competition_launch/final_label_distribution.png`
  - Slide-ready chart for the final label distribution.
- `docs/competition_launch/final_label_distribution.svg`
  - Vector version of the same chart.
- `docs/competition_launch/data_preparation.md`
  - Reusable Step 2 write-up for Codabench, the report, and the slide.

## Instructions For Person B

Use `data/processed/final_gold_labels.csv` as the only source of truth for:

- train/validation/test splitting
- baseline training
- validation and test labels
- any published metric tables

Do not derive splits or baselines from `annotations_labeled_long.csv` or any pre-freeze file.

The label set is fixed to:

1. `REQUEST`
2. `INFORM_CONSTRAINT`
3. `CONFIRM_ACCEPT`
4. `CORRECT_CLARIFY`
5. `SOCIAL`

## Instructions For Person C

Use these files directly:

- `docs/competition_launch/data_preparation.md`
  - Copy/adapt this text into the Codabench data section and any report text.
- `data/processed/final_label_distribution.csv`
  - Source of truth for counts and percentages in tables.
- `docs/competition_launch/final_label_distribution.png`
  - Preferred chart for Google Slides.
- `docs/competition_launch/final_label_distribution.svg`
  - Fallback chart for documents that prefer vector graphics.

Do not describe the disagreement resolution as full group adjudication unless the other teammates explicitly sign off on the recommended labels in `data/processed/adjudication_log.csv`.

## How To Regenerate

Run:

```bash
python3 scripts/freeze_gold_labels.py
```

That command rebuilds:

- `final_gold_labels.csv`
- `adjudication_log.csv`
- `final_label_distribution.csv`
- `final_label_distribution.png`
- `final_label_distribution.svg`
- `data_preparation.md`

## Current Frozen Counts

- `REQUEST`: 428
- `INFORM_CONSTRAINT`: 206
- `CONFIRM_ACCEPT`: 136
- `CORRECT_CLARIFY`: 84
- `SOCIAL`: 196

Total frozen instances: `1050`
