# 4NL3 Project: Turn-Level User Intent Classification (MultiWOZ 2.2)

## Team

1. Mayowa Adesanya (`adesanym`)
2. Nicholas Zajkeskovic (`zajkeskn`)
3. Divij Dhiraaj (`dhiraajd`)

## Project Goal

Build a manually annotated dataset and baseline workflow for classifying the
main intent of a user turn in task-oriented dialogue using one-turn system
context.

## Final Model Ownership (Start Here)

For the final report stage, each teammate should work in only their own model
file:

1. Mayowa: `models/mayowa_model.py`
2. Nicholas: `models/nicholas_model.py`
3. Divij: `models/divij_model.py`

These starter files already use the shared dataset and split scaffold. Keep the
following rules fixed across all teammate models:

1. Use `data/processed/final_gold_labels.csv` as the only modeling dataset.
2. Keep the dialogue-level 80/10/10 split with `random_state=42`.
3. Report validation `accuracy` and `macro F1`.
4. Do not change another teammate's model file unless the group explicitly
   agrees.

### Shared Baseline Files

These are shared comparison rows for the final report table and should not be
used as personal model files:

1. `Baselines/Simple/random+majority.py`
2. `Baselines/Trained/trained.py`

### Report Files

If you are helping with the final write-up, use these files:

1. Final report template: `Final Project TeX/main.tex`
2. Existing working draft: `docs/report/final_report.tex`

## Competition Launch Handoff (Start Here)

If you are working on the Competition Launch deliverables, start with:

1. `docs/competition_launch/README.md`
2. `docs/competition_launch/data_preparation.md`
3. `data/processed/final_gold_labels.csv`

### Person B: Baselines and Splits

Use `data/processed/final_gold_labels.csv` as the only source of truth for:

1. train/validation/test splits
2. baseline training
3. validation and test labels
4. any published metric tables

Do not use `annotations_labeled_long.csv` or any pre-freeze annotation file for
modeling or reporting.

### Person C: Codabench and Slide

Use these files directly:

1. `docs/competition_launch/data_preparation.md` for Step 2 write-up text
2. `data/processed/final_label_distribution.csv` for counts and percentages
3. `docs/competition_launch/final_label_distribution.png` for the slide chart
4. `data/processed/adjudication_log.csv` for disagreement-case reference

### Regenerate Person A Artifacts

Run:

```bash
python3 scripts/freeze_gold_labels.py
```

This rebuilds the frozen gold labels, adjudication log, distribution table, and
chart assets.

## Current Status (February 17, 2026)

1. Phase 1 setup is complete.
2. MultiWOZ 2.2 was downloaded and preprocessed.
3. Annotation pool generated: `data/processed/annotation_pool.csv` (1200 rows).
4. Initial assignments generated: `data/processed/assignments_initial.csv`
   (1050 rows, 350 per annotator).
5. Reannotation assignments generated:
   `data/processed/assignments_reannotation.csv` (135 rows, 45 per annotator).
6. Excel annotation workbooks generated in `annotation/excel/`.

## What To Do Now (Phase 2)

Each teammate should annotate in their own workbook only:

1. `annotation/excel/adesanym_annotation.xlsx`
2. `annotation/excel/zajkeskn_annotation.xlsx`
3. `annotation/excel/dhiraajd_annotation.xlsx`

Within your workbook:

1. Label rows in `initial` first.
2. Label rows in `reannotation` second.
3. Use only dropdown labels in the `label` column.
4. Add comments for unclear cases in `notes`.
5. Do not change `annotator_id`, `annotation_pass`, or `source_annotator_id`.

## Reannotation Mapping

1. `adesanym` reannotates rows originally from `dhiraajd`
2. `zajkeskn` reannotates rows originally from `adesanym`
3. `dhiraajd` reannotates rows originally from `zajkeskn`

The `source_annotator_id` column in the `reannotation` sheet shows the original
owner of each row.

## Label Set

1. `REQUEST`
2. `INFORM_CONSTRAINT`
3. `CONFIRM_ACCEPT`
4. `CORRECT_CLARIFY`
5. `SOCIAL`

Full decision rules are in `docs/phase1/annotation-guidelines-v1.md`.

## Key Files

1. Phase 1 runbook: `docs/phase1/runbook.md`
2. Guidelines: `docs/phase1/annotation-guidelines-v1.md`
3. Assignment plan: `docs/phase1/assignment-overlap-plan.md`
4. Annotation instructions: `annotation/instructions.md`
5. Phase checklist: `docs/phase1/phase1-checklist.md`

## If You Need To Regenerate Files

Install dependency first (needed for workbook generation):

```bash
python3 -m pip install openpyxl
```

Regenerate data + assignments + workbooks:

```bash
python3 scripts/download_multiwoz22.py \
  --raw-dir data/raw/multiwoz22 \
  --combined-out data/raw/multiwoz22.json \
  --manifest-out data/raw/multiwoz22_manifest.json

python3 scripts/extract_multiwoz22_pool.py \
  --input data/raw/multiwoz22.json \
  --output data/processed/annotation_pool.csv \
  --max-instances 1200 \
  --min-tokens 1 \
  --max-tokens 40 \
  --seed 13

python3 scripts/create_assignment_plan.py \
  --pool data/processed/annotation_pool.csv \
  --annotators adesanym,zajkeskn,dhiraajd \
  --per-annotator 350 \
  --overlap-total 135 \
  --seed 13

python3 scripts/build_excel_annotation_workbooks.py \
  --initial-csv data/processed/assignments_initial.csv \
  --reannotation-csv data/processed/assignments_reannotation.csv \
  --out-dir annotation/excel
```

## Definition of Done for Phase 2

1. Every annotator completes both sheets in their workbook.
2. No blank labels in assigned rows.
3. Team reviews disagreement cases and refines guideline text if needed.
4. Project is ready for agreement computation and report write-up.

## Compute Agreement (Phase 4.1)

Use the completed Excel workbooks to compile all labels and compute nominal
Krippendorff's Alpha on the overlap subset.

Run:

```bash
./.venv/bin/python scripts/compute_agreement.py
```

Outputs:

1. `data/processed/annotations_labeled_long.csv`
2. `data/processed/annotations_overlap_pairs.csv`
3. `data/processed/agreement_summary.json`

`agreement_summary.json` contains:

1. `initial_rows`, `reannotation_rows`, `overlap_rows`
2. `percent_agreement`
3. `krippendorff_alpha_nominal`
4. Label distributions and confusion matrix for overlap labels
