# Phase 1 Runbook

## 1. Download MultiWOZ 2.2

```bash
python3 scripts/download_multiwoz22.py \
  --raw-dir data/raw/multiwoz22 \
  --combined-out data/raw/multiwoz22.json \
  --manifest-out data/raw/multiwoz22_manifest.json
```

## 2. Build annotation pool

```bash
python3 scripts/extract_multiwoz22_pool.py \
  --input data/raw/multiwoz22.json \
  --output data/processed/annotation_pool.csv \
  --max-instances 1200 \
  --min-tokens 1 \
  --max-tokens 40 \
  --seed 13
```

## 3. Create assignment files

```bash
python3 scripts/create_assignment_plan.py \
  --pool data/processed/annotation_pool.csv \
  --annotators adesanym,zajkeskn,dhiraajd \
  --per-annotator 350 \
  --overlap-total 135 \
  --seed 13
```

## 4. Build dropdown-enabled Excel workbooks

```bash
python3 scripts/build_excel_annotation_workbooks.py \
  --initial-csv data/processed/assignments_initial.csv \
  --reannotation-csv data/processed/assignments_reannotation.csv \
  --out-dir annotation/excel
```

## 5. Start annotation

1. Use `annotation/instructions.md`.
2. Each annotator works in their own file:
   - `annotation/excel/adesanym_annotation.xlsx`
   - `annotation/excel/zajkeskn_annotation.xlsx`
   - `annotation/excel/dhiraajd_annotation.xlsx`
3. Save frequently.
