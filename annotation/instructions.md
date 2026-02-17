# Annotation Interface Instructions (Excel Workflow)

## 1. Files to use

1. Keep CSV files in `data/processed/` as source-of-truth assignment data.
2. Annotate in the generated Excel workbook assigned to your annotator ID:
   - `annotation/excel/adesanym_annotation.xlsx`
   - `annotation/excel/zajkeskn_annotation.xlsx`
   - `annotation/excel/dhiraajd_annotation.xlsx`
3. Use both sheets in your workbook:
   - `initial`
   - `reannotation`
4. `reannotation` includes `source_annotator_id` so you can see who originally
   owned that row in the initial pass.

## 2. How to annotate each row

1. Read `system_context`.
2. Read `user_utterance`.
3. Apply `docs/phase1/annotation-guidelines-v1.md`.
4. Select exactly one value from the `label` dropdown.
5. Add edge-case rationale in `notes` when uncertain.
6. Avoid editing assignment fields unless you are intentionally fixing an issue.

## 3. Allowed label values

- `REQUEST`
- `INFORM_CONSTRAINT`
- `CONFIRM_ACCEPT`
- `CORRECT_CLARIFY`
- `SOCIAL`

`annotation_pass` is controlled by dropdown with:

- `initial`
- `reannotation`

## 4. Saving and progress policy

1. Save every 10 to 20 rows.
2. Keep one backup copy at midpoint and endpoint.
3. Do not reorder rows or rename sheets.
4. Keep `annotator_id` and `annotation_pass` as assigned.
5. Do not change `source_annotator_id`; it is metadata for agreement analysis.

## 5. Completion checklist per annotator

1. No blank labels.
2. Labels are valid enum values only.
3. `annotator_id` and `annotation_pass` remain unchanged from assigned values.
4. Notes added for unclear edge cases.
