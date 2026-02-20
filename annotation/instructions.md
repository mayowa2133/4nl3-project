# Annotation Interface Instructions (Excel Workflow)

This file is sufficient for day-to-day annotation in Excel.

## 1. Files to Use

1. Keep CSV files in `data/processed/` as source-of-truth assignment data.
2. Annotate only your assigned workbook:
   - `annotation/excel/adesanym_annotation.xlsx`
   - `annotation/excel/zajkeskn_annotation.xlsx`
   - `annotation/excel/dhiraajd_annotation.xlsx`
3. Use both sheets in this order:
   - `initial` first
   - `reannotation` second
4. `reannotation` includes `source_annotator_id` to show who owned the row in
   the initial pass.

## 2. How to Annotate Each Row (Exact Procedure)

1. Read `system_context`.
2. Read `user_utterance`.
3. Choose exactly one label from the `label` dropdown using Section 3
   (Decision Order).
4. If the turn is mixed/unclear, still choose one best label and add a concise
   rationale in `notes`.
5. Do not leave any `label` blank.
6. Do not edit `annotator_id`, `annotation_pass`, or `source_annotator_id`.
7. Do not reorder rows or rename sheets.

## 3. Labeling Decision Order (Use This Every Row)

Pick the user's dominant intent (the main intent that should drive the next
system response) using this order:

1. `CORRECT_CLARIFY`
2. `CONFIRM_ACCEPT`
3. `REQUEST`
4. `INFORM_CONSTRAINT`
5. `SOCIAL`

## 4. Tie-Break Rules (When Multiple Intents Appear)

1. Correction beats all other intents.
2. Explicit acceptance beats request/inform.
3. Request beats inform when the user asks for the next system action.
4. Inform beats social when any task content is present.
5. Use social only when there is no task-directed intent.

## 5. Allowed Label Values

- `REQUEST`: asks for information/options or asks the system to do an action.
- `INFORM_CONSTRAINT`: provides preferences/details (price, area, day, time,
  people, etc.).
- `CONFIRM_ACCEPT`: explicitly accepts/confirms a proposal.
- `CORRECT_CLARIFY`: corrects wrong details or clarifies after misunderstanding.
- `SOCIAL`: greeting/thanks/closing with no task intent.

`annotation_pass` values are:

- `initial`
- `reannotation`

## 6. Quick Examples

1. "No, not Tuesday, Thursday." -> `CORRECT_CLARIFY`
2. "Yes, please book it." -> `CONFIRM_ACCEPT`
3. "Can you find one downtown?" -> `REQUEST`
4. "Cheap in the north for two nights." -> `INFORM_CONSTRAINT`
5. "Thanks, bye." -> `SOCIAL`
6. "Yes, and make it cheap." -> `INFORM_CONSTRAINT` when the new constraint is
   central.
7. "Thanks, can you find one downtown?" -> `REQUEST` (task request beats social
   politeness).

## 7. Saving and Progress Policy

1. Save every 10 to 20 rows.
2. Keep one backup copy at midpoint and endpoint.
3. Do not reorder rows or rename sheets.
4. Keep `annotator_id` and `annotation_pass` as assigned.
5. Do not change `source_annotator_id`; it is metadata for agreement analysis.

## 8. Completion Checklist (Must Pass)

1. Zero blank labels in assigned rows.
2. Every label is one of the 5 allowed enum values.
3. `annotator_id`, `annotation_pass`, and `source_annotator_id` are unchanged.
4. `notes` are present only for ambiguous/unclear rows.

## 9. Full Reference

For full rationale and additional examples, see:
`docs/phase1/annotation-guidelines-v1.md`.
