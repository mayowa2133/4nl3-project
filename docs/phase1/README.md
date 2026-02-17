# Phase 1 Foundation (Step 1)

This folder contains the Phase 1 setup artifacts for the COMPSCI 4NL3
annotation assignment.

## Files

- `annotation-guidelines-v1.md`
  - Fixed label set and deterministic tie-break rules.
- `dataset-spec.md`
  - Dataset schema, field requirements, and data quality checks.
- `assignment-overlap-plan.md`
  - Team split and reannotation overlap plan.
- `phase1-checklist.md`
  - Completion checklist for Phase 1.

## Related folders

- `scripts/`
  - `download_multiwoz22.py`: download MultiWOZ 2.2 shards and create one
    combined JSON file.
  - `extract_multiwoz22_pool.py`: build annotation pool CSV from MultiWOZ 2.2.
  - `create_assignment_plan.py`: split pool into per-annotator assignments with
    overlap.
  - `build_excel_annotation_workbooks.py`: generate one dropdown-enabled Excel
    workbook per annotator.
- `annotation/`
  - `annotation_template.csv`: spreadsheet template for manual labeling.
  - `instructions.md`: Excel annotation workflow and quality checks.
  - `excel/*.xlsx`: annotator workbooks with label dropdowns and
    `source_annotator_id` metadata for reannotation rows.
