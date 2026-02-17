# Assignment and Overlap Plan (Group 13)

## 1. Annotators

- `adesanym` (Mayowa)
- `zajkeskn` (Nicholas)
- `dhiraajd` (Divij)

## 2. Initial annotation target

- Minimum per member: 1 hour
- Recommended throughput baseline: about 300 instances/hour
- Team unique target for Step 1: 1000 to 1200 unique instances

Default allocation for 1050 unique instances:

- `adesanym`: 350 initial instances
- `zajkeskn`: 350 initial instances
- `dhiraajd`: 350 initial instances

## 3. Reannotation overlap target

Course guidance: overlap should be about 15% of one-hour-equivalent annotation
volume.

For 3 annotators at roughly 300/hour each:

- One-hour-equivalent total = 900
- 15% overlap target = 135 duplicate annotations

Recommended plan:

- 45 instances originally by `adesanym` reannotated by `zajkeskn`
- 45 instances originally by `zajkeskn` reannotated by `dhiraajd`
- 45 instances originally by `dhiraajd` reannotated by `adesanym`

## 4. Assignment rules

1. No annotator reannotates their own initial items.
2. Keep overlap distribution balanced across label space when possible.
3. Use frozen guideline version for both passes.
4. Do not change initial labels after seeing reannotation labels.
5. Log uncertain cases in `notes`.

## 5. Output files

1. `data/processed/annotation_pool.csv`
2. `data/processed/assignments_initial.csv`
3. `data/processed/assignments_reannotation.csv`
4. `data/processed/annotations_combined.csv`
