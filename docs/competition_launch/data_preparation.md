# Data Preparation Handoff

## Summary

This dataset is derived from MultiWOZ 2.2 and uses one `(previous system turn, current user turn)` pair as the annotation unit.
The frozen gold dataset contains **1050** unique instances with one final intent label per `instance_id`.

## Ground-Truth Policy

Ground truth was frozen with a single-label policy:

1. Non-overlap items keep their original initial annotation.
2. Overlap items with matching labels keep the agreed label.
3. Overlap items with disagreement receive an owner adjudication label based on the published annotation guidelines.

Overlap agreement before adjudication was **70.37%** with nominal Krippendorff's alpha **0.5997**.
There were **40** overlap disagreements requiring adjudication.

## Final Label Distribution

| Label | Count | Percentage |
| --- | ---: | ---: |
| REQUEST | 428 | 40.8% |
| INFORM_CONSTRAINT | 206 | 19.6% |
| CONFIRM_ACCEPT | 136 | 13.0% |
| CORRECT_CLARIFY | 84 | 8.0% |
| SOCIAL | 196 | 18.7% |

The class distribution is **imbalanced**: `REQUEST` is the largest class and `CORRECT_CLARIFY` remains the smallest.

## Disagreement Hotspots

The most common pre-adjudication disagreement patterns were:
- `REQUEST -> CONFIRM_ACCEPT`: 8 cases
- `INFORM_CONSTRAINT -> REQUEST`: 6 cases
- `REQUEST -> CORRECT_CLARIFY`: 4 cases

These disagreements cluster around turns that mix acceptance, correction, and a new downstream request in the same utterance.

## Handoff Notes

- Person B should use `data/processed/final_gold_labels.csv` as the only source of truth for splits and baselines.
- Person C should use the counts in `data/processed/final_label_distribution.csv` and the chart in `docs/competition_launch/final_label_distribution.png` (fallback: `final_label_distribution.svg`).
- The adjudication log in `data/processed/adjudication_log.csv` marks each disputed overlap case as `owner_adjudicated_pending_team_ratification`; do not describe this as group adjudication until teammates sign off.
