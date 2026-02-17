# Dataset Specification (Phase 1)

## 1. Source and unit

- Source corpus: MultiWOZ 2.2
- Instance unit: one `(previous system turn, current user turn)` pair.

## 2. File formats

- Primary annotation file: CSV
- Optional mirror export: JSON

## 3. Required schema (fixed)

Use this exact column order in CSV:

1. `instance_id`
2. `dialogue_id`
3. `turn_id`
4. `system_context`
5. `user_utterance`
6. `label`
7. `annotator_id`
8. `annotation_pass`
9. `notes`

## 4. Field definitions

1. `instance_id`
   - String, unique globally.
   - Format: `mw22_<dialogue_id>_<turn_id>`.
2. `dialogue_id`
   - String ID from source dialogue.
3. `turn_id`
   - Integer index of user turn in source dialogue.
4. `system_context`
   - Normalized previous system utterance text.
5. `user_utterance`
   - Normalized current user utterance text.
6. `label`
   - One of:
     - `REQUEST`
     - `INFORM_CONSTRAINT`
     - `CONFIRM_ACCEPT`
     - `CORRECT_CLARIFY`
     - `SOCIAL`
   - Empty only before annotation is completed.
7. `annotator_id`
   - Annotator username/ID.
8. `annotation_pass`
   - `initial` or `reannotation`.
9. `notes`
   - Optional free-text note for edge-case reasoning.

## 5. Filtering defaults for annotation pool

1. Keep only records where user length is 1 to 40 whitespace tokens.
2. Normalize whitespace.
3. Keep punctuation unchanged.
4. Exclude instances with empty system context or empty user text.

## 6. Quality checks before freeze

1. No duplicate `instance_id`.
2. No null `system_context`/`user_utterance`.
3. `label` values restricted to label enum.
4. `annotation_pass` values restricted to `initial|reannotation`.
5. Overlap subset rows duplicated only by design (same `instance_id`, different
   `annotator_id`).
