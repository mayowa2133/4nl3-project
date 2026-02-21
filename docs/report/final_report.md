# Group 13 Report: Turn-Level User Intent Annotation (MultiWOZ 2.2)

## 1. Group Number and Members

- Group Number: 13
- Members:
  - Mayowa Adesanya (`adesanym`) — `400233110`
  - Nicholas Zajkeskovic (`zajkeskn`) — `400402381`
  - Divij Dhiraaj (`dhiraajd`) — `400360225`

## 2. Annotation Instructions (Detailed)

### 2.1 Task definition

For each instance, annotators read:

1. `system_context` (previous system turn)
2. `user_utterance` (current user turn)

Annotators assign exactly one dominant intent label to the user turn.

### 2.2 Label schema

The label set is fixed to five classes:

1. `REQUEST`:
- User asks for information/options or asks the system to perform an action.

2. `INFORM_CONSTRAINT`:
- User provides constraints or preferences (price, area, time, number of people, etc.).

3. `CONFIRM_ACCEPT`:
- User explicitly accepts/confirms a proposal.

4. `CORRECT_CLARIFY`:
- User corrects earlier details or clarifies a misunderstanding.

5. `SOCIAL`:
- Greeting/thanks/closing with no task-directed intent.

### 2.3 Operational annotation procedure

Each annotator used their assigned workbook:

- `annotation/excel/adesanym_annotation.xlsx`
- `annotation/excel/zajkeskn_annotation.xlsx`
- `annotation/excel/dhiraajd_annotation.xlsx`

Each workbook contains:

- `initial` sheet (primary labels)
- `reannotation` sheet (secondary overlap labels)

Procedure per row:

1. Read context and utterance.
2. Apply decision order below.
3. Select one label from dropdown.
4. If ambiguous, keep one label and add rationale in `notes`.
5. Do not edit `annotator_id`, `annotation_pass`, `source_annotator_id`.
6. Do not reorder rows or rename sheets.

### 2.4 Decision order and tie-break policy

Primary decision order:

1. `CORRECT_CLARIFY`
2. `CONFIRM_ACCEPT`
3. `REQUEST`
4. `INFORM_CONSTRAINT`
5. `SOCIAL`

Tie-break policy for mixed-intent turns:

1. Correction beats all.
2. Explicit acceptance beats request/inform.
3. Request beats inform when next action is being asked.
4. Inform beats social when task content exists.
5. Social is used only when no task intent remains.

### 2.5 Quality controls

- Zero blank labels in assigned rows.
- Labels restricted to valid enum.
- Metadata columns unchanged.
- Notes reserved for unclear edge cases.

## 3. Dataset Description and Sensitive-Content Disclaimer

### 3.1 Data source and provenance

Our dataset is built from **MultiWOZ 2.2** task-oriented dialogues.

Source provenance:

- Dataset project (GitHub): [https://github.com/budzianowski/multiwoz](https://github.com/budzianowski/multiwoz)
- Raw files used by downloader: [https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2](https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2)

### 3.2 Unit of annotation

One instance is a pair:

- previous system turn (`system_context`)
- current user turn (`user_utterance`)

### 3.3 Final annotated data volume

- Initial (unique) annotations: 1050
- Reannotation overlap: 135
- Total label entries: 1185
- Blank labels after completion: 0

### 3.4 Sensitive-content disclaimer

The corpus contains natural user language from dialogue systems. Although primarily benign, it may include:

- rude or insulting phrasing,
- abrupt conversational tone shifts,
- potentially offensive wording in isolated turns.

This dataset is used strictly for academic annotation and model-analysis purposes.

## 4. Interesting Data Points, Estimates, and Agreement Metric

### 4.1 Required estimates from annotation stage

- Total unique instances annotated: 1050
- Total label operations including overlap: 1185

Timing estimate (based on agreed 1-hour baseline per annotator):

- Each annotator completed ~350 initial labels in 60 minutes.
- Estimated average time per initial instance:
  - `60 * 60 / 350 = 10.29` seconds per instance.

### 4.2 Agreement metric selection

Because overlap was distributed among **3 annotators** (not a single fixed pair), we used:

- **Krippendorff's Alpha (nominal)** on overlap instances.

Computed values:

- Overlap instances: 135
- Percent agreement: 0.7037
- Krippendorff's Alpha: 0.5997

Interpretation:

- Reliability is **moderate** (usable for analysis, but not yet strong).
- Main disagreements occur near boundaries between `REQUEST`, `CONFIRM_ACCEPT`, and `CORRECT_CLARIFY`.

### 4.3 Three interesting data points

#### Example 1: Hostility + task redirection in one turn

- `instance_id`: `mw22_PMUL4392.json_2`
- User: "uhm, no. that's your job, weirdo. also, i need a train."
- Labels observed:
  - Original: `REQUEST`
  - Reannotation: `CORRECT_CLARIFY`

Why interesting:

- The turn combines social hostility, rejection/correction, and a new task request.
- It exposes ambiguity around whether correction intent or request intent is dominant.

#### Example 2: "instead" ambiguity (correction vs request)

- `instance_id`: `mw22_MUL1543.json_6`
- User: "Can you help me find a train going to Stevenage and leaving Thursday instead?"
- Labels observed:
  - Original: `REQUEST`
  - Reannotation: `CORRECT_CLARIFY`

Why interesting:

- The phrase "instead" can signal correction of prior constraints while still being phrased as a request.
- This is a core disagreement pattern in dialogue-act annotation.

#### Example 3: Social preface + new actionable intent

- `instance_id`: `mw22_PMUL0719.json_12`
- User: "Thank you. I also want a train for friday, from broxbourne to cambridge"
- Labels observed:
  - Original: `INFORM_CONSTRAINT`
  - Reannotation: `REQUEST`

Why interesting:

- The utterance starts socially but contains a fresh task objective.
- Distinguishing constraint-provision from direct request remains non-trivial.

### 4.4 Disagreement concentration

Most frequent overlap disagreements:

1. `REQUEST` -> `CONFIRM_ACCEPT` (8)
2. `INFORM_CONSTRAINT` -> `REQUEST` (6)
3. `REQUEST` -> `CORRECT_CLARIFY` (4)

Per-original-label agreement rates:

- `REQUEST`: 72.7%
- `INFORM_CONSTRAINT`: 69.0%
- `CONFIRM_ACCEPT`: 52.6%
- `CORRECT_CLARIFY`: 25.0%
- `SOCIAL`: 95.8%

This suggests the biggest boundary problem is pragmatic action intent vs correction/acceptance intent, while pure social turns are highly consistent.

## 5. Required Reflection Questions

### (a) What did you learn about the task from doing the annotation?

The core learning is that intent labeling is less about isolated keywords and more about discourse function in context. Short responses ("yes", "okay", "thanks") become labelable only when interpreted against the previous system turn. We also learned that deterministic tie-break rules significantly improve consistency when turns contain multiple functions.

### (b) What challenges do you expect models to face when learning from your data?

Expected model challenges:

1. Mixed-intent utterances (acceptance + new request in one turn).
2. Pragmatic variation and informal wording (typos, slang, terse responses).
3. Context dependence for short confirmations/corrections.
4. Class-boundary overlap between `REQUEST`, `INFORM_CONSTRAINT`, and `CORRECT_CLARIFY`.
5. Relatively fewer examples of `CORRECT_CLARIFY`, making that class harder to learn robustly.

### (c) What surprising things did you observe in the data?

Three notable surprises:

1. Very high consistency for `SOCIAL` despite stylistic variability.
2. Low consistency for `CORRECT_CLARIFY`, even with explicit tie-break rules.
3. Frequent pragmatic shifts where one utterance both continues and redirects task flow.

### (d) Which features do you expect to be useful?

Useful features likely include:

1. Lexical/action cues ("book", "find", "need", "instead", "no").
2. Confirmation/rejection markers ("yes", "no", "that works").
3. Constraint expressions (time, day, location, number of people).
4. Sequence/context features linking user turn to previous system question/proposal.
5. Intent transition history within a dialogue.

### (e) Describe the mental model you came up with to do the task.

Our annotation mental model was a decision ladder:

1. Check for explicit correction of prior content.
2. If not correction, check for explicit acceptance/confirmation.
3. If not acceptance, check whether the user is requesting next action.
4. Else check if user is supplying constraints/details.
5. Use social only for non-task utterances.

For mixed cases, we applied tie-break rules and wrote notes for review.

### (f) Was there anything unclear about the instructions? How would you improve them?

Initially, high-level instructions were workable but not fully self-contained for ambiguous turns. The most helpful improvements were:

1. Explicit decision order in the instructions document.
2. Explicit tie-break policy for mixed intents.
3. Compact worked examples tied to edge cases.
4. Explicit statement that only label/notes fields should be edited.

These changes made annotation faster and reduced process-level disagreements.

## 6. Conclusion

The team successfully produced a complete annotated dataset with required overlap and computed reliability metrics. Agreement is moderate (alpha = 0.5997), with strongest consistency in clearly social turns and weakest consistency at correction/request boundaries. The dataset is suitable for baseline modeling and detailed error analysis, and the identified disagreement patterns directly inform future guideline refinement.
