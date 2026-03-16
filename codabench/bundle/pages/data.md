# Data

## Source

User turns were sampled from MultiWOZ 2.2 dialogues and filtered to utterances
between 1 and 40 tokens. Each instance includes the user utterance and the
immediately preceding system turn as context. Instances were drawn from the train
split of MultiWOZ 2.2 to avoid overlap with standard evaluation sets.

## What Is a Data Point?

Each data point consists of the following fields:

- **instance_id** — unique identifier for the instance
- **dialogue_id** — unique identifier for the source dialogue
- **turn_id** — turn index within the dialogue
- **system_context** — the immediately preceding system utterance
- **user_utterance** — the user's response (≤40 tokens)
- **label** — the gold-standard intent label

Example:

```
instance_id,dialogue_id,turn_id,system_context,user_utterance,label
mw22_MUL0592.json_12,MUL0592.json,12,"Indeed I can book that for 4 people. Your booking was successful, the total fee is 40.4 GBP payable at the station. Reference number is : 2I1YOWD4","Great that's all that I need, thank you!",SOCIAL
```

## Labeled Examples

| system_context | user_utterance | label |
|---|---|---|
| I have found several restaurants. Do you have a price range in mind? | I would like something in the cheap price range. | `INFORM_CONSTRAINT` |
| The Curry Garden is a cheap Indian restaurant located in the centre. | Can you give me their phone number and address? | `REQUEST` |
| I have booked a table for 2 at 19:00. Is there anything else I can help you with? | No, that is all. Thank you! | `SOCIAL` |
| I found a cheap restaurant called The Eagle. Shall I book a table? | Actually, I wanted a moderately priced place, not cheap. | `CORRECT_CLARIFY` |
| I have booked a taxi for you at 14:00. The contact number is 07700900461. | Yes, that sounds great. Thank you. | `CONFIRM_ACCEPT` |

## Ground Truth Method

Ground truth labels were determined through adjudication. Three annotators each
labeled 350 turns during an initial pass, with a 135-turn overlap subset (45 per
annotator pair) used to measure agreement. Disagreements on the overlap set were
resolved through manual adjudication: each contested instance was reviewed by the
full team and assigned a final label by consensus. Inter-annotator agreement was
computed using Krippendorff's Alpha (nominal) on the overlap subset.

## Dataset Splits

| Split | Instances | Proportion |
|---|---|---|
| Train | 840 | ~80% |
| Validation | 105 | ~10% |
| Test | 105 | ~10% (withheld) |

Splits are stratified by label to preserve class distribution. Train and validation
sets are available for download via the starting kit. Test labels are withheld and
used only for scoring.

## Label Distribution

The dataset contains five intent classes and is imbalanced. REQUEST is the
dominant class, accounting for over 40% of all instances, while CORRECT_CLARIFY
is the rarest at 8%.

```
Label             Count  │
──────────────────────────────────────────────────────
REQUEST           428    │████████████████████████████████████████  40.76%
INFORM_CONSTRAINT 206    │████████████████████                      19.62%
SOCIAL            196    │███████████████████                       18.67%
CONFIRM_ACCEPT    136    │█████████████                             12.95%
CORRECT_CLARIFY    84    │████████                                   8.00%
```

Due to this imbalance, macro F1 is used as the primary evaluation metric rather
than accuracy, as it penalizes models that ignore minority classes like
CORRECT_CLARIFY.