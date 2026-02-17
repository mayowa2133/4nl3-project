# Annotation Guidelines v1

## 1. Annotator task

For each instance, read:

1. `system_context` (previous system utterance)
2. `user_utterance` (current user utterance)

Assign exactly one label that best captures the user's main communicative
intent in the current turn.

## 2. Label set (fixed)

Use only these 5 labels:

1. `REQUEST`
   - User asks for information, options, availability, or asks system to do an
     action.
2. `INFORM_CONSTRAINT`
   - User provides constraints/preferences/details (price, area, time, people,
     day, etc.).
3. `CONFIRM_ACCEPT`
   - User accepts, confirms, or approves a proposal/detail.
4. `CORRECT_CLARIFY`
   - User corrects a wrong detail or clarifies after misunderstanding.
5. `SOCIAL`
   - Greeting, thanks, closing, politeness-only turn with no task content.

## 3. Decision process

Apply this order to reduce ambiguity:

1. If user explicitly rejects/corrects prior system content, label
   `CORRECT_CLARIFY`.
2. Else if the turn is primarily acceptance/confirmation of a proposal, label
   `CONFIRM_ACCEPT`.
3. Else if the user is asking a question/requesting an action, label `REQUEST`.
4. Else if user is mainly supplying preferences/details, label
   `INFORM_CONSTRAINT`.
5. Else if none apply and content is social only, label `SOCIAL`.

## 4. Tie-break rules (single-label policy)

When a turn contains multiple functions, choose the dominant intent with these
rules:

1. Correction beats all other intents.
2. Acceptance beats request/inform when acceptance is explicit.
3. Request beats inform when the user is asking for next system action.
4. Inform beats social when task content is present.
5. Use `SOCIAL` only when there is no task-directed intent.

## 5. Borderline handling

1. "Yes" after a booking proposal -> `CONFIRM_ACCEPT`.
2. "Yes, and make it cheap" -> `INFORM_CONSTRAINT` if new constraint is central.
3. "No, not Tuesday, Thursday" -> `CORRECT_CLARIFY`.
4. "Thanks, can you find one downtown?" -> `REQUEST`.
5. "Hello there" / "Thanks bye" -> `SOCIAL`.

## 6. Worked examples

1. `REQUEST`
   - System: "What area do you prefer?"
   - User: "Do you have anything downtown?"
   - Why: user asks for options; not just giving a constraint.
2. `INFORM_CONSTRAINT`
   - System: "What kind of hotel should I look for?"
   - User: "Cheap, in the north, for two nights."
   - Why: user supplies constraints; no direct action request.
3. `CONFIRM_ACCEPT`
   - System: "I can book the 6:15 train."
   - User: "Yes, please book it."
   - Why: explicit acceptance/approval.
4. `CORRECT_CLARIFY`
   - System: "Booked for Tuesday at 7 pm."
   - User: "No, I said Thursday at 7."
   - Why: correction of prior system misunderstanding.
5. `SOCIAL`
   - System: "Anything else I can help with?"
   - User: "No thanks, bye."
   - Why: closing/politeness only.

## 7. Annotation quality notes

1. Use context to interpret short turns ("yes", "okay", "that works").
2. Do not invent missing details beyond visible context.
3. If uncertain, choose best label by decision order and add a note in the
   optional comment field.
