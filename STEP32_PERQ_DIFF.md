# Per-question diff — Step 32 vs bd2693e2

Compared the Step 32 bench to the bd2693e2 baseline question by question.

LME metrics: S@1, S@5, S@10, T@1, T@5, T@10, Turn All@5.
LoCoMo metrics: S@1, S@5, T@1, T@5, T@10.
BEAM metrics: T@1, T@5, T@10, All@5, S@1, S@5, Session All@5.

Losses: 0. The only flips are the nine Turn@1 gains below. Top-5 membership is unchanged on each; slot 1 moves inside that window. Session metrics do not move.

Slot 1 before is the bd2693e2 head. Slot 1 after is the turn `retrieval_bench` returned for that question.

## Opener arm

A question at slot 1 yields to an earlier-or-equal session opener that posts the rarest missing surface lexeme.

| bench | question | slot 1 before | slot 1 after |
|---|---|---|---|
| LME `c4a1ceb8` (multi-session) | How many different types of citrus fruits have I used in my cocktail recipes? | `answer_56d02cab_3#2` | `answer_56d02cab_3#0` |
| LoCoMo `5:13` (Temporal) | When did Audrey move to a new place? | `D9:17` | `D9:1` |
| BEAM `9:temporal_reasoning:1` | How many days are there between my meeting with Professor Danielle to review my draft and my mock interview with her? | `154` | `44` |
| BEAM `11:knowledge_update:0` | When is the webinar on AI ethics in hiring scheduled to take place? | `84` | `70` |
| BEAM `20:temporal_reasoning:1` | How many days do I have between my meeting with Ashlee and the patent response deadline? | `154` | `152` |

## Echo arm

The reply at slot 1 yields to the declarative turn it answers.

| bench | question | slot 1 before | slot 1 after |
|---|---|---|---|
| LoCoMo `2:106` (Single-Hop) | What important values does John want to teach his kids through adopting a rescue dog? | `D17:12` | `D17:11` |
| LoCoMo `3:143` (Single-Hop) | How did Joanna feel when someone wrote her a letter after reading her blog post? | `D18:6` | `D18:5` |
| LoCoMo `9:147` (Single-Hop) | What tools does Calvin use to boost his motivation for music? | `D28:35` | `D28:34` |
| LoCoMo `6:25` (Open-Domain) | Are John and James fans of the same football team? | `D13:13` | `D13:12` |
