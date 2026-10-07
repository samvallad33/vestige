# Mattar and Daw's replay rule, tested on an AI agent's record

## The question

When an AI agent is idle and goes back over what it has learned, which memory should it replay first?

## The rule

Mattar and Daw say to replay the memory with the highest Gain times Need. Gain is how much the replay would improve the next decision. Need is how soon the agent is likely to be back at that memory.

## How it maps onto Vestige

Vestige keeps an append-only, hash-chained record of what an agent learns and how those things connect.

| In the paper | In Vestige |
| --- | --- |
| State | One memory |
| Experience | One recorded link between two memories |
| Reward | +1 when a memory is marked helpful, −1 when it is marked wrong |
| Need | Learned from the order the agent touched its memories |
| Gain | How much replaying one link changes what the agent picks next |
| Priority | Gain times Need, highest first |

The rule runs as a test beside the shipped version of Vestige.

## The test

Two agent histories were built for this test. Each one was stopped at 32 points. At every stop, each rule got up to 20 replays. The score counts how many of the agent's next real steps the rule got right, with later steps counting a little less.

Nine rules were compared. They include Gain alone, most recent first, random order and the two orderings Vestige ships today.

## Results

- Gain times Need beat the best ordering Vestige ships today on both tests. It scored 1.13 against 0.07 on the first and 0.48 against 0.16 on the second.
- Against Gain alone it won the first test, 1.13 against 0.57. The second test was too close to call.
- Need alone was better at predicting which memories the agent touched next than the recall score Vestige uses today. It won the second test, 0.64 against 0.52, and the first was too close to call.

The two questions fixed in advance each got one clear win and one result too close to call. Neither came out against the rule.

## What comes next

1. Run it on a real agent's history.
2. Replay more than one step at a time, to look at forward and reverse sequences.
3. Record what the agent looks up as well as what it writes down, for a truer read on Need.

## The fine print

The full protocol was committed before the results and has not changed since. Every number on this page comes from the results table.

- [Full protocol](MATTAR-EVB-PREREGISTRATION.md)
- [Full results table](results/mattar-evb-v1.md)

Paper: Mattar, M. G. and Daw, N. D., 2018. Prioritized memory access explains planning and hippocampal replay. Nature Neuroscience 21, 1609 to 1617.
