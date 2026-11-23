# Mentor feedback on project proposal (Saif)

Hey, I'm Saif and I will be your mentor for your final project.

This is a very well thought out project. Going to the register machine task where you know the minimal state is a very clever idea to see what the model retains, something that can't be done in the large scale papers. The main things to sort out are whether your two memory modes stay meaningfully different after training, budget matching, and getting the run count right.

## What's working well

- The sandbox allows you to have precise labels, a tunable difficulty level (N, k), an out-of-distribution length regime, and a known minimal state, making state fidelity an exact measurement.
- By keeping the per-step arithmetic simple, you'll make sure that any failure would be due to loss of memory and not poor arithmetic.
- Your baselines (untrained tail clipping, full context, oracle-state SFT, and chance) are nicely bracketing the RL conditions, and the state corruption test is a good sanity check to make sure the model uses its state.
- The compute plan is consistent, has a lean fallback, and starts with a pilot.

## Things I'd like you to address

- **Tail vs. learned memory.** Under RL, a tail-mode policy could learn to end each chunk by writing out all its registers. At that point the two modes only differ in which end of a span survives truncation. How will you tell them apart after training? Measuring state fidelity on the tail window would show whether the tail policy has learned to write a state.
- **Budget matching.** The full context baseline should have the same generation budget (12 chunks x 256 tokens) and the same number of warm-up and RL steps. Since oracle-state SFT gets supervision the RL runs never see, I'd treat it as a reference point rather than an upper bound.
- **Run count.** For tail and learned memory with RL for four values of m, you have 8 runs per seed, plus full-context RL. Two seeds would mean more than the 6 main runs plus 4 sweep runs that you have in your computation section. List the runs along with their costs in a table. Also make sure to clarify if k is mixed within a run or swept separately.
- **Learned truncation.** If the span of a `<mem>` is larger than m, which end will be preserved, and will the program counter be stored first? It is important for m = 25, where k = 4 is already just enough for the minimal state size.
- **The probe.** Which hidden states does it use, and where, and does it have access to the whole program? Without the program, the final answer cannot be derived from the state alone, so the probe would not actually measure sufficiency.
- **Credit assignment.** The binary final reward assigned to all chunks is punishing the early chunks which have the correct state. This is okay as an initial approach, but if training stalls at large N, consider moving the dense state-correctness reward out of your stretch goals.

## For the milestone, I'd like to see

- The warm-up model reliably following the format and doing the arithmetic right at full context.
- The number of GPU-hours per RL run and the resulting run table.
- Dates for the milestone, poster, and final deliverables.

## General

Don't worry if the research direction drifts, or not great results and you have to pivot because of that. That is part of the process, just make sure you settle on something you think is good, and if anything changes explain why, and then what metrics make this new idea better/stronger or such.
