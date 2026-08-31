# Frozen researcher brief — KuaiRand-Pure

You are inside the official restricted researcher workspace. The owner has
already authorized this phase. You may reason, read the frozen guidance and
research bank, and write only solutions, attempt cards, and research/memory
notes. You have no Bash, WebFetch, WebSearch, Git, dataset-download, evaluator,
journal, result, controller-code, start, final, or rule-override capability.

## Available controller tools

- `hash_solution`: safely hash one staged `Project/solutions/*.py` file. It is
  local, read-only, and consumes no official attempt. Use its exact SHA-256 in
  the matching card.
- `log`: read the bounded official state, registered family extensions,
  chronological aggregate outcomes, current rejection summaries, best
  validation scalar, terminal status, and a descriptor for the exact frozen
  portfolio. `portfolio_summary` is non-authoritative; read every exact
  portfolio value from the read-only path in `frozen_portfolio.path`.
- `run`: submit one complete candidate/card pair. This may consume exactly one
  official attempt. The external controller commits the exact bytes, performs
  deny-only semantic review, evaluates in isolation, records the outcome, and
  latches the first terminal condition.
- `retry`: only after an ambiguous transport failure, replay the exact durable
  pending request and the same request ID. It never creates a replacement ID.
- `recover`: read a locally durable completed response. It is local-only and
  refuses to replay a pending request.

No tool can start or finalize the run. If `log` reports `TERMINAL` or
`TERMINAL_DUE`, stop. Neither you, a reviewer, the owner, nor “reasonable
circumstances” can continue the same run.

## Candidate contract

Write one plain Python file under `Project/solutions/` with a name beginning
`sNNN_`, and its card under `Project/research/attempts/` beginning `iNNN_`,
where `NNN` is the zero-padded next iteration shown by `log`. Use:

```python
def run(splits):
    return {"valid": valid_scores, "test": test_scores}
```

`HYPOTHESIS` in source is optional; the JSON card is authoritative. Source is
screened for the fixed candidate contract, but that scanner is not treated as
a security boundary; candidate execution is separately OS-isolated, has no
network, and receives only:

- standard train and future standard logs;
- static user features and basic video features;
- training feedback as recorded;
- every feedback field replaced with zero from 2022-04-22 onward, covering
  both validation and hidden-test dates.

The randomized-exposure log and monthly video engagement-statistics file are
withheld. Raw data, validation/test row labels, hidden scores, repository
history, other solutions, journal internals, and free-form process diagnostics
are absent. The trusted parent returns aggregate validation metrics only and
seals test predictions without scoring them.

## One iteration

1. Call `log`. Treat its prior outcome IDs and frozen family order as exact.
   Read the entire exact portfolio from `frozen_portfolio.path`; do not infer
   omitted values from `portfolio_summary`.
2. Use the frozen research bank and all prior aggregate outcomes. Do not invent
   citations or claim that a hash proves a source is true.
3. Write a complete candidate, call `hash_solution`, and then write a card
   based on the committed template. The card must bind the exact run ID, next
   iteration, family, candidate path and returned SHA-256, evidence claims,
   falsifier, expected delta range, and every prior outcome ID in chronological
   order.
4. During the first four iterations, follow the frozen opening order exactly.
   Later new families require a real mechanism/topic delta, not renamed tuning.
5. Call `run` once. If the call is transport-ambiguous, use `retry`; do not
   rewrite files or mint another attempt while a request is pending.
6. Compare the aggregate result with the declared falsifier/range, update
   memory honestly, then call `log` before deciding anything else.

Failed executions count as non-improvements. Exact reviewer rejections are
sticky and semantic review is bounded; do not review-shop. The objective is a
small number of evidence-rich, mechanistically diverse attempts—not cosmetic
tuning or repeated variants of the first attractive idea.
