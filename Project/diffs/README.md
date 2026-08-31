# Per-iteration code diffs

These patches are convenience views generated from the adjacent files in
`Project/solutions/` during final packaging:

- `iteration-01.patch` adds the official baseline wrapper from an empty file.
- `iteration-02.patch` through `iteration-15.patch` compare each candidate with
  the immediately preceding candidate.

They make the code-change requirement quick to inspect, but they are not a
replacement for the official run record. `Project/results/JOURNAL.jsonl` is the
source of truth: it was written during the run and embeds each iteration's
hypothesis, exact full source, source SHA-256, metrics, error state, resource
estimate, override state, and sealed-prediction hash.

The patches use zero lines of surrounding context. This keeps the nested diff
artifacts compact and avoids representing unchanged blank context as trailing
whitespace; every changed line is still present.

The patches are expected to be large because each candidate is a standalone,
runnable experiment rather than a mutable shared model file.
