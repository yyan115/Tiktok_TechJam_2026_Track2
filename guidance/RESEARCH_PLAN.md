# Research plan format

Write `/workspace/research_plan.json` with this structure:

```json
{
  "problem_summary": "...",
  "eda_observations": [
    {"result_id": "...", "fact": "measured fact", "implication": "decision it changes"}
  ],
  "sources": [
    {
      "url": "https://...",
      "title": "...",
      "claim": "what the source actually supports",
      "limitations": "why it may not transfer",
      "decision": "specific local design choice"
    }
  ],
  "portfolio": [
    {
      "family": "...",
      "mechanism": "...",
      "evidence": ["EDA/source references"],
      "falsifier": "...",
      "resource_plan": "..."
    }
  ],
  "first_attempt": "family and reason"
}
```

Require at least three concrete EDA observations, three primary/official sources,
and three genuinely different candidate families. This is a research portfolio,
not a requirement to spend official attempts on every family. Promote only ideas
that remain strong after evidence and resource review.
