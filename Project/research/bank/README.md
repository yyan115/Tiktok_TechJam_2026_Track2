# Frozen research bank

Prepare this directory **before** `start-run`. The official controller expects
`catalog.json` plus one or more flat Markdown notes under `notes/`. At run
start it verifies that every byte is committed at `HEAD`, resolves the exact
cited line ranges, and freezes the complete descriptor. During the official
run this directory is mounted read-only for the researcher.

Use `catalog.template.json` as the shape reference. Replace every placeholder,
compute each note's lowercase SHA-256 from its exact UTF-8/LF bytes, and save
the finished file as `catalog.json`. The finished catalog must contain 1–256
claims and may reference at most 64 notes.

Hashes establish which note bytes were cited; they do not establish that the
note's claim is true. Each note should therefore record the primary source,
URL or publication identifier, access date, precise supported claim, caveats,
and enough quoted/paraphrased context for the independent reviewer to assess
relevance.
