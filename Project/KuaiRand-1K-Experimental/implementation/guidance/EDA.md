# Bounded EDA request format

Write `/workspace/eda_request.json` as:

```json
{"queries": [{"id": "unique-name", "type": "overview"}]}
```

Use at most 20 queries. Available query types:

- `schema`
- `overview`
- `label_by_date`
- `user_history`
- `item_frequency`
- `feedback_correlations`
- `cardinality` with `columns` (1-10 training columns)
- `numeric_quantiles` with `column`
- `top_values` with `column` and `top_k` up to 30
- `cold_item_rate`
- `metadata_overview`
- `sample_train` with `rows` up to 50

Choose questions because their answers would change a modelling decision. Do not
request every query mechanically. The trusted service reads training labels and
label-free validation structure only; it never reads hidden-test labels.
