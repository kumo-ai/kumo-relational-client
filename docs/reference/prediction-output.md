---
title: "NVIDIA Nemotron Predict Client Prediction Output"
description: "Reference for the shape and columns of the DataFrame returned by Nemotron Relational predictions, which vary by task type."
template-library-version: "1.0.0"
---

# NVIDIA Nemotron Predict Client Prediction Output

Both `predict()` and `predict_task()` return a `pandas.DataFrame`, but **the
number of rows and the set of columns depend on the task type**. Only binary
classification and regression return one row per entity. Multiclass
classification, temporal link prediction and forecasting each return several
rows per entity, so code that assumes a one-to-one mapping between the entities
it asked for and the rows it gets back will be wrong for those tasks.

Every row carries `ENTITY` and `ANCHOR_TIMESTAMP`. Use `ENTITY` to join results
back to your input rather than relying on row order or position.

## Rows returned

| Task type | Rows returned | Reachable from |
| --- | --- | --- |
| `binary_classification` | One per entity | `predict()`, `predict_task()` |
| `regression` | One per entity | `predict()`, `predict_task()` |
| `multiclass_classification` | One per entity **per class** | `predict()`, `predict_task()` |
| `temporal_link_prediction` | One per entity **per ranked item** (`RANK TOP k` → `k` rows) | `predict()`, `predict_task()` |
| `forecasting` | One per entity **per forecast step** (`num_forecasts` rows) | `predict_task()` |

## Columns returned

| Task type | Columns |
| --- | --- |
| `binary_classification` | `ENTITY`, `ANCHOR_TIMESTAMP`, `PREDICTION`, `FALSE_PROB`, `TRUE_PROB` |
| `regression` | `ENTITY`, `ANCHOR_TIMESTAMP`, `PREDICTION` |
| `multiclass_classification` | `ENTITY`, `ANCHOR_TIMESTAMP`, `CLASS`, `SCORE`, `PREDICTED` |
| `temporal_link_prediction` | `ENTITY`, `ANCHOR_TIMESTAMP`, `CLASS`, `SCORE` |
| `forecasting` | `ENTITY`, `ANCHOR_TIMESTAMP`, `PREDICTION`, `FORECAST_STEP` |

For the tasks that return one row per class or per ranked item, `CLASS` holds
the class label or item id and `SCORE` its probability, ordered highest first.
Multiclass additionally sets `PREDICTED` to `True` on the single winning row per
entity, so `frame[frame['PREDICTED']]` recovers one row per entity.

Passing `return_embeddings=True` appends an `EMBEDDINGS` column; it does not
change the number of rows.

## Worked examples

Binary classification, three entities in, three rows out:

```python
frame = client.relational(graph).predict(
    'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
    indices=['u1', 'u2', 'u3'],
)
#   ENTITY          ANCHOR_TIMESTAMP  PREDICTION  FALSE_PROB  TRUE_PROB
# 0     u1 2026-03-01 00:00:00+00:00        True       0.004      0.996
```

Multiclass over three classes, two entities in, **six** rows out:

```python
frame = client.relational(graph).predict_task(
    context=context_df,
    predict=predict_df,
    task_type='multiclass_classification',
    entity_table='users',
)
#   ENTITY          ANCHOR_TIMESTAMP CLASS     SCORE  PREDICTED
# 0     u1 2026-03-01 00:00:00+00:00     b  0.999914       True
# 1     u1 2026-03-01 00:00:00+00:00     c  0.000053      False
# 2     u1 2026-03-01 00:00:00+00:00     a  0.000033      False
# 3     u2 2026-03-01 00:00:00+00:00     c  0.999951       True
# ...

winners = frame[frame['PREDICTED']]  # one row per entity
```

Temporal link prediction with `RANK TOP 3`, three entities in, nine rows out:

```python
frame = client.relational(graph).predict(
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 3 '
    'FOR EACH users.user_id',
    indices=['u1', 'u2', 'u3'],
)
top_1 = frame.groupby('ENTITY').head(1)  # best item per entity
```

## Link prediction takes a list-valued target

When you reach temporal link prediction through `predict_task()`, each value in
the context's target column must be a **list** of target ids rather than a
single id, and `entity_table` must be the `(source, target)` pair. A scalar
target is rejected with "Link prediction target values must be stringlist
arrays".

```python
context = pd.DataFrame(
    {
        'ENTITY': [...],
        'TARGET': [['i0', 'i1'], ['i3'], ...],  # a list per row
        'ANCHOR_TIMESTAMP': [...],
    }
)
frame = client.relational(graph).predict_task(
    context=context,
    predict=predict_df,
    task_type='temporal_link_prediction',
    entity_table=('users', 'items'),
    top_k=3,
)  # 3 rows per entity
```

## Multiclass

A predictive query whose target is a categorical column is a multiclass task,
so `predict()` reaches it as well as `predict_task()`:

```python
frame = client.relational(graph).predict(
    'PREDICT users.segment FOR users.user_id IN (0, 1, 2)'
)  # 3 entities x N classes
```

Use `predict_task(task_type="multiclass_classification", ...)` when the labels
are not a column of the graph and you want to supply the labelled rows
directly.

## Explanations

When `explain` is set, the call returns a `nemotron_relational` `Explanation` instead of a
`DataFrame`. The frame described above is still available on its `prediction`
attribute, with the same shape rules.

An explanation covers exactly one entity. That entity has to be selected with
`indices=`, not in the query text, because a one-element `IN (...)` array is
not valid PQL:

```python
explanation = client.relational(graph).predict(
    'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id IN (0, 1)',
    indices=[0],
    explain=True,
)
```
