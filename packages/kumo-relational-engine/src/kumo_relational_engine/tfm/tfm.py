# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import pandas as pd

from kumo_relational_engine.api.pquery import ValidatedPredictiveQuery
from kumo_relational_engine.pql.parser.parser import QueryValidationType
from kumo_relational_engine.rfm import Graph
from kumo_relational_engine.rfm.query_parser import parse_query_locally


class KumoTabular:
    r"""Run Kumo Tabular predictions.

    :class:`KumoTabular` is the tabular counterpart of
    :class:`~kumo_relational_engine.rfm.KumoRelational`: it takes the same
    :class:`~kumo_relational_engine.rfm.Graph` and the same PQL. It processes
    the data, converts it to a single table, and builds the context table the
    tabular foundation model predicts from. The model scores one label per
    entity, so it serves binary classification, multiclass classification and
    regression.

    .. code-block:: python

        from kumo_relational_engine.rfm import Graph
        from kumo_relational_engine.tfm import KumoTabular

        graph = Graph.from_data({
            'users': df_users,
            'orders': df_orders,
        })

        tabular = KumoTabular(graph)

        query = ("PREDICT COUNT(orders.*, 0, 30, days) > 0 "
                 "FOR EACH users.user_id")
        result = tabular.predict(query, indices=[1, 2])

    Args:
        graph: The graph.
    """

    def __init__(self, graph: Graph) -> None:
        graph = graph.validate()
        self._graph_def = graph._to_api_graph_definition()

    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
    ) -> pd.DataFrame:
        r"""Returns predictions for a predictive query.

        Args:
            query: The predictive query.
            indices: The entity primary keys to predict for. Will override the
                indices given as part of the predictive query.

        Raises:
            ValueError: If the query is not valid for the tabular model.
            NotImplementedError: Always, once the query validates. Prediction
                is not wired up yet.
        """
        if isinstance(query, str):
            parse_query_locally(
                query,
                self._graph_def,
                QueryValidationType.TFM,
            )
        raise NotImplementedError(
            'KumoTabular cannot answer a query yet. This release validates a '
            'predictive query against the graph and the tabular task gate; '
            'label generation, feature assembly and prediction land in a '
            'later release. Use KumoRelational to make predictions today.'
        )
