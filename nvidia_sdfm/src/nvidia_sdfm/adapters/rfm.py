from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from nvidia_sdfm.base import ModelAdapter
from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import MissingExtraError


class RFMAdapter(ModelAdapter):
    name = 'kumo-rfm'

    def predict(
        self,
        client: TFMClient,
        *,
        graph: Any,
        query: str,
        indices: Sequence[Any] | None = None,
        run_mode: str = 'fast',
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import kumoai.rfm as kumoai_rfm
        except (ImportError, RuntimeError) as error:
            raise MissingExtraError('rfm', 'kumoai') from error

        kumoai_rfm.init(url=client.url, api_key=client.api_key)
        model = kumoai_rfm.KumoRFM(graph)
        result = model.predict(
            query,
            indices=indices,
            run_mode=run_mode,
            **kwargs,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                'expected a DataFrame result; pass explain=False (the '
                'default) to get a plain prediction DataFrame',
            )
        return result
