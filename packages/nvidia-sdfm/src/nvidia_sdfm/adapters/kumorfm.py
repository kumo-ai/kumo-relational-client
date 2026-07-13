from __future__ import annotations

import pandas as pd

from nvidia_sdfm.base import ModelAdapter, ModelCapabilities
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, SdfmError
from nvidia_sdfm.requests import KumoRFMRequest


class KumoRFMAdapter(ModelAdapter):
    name = 'kumo-rfm'
    request_type = KumoRFMRequest

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='kumo-rfm',
            request_type=self.request_type.__name__,
            tasks=('relational',),
            outputs=('prediction', 'probabilities'),
        )

    def predict(
        self,
        transport: Transport,
        request: KumoRFMRequest,
    ) -> pd.DataFrame:
        try:
            import kumorfm.rfm as rfm_engine
        except ModuleNotFoundError as error:
            if error.name != 'kumorfm':
                raise
            raise MissingExtraError('kumorfm', 'kumorfm') from error

        reserved = {'indices', 'run_mode'} & set(request.options)
        if reserved:
            raise SdfmError(
                f'KumoRFMRequest.options contains reserved keys {sorted(reserved)}; '
                'set them as request fields instead',
                code='INVALID_REQUEST',
            )

        rfm_engine.init(url=transport.url, api_key=transport.api_key,
                        verify_ssl=transport.verify_ssl)
        model = rfm_engine.KumoRFM(request.graph)
        result = model.predict(
            request.query,
            indices=request.indices,
            run_mode=request.run_mode,
            **request.options,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                'expected a DataFrame result; pass explain=False (the '
                'default) to get a plain prediction DataFrame',
            )
        return result
