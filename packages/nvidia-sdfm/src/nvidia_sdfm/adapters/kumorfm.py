from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pandas as pd

from nvidia_sdfm.base import ModelAdapter, ModelCapabilities
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, SdfmError
from nvidia_sdfm.requests import KumoRFMRequest

if TYPE_CHECKING:
    from kumorfm.rfm.rfm import Explanation

_UNSET = object()


def _is_explain_config(value: Any) -> bool:
    r"""Return ``True`` when ``value`` is a driver ``ExplainConfig`` instance.

    Imported lazily so this adapter still imports without the optional
    ``kumorfm`` engine installed.
    """
    try:
        from kumorfm.rfm.rfm import ExplainConfig
    except Exception:
        return False
    return isinstance(value, ExplainConfig)


def _resolve_explain(field_value, options):
    r"""Resolve the effective ``explain`` setting from the request field and the
    legacy ``options['explain']`` compatibility path.

    ``KumoRFMRequest.explain`` is canonical. ``options['explain']`` is still
    accepted for backward compatibility, but specifying both is rejected so a
    stale option cannot silently override the first-class field. The resulting
    value must be a ``bool``, an ``ExplainConfig``, or an ``ExplainConfig``
    dict (all accepted by the driver's ``KumoRFM.predict``); anything else is
    an ``INVALID_REQUEST`` rather than a deep engine-level failure.
    """
    option_value = options.pop('explain', _UNSET)
    explain = field_value
    if option_value is not _UNSET:
        if (field_value is True or isinstance(field_value, dict)
                or _is_explain_config(field_value)):
            raise SdfmError(
                "explain is set both as a KumoRFMRequest field and in options; "
                "specify it once (prefer the request field)",
                code='INVALID_REQUEST',
            )
        explain = option_value
    if (explain is not True and explain is not False
            and not isinstance(explain, dict)
            and not _is_explain_config(explain)):
        raise SdfmError(
            "explain must be a bool, an ExplainConfig, or an ExplainConfig "
            f"dict; got {type(explain).__name__}",
            code='INVALID_REQUEST',
        )
    return explain


class KumoRFMAdapter(ModelAdapter):
    name = 'kumo-rfm'
    request_type = KumoRFMRequest

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='kumo-rfm',
            request_type=self.request_type.__name__,
            tasks=('relational',),
            outputs=('prediction', 'probabilities', 'explanation'),
        )

    def predict(
        self,
        transport: Transport,
        request: KumoRFMRequest,
    ) -> 'pd.DataFrame | Explanation':
        try:
            import kumorfm.rfm as rfm_engine
        except ModuleNotFoundError as error:
            if error.name != 'kumorfm':
                raise
            raise MissingExtraError('kumorfm', 'kumorfm') from error

        reserved = ({'indices', 'run_mode', 'batch_size', 'num_retries'}
                    & set(request.options))
        if reserved:
            raise SdfmError(
                f'KumoRFMRequest.options contains reserved keys {sorted(reserved)}; '
                'set them as request fields instead',
                code='INVALID_REQUEST',
            )
        batch_size = request.batch_size
        if (batch_size is not None and batch_size != 'max'
                and not (isinstance(batch_size, int)
                         and not isinstance(batch_size, bool)
                         and batch_size > 0)):
            raise SdfmError(
                "batch_size must be a positive int or the literal 'max'; got "
                f"{batch_size!r}",
                code='INVALID_REQUEST',
            )
        options = dict(request.options)
        explain = _resolve_explain(request.explain, options)

        wants_explanation = explain is not False

        rfm_engine.init(url=transport.url, api_key=transport.api_key,
                        verify_ssl=transport.verify_ssl,
                        _token=rfm_engine._SDFM_CLIENT_TOKEN)
        model = rfm_engine.KumoRFM(request.graph)
        if batch_size is not None:
            batch_ctx = model.batch_mode(batch_size,
                                         num_retries=request.num_retries)
        else:
            batch_ctx = contextlib.nullcontext()
        with batch_ctx:
            result = model.predict(
                request.query,
                indices=request.indices,
                run_mode=request.run_mode,
                explain=explain,
                **options,
            )
        if wants_explanation:
            from kumorfm.rfm.rfm import Explanation
            if not isinstance(result, Explanation):
                raise TypeError(
                    'expected an Explanation result for explain=True; got '
                    f'{type(result).__name__}',
                )
            return result
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                'expected a DataFrame result; pass explain=False (the '
                'default) to get a plain prediction DataFrame',
            )
        return result
