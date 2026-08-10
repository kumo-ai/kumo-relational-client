# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SDK-owned, not upstream's. Upstream also defined validate/parse/evaluate
# request and response types; this SDK only ever POSTs a prediction, so only
# the prediction request and response are kept. See api/SOURCE.md.

from typing import Any

from pydantic.dataclasses import dataclass

from kumorfm.api.rfm import Context
from kumorfm.api.rfm.inference import InferenceConfig
from kumorfm.runmode import RunMode


@dataclass
class RFMPredictRequest:
    context: Context
    run_mode: RunMode
    use_prediction_time: bool = False
    query: str = ''
    inference_config: InferenceConfig | None = None
    return_embeddings: bool = False


@dataclass
class RFMPredictResponse:
    prediction: dict[str, Any]
