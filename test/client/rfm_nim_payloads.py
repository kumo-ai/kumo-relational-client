from __future__ import annotations

from copy import deepcopy
from typing import Any

NIM_V0_PREDICTION_PATH = '/v0/predictions'
NIM_V0_SESSIONS_PATH = '/v0/sessions'
SDK_V1_PREDICTION_PATH = '/v1/predictions'

_NIM_V0_SMOKE_PAYLOAD: dict[str, Any] = {
    'model': 'kumo-rfm',
    'task': {
        'kind': 'classification',
        'target': {
            'column_name': 'status',
            'dtype': 'bool',
        },
        'entity_table_names': ['accounts'],
    },
    'schema': {
        'instance_table': {
            'columns': {
                'account_id': {
                    'dtype': 'int64',
                    'stype': 'ID',
                },
                'status': {
                    'dtype': 'bool',
                    'stype': 'categorical',
                },
                'anchor_time': {
                    'dtype': 'timestamp[us]',
                    'stype': 'timestamp',
                },
            },
            'primary_key': 'account_id',
        },
        'related_tables': {
            'accounts': {
                'columns': {
                    'account_id': {
                        'dtype': 'int64',
                        'stype': 'ID',
                    },
                    'segment': {
                        'dtype': 'string',
                        'stype': 'categorical',
                    },
                },
                'primary_key': 'account_id',
            },
        },
        'relationships': [{
            'source_columns': ['account_id'],
            'target_table': 'accounts',
            'target_columns': ['account_id'],
        }],
    },
    'context': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['account_id', 'status', 'anchor_time'],
            'rows': [
                [501, True, '2025-01-01T00:00:00Z'],
                [502, False, '2025-01-02T00:00:00Z'],
            ],
        },
        'related_tables': {
            'accounts': {
                'format': 'arrays',
                'columns': ['account_id', 'segment'],
                'rows': [
                    [501, 'enterprise'],
                    [502, 'startup'],
                ],
            },
        },
    },
    'predict': {
        'instance_table': {
            'format': 'arrays',
            'columns': ['account_id', 'anchor_time'],
            'rows': [[601, '2025-02-01T00:00:00Z']],
        },
        'related_tables': {
            'accounts': {
                'format': 'arrays',
                'columns': ['account_id', 'segment'],
                'rows': [[601, 'enterprise']],
            },
        },
    },
    'output': {
        'fields': ['prediction', 'probabilities'],
    },
    'inference': {
        'run_mode': 'best',
    },
}


def nim_v0_smoke_payload() -> dict[str, Any]:
    """Payload mirrored from kumo_rfm_nim/scripts/run_container_smoke.sh."""
    return deepcopy(_NIM_V0_SMOKE_PAYLOAD)


def sdk_v1_smoke_payload() -> dict[str, Any]:
    payload = nim_v0_smoke_payload()
    payload['version'] = 'v1'
    payload['metadata'] = {
        'source_query': 'rfm-sdk-nim-contract-smoke',
    }
    return payload


def nim_v0_session_create_payload() -> dict[str, Any]:
    payload = nim_v0_smoke_payload()
    payload.pop('predict')
    payload.pop('output')
    payload.pop('inference')
    return payload


def nim_v0_session_predict_minimal_payload() -> dict[str, Any]:
    payload = nim_v0_smoke_payload()
    return {
        'predict': payload['predict'],
        'output': payload['output'],
    }
