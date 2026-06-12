import asyncio
import logging
import os
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from snowflake.snowpark import DataFrame, Session

logger = logging.getLogger('kumoai')


def _parse_private_key(
    private_key: str,
    passphrase: Optional[str] = None,
) -> bytes:
    r"""Load a PEM private key and return DER-encoded bytes for Snowflake."""
    from cryptography.hazmat.primitives import serialization

    # Replace escaped newlines for keys from env vars or secrets
    if isinstance(private_key, str) and '\\n' in private_key:
        private_key = private_key.replace('\\n', '\n')

    pkey = serialization.load_pem_private_key(
        private_key.encode(),
        password=passphrase.encode() if passphrase else None,
    )
    return pkey.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _build_snowflake_connector_params(
        credentials: Dict[str, Any]) -> Dict[str, Any]:
    r"""Build connection parameters for snowflake.connector.connect from
    credentials. Supports password-based or key-pair authentication.
    """
    params: Dict[str, Any] = {
        'user': credentials['user'],
        'account': credentials['account'],
        'session_parameters': {
            'PYTHON_CONNECTOR_QUERY_RESULT_FORMAT': 'json'
        },
    }
    if 'password' in credentials and credentials['password']:
        params['password'] = credentials['password']
    elif 'private_key' in credentials and credentials['private_key']:
        private_key = credentials['private_key']
        passphrase = credentials.get('private_key_passphrase')
        if isinstance(private_key, str):
            # PEM string: load with passphrase if encrypted, output DER bytes
            params['private_key'] = _parse_private_key(private_key, passphrase)
        else:
            # Already bytes (e.g. from DER)
            params['private_key'] = private_key
    else:
        raise ValueError(
            "Snowflake credentials must include either 'password' or "
            "'private_key' for SPCS authentication.")
    return params


def _get_spcs_token(snowflake_credentials: Dict[str, Any]) -> str:
    r"""Fetches a token to access a Kumo application deployed in Snowflake
    Snowpark Container Services (SPCS). This token is valid for 1 hour, after
    which the token must be re-generated.

    Supports both password-based and key-pair authentication. For key-pair,
    credentials should include ``user``, ``private_key``, and ``account``,
    with optional ``private_key_passphrase`` for encrypted keys.
    """
    # Create a request to the ingress endpoint with authz:
    active_session = _get_active_session()
    if active_session is not None:
        ctx = active_session.connection
    else:
        import snowflake.connector

        params = _build_snowflake_connector_params(snowflake_credentials)
        ctx = snowflake.connector.connect(**params)

    # Obtain a session token:
    assert ctx._rest is not None
    token_data = ctx._rest._token_request('ISSUE')
    token_extract = token_data['data']['sessionToken']
    return f'\"{token_extract}\"'


async def _run_refresh_spcs_token(minutes: int) -> None:
    r"""Runs the SPCS token refresh loop every `minutes` minutes."""
    while True:
        await asyncio.sleep(minutes * 60)
        try:
            logger.debug("triggering periodic SPCS token refresh")
            refresh_spcs_token()
        except Exception as e:
            logger.error(f"Failed to refresh SPCS token: {e}. "
                         f"Will retry in {minutes} minutes.")


def refresh_spcs_token():
    r"""Attempts to refresh the SPCS token and sets it in the global state
    if successful. If the refresh fails, the existing token is preserved.
    """
    from kumoai import KumoClient, global_state
    if (not global_state.initialized
            or (not global_state._snowflake_credentials
                and not global_state._snowpark_session)):
        logger.warning("Cannot refresh SPCS token: not initialized with "
                       "snowflake credentials")
        return

    new_token = _get_spcs_token(global_state._snowflake_credentials or {})
    assert global_state._url is not None
    client = KumoClient(
        url=global_state._url,
        api_key=None,
        spcs_token=new_token,
    )
    client.authenticate()
    global_state.set_spcs_token(new_token)


def _get_active_session() -> 'Optional[Session]':
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        return None


def _get_session() -> 'Session':
    import snowflake.snowpark as snowpark

    from kumoai import global_state
    assert global_state._snowflake_credentials is not None
    creds = global_state._snowflake_credentials

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    if not database or not schema:
        raise ValueError("Please set the SNOWFLAKE_DATABASE and "
                         "SNOWFLAKE_SCHEMA environment variables.")

    params: Dict[str, Any] = {
        'user': creds['user'],
        'account': creds['account'],
        'database': database,
        'schema': schema,
        'client_session_keep_alive': True,
    }

    if 'password' in creds and creds.get('password'):
        params['password'] = creds['password']
    elif 'private_key' in creds and creds.get('private_key'):
        private_key = creds['private_key']
        passphrase = creds.get('private_key_passphrase')
        if isinstance(private_key, str):
            params['private_key'] = _parse_private_key(private_key, passphrase)
        else:
            params['private_key'] = private_key
    else:
        raise ValueError(
            "Snowflake credentials must include either 'password' or "
            "'private_key' for session creation.")

    return snowpark.Session.builder.configs(params).create()


def _remove_path(session: 'Session', stage_path: str, file_path: str) -> None:
    stage_prefix = '.'.join(stage_path.split('.')[:2])
    name_remove = '.'.join([stage_prefix, file_path])
    session.sql(f"REMOVE {name_remove}").collect()


def _parquet_to_df(path: str) -> 'DataFrame':
    r"""Reads parquet from the given path and returns a snowpark DataFrame."""
    session = _get_session()
    if not path.endswith(os.path.sep):
        path += os.path.sep
    file_list = session.sql(f"LIST {path}").collect()
    for file_row in file_list:
        if file_row.name.endswith('.parquet'):
            continue
        _remove_path(session, path, file_row.name)
    df = session.read.parquet(path)
    return df


def _parquet_dataset_to_df(paths: List[str]) -> 'DataFrame':
    r"""Reads parquet from the given paths and returns a snowpark DataFrame."""
    from snowflake.snowpark import DataFrame
    df_list = [_parquet_to_df(url) for url in paths]
    return reduce(DataFrame.union_all, df_list)
