from typing import Generator

import pytest
import requests_mock

from kumoai import global_state, init
from kumoai.client.endpoints import Endpoint, HTTPMethod

# Not mock:// due to https://stackoverflow.com/a/76056002
MOCK_URL = "http://kumo.ai"


def pytest_addoption(parser):
    parser.addoption('--runintegration', action='store_true', default=False,
                     help="run integration tests")


def pytest_collection_modifyitems(config, items):
    # check if you got an option like --key=snowflake
    if not config.getoption("--runintegration"):
        skip_integ = pytest.mark.skip(reason="integration test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integ)
    else:
        skip_integ = pytest.mark.skip(reason="no integration test")
        for item in items:
            if ("integration" not in item.keywords):
                item.add_marker(skip_integ)


@pytest.fixture(scope="class")
def mock_api() -> Generator[requests_mock.Mocker, None, None]:
    with requests_mock.Mocker() as m:
        yield m


@pytest.fixture(scope="class")
def setup_mock_client(mock_api):
    # Create the client:
    mock_api.get(f"{MOCK_URL}/v1/connectors", status_code=200)
    mock_api.get(f"{MOCK_URL}/config", json={})
    init(url=MOCK_URL, api_key="DISABLED")

    # Run the test:
    yield

    # Cleanup, do not re-use clients across unit tests:
    global_state.clear()


@pytest.fixture(scope="class")
def setup_integration_client():
    # Create the client:
    init(url="http://localhost:10002", api_key="test:DISABLED")

    # Run the test:
    yield

    # Cleanup, do not re-use clients across unit tests:
    global_state.clear()


def get_mock_method(mock_api, endpoint: Endpoint):
    method_map = {
        HTTPMethod.GET: mock_api.get,
        HTTPMethod.POST: mock_api.post,
        HTTPMethod.PATCH: mock_api.patch,
        HTTPMethod.DELETE: mock_api.delete,
    }
    if endpoint.method not in method_map:
        raise ValueError(f"Unsupported HTTP method: {endpoint.method}")

    return method_map[endpoint.method]
