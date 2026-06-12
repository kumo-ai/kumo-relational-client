from __future__ import annotations

import os
import uuid

import pandas as pd
import pytest

from kumoai import rfm

try:
    import boto3
    import botocore.exceptions
    from sagemaker.model import ModelPackage
    from sagemaker.session import Session
except ImportError:
    pytest.skip("'sagemaker' not installed", allow_module_level=True)

# ---------- Configuration ----------
INSTANCE_TYPE = "ml.g4dn.xlarge"
INSTANCE_COUNT = 1
DEFAULT_ROLE_NAME = "AmazonSageMaker-ExecutionRole-20251029T122944"
DEFAULT_MODEL_PACKAGE_NAME = "kumo-rfm-20260414"


@pytest.fixture(scope="module")
def sage_session() -> Session:
    """Return a SageMaker session."""
    return Session(
        boto_session=None,
        sagemaker_client=None,
        sagemaker_runtime_client=None,
    )


@pytest.fixture(scope="module")
def role_arn() -> str:
    """SageMaker execution role ARN from ENV or the caller identity."""
    role_arn = os.getenv("RFM_SAGEMAKER_EXECUTION_ROLE_ARN")
    if role_arn:
        return role_arn

    # Compose default ROLE ARN from caller identity for local testing.
    account_id = boto3.client("sts").get_caller_identity().get("Account")
    return f"arn:aws:iam::{account_id}:role/service-role/{DEFAULT_ROLE_NAME}"


@pytest.fixture(scope="module")
def model_package_arn(sage_session: Session) -> str:
    """SageMaker ModelPackage ARN from ENV or the caller identity."""
    role_arn = os.getenv("RFM_SAGEMAKER_MODEL_PACKAGE_ARN")
    if role_arn:
        return role_arn

    # Compose default ROLE ARN
    account_id = boto3.client("sts").get_caller_identity().get("Account")
    region = sage_session.boto_region_name
    return (f"arn:aws:sagemaker:{region}:{account_id}:model-package/"
            f"{DEFAULT_MODEL_PACKAGE_NAME}")


@pytest.fixture(scope="module", autouse=True)
def setup(
    role_arn: str,
    model_package_arn: str,
    sage_session: Session,
):
    """Setup an SageMaker endpoint, yield, and teardown afterwards."""
    region = sage_session.boto_region_name

    unique_id: str = uuid.uuid4().hex[:8]
    endpoint_name: str = f"test-rfm-endpoint-{unique_id}"
    model_name = f"test-rfm-model-{unique_id}"

    # 1. Create Model object from existing model package
    model = ModelPackage(
        name=model_name,
        model_package_arn=model_package_arn,
        role=role_arn,
        sagemaker_session=sage_session,
    )

    # 2. Deploy model to endpoint
    try:
        model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            wait=True,
        )
    except botocore.exceptions.ClientError:
        pytest.skip("SageMaker deploy failed")

    # 3. Initialize SDK with the endpoint URL
    endpoint_url: str = (
        f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/"
        f"{endpoint_name}/invocations")
    rfm.init(url=endpoint_url)

    try:
        yield
    finally:
        # 4. Cleanup
        print(f"[INFO] Deleting endpoint {endpoint_name} ...")
        sage_session.delete_model(model_name)
        sage_session.delete_endpoint(endpoint_name)
        print("[INFO] Cleanup complete.")


@pytest.mark.integration
def test_sagemaker_integ() -> None:
    """Test KumoRFM predict using live SageMaker endpoint."""
    path = 's3://kumo-sdk-public/rfm-datasets/online-shopping'
    df_dict = {
        'users': pd.read_parquet(f'{path}/users.parquet'),
        'items': pd.read_parquet(f'{path}/items.parquet'),
        'orders': pd.read_parquet(f'{path}/orders.parquet')
    }

    graph = rfm.Graph.from_data(df_dict)
    model = rfm.KumoRFM(graph)

    query = "PREDICT COUNT(orders.*, 0, 30, days)>0 FOR EACH users.user_id"
    result = model.predict(query, indices=[0])
    print(result)
