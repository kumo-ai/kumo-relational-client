# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared RBAC dataclasses used by both the Kumo REST service and the SDK."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from typing_extensions import Literal


@dataclass
class DatabricksConnectorScope:
    """Scope constraints for Databricks group connectors.

    Attributes:
        connector_type: Discriminator, always ``"databricks"``.
        cluster_id: Databricks cluster ID override (optional).
        warehouse_id: Databricks warehouse ID override (optional).
        catalog: Unity Catalog name override (optional).
        allowed_schemas: Full access to all tables in listed schemas.
            ``None`` means all schemas are permitted.
        allowed_tables: Partial access — mapping of schema name to list of
            permitted table names. Unioned with *allowed_schemas*.
    """
    connector_type: Literal["databricks"] = "databricks"
    cluster_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    catalog: Optional[str] = None
    allowed_schemas: Optional[List[str]] = None
    allowed_tables: Optional[Dict[str, List[str]]] = None


@dataclass
class FileConnectorScope:
    """Scope constraints for file-based group connectors.

    Attributes:
        connector_type: Discriminator, always ``"file"``.
        allowed_paths: Restricts access to specific file paths.
            ``None`` means all paths are permitted.
    """
    root_dir: str
    connector_type: Literal["file"] = "file"


ConnectorScope = Union[DatabricksConnectorScope, FileConnectorScope]
