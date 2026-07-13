from dataclasses import dataclass

from kumoapi.typing import Dtype


@dataclass
class SourceColumn:
    name: str
    dtype: Dtype | None
    is_primary_key: bool
    is_unique_key: bool
    is_nullable: bool


@dataclass
class SourceForeignKey:
    name: str
    dst_table: str
    primary_key: str
