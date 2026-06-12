from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Optional


class HTTPMethod(Enum):
    r"""HTTP methods supported by the API."""

    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass(frozen=True)
class Endpoint:
    r"""Represents an API endpoint with its path and HTTP method."""

    path: Optional[str] = field(default=None)
    method: HTTPMethod = HTTPMethod.GET

    def validate(self) -> None:
        pass

    def get_path(self) -> str:
        if self.path is None:
            raise ValueError("Endpoint requires a path")
        return self.path


class RFMEndpoints:
    BASE: Final[str] = "/rfm"

    predict = Endpoint(f"{BASE}/predict", HTTPMethod.POST)
    explain = Endpoint(f"{BASE}/explain", HTTPMethod.POST)
    evaluate = Endpoint(f"{BASE}/evaluate", HTTPMethod.POST)
    validate_query = Endpoint(f"{BASE}/validate_query", HTTPMethod.POST)
    parse_query = Endpoint(f"{BASE}/parse_query", HTTPMethod.POST)
