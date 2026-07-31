# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import requests
from kumorfm.api.json_serde import from_json

from kumorfm.exceptions import HTTPException

logger = logging.getLogger(__name__)

_DataclassT = TypeVar('_DataclassT')
ResponseT = TypeVar('ResponseT')


# See https://github.com/python/mypy/issues/9003
class Returns(Generic[ResponseT]):
    r"""Use this class if you'd like to use Python types as annotations for
    mypy, where `Type` is expected. See `parse_response` as an example.
    """
    if not TYPE_CHECKING:

        def __class_getitem__(cls, item: object) -> object:
            """Called when Returns is used as an Annotation at runtime.
            We just return the type we're given
            """
            return item


def _parse_dataclass_list(
    data_class: Type[_DataclassT],
    response: requests.Response,
) -> List[_DataclassT]:
    json_response = response.json()
    assert isinstance(json_response, list)

    def _parse_elem(v: Any) -> _DataclassT:
        # `bool` first: it is a subclass of `int`, so the cast below would
        # otherwise claim it and turn the string 'false' into True.
        if data_class is bool:
            return cast(_DataclassT, str(v).lower() == 'true')
        if issubclass(data_class, (str, int)):
            return cast(_DataclassT, data_class(v))
        return from_json(v, data_class)

    return [_parse_elem(item) for item in json_response]


@overload
def parse_response(response_type: Type[Returns[ResponseT]],
                   response: requests.Response) -> ResponseT:
    ...


@overload
def parse_response(response_type: Type[ResponseT],
                   response: requests.Response) -> ResponseT:
    ...


def parse_response(response_type: Union[Type[ResponseT],
                                        Type[Returns[ResponseT]]],
                   response: requests.Response) -> ResponseT:
    """Parse the HTTP response body into a Pydantic data structure.

    Args:
        response_type: a class definition for a Pydantic dataclass, or a
            :class:`list` of Pydantic dataclasses.
        response: a requests.Response object containing
            json data in the body

    Returns:
        The parsed response.
    """
    # Case 0: no parsing needed
    if response_type is str:
        return cast(ResponseT, response.text)
    if response_type is bool:
        return cast(ResponseT, response.text.lower() == 'true')

    # Case 1: dict[str, V] type
    if response_type is dict:
        k_type, v_type = get_args(response_type)
        assert issubclass(k_type, str)
        data = response.json()
        assert isinstance(data, dict)
        data = {k: from_json(v, v_type) for k, v in data.items()}
        return cast(ResponseT, data)

    # Case 2: response is a list of pydantic data classes
    if get_origin(response_type) is list:
        list_inner_type = get_args(response_type)[0]
        result = _parse_dataclass_list(list_inner_type, response)
        return cast(ResponseT, result)

    # Case 3: response is pydantic data classes
    try:
        return cast(ResponseT, from_json(response.text, response_type))
    except Exception:
        logger.error(f"Unable to parse response {response.text}")
        raise


def parse_id_response(response: requests.Response) -> str:
    return parse_response(Dict[str, str], response)['id']


def parse_patch_response(response: requests.Response) -> bool:
    """Parse PATCH response to indicate whether the resource was modified."""
    return parse_response(Dict[str, bool], response)['resource_updated']


def raise_on_error(response: requests.Response) -> None:
    r"""Raises an :class:`~kumorfm.exceptions.HTTPException` if a response does
    not return with an OK status code.
    """
    if not response.ok:
        assert response.status_code is not None
        raise HTTPException(response.status_code, response.text)
