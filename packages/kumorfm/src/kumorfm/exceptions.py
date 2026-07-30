# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http
from typing import Dict, Optional


class FeatureNotEnabledError(Exception):
    r"""Raised when an RBAC feature is not enabled for this workspace."""
    def __init__(self) -> None:
        super().__init__(
            "The projects/groups feature is not enabled for this workspace. ")


class GroupNotSetError(Exception):
    r"""Raised when a group operation is attempted without a group set."""
    def __init__(self) -> None:
        super().__init__(
            "No group is set. Call kumorfm.set_group() or pass group= to "
            "kumorfm.init() before performing this operation.")


class ProjectNotSetError(Exception):
    r"""Raised when an object operation is attempted without a project
    set.
    """
    def __init__(self) -> None:
        super().__init__(
            "No project is set. Call kumorfm.set_project() or pass "
            "project= to kumorfm.init() before performing object "
            "operations.")


class InvalidResponseError(ValueError):
    r"""The NIM answered, but its response does not match the contract.

    Subclasses :class:`ValueError` so existing callers keep working, while
    letting the SDK report a malformed *server* response as such instead of
    blaming the caller's request — and stopping a body that will never parse
    from being retried as though the NIM were merely busy.
    """


class NimFailureError(RuntimeError):
    r"""A failed Kumo RFM NIM prediction, classified and ready to re-wrap.

    Subclasses :class:`RuntimeError` so existing callers keep working, and
    carries the structured facts a caller (or the ``nvidia-sdfm`` adapter) needs
    to map it onto its own error hierarchy without re-parsing the message:
    the HTTP status, the problem document's ``detail``, its RFC-9457
    ``invalid_params`` entries, and whether the failure looks transient.
    """
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Optional[str] = None,
        invalid_params: Optional[list] = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.invalid_params = invalid_params or []
        self.transient = transient


class HTTPException(Exception):
    r"""An HTTP exception, with detailed information and headers."""
    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        # Derived from starlette/blob/master/starlette/exceptions.py
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (f"{class_name}(status_code={self.status_code!r}, "
                f"detail={self.detail!r})")
