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
            "No group is set. Call kumoai.set_group() or pass group= to "
            "kumoai.init() before performing this operation.")


class ProjectNotSetError(Exception):
    r"""Raised when an object operation is attempted without a project
    set.
    """
    def __init__(self) -> None:
        super().__init__(
            "No project is set. Call kumoai.set_project() or pass "
            "project= to kumoai.init() before performing object "
            "operations.")


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
