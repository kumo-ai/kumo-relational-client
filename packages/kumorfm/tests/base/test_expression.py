import pytest

from kumorfm.rfm.base import Expression, LocalExpression


def test_coerce() -> None:
    assert Expression.coerce(None) is None
    assert Expression.coerce(LocalExpression('A')) == LocalExpression('A')
    assert Expression.coerce('A') == LocalExpression('A')
    assert Expression.coerce(dict(value='A')) == LocalExpression('A')

    with pytest.raises(TypeError):
        Expression.coerce(dict(value='A', on='B'))
    with pytest.raises(TypeError):
        Expression.coerce(['A', 'B'])  # type: ignore
