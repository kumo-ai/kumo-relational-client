import pytest
from kumoapi.typing import Dtype, Stype

from kumoai.rfm.base import LocalExpression

try:
    from kumoai.rfm.backend.sqlite import Connection, SQLiteTable
except ImportError:
    pytest.skip("'sqlite' extension not installed", allow_module_level=True)


def test_invalid_name(connection: Connection) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SQLiteTable(connection, name='foo')


def test_empty(connection: Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE EMPTY (ID INTEGER)")

    with pytest.raises(RuntimeError, match="is empty"):
        SQLiteTable(connection, name='EMPTY')

    with pytest.raises(RuntimeError, match="is empty"):
        SQLiteTable(
            connection,
            name='EMPTY',
            columns=[dict(name='EVEN_ID', expr='2*ID')],
        )

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS EMPTY")


def test_basic(connection: Connection) -> None:
    table = SQLiteTable(connection, name='USERS')

    assert table.name == 'USERS'
    assert table.primary_key is None
    assert table.time_column is None
    assert table.end_time_column is None

    assert len(table.columns) == 5
    assert table['USER_ID'].dtype == Dtype.int
    assert table['USER_ID'].stype == Stype.ID
    assert table['IS_FLAG'].dtype == Dtype.int
    assert table['IS_FLAG'].stype == Stype.categorical
    assert table['AGE'].dtype == Dtype.float
    assert table['AGE'].stype == Stype.numerical
    assert table['GENDER'].dtype == Dtype.string
    assert table['GENDER'].stype == Stype.categorical
    assert table['DOB'].dtype == Dtype.string
    assert table['DOB'].stype == Stype.timestamp


def test_column_expression(connection: Connection) -> None:
    table = SQLiteTable(
        connection=connection,
        name='USER',
        source_name='USERS',
        columns=[
            dict(name='ID', expr="2*USER_ID"),
            dict(name='GENDER', expr="CONCAT(GENDER, '-', GENDER)"),
            dict(name='flagged', expr="IS_FLAG"),
            dict(name='AGE', expr="AGE"),
            dict(name='date_of_birth', expr="DOB"),
        ],
    )
    table.infer_metadata(verbose=False)

    # No source information was queried:
    assert '_source_column_dict' not in table.__dict__
    assert '_source_sample_df' not in table.__dict__

    assert table.primary_key is not None
    assert table.primary_key.name == 'ID'
    assert table.time_column is not None
    assert table.time_column.name == 'date_of_birth'
    assert table.end_time_column is None

    assert len(table.columns) == 5

    column = table['ID']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == '2*USER_ID'
    assert column.dtype == Dtype.int
    assert column.stype == Stype.ID
    assert column._is_primary_key

    column = table['GENDER']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == "CONCAT(GENDER, '-', GENDER)"
    assert column.dtype == Dtype.string
    assert column.stype == Stype.categorical

    column = table['flagged']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == "IS_FLAG"
    assert column.dtype == Dtype.int
    assert column.stype == Stype.categorical

    column = table['AGE']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == "AGE"
    assert column.dtype == Dtype.float
    assert column.stype == Stype.numerical

    column = table['date_of_birth']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == "DOB"
    assert column.dtype == Dtype.string
    assert column.stype == Stype.timestamp
    assert column._is_time_column
