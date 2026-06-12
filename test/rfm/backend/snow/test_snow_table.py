import pytest
from kumoapi.typing import Dtype, Stype

from kumoai import rfm
from kumoai.rfm.backend.local import LocalTable
from kumoai.rfm.base import LocalExpression

try:
    from kumoai.rfm.backend.snow import Connection, SnowTable
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)


def test_invalid_name(connection: Connection) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SnowTable(connection, name='foo')


def test_empty(connection: Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE EMPTY (ID INTEGER)")

    with pytest.raises(RuntimeError, match="is empty"):
        SnowTable(connection, name='EMPTY')

    with pytest.raises(RuntimeError, match="is empty"):
        SnowTable(
            connection,
            name='EMPTY',
            columns=[dict(name='EVEN_ID', expr='2*ID')],
        )

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS EMPTY")


def test_basic(connection: Connection) -> None:
    table = SnowTable(connection, name='USERS')

    assert table.name == 'USERS'
    assert table.primary_key is not None
    assert table.primary_key.name == 'USER_ID'
    assert table.time_column is None
    assert table.end_time_column is None

    assert len(table.columns) == 5
    assert table['USER_ID'].dtype == Dtype.int
    assert table['USER_ID'].stype == Stype.ID
    assert table['IS_FLAG'].dtype == Dtype.bool
    assert table['IS_FLAG'].stype == Stype.categorical
    assert table['AGE'].dtype == Dtype.float
    assert table['AGE'].stype == Stype.numerical
    assert table['GENDER'].dtype == Dtype.string
    assert table['GENDER'].stype == Stype.categorical
    assert table['DOB'].dtype == Dtype.date
    assert table['DOB'].stype == Stype.timestamp


def test_db_schema(connection: Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        result = cursor.fetchone()
        assert result is not None
        database, schema = result

    table = SnowTable(
        connection,
        name='USERS',
        database=database,
        schema=schema,
    )
    assert table.source_name == f'{database}.{schema}.USERS'
    assert table._quoted_source_name == f'"{database}"."{schema}"."USERS"'


def test_column_expression(connection: Connection) -> None:
    table = SnowTable(
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
    assert column.dtype == Dtype.bool
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
    assert column.dtype == Dtype.date
    assert column.stype == Stype.timestamp
    assert column._is_time_column


@pytest.mark.rfm_backend
def test_dtype(connection: Connection) -> None:
    # https://docs.snowflake.com/en/sql-reference/intro-summary-data-types
    dtypes = [
        ('NUMBER', "1", Dtype.int),
        ('NUMBER(2, 0)', "1", Dtype.int),
        ('NUMBER(10, 1)', "1.1", Dtype.float),
        ('DECIMAL(20, 2)', "1.23", Dtype.float),
        ('NUMERIC(30, 3)', "1.234", Dtype.float),
        ('INT', "1", Dtype.int),
        ('INTEGER', "1", Dtype.int),
        ('BIGINT', "1", Dtype.int),
        ('SMALLINT', "1", Dtype.int),
        ('TINYINT', "1", Dtype.int),
        ('BYTEINT', "1", Dtype.int),
        ('FLOAT', "1.0", Dtype.float),
        ('FLOAT4', "1.0", Dtype.float),
        ('FLOAT8', "1.0", Dtype.float),
        ('DOUBLE', "1.0", Dtype.float),
        ('DOUBLE PRECISION', "1.0", Dtype.float),
        ('REAL', "1.0", Dtype.float),
        ('VARCHAR', "'a'", Dtype.string),
        ('VARCHAR(2)', "'a'", Dtype.string),
        ('CHAR', "'a'", Dtype.string),
        ('CHARACTER', "'a'", Dtype.string),
        ('STRING', "'a'", Dtype.string),
        ('TEXT', "'a'", Dtype.string),
        ('BINARY', "TO_BINARY('01')", Dtype.binary),
        ('VARBINARY', "TO_BINARY('01')", Dtype.binary),
        ('BOOLEAN', "TRUE", Dtype.bool),
        ('DATE', "DATE '2024-01-01'", Dtype.date),
        ('DATETIME', "'2024-01-01 00:00:00'", Dtype.date),
        ('TIME', "TIME '00:00:00'", Dtype.time),
        ('TIMESTAMP', "'2024-01-01 00:00:00'", Dtype.date),
        ('TIMESTAMP_LTZ', "'2024-01-01 00:00:00'", Dtype.date),
        ('TIMESTAMP_NTZ', "'2024-01-01 00:00:00'", Dtype.date),
        ('TIMESTAMP_TZ', "'2024-01-01 00:00:00 +00:00'", Dtype.date),
        ('VARIANT', "PARSE_JSON('{}')", Dtype.unsupported),
        ('OBJECT', "OBJECT_CONSTRUCT()", Dtype.unsupported),
        ('ARRAY', "ARRAY_CONSTRUCT(12, 'twelve', NULL)", Dtype.unsupported),
        ('ARRAY(INT)', "[1,2]::ARRAY(INT)", Dtype.intlist),
        ('ARRAY(FLOAT)', "[1.0,2.0]::ARRAY(FLOAT)", Dtype.floatlist),
        ('ARRAY(VARCHAR)', "['a','b']::ARRAY(VARCHAR)", Dtype.stringlist),
        ('MAP(STRING, INT)', "NULL", Dtype.unsupported),
        ('FILE', "NULL", Dtype.unsupported),
        ('GEOGRAPHY', "'POINT(0 0)'", Dtype.unsupported),
        ('GEOMETRY', "'POINT(0 0)'", Dtype.unsupported),
        ('VECTOR(INT, 2)', "[1,2]::VECTOR(INT,2)", Dtype.intlist),
        ('VECTOR(FLOAT, 2)', "[1.0,2.0]::VECTOR(FLOAT,2)", Dtype.floatlist),
    ]

    with connection.cursor() as cursor:
        columns = [f'COL_{i} {dtype}' for i, (dtype, *_) in enumerate(dtypes)]
        cursor.execute(f"CREATE TABLE DTYPE ({', '.join(columns)})")

        values = [value for _, value, _ in dtypes]
        cursor.execute(f"INSERT INTO DTYPE SELECT {', '.join(values)}")

        cursor.execute("DESCRIBE TABLE DTYPE")
        snowflake_dtypes = [row[1] for row in cursor.fetchall()]

    # Test data and semantic type handling ####################################

    kumo_dtypes = [SnowTable._to_dtype(dtype) for dtype in snowflake_dtypes]
    for dtype, (_, _, expected_dtype) in zip(kumo_dtypes, dtypes):
        assert dtype == expected_dtype

    with pytest.raises(ValueError, match="unsupported data type"):
        SnowTable(connection, name='DTYPE')

    columns = [
        f'COL_{i}' for i, (_, _, expected_dtype) in enumerate(dtypes)
        if expected_dtype != Dtype.unsupported
    ]
    table = SnowTable(connection, name='DTYPE', columns=columns)

    for i, (_, _, expected_dtype) in enumerate(dtypes):
        if expected_dtype != Dtype.unsupported:
            assert table[f'COL_{i}'].dtype == expected_dtype
        else:
            assert f'COL_{i}' not in table

    # Test parity with `LocalTable` variant:
    local_table = LocalTable(table._source_sample_df[columns], name='DTYPE')
    for column in table.columns:
        local_column = local_table[column.name]
        # NOTE `time` is converted internally to `datetime`:
        if column.dtype == Dtype.time:
            assert local_column.dtype == Dtype.date
        else:
            assert column.dtype == local_column.dtype
        assert column.stype == local_column.stype

    # Test data processing end-to-end #########################################
    table.primary_key = 'COL_0'
    graph = rfm.Graph([table])

    # Test default semantic types:
    model = rfm.KumoRFM(graph)
    with model.retry(3):
        model.predict(
            'PREDICT DTYPE.COL_1 FOR EACH DTYPE.COL_0',
            indices=[1],
            verbose=False,
        )

    # Test custom semantic types:
    for column in table.columns:
        if column.stype == Stype.categorical and column.dtype.is_numerical():
            column.stype = Stype.numerical
        if column.dtype in {Dtype.intlist, Dtype.floatlist}:
            column.stype = Stype.sequence

    model = rfm.KumoRFM(graph)
    with model.retry(3):
        model.predict(
            'PREDICT DTYPE.COL_1 FOR EACH DTYPE.COL_0',
            indices=[1],
            verbose=False,
        )

    # Clean-up ################################################################

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS DTYPE")
