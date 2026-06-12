import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from kumoai.rfm.backend.snow import Connection


@pytest.fixture(scope='session')
def connection() -> Generator['Connection', None, None]:
    snow = pytest.importorskip(
        "kumoai.testing.snow",
        reason="'snowflake' extension not installed",
    )

    connection = snow.connect(
        region='us-west-2',
        id='926922431314',
        account='xva19026',
        user='kumo',
        warehouse='WH_XS',
        database='KUMO',
    )

    with connection.cursor() as cursor:
        schema_name = f"RFM_TEST_SCHEMA_{uuid.uuid4().hex[:6].upper()}"
        cursor.execute(f"CREATE SCHEMA {schema_name}")
        cursor.execute(f"USE SCHEMA {schema_name}")

        cursor.execute("""
            CREATE TABLE USERS (
                USER_ID INTEGER PRIMARY KEY,
                IS_FLAG BOOLEAN,
                AGE FLOAT,
                GENDER TEXT,
                DOB TIMESTAMP
            )
        """)
        cursor.executemany(
            ("INSERT INTO USERS "
             "(USER_ID, IS_FLAG, AGE, GENDER, DOB) "
             "VALUES (%s, %s, %s, %s, %s)"),
            [
                (0, True, 60, 'male', '1960-01-01'),
                (1, False, 50, 'female', '1970-01-01'),
                (2, True, 40, 'female', '1980-01-01'),
                (3, False, 30, 'female', '1990-01-01'),
                (3, None, None, None, None),
            ],
        )

        cursor.execute("""
            CREATE TABLE ITEMS (
                ITEM_ID TEXT PRIMARY KEY,
                CATEGORY VARCHAR
            )
        """)
        cursor.executemany(
            "INSERT INTO ITEMS "
            "(ITEM_ID, CATEGORY) "
            "VALUES (%s, %s)",
            [
                ('A', 'Electronics'),
                ('B', 'Cloth'),
                ('C', 'Food'),
            ],
        )

        cursor.execute("""
            CREATE TABLE ORDERS (
                DATE STRING,
                USER_ID INTEGER,
                ITEM_ID TEXT,
                PRICE REAL,
                FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID),
                FOREIGN KEY (ITEM_ID) REFERENCES ITEMS(ITEM_ID)
            )
        """)
        cursor.executemany(
            "INSERT INTO ORDERS "
            "(DATE, USER_ID, ITEM_ID, PRICE) "
            "VALUES (%s, %s, %s, %s)",
            [
                ('2023-01-01', 0, 'A', 19.99),
                ('2023-01-02', 1, 'B', 45.50),
                ('2023-01-03', 2, 'A', 12.00),
                ('2023-01-04', 1, 'D', 99.95),
                ('2023-01-05', 1, 'C', 60.00),
                ('2023-01-06', 2, 'B', 7.25),
                ('2023-01-07', 3, 'D', 250.00),
                ('2023-01-08', 0, 'A', 15.75),
            ],
        )

    yield connection

    with connection.cursor() as cursor:
        cursor.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")

    connection.close()
