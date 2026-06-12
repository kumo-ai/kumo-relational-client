from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from kumoai.rfm.backend.duckdb import Connection


@pytest.fixture
def connection(tmp_path: Path) -> Generator['Connection', None, None]:
    duckdb = pytest.importorskip(
        "kumoai.rfm.backend.duckdb",
        reason="'duckdb' extension not installed",
    )

    connection = duckdb.connect(tmp_path / 'data.duckdb')

    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE USERS (
                USER_ID INTEGER UNIQUE,
                IS_FLAG BOOLEAN,
                AGE FLOAT,
                GENDER TEXT,
                DOB TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO USERS
            (USER_ID, IS_FLAG, AGE, GENDER, DOB)
            VALUES
                (0, TRUE, 60, 'male', '1960-01-01'),
                (1, FALSE, 50, 'female', '1970-01-01'),
                (2, TRUE, 40, 'female', '1980-01-01'),
                (3, FALSE, 30, 'female', '1990-01-01')
        """)

        cursor.execute("""
            CREATE TABLE ITEMS (
                ITEM_ID TEXT PRIMARY KEY,
                CATEGORY VARCHAR
            )
        """)
        cursor.execute("""
            INSERT INTO ITEMS
            (ITEM_ID, CATEGORY)
            VALUES
                ('A', 'Electronics'),
                ('B', 'Cloth'),
                ('C', 'Food'),
                ('D', 'Unknown')
        """)

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
        cursor.execute("""
            INSERT INTO ORDERS
            (DATE, USER_ID, ITEM_ID, PRICE)
            VALUES
                ('2023-01-01', 0, 'A', 19.99),
                ('2023-01-02', 1, 'B', 45.50),
                ('2023-01-03', 2, 'A', 12.00),
                ('2023-01-04', 1, 'D', 99.95),
                ('2023-01-05', 1, 'C', 60.00),
                ('2023-01-06', 2, 'B', 7.25),
                ('2023-01-07', 3, 'D', 250.00),
                ('2023-01-08', 0, 'A', 15.75)
        """)

        connection.commit()

    yield connection

    try:
        connection.close()
    except Exception:
        pass  # Connection may have been closed by the test.
