from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from kumoai.rfm.backend.sqlite import Connection


@pytest.fixture
def connection(tmp_path: Path) -> Generator['Connection', None, None]:
    sqlite = pytest.importorskip(
        "kumoai.rfm.backend.sqlite",
        reason="'sqlite' extension not installed",
    )

    connection = sqlite.connect(tmp_path / 'data.db')

    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE USERS (
                USER_ID INTEGER,
                IS_FLAG BOOLEAN,
                AGE FLOAT,
                GENDER TEXT,
                DOB TIMESTAMP
            )
        """)
        cursor.executemany(
            ("INSERT INTO USERS "
             "(USER_ID, IS_FLAG, AGE, GENDER, DOB) "
             "VALUES (?, ?, ?, ?, ?)"),
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
            "VALUES (?, ?)",
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
            "VALUES (?, ?, ?, ?)",
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

        connection.commit()

    yield connection

    connection.close()
