from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from .predict import PredictionResult


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "predictions.sqlite3"


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def open_db(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    connection = connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with open_db(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                class_name TEXT NOT NULL,
                display_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendation TEXT NOT NULL,
                source TEXT NOT NULL,
                model_path TEXT,
                probabilities_json TEXT NOT NULL,
                note TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions(created_at DESC)
            """
        )


def insert_prediction(
    filename: str,
    image_sha256: str,
    result: PredictionResult,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    probabilities_json = json.dumps(result.probabilities, sort_keys=True)

    with open_db(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                filename,
                image_sha256,
                class_name,
                display_label,
                confidence,
                recommendation,
                source,
                model_path,
                probabilities_json,
                note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                image_sha256,
                result.class_name,
                result.display_label,
                float(result.confidence),
                result.recommendation,
                result.source,
                result.model_path,
                probabilities_json,
                result.note,
            ),
        )
        return int(cursor.lastrowid)


def fetch_recent_predictions(
    limit: int = 25,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    with open_db(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                filename,
                display_label,
                confidence,
                recommendation,
                source,
                model_path
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def count_predictions(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    with open_db(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()

    return int(row["total"])


def clear_predictions(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with open_db(db_path) as connection:
        connection.execute("DELETE FROM predictions")
