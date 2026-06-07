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
DEFAULT_SUPERIOR_PARAMETERS: tuple[dict[str, Any], ...] = (
    {
        "code": "kelas_prediksi",
        "label": "Tingkat kematangan",
        "target_value": "matang",
        "unit": "",
        "weight": 0.35,
        "description": "Kandidat unggul harus terdeteksi pada tingkat kematangan matang.",
    },
    {
        "code": "confidence_minimum",
        "label": "Confidence minimum",
        "target_value": "75",
        "unit": "%",
        "weight": 0.25,
        "description": "Batas minimum keyakinan model untuk status unggul.",
    },
    {
        "code": "rekomendasi",
        "label": "Rekomendasi",
        "target_value": "Layak / Direkomendasikan",
        "unit": "",
        "weight": 0.10,
        "description": "Status rekomendasi yang dianggap memenuhi kriteria unggul.",
    },
    {
        "code": "ukuran_area_minimum",
        "label": "Ukuran area objek minimum",
        "target_value": "20",
        "unit": "%",
        "weight": 0.15,
        "description": "Perkiraan luas area objek sawit minimum di dalam gambar.",
    },
    {
        "code": "warna_dominan",
        "label": "Warna dominan",
        "target_value": "merah,oranye,coklat",
        "unit": "",
        "weight": 0.10,
        "description": "Warna dominan yang dianggap sesuai untuk kandidat unggul.",
    },
)


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
                note TEXT,
                quality_status TEXT,
                quality_score REAL,
                parameter_snapshot_json TEXT
            )
            """
        )
        _ensure_prediction_columns(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions(created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS superior_seed_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                target_value TEXT NOT NULL,
                unit TEXT NOT NULL DEFAULT '',
                weight REAL NOT NULL DEFAULT 1.0,
                description TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        seed_superior_parameters(connection)


def _ensure_prediction_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
    }
    additions = {
        "quality_status": "TEXT",
        "quality_score": "REAL",
        "parameter_snapshot_json": "TEXT",
    }

    for column_name, column_type in additions.items():
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE predictions ADD COLUMN {column_name} {column_type}"
            )


def seed_superior_parameters(connection: sqlite3.Connection) -> None:
    for parameter in DEFAULT_SUPERIOR_PARAMETERS:
        connection.execute(
            """
            INSERT OR IGNORE INTO superior_seed_parameters (
                code,
                label,
                target_value,
                unit,
                weight,
                description
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                parameter["code"],
                parameter["label"],
                parameter["target_value"],
                parameter["unit"],
                float(parameter["weight"]),
                parameter["description"],
            ),
        )
        connection.execute(
            """
            UPDATE superior_seed_parameters
            SET label = ?,
                unit = ?,
                description = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE code = ?
            """,
            (
                parameter["label"],
                parameter["unit"],
                parameter["description"],
                parameter["code"],
            ),
        )


def insert_prediction(
    filename: str,
    image_sha256: str,
    result: PredictionResult,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    quality_status: str | None = None,
    quality_score: float | None = None,
    parameter_snapshot: list[dict[str, Any]] | None = None,
) -> int:
    probabilities_json = json.dumps(result.probabilities, sort_keys=True)
    parameter_snapshot_json = (
        json.dumps(parameter_snapshot, sort_keys=True) if parameter_snapshot else None
    )

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
                note,
                quality_status,
                quality_score,
                parameter_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                quality_status,
                quality_score,
                parameter_snapshot_json,
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
                model_path,
                quality_status,
                quality_score
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_superior_parameters(
    db_path: str | Path = DEFAULT_DB_PATH,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    where_clause = "WHERE is_active = 1" if active_only else ""

    with open_db(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                id,
                code,
                label,
                target_value,
                unit,
                weight,
                description,
                is_active
            FROM superior_seed_parameters
            {where_clause}
            ORDER BY id ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def update_superior_parameter(
    code: str,
    target_value: str,
    weight: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    with open_db(db_path) as connection:
        connection.execute(
            """
            UPDATE superior_seed_parameters
            SET target_value = ?,
                weight = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE code = ?
            """,
            (target_value, float(weight), code),
        )


def count_predictions(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    with open_db(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()

    return int(row["total"])


def clear_predictions(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with open_db(db_path) as connection:
        connection.execute("DELETE FROM predictions")
