import sqlite3
import json
from pathlib import Path


# ==========================================
# DATABASE SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "smart_helmet.db"


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ==========================================
# CHECK / ADD DATABASE COLUMN
# ==========================================

def add_column_if_missing(cursor, column_name, column_type):
    cursor.execute("PRAGMA table_info(violations)")
    columns = [row[1] for row in cursor.fetchall()]

    if column_name not in columns:
        cursor.execute(
            f"ALTER TABLE violations "
            f"ADD COLUMN {column_name} {column_type}"
        )


# ==========================================
# CREATE / UPDATE DATABASE TABLE
# ==========================================

def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            detected_at TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,

            violation_type TEXT NOT NULL,

            motorcycle_id INTEGER,
            rider_id INTEGER,
            rider_count INTEGER,

            confidence REAL,

            vehicle_number TEXT,

            vehicle_number_status TEXT
                DEFAULT 'UNCONFIRMED',

            plate_candidates TEXT,

            evidence_image TEXT,

            location TEXT,

            latitude REAL,
            longitude REAL,

            camera_id TEXT,

            status TEXT
                DEFAULT 'PENDING',

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------
    # MIGRATE EXISTING DATABASE
    # --------------------------------------

    add_column_if_missing(
        cursor,
        "latitude",
        "REAL"
    )

    add_column_if_missing(
        cursor,
        "longitude",
        "REAL"
    )

    connection.commit()
    connection.close()

    print("Smart Helmet database initialized.")


# ==========================================
# SAVE VIOLATION
# ==========================================

def save_violation(
    detected_at,
    date,
    time,
    violation_type,
    motorcycle_id=None,
    rider_id=None,
    rider_count=None,
    confidence=None,
    vehicle_number=None,
    vehicle_number_status="UNCONFIRMED",
    plate_candidates=None,
    evidence_image=None,
    location=None,
    latitude=None,
    longitude=None,
    camera_id=None
):
    connection = get_connection()
    cursor = connection.cursor()

    if plate_candidates is None:
        plate_candidates = []

    plate_candidates_json = json.dumps(
        plate_candidates
    )

    cursor.execute(
        """
        INSERT INTO violations (
            detected_at,
            date,
            time,
            violation_type,
            motorcycle_id,
            rider_id,
            rider_count,
            confidence,
            vehicle_number,
            vehicle_number_status,
            plate_candidates,
            evidence_image,
            location,
            latitude,
            longitude,
            camera_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            detected_at,
            date,
            time,
            violation_type,
            motorcycle_id,
            rider_id,
            rider_count,
            confidence,
            vehicle_number,
            vehicle_number_status,
            plate_candidates_json,
            evidence_image,
            location,
            latitude,
            longitude,
            camera_id
        )
    )

    violation_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return violation_id


# ==========================================
# CONVERT DATABASE ROW
# ==========================================

def row_to_violation(row):
    if row is None:
        return None

    item = dict(row)

    try:
        item["plate_candidates"] = json.loads(
            item.get("plate_candidates") or "[]"
        )

    except Exception:
        item["plate_candidates"] = []

    return item


# ==========================================
# GET ALL VIOLATIONS
# ==========================================

def get_all_violations():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM violations
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    connection.close()

    return [
        row_to_violation(row)
        for row in rows
    ]


# ==========================================
# GET ONE VIOLATION
# ==========================================

def get_violation(violation_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM violations
        WHERE id = ?
        """,
        (violation_id,)
    )

    row = cursor.fetchone()
    connection.close()

    return row_to_violation(row)


# ==========================================
# GET VIOLATIONS WITH LOCATION
# ==========================================

def get_hotspot_violations():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM violations
        WHERE latitude IS NOT NULL
        AND longitude IS NOT NULL
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    connection.close()

    return [
        row_to_violation(row)
        for row in rows
    ]


# ==========================================
# SEARCH VEHICLE
# ==========================================

def search_vehicle(vehicle_number):
    search_value = (
        str(vehicle_number)
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )

    if not search_value:
        return []

    violations = get_all_violations()

    matches = []

    for violation in violations:
        matched = False

        # ----------------------------------
        # CONFIRMED VEHICLE NUMBER
        # ----------------------------------

        stored_number = violation.get(
            "vehicle_number"
        )

        if stored_number:
            clean_number = (
                str(stored_number)
                .upper()
                .replace(" ", "")
                .replace("-", "")
            )

            if search_value in clean_number:
                matched = True

        # ----------------------------------
        # OCR CANDIDATES
        # ----------------------------------

        if not matched:
            candidates = (
                violation.get(
                    "plate_candidates"
                )
                or []
            )

            for candidate in candidates:
                candidate_number = (
                    candidate.get(
                        "vehicle_number"
                    )
                    or candidate.get(
                        "original_text"
                    )
                    or ""
                )

                clean_candidate = (
                    str(candidate_number)
                    .upper()
                    .replace(" ", "")
                    .replace("-", "")
                )

                if search_value in clean_candidate:
                    matched = True
                    break

        if matched:
            matches.append(violation)

    return matches


# ==========================================
# DATABASE STATISTICS
# ==========================================

def get_statistics():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM violations
        """
    )

    total = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM violations
        WHERE violation_type = 'NO HELMET'
        """
    )

    no_helmet = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM violations
        WHERE violation_type = 'TRIPLE RIDING'
        """
    )

    triple_riding = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM violations
        WHERE latitude IS NOT NULL
        AND longitude IS NOT NULL
        """
    )

    geotagged_violations = cursor.fetchone()[0]

    connection.close()

    return {
        "total_violations": total,
        "no_helmet": no_helmet,
        "triple_riding": triple_riding,
        "geotagged_violations":
            geotagged_violations
    }
# ==========================================
# DELETE VIOLATION
# ==========================================

def delete_violation(violation_id):
    connection = get_connection()
    cursor = connection.cursor()

    # Get the record first so app.py can also
    # delete its evidence image.
    cursor.execute(
        """
        SELECT *
        FROM violations
        WHERE id = ?
        """,
        (violation_id,)
    )

    row = cursor.fetchone()

    if row is None:
        connection.close()
        return None

    violation = row_to_violation(row)

    cursor.execute(
        """
        DELETE FROM violations
        WHERE id = ?
        """,
        (violation_id,)
    )

    connection.commit()
    connection.close()

    return violation

# ==========================================
# START DATABASE
# ==========================================

initialize_database()
