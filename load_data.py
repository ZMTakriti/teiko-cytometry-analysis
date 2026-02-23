"""
load_data.py
Initializes a SQLite database and loads cell-count.csv into a normalized schema.

Schema:
  projects  (project_id)
  subjects  (subject_id, project_id, condition, age, sex)
  samples   (sample_id, subject_id, sample_type, time_from_treatment_start,
             treatment, response, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)

Run: python load_data.py
"""

import csv
import sqlite3
import os

DB_PATH = "teiko.db"
CSV_PATH = "cell-count.csv"

CREATE_PROJECTS = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY
);
"""

CREATE_SUBJECTS = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id  TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(project_id),
    condition   TEXT,
    age         INTEGER,
    sex         TEXT
);
"""

CREATE_SAMPLES = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id                   TEXT PRIMARY KEY,
    subject_id                  TEXT NOT NULL REFERENCES subjects(subject_id),
    sample_type                 TEXT,
    time_from_treatment_start   INTEGER,
    treatment                   TEXT,
    response                    TEXT,
    b_cell                      INTEGER,
    cd8_t_cell                  INTEGER,
    cd4_t_cell                  INTEGER,
    nk_cell                     INTEGER,
    monocyte                    INTEGER
);
"""


def init_db(conn):
    cur = conn.cursor()
    cur.executescript(CREATE_PROJECTS + CREATE_SUBJECTS + CREATE_SAMPLES)
    conn.commit()


def load_csv(conn, csv_path):
    cur = conn.cursor()

    projects_seen = set()
    subjects_seen = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            project_id = row["project"]
            subject_id = row["subject"]

            if project_id not in projects_seen:
                cur.execute(
                    "INSERT OR IGNORE INTO projects (project_id) VALUES (?)",
                    (project_id,),
                )
                projects_seen.add(project_id)

            if subject_id not in subjects_seen:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO subjects
                        (subject_id, project_id, condition, age, sex)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        subject_id,
                        project_id,
                        row["condition"],
                        int(row["age"]) if row["age"] else None,
                        row["sex"],
                    ),
                )
                subjects_seen.add(subject_id)

            response = row["response"] if row["response"] else None
            cur.execute(
                """
                INSERT OR IGNORE INTO samples
                    (sample_id, subject_id, sample_type,
                     time_from_treatment_start, treatment, response,
                     b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["sample"],
                    subject_id,
                    row["sample_type"],
                    int(row["time_from_treatment_start"]) if row["time_from_treatment_start"] else None,
                    row["treatment"],
                    response,
                    int(row["b_cell"]),
                    int(row["cd8_t_cell"]),
                    int(row["cd4_t_cell"]),
                    int(row["nk_cell"]),
                    int(row["monocyte"]),
                ),
            )

    conn.commit()


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        print("Initializing schema...")
        init_db(conn)

        print(f"Loading data from {CSV_PATH}...")
        load_csv(conn, CSV_PATH)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM projects")
        n_projects = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM subjects")
        n_subjects = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM samples")
        n_samples = cur.fetchone()[0]

        print(f"Done. Loaded into {DB_PATH}:")
        print(f"  {n_projects} project(s)")
        print(f"  {n_subjects} subject(s)")
        print(f"  {n_samples} sample(s)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
