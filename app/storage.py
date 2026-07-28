"""
storage.py
SQLite-based session persistence for the AI Governance Dashboard.
Stores every analysis result for trend tracking and audit trails.
"""

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager


# ── Database path ─────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "data/sessions.db")


# ── Connection manager ────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Creates tables if they do not already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                text            TEXT    NOT NULL,
                word_count      INTEGER,
                risk_level      TEXT,
                toxicity        REAL,
                severe_toxicity REAL,
                identity_attack REAL,
                threat          REAL,
                sentiment       REAL,
                gender_bias     REAL,
                racial_bias     REAL,
                is_compliant    INTEGER,
                violation_count INTEGER,
                violations_json TEXT,
                flags_json      TEXT,
                report_path     TEXT
            );

            CREATE TABLE IF NOT EXISTS violations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                category    TEXT,
                severity    TEXT,
                matched_on  TEXT,
                confidence  REAL
            );
        """)
    print(f"Database initialised at {DB_PATH}")


# ── Write ─────────────────────────────────────────────────────────────────────

def save_session(
    analysis_result: dict,
    policy_result,
    report_path: str = None,
) -> int:
    """
    Saves a full analysis session to the database.
    Returns the session ID.
    """
    tox   = analysis_result.get("toxicity", {})
    bias  = analysis_result.get("bias", {})
    sent  = analysis_result.get("sentiment", {})

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO sessions (
                timestamp, text, word_count, risk_level,
                toxicity, severe_toxicity, identity_attack, threat,
                sentiment, gender_bias, racial_bias,
                is_compliant, violation_count,
                violations_json, flags_json, report_path
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            analysis_result.get("text", ""),
            analysis_result.get("word_count", 0),
            analysis_result.get("risk_level", "UNKNOWN"),
            tox.get("toxicity", 0.0),
            tox.get("severe_toxicity", 0.0),
            tox.get("identity_attack", 0.0),
            tox.get("threat", 0.0),
            sent.get("compound", 0.0),
            bias.get("gender_bias", None),
            bias.get("racial_bias", None),
            int(policy_result.is_compliant),
            len(policy_result.violations),
            json.dumps([v.category for v in policy_result.violations]),
            json.dumps(analysis_result.get("flags", [])),
            report_path,
        ))
        session_id = cursor.lastrowid

        # Save individual violations
        for v in policy_result.violations:
            conn.execute("""
                INSERT INTO violations (session_id, category, severity, matched_on, confidence)
                VALUES (?,?,?,?,?)
            """, (session_id, v.category, v.severity.value, v.matched_on, v.confidence))

    return session_id


# ── Read ──────────────────────────────────────────────────────────────────────

def get_all_sessions(limit: int = 500) -> list[dict]:
    """Returns all sessions ordered by most recent first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM sessions ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_session_by_id(session_id: int) -> dict | None:
    """Returns a single session by ID, including its violations."""
    with get_connection() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        if not session:
            return None

        violations = conn.execute(
            "SELECT * FROM violations WHERE session_id = ?", (session_id,)
        ).fetchall()

    result = dict(session)
    result["violations"] = [dict(v) for v in violations]
    return result


def get_stats() -> dict:
    """Returns aggregate statistics across all sessions."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        if total == 0:
            return {
                "total_sessions":    0,
                "high_risk_count":   0,
                "flagged_count":     0,
                "avg_toxicity":      0.0,
                "avg_sentiment":     0.0,
                "top_violation":     None,
            }

        high_risk = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE risk_level = 'HIGH'"
        ).fetchone()[0]

        flagged = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE is_compliant = 0"
        ).fetchone()[0]

        avg_tox = conn.execute(
            "SELECT AVG(toxicity) FROM sessions"
        ).fetchone()[0]

        avg_sent = conn.execute(
            "SELECT AVG(sentiment) FROM sessions"
        ).fetchone()[0]

        top_violation = conn.execute("""
            SELECT category, COUNT(*) as cnt
            FROM violations
            GROUP BY category
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

    return {
        "total_sessions":  total,
        "high_risk_count": high_risk,
        "flagged_count":   flagged,
        "avg_toxicity":    round(avg_tox or 0.0, 4),
        "avg_sentiment":   round(avg_sent or 0.0, 4),
        "top_violation":   dict(top_violation) if top_violation else None,
    }


def get_violation_counts() -> list[dict]:
    """Returns violation frequency by category for charting."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT category, severity, COUNT(*) as count
            FROM violations
            GROUP BY category, severity
            ORDER BY count DESC
        """).fetchall()
    return [dict(row) for row in rows]


def export_sessions_csv() -> str:
    """Exports all sessions to a CSV string."""
    import csv
    import io
    sessions = get_all_sessions()
    if not sessions:
        return ""
    fieldnames = list(sessions[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sessions)
    return output.getvalue()


def delete_session(session_id: int) -> bool:
    """Deletes a session and its violations. Returns True if deleted."""
    with get_connection() as conn:
        conn.execute("DELETE FROM violations WHERE session_id = ?", (session_id,))
        result = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return result.rowcount > 0


def clear_all_sessions():
    """Deletes all sessions and violations. Use with caution."""
    with get_connection() as conn:
        conn.execute("DELETE FROM violations")
        conn.execute("DELETE FROM sessions")
    print("All sessions cleared.")
