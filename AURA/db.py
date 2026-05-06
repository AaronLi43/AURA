"""SQLite storage for the AURA privacy pipeline."""
from __future__ import annotations

import json
import sqlite3
import sys

import pipeline_config as cfg


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(cfg.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id       TEXT PRIMARY KEY,
            original_text     TEXT NOT NULL,
            blacklist_json    TEXT,
            insight_profile_json TEXT,
            privacy_inferences_json TEXT,
            final_text        TEXT,
            final_privacy_score REAL,
            status            TEXT DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS iterations (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id             TEXT REFERENCES documents(document_id),
            iteration_num           INTEGER,
            template_text           TEXT,
            mask_map_json           TEXT,
            variations_json         TEXT,
            best_variation_idx      INTEGER,
            attacker_report_json    TEXT,
            keeper_report_json      TEXT,
            modulator_output_json   TEXT,
            masker_rules_json       TEXT,
            refiller_rules_json     TEXT,
            blacklist_snapshot_json TEXT
        );
    """)
    conn.commit()
    conn.close()
    print(f"Database initialised at {cfg.DB_PATH}")


# ── Document CRUD ──────────────────────────────────────────────────────

def upsert_document(document_id: str, **kwargs):
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM documents WHERE document_id=?", (document_id,)
    ).fetchone()

    json_keys = {
        "blacklist": "blacklist_json",
        "evidence_spans": "blacklist_json",
        "insight_profile": "insight_profile_json",
        "privacy_inferences": "privacy_inferences_json",
    }

    if existing:
        updates, vals = [], []
        for key, val in kwargs.items():
            if val is None:
                continue
            col = json_keys.get(key, key)
            if key in json_keys:
                updates.append(f"{col}=?")
                vals.append(json.dumps(val))
            else:
                updates.append(f"{col}=?")
                vals.append(val)
        if updates:
            vals.append(document_id)
            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE document_id=?",
                vals,
            )
    else:
        evidence_spans = kwargs.get("evidence_spans")
        blacklist = kwargs.get("blacklist")
        blacklist_payload = blacklist if blacklist is not None else evidence_spans
        conn.execute(
            "INSERT INTO documents "
            "(document_id, original_text, blacklist_json, insight_profile_json, "
            "privacy_inferences_json, final_text, final_privacy_score, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                document_id,
                kwargs.get("original_text", ""),
                json.dumps(blacklist_payload) if blacklist_payload is not None else None,
                json.dumps(kwargs.get("insight_profile")) if kwargs.get("insight_profile") is not None else None,
                json.dumps(kwargs.get("privacy_inferences")) if kwargs.get("privacy_inferences") is not None else None,
                kwargs.get("final_text"),
                kwargs.get("final_privacy_score"),
                kwargs.get("status", "pending"),
            ),
        )
    conn.commit()
    conn.close()


def get_document(document_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM documents WHERE document_id=?", (document_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    for json_col, key in [
        ("blacklist_json", "blacklist"),
        ("insight_profile_json", "insight_profile"),
        ("privacy_inferences_json", "privacy_inferences"),
    ]:
        d[key] = json.loads(d[json_col]) if d.get(json_col) else ([] if key == "blacklist" else None)
    d["evidence_spans"] = list(d.get("blacklist") or [])
    return d


def get_all_documents(status: str | None = None) -> list[dict]:
    conn = _connect()
    if status:
        rows = conn.execute("SELECT * FROM documents WHERE status=?", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents").fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        for json_col, key in [
            ("blacklist_json", "blacklist"),
            ("insight_profile_json", "insight_profile"),
            ("privacy_inferences_json", "privacy_inferences"),
        ]:
            d[key] = json.loads(d[json_col]) if d.get(json_col) else ([] if key == "blacklist" else None)
        d["evidence_spans"] = list(d.get("blacklist") or [])
        results.append(d)
    return results


# ── Iteration CRUD ─────────────────────────────────────────────────────

def insert_iteration(document_id: str, iteration_num: int, **kwargs) -> int:
    conn = _connect()
    json_fields = {}
    for key in (
        "template_text", "mask_map_json", "variations_json",
        "best_variation_idx", "attacker_report_json", "keeper_report_json",
        "modulator_output_json", "masker_rules_json", "refiller_rules_json",
        "blacklist_snapshot_json",
    ):
        val = kwargs.get(key)
        if val is not None and not isinstance(val, (str, int)):
            json_fields[key] = json.dumps(val)
        elif val is not None:
            json_fields[key] = val

    cols = ["document_id", "iteration_num"] + list(json_fields.keys())
    placeholders = ",".join(["?"] * len(cols))
    vals = [document_id, iteration_num] + list(json_fields.values())
    cur = conn.execute(
        f"INSERT INTO iterations ({','.join(cols)}) VALUES ({placeholders})", vals
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_iterations(document_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM iterations WHERE document_id=? ORDER BY iteration_num",
        (document_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Export ─────────────────────────────────────────────────────────────

def export_document(document_id: str) -> dict:
    doc = get_document(document_id)
    if doc is None:
        return {}
    doc["iterations"] = get_iterations(document_id)
    for key in list(doc.keys()):
        if key.endswith("_json"):
            del doc[key]
    return doc


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 db.py {init|status}")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "init":
        init_db()
    elif cmd == "status":
        conn = _connect()
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM documents GROUP BY status"
        ).fetchall()
        conn.close()
        for row in rows:
            print(f"  {row['status']}: {row['cnt']}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
