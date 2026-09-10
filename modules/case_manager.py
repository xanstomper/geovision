"""
GeoVision Case Manager
========================
SQLite-backed case management system.
Tables: cases, case_scans, case_notes
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',
    status      TEXT DEFAULT 'active',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id       INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    scan_id       TEXT NOT NULL,
    image_path    TEXT,
    best_lat      REAL,
    best_lon      REAL,
    best_conf     REAL,
    summary       TEXT DEFAULT '{}',
    scan_json_path TEXT,
    scan_html_path TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    note       TEXT NOT NULL,
    author     TEXT DEFAULT 'analyst',
    created_at TEXT NOT NULL
);
"""


class CaseManager:
    def __init__(self, db_path: str = "data/cases.db"):
        self._db_path = Path(db_path).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init_db(self):
        with self._conn() as con:
            con.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    # Cases                                                                #
    # ------------------------------------------------------------------ #

    def create_case(self, name: str, description: str = "", tags: List[str] = None) -> Dict[str, Any]:
        tags_json = json.dumps(tags or [])
        now = _now()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO cases (name,description,tags,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (name, description, tags_json, "active", now, now)
            )
            new_id = cur.lastrowid
        # Call get_case after the with-block so the INSERT is fully committed
        return self.get_case(new_id)

    def list_cases(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT c.*, COUNT(DISTINCT s.id) scan_count, COUNT(DISTINCT n.id) note_count "
                "FROM cases c "
                "LEFT JOIN case_scans s ON s.case_id=c.id "
                "LEFT JOIN case_notes n ON n.case_id=c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC"
            ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def get_case(self, case_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            row = con.execute(
                "SELECT c.*, "
                "(SELECT COUNT(*) FROM case_scans WHERE case_id=c.id) scan_count, "
                "(SELECT COUNT(*) FROM case_notes WHERE case_id=c.id) note_count "
                "FROM cases c WHERE c.id=?", (case_id,)
            ).fetchone()
            if not row:
                return None
            case = self._row_to_case(row)
            case["scans"] = [dict(s) for s in con.execute(
                "SELECT * FROM case_scans WHERE case_id=? ORDER BY created_at DESC", (case_id,)
            ).fetchall()]
            case["notes"] = [dict(n) for n in con.execute(
                "SELECT * FROM case_notes WHERE case_id=? ORDER BY created_at ASC", (case_id,)
            ).fetchall()]
            for s in case["scans"]:
                try:
                    s["summary"] = json.loads(s.get("summary") or "{}")
                except Exception:
                    s["summary"] = {}
        return case

    def update_case(self, case_id: int, **fields) -> Optional[Dict[str, Any]]:
        allowed = {"name","description","tags","status"}
        updates = {k: v for k,v in fields.items() if k in allowed}
        if "tags" in updates and isinstance(updates["tags"], list):
            updates["tags"] = json.dumps(updates["tags"])
        updates["updated_at"] = _now()
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [case_id]
        with self._conn() as con:
            con.execute(f"UPDATE cases SET {sets} WHERE id=?", vals)
        return self.get_case(case_id)

    def delete_case(self, case_id: int) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM cases WHERE id=?", (case_id,))
        return cur.rowcount > 0

    def search_cases(self, query: str) -> List[Dict[str, Any]]:
        q = f"%{query}%"
        with self._conn() as con:
            rows = con.execute(
                "SELECT c.*, COUNT(DISTINCT s.id) scan_count, COUNT(DISTINCT n.id) note_count "
                "FROM cases c "
                "LEFT JOIN case_scans s ON s.case_id=c.id "
                "LEFT JOIN case_notes n ON n.case_id=c.id "
                "WHERE c.name LIKE ? OR c.description LIKE ? OR c.tags LIKE ? "
                "GROUP BY c.id ORDER BY c.updated_at DESC",
                (q, q, q)
            ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def export_case(self, case_id: int) -> Optional[Dict[str, Any]]:
        case = self.get_case(case_id)
        if case:
            case["exported_at"] = _now()
            case["geovision_version"] = "2.0"
        return case

    # ------------------------------------------------------------------ #
    # Scans                                                                #
    # ------------------------------------------------------------------ #

    def add_scan_to_case(self, case_id: int, scan_id: str, image_path: str = None,
                          best_lat: float = None, best_lon: float = None,
                          best_conf: float = None, summary: dict = None,
                          scan_json_path: str = None, scan_html_path: str = None) -> Dict[str, Any]:
        now = _now()
        summary_json = json.dumps(summary or {})
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO case_scans (case_id,scan_id,image_path,best_lat,best_lon,"
                "best_conf,summary,scan_json_path,scan_html_path,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (case_id, scan_id, image_path, best_lat, best_lon, best_conf,
                 summary_json, scan_json_path, scan_html_path, now)
            )
            con.execute("UPDATE cases SET updated_at=? WHERE id=?", (now, case_id))
            row = con.execute("SELECT * FROM case_scans WHERE id=?", (cur.lastrowid,)).fetchone()
        r = dict(row)
        try: r["summary"] = json.loads(r.get("summary") or "{}")
        except Exception: r["summary"] = {}
        return r

    # ------------------------------------------------------------------ #
    # Notes                                                                #
    # ------------------------------------------------------------------ #

    def add_note(self, case_id: int, note: str, author: str = "analyst") -> Dict[str, Any]:
        now = _now()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO case_notes (case_id,note,author,created_at) VALUES (?,?,?,?)",
                (case_id, note, author, now)
            )
            con.execute("UPDATE cases SET updated_at=? WHERE id=?", (now, case_id))
            row = con.execute("SELECT * FROM case_notes WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _row_to_case(self, row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        return d
