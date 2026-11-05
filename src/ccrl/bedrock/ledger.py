"""The ledger: every request, every API attempt, every cent — durable.

SQLite (stdlib, WAL, synchronous=FULL), one file per workspace
(runs/ledger.sqlite). Two levels:

  requests  one row per LOGICAL request (job x item x model x effort x replicate);
            its identity is a content hash, so re-running a job skips what is done.
  attempts  one row per ACTUAL API call (retries and sequential turns are
            separate rows). Spend is summed over attempts, so a retried call
            is billed twice in the ledger exactly as it is billed by AWS.

Attempt status:
  inflight        dispatched, no outcome yet (counted at its reservation)
  ok              response received (parse outcome stored separately)
  truncated       response stopped at max_tokens (billed)
  refused         refusal / content filter (billed, final, never retried)
  uncertain       timeout or connection loss after sending: AWS may have
                  billed it, so it is counted at its reservation forever
  unbilled_error  rejected before inference (validation, access, quota) — 0 cost

On start-up `recover()` converts attempts left `inflight` by a crash into
`uncertain`: their cost is unknown, so the guard keeps counting them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ccrl.util import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests(
  request_id TEXT PRIMARY KEY,
  job TEXT NOT NULL, priority TEXT, model_key TEXT NOT NULL, model_id TEXT,
  item_id TEXT NOT NULL, case_id TEXT, transform TEXT, dataset TEXT,
  effort TEXT, replicate INTEGER DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',      -- pending|done|failed|refused
  n_attempts INTEGER DEFAULT 0, parse_ok INTEGER, informative INTEGER,
  final_attempt INTEGER, note TEXT,
  created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_req_job ON requests(job, model_key, status);
CREATE TABLE IF NOT EXISTS attempts(
  request_id TEXT NOT NULL, attempt INTEGER NOT NULL, turn INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  reservation REAL DEFAULT 0, cost_reported REAL DEFAULT 0, cost_guard REAL DEFAULT 0,
  input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, cache_write INTEGER,
  usage_semantics TEXT, stop_reason TEXT, served_tier TEXT, latency_ms INTEGER,
  throttles INTEGER DEFAULT 0, error_code TEXT, error_msg TEXT,
  request_json TEXT, response_json TEXT, text TEXT, reasoning TEXT, parsed_json TEXT,
  PRIMARY KEY(request_id, attempt, turn));
CREATE INDEX IF NOT EXISTS ix_att_status ON attempts(status);
CREATE TABLE IF NOT EXISTS events(ts TEXT, kind TEXT, job TEXT, model_key TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

BILLED = ("ok", "truncated", "refused")
RESERVED = ("inflight", "uncertain")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60,
                                    isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=FULL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(SCHEMA)

    # ------------------------------------------------------------ plumbing
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield self._con
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._con.execute(sql, args))

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def backup(self, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            out = sqlite3.connect(str(dest))
            self._con.backup(out)
            out.close()
        return dest

    # ------------------------------------------------------------ meta / events
    def set_meta(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        r = self.q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def event(self, kind: str, detail: Any = None, job: str | None = None, model_key: str | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                      (now_iso(), kind, job, model_key, json.dumps(detail, default=str) if detail is not None else None))

    # ------------------------------------------------------------ requests
    def ensure_requests(self, rows: list[dict]) -> int:
        """Insert logical requests that are not yet known. Returns #new."""
        now = now_iso()
        n = 0
        with self.tx() as c:
            for r in rows:
                cur = c.execute(
                    "INSERT OR IGNORE INTO requests(request_id,job,priority,model_key,model_id,item_id,case_id,"
                    "transform,dataset,effort,replicate,status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)",
                    (r["request_id"], r["job"], r.get("priority"), r["model_key"], r.get("model_id"), r["item_id"],
                     r.get("case_id"), r.get("transform"), r.get("dataset"), r.get("effort"),
                     int(r.get("replicate", 0)), now, now))
                n += cur.rowcount
        return n

    def request(self, request_id: str) -> sqlite3.Row | None:
        r = self.q("SELECT * FROM requests WHERE request_id=?", (request_id,))
        return r[0] if r else None

    def statuses(self, job: str) -> dict[str, str]:
        return {r["request_id"]: r["status"] for r in self.q("SELECT request_id,status FROM requests WHERE job=?", (job,))}

    def finish_request(self, request_id: str, status: str, final_attempt: int | None,
                       parse_ok: bool | None = None, informative: bool | None = None, note: str | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE requests SET status=?, final_attempt=?, parse_ok=?, informative=?, note=?, updated_at=? "
                      "WHERE request_id=?",
                      (status, final_attempt, None if parse_ok is None else int(parse_ok),
                       None if informative is None else int(informative), note, now_iso(), request_id))

    # ------------------------------------------------------------ attempts
    def next_attempt_no(self, request_id: str) -> int:
        r = self.q("SELECT COALESCE(MAX(attempt),0) AS m FROM attempts WHERE request_id=?", (request_id,))
        return int(r[0]["m"]) + 1

    def open_attempt(self, request_id: str, attempt: int, turn: int, reservation: float, request_json: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO attempts(request_id,attempt,turn,status,started_at,reservation,request_json) "
                      "VALUES(?,?,?,'inflight',?,?,?)", (request_id, attempt, turn, now_iso(), reservation, request_json))
            c.execute("UPDATE requests SET n_attempts=MAX(n_attempts,?), updated_at=? WHERE request_id=?",
                      (attempt, now_iso(), request_id))

    def close_attempt(self, request_id: str, attempt: int, turn: int, **f: Any) -> None:
        cols = ["status", "finished_at", "cost_reported", "cost_guard", "input_tokens", "output_tokens", "cache_read",
                "cache_write", "usage_semantics", "stop_reason", "served_tier", "latency_ms", "throttles",
                "error_code", "error_msg", "response_json", "text", "reasoning", "parsed_json"]
        f.setdefault("finished_at", now_iso())
        sets = ", ".join(f"{k}=?" for k in cols if k in f)
        vals = [f[k] for k in cols if k in f]
        with self.tx() as c:
            c.execute(f"UPDATE attempts SET {sets} WHERE request_id=? AND attempt=? AND turn=?",
                      (*vals, request_id, attempt, turn))

    def attempts(self, request_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM attempts WHERE request_id=? ORDER BY attempt, turn", (request_id,))

    def recover(self) -> int:
        """Attempts left in flight by a crash -> uncertain (still counted)."""
        with self.tx() as c:
            cur = c.execute("UPDATE attempts SET status='uncertain', error_code='crash_recovered', finished_at=? "
                            "WHERE status='inflight'", (now_iso(),))
            n = cur.rowcount
        if n:
            self.event("recovered_inflight", {"n": n})
        return n

    # ------------------------------------------------------------ spend
    def _spend(self, kind: str, job: str | None = None, model_key: str | None = None,
               priority: str | None = None) -> float:
        where, args = [], []
        if job:
            where.append("r.job=?")
            args.append(job)
        if model_key:
            where.append("r.model_key=?")
            args.append(model_key)
        if priority:
            where.append("r.priority=?")
            args.append(priority)
        w = (" AND " + " AND ".join(where)) if where else ""
        if kind == "guard":
            expr = ("CASE WHEN a.status IN ('ok','truncated','refused') THEN a.cost_guard "
                    "WHEN a.status IN ('inflight','uncertain') THEN a.reservation ELSE 0 END")
        else:
            expr = ("CASE WHEN a.status IN ('ok','truncated','refused') THEN a.cost_reported "
                    "WHEN a.status='uncertain' THEN a.reservation ELSE 0 END")
        r = self.q(f"SELECT COALESCE(SUM({expr}),0) AS s FROM attempts a JOIN requests r USING(request_id) WHERE 1=1{w}",
                   tuple(args))
        return float(r[0]["s"])

    def committed(self, job: str | None = None, model_key: str | None = None, priority: str | None = None) -> float:
        """Guard spend: what the budget guard compares against caps."""
        return self._spend("guard", job, model_key, priority)

    def reported(self, job: str | None = None, model_key: str | None = None) -> float:
        return self._spend("reported", job, model_key)

    def summary(self) -> list[dict]:
        rows = self.q(
            "SELECT r.job, r.model_key, r.status, COUNT(DISTINCT r.request_id) AS n FROM requests r "
            "GROUP BY r.job, r.model_key, r.status ORDER BY r.job, r.model_key")
        return [dict(x) for x in rows]

    def spend_table(self) -> list[dict]:
        rows = self.q(
            "SELECT r.job, r.model_key, COUNT(*) AS attempts, "
            "SUM(CASE WHEN a.status IN ('ok','truncated','refused') THEN a.cost_reported ELSE 0 END) AS reported, "
            "SUM(CASE WHEN a.status IN ('ok','truncated','refused') THEN a.cost_guard "
            "WHEN a.status IN ('inflight','uncertain') THEN a.reservation ELSE 0 END) AS guard, "
            "SUM(COALESCE(a.input_tokens,0)) AS in_tok, SUM(COALESCE(a.output_tokens,0)) AS out_tok, "
            "SUM(CASE WHEN a.status='uncertain' THEN 1 ELSE 0 END) AS uncertain "
            "FROM attempts a JOIN requests r USING(request_id) GROUP BY r.job, r.model_key ORDER BY r.job, r.model_key")
        return [dict(x) for x in rows]
