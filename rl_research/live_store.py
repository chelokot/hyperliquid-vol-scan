"""Local source of truth for the live system (sqlite, WAL so the dashboard can
read while the engine writes): per-second bars (restart-resilient warmup + future
retrain data), trades with realized PnL/fees, latest state snapshot, model
versions."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from features_v2 import OUT_DIR

DB_PATH = OUT_DIR / "live_store.db"


class LiveStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = str(path or DB_PATH)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS bars(
              date TEXT, symbol TEXT, session_second INTEGER, ts_ms INTEGER,
              perp_last REAL, perp_buy REAL, perp_sell REAL, perp_count INTEGER,
              stock_last REAL, stock_count INTEGER,
              PRIMARY KEY(date, symbol, session_second));
            CREATE TABLE IF NOT EXISTS trades(
              ts_ms INTEGER, symbol TEXT, side TEXT, size REAL, price REAL,
              target REAL, notional REAL, kind TEXT, live INTEGER, result TEXT);
            CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1), ts_ms INTEGER, payload TEXT);
            CREATE TABLE IF NOT EXISTS models(ts_ms INTEGER, path TEXT, summary TEXT);
            """
        )
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    # ---- bars ----
    def record_bar(self, date, symbol, sec, ts_ms, perp_last, perp_buy, perp_sell, perp_count, stock_last, stock_count) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO bars VALUES(?,?,?,?,?,?,?,?,?,?)",
            (date, symbol, int(sec), int(ts_ms), perp_last, perp_buy, perp_sell, int(perp_count), stock_last, int(stock_count)),
        )

    def commit(self) -> None:
        self._conn().commit()

    def load_bars(self, date, symbol):
        cur = self._conn().execute(
            "SELECT session_second, perp_last, perp_buy, perp_sell, perp_count, stock_last, stock_count "
            "FROM bars WHERE date=? AND symbol=? ORDER BY session_second",
            (date, symbol),
        )
        return cur.fetchall()

    # ---- trades ----
    def record_trade(self, symbol, side, size, price, target, notional, kind, live, result) -> None:
        c = self._conn()
        c.execute(
            "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?)",
            (int(time.time() * 1000), symbol, side, size, price, target, notional, kind, int(live), result),
        )
        c.commit()

    def recent_trades(self, limit=200):
        cur = self._conn().execute("SELECT ts_ms,symbol,side,size,price,target,notional,kind,result FROM trades ORDER BY ts_ms DESC LIMIT ?", (limit,))
        cols = ["ts_ms", "symbol", "side", "size", "price", "target", "notional", "kind", "result"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ---- state snapshot (dashboard) ----
    def record_state(self, payload: dict) -> None:
        c = self._conn()
        c.execute("INSERT OR REPLACE INTO state VALUES(1,?,?)", (int(time.time() * 1000), json.dumps(payload)))
        c.commit()

    def latest_state(self) -> dict | None:
        cur = self._conn().execute("SELECT ts_ms, payload FROM state WHERE id=1")
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row[1])
        data["ts_ms"] = row[0]
        return data

    # ---- models ----
    def record_model(self, path, summary) -> None:
        c = self._conn()
        c.execute("INSERT INTO models VALUES(?,?,?)", (int(time.time() * 1000), str(path), json.dumps(summary)))
        c.commit()

    def model_history(self, limit=50):
        cur = self._conn().execute("SELECT ts_ms,path,summary FROM models ORDER BY ts_ms DESC LIMIT ?", (limit,))
        return [{"ts_ms": r[0], "path": r[1], "summary": json.loads(r[2])} for r in cur.fetchall()]
