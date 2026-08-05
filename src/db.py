"""SQLite schema + helpers for data/test.db (live candles, daily bars, predictions, signals)."""

import sqlite3

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS regime_predictions (
    ts INTEGER PRIMARY KEY,
    date TEXT,
    label TEXT,
    prob_bearish REAL,
    prob_ranging REAL,
    prob_bullish REAL
);

CREATE TABLE IF NOT EXISTS signals (
    ts INTEGER PRIMARY KEY,
    signal_type TEXT,
    regime TEXT,
    confidence REAL,
    price REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS candles_m1 (
    time INTEGER PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    tick_volume INTEGER, spread INTEGER
);

CREATE TABLE IF NOT EXISTS candles_m5 (
    time INTEGER PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    tick_volume INTEGER, spread INTEGER
);

CREATE TABLE IF NOT EXISTS intraday_alerts (
    ts INTEGER PRIMARY KEY,
    event TEXT,
    direction TEXT,
    magnitude_atr REAL,
    horizon_min INTEGER,
    continuation_prob REAL,
    in_window INTEGER,
    window_name TEXT,
    price REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS next_bar_predictions (
    ts INTEGER PRIMARY KEY,
    direction TEXT,
    confidence REAL,
    price REAL
);

-- Replaces next_bar_predictions on the live widget (retired 2026-08-05: that
-- model's holdout AUC was ~0.51 across every variant tried -- full history,
-- balanced weights, recent-only data, deeper trees -- none of it moved the
-- needle, and it ran bearish on 14/15 live calls, 27% hit rate, straight
-- through a sustained rally. This one asks a more tractable question --
-- 3-class bearish/ranging/bullish over a 15-min/3-bar window with a min-move
-- threshold, same shape as the daily regime model -- rather than raw
-- single-candle direction.
CREATE TABLE IF NOT EXISTS near_term_predictions (
    ts INTEGER PRIMARY KEY,
    label TEXT,
    confidence REAL,
    price REAL
);

-- Single pinned row: last candles_m5.time the burst-assessment state machine
-- has already run next_event() on. Lets every tick pick up exactly where it
-- left off (routine 5-min step, or a multi-day gap after downtime) and
-- replay any missed bars through the same logic instead of just storing them.
CREATE TABLE IF NOT EXISTS intraday_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_processed_time INTEGER
);
"""


def get_connection() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.executescript(SCHEMA)
    return conn


def upsert_daily_bar(conn: sqlite3.Connection, date, open_, high, low, close, source) -> None:
    conn.execute(
        """
        INSERT INTO daily_bars (date, open, high, low, close, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            high=MAX(high, excluded.high), low=MIN(low, excluded.low),
            close=excluded.close, source=excluded.source
        """,
        (date, open_, high, low, close, source),
    )
    conn.commit()


def insert_prediction(conn: sqlite3.Connection, ts, date, label, probs) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO regime_predictions
        (ts, date, label, prob_bearish, prob_ranging, prob_bullish)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ts, date, label, float(probs[0]), float(probs[1]), float(probs[2])),
    )
    conn.commit()


def insert_signal(conn: sqlite3.Connection, ts, signal_type, regime, confidence, price, note) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO signals (ts, signal_type, regime, confidence, price, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ts, signal_type, regime, confidence, price, note),
    )
    conn.commit()


def get_last_signal(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT ts, signal_type, regime, confidence, price, note FROM signals ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    return row


def upsert_m5_candles(conn: sqlite3.Connection, rows) -> None:
    """rows: iterable of (time, open, high, low, close, tick_volume, spread).
    No pruning, ever -- kept forever so the company can review the full
    history for any trend burst that might have affected client trades."""
    conn.executemany(
        """
        INSERT INTO candles_m5 (time, open, high, low, close, tick_volume, spread)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(time) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, tick_volume=excluded.tick_volume, spread=excluded.spread
        """,
        rows,
    )
    conn.commit()


def get_last_m5_time(conn: sqlite3.Connection):
    row = conn.execute("SELECT MAX(time) FROM candles_m5").fetchone()
    return row[0] if row and row[0] is not None else None


def get_last_closed_m5_time(conn: sqlite3.Connection, closed_before_epoch: int):
    """MAX(time) among bars that have actually finished forming as of
    closed_before_epoch (typically server_now - bar_duration). candles_m5's
    newest row is continuously re-upserted while its bar is still forming, so
    plain MAX(time) always reflects that in-progress bar, never a closed one."""
    row = conn.execute(
        "SELECT MAX(time) FROM candles_m5 WHERE time <= ?", (closed_before_epoch,)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def get_intraday_cursor(conn: sqlite3.Connection):
    row = conn.execute("SELECT last_processed_time FROM intraday_cursor WHERE id = 1").fetchone()
    return row[0] if row else None


def set_intraday_cursor(conn: sqlite3.Connection, last_processed_time: int) -> None:
    conn.execute(
        """
        INSERT INTO intraday_cursor (id, last_processed_time) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET last_processed_time = excluded.last_processed_time
        """,
        (last_processed_time,),
    )
    conn.commit()


def insert_intraday_alert(
    conn: sqlite3.Connection, ts, event, direction, magnitude_atr, horizon_min,
    continuation_prob, in_window, window_name, price, note, commit: bool = True,
) -> None:
    """commit=False lets a bulk historical replay (thousands of events in one
    tick) batch everything into a single transaction instead of fsync'ing
    per-row -- callers doing that must commit() once themselves afterward."""
    conn.execute(
        """
        INSERT OR REPLACE INTO intraday_alerts
        (ts, event, direction, magnitude_atr, horizon_min, continuation_prob, in_window, window_name, price, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts, event, direction,
            None if magnitude_atr is None else float(magnitude_atr),
            horizon_min,
            None if continuation_prob is None else float(continuation_prob),
            int(bool(in_window)), window_name, float(price), note,
        ),
    )
    if commit:
        conn.commit()


def get_last_intraday_alert(conn: sqlite3.Connection):
    row = conn.execute(
        """SELECT ts, event, direction, magnitude_atr, horizon_min, continuation_prob,
                  in_window, window_name, price, note
           FROM intraday_alerts ORDER BY ts DESC LIMIT 1"""
    ).fetchone()
    return row


def insert_near_term_prediction(conn: sqlite3.Connection, ts, label, confidence, price, commit: bool = True) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO near_term_predictions (ts, label, confidence, price)
        VALUES (?, ?, ?, ?)
        """,
        (ts, label, float(confidence), float(price)),
    )
    if commit:
        conn.commit()


def get_last_near_term_prediction(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT ts, label, confidence, price FROM near_term_predictions ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    return row


def get_current_burst_start_ts(conn: sqlite3.Connection):
    """ts of the BURST_START that began the currently-active (not yet ended) burst, or None."""
    row = conn.execute(
        """
        SELECT ts FROM intraday_alerts
        WHERE event = 'BURST_START'
          AND ts > COALESCE((SELECT MAX(ts) FROM intraday_alerts WHERE event = 'BURST_END'), 0)
        ORDER BY ts DESC LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None