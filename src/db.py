"""PostgreSQL schema + helpers for the live DB (candles, daily bars, predictions, signals).

Connects to config.PG_HOST/PG_PORT/PG_DBNAME as config.PG_USER -- see
config.py for how those are set. The live loops (realtime_loop.py/
intraday_loop.py) and Postgres itself run on the same machine, so their
connection is effectively local; a dev machine reaches the same server over
the LAN by setting PG_HOST to that machine's LAN IP.
"""

import warnings

import psycopg2
import psycopg2.extras

from . import config

# pandas.read_sql_query only officially tests SQLAlchemy engines/sqlite3 --
# a raw psycopg2 connection works fine through its generic DBAPI2 fallback,
# but warns on every single call otherwise. Every caller here (widget.py,
# dashboard.py, the live loops) passes this module's connections straight
# into read_sql_query, so left unfiltered this would repeat in every log
# file forever.
warnings.filterwarnings(
    "ignore", message="pandas only supports SQLAlchemy connectable.*", category=UserWarning,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    date TEXT PRIMARY KEY,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION,
    source TEXT
);

CREATE TABLE IF NOT EXISTS regime_predictions (
    ts BIGINT PRIMARY KEY,
    date TEXT,
    label TEXT,
    prob_bearish DOUBLE PRECISION,
    prob_ranging DOUBLE PRECISION,
    prob_bullish DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS signals (
    ts BIGINT PRIMARY KEY,
    signal_type TEXT,
    regime TEXT,
    confidence DOUBLE PRECISION,
    price DOUBLE PRECISION,
    note TEXT
);

CREATE TABLE IF NOT EXISTS candles_m1 (
    time BIGINT PRIMARY KEY,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION,
    tick_volume INTEGER, spread INTEGER
);

CREATE TABLE IF NOT EXISTS candles_m5 (
    time BIGINT PRIMARY KEY,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION,
    tick_volume INTEGER, spread INTEGER
);

CREATE TABLE IF NOT EXISTS intraday_alerts (
    ts BIGINT PRIMARY KEY,
    event TEXT,
    direction TEXT,
    magnitude_atr DOUBLE PRECISION,
    horizon_min INTEGER,
    continuation_prob DOUBLE PRECISION,
    in_window INTEGER,
    window_name TEXT,
    price DOUBLE PRECISION,
    note TEXT
);

CREATE TABLE IF NOT EXISTS next_bar_predictions (
    ts BIGINT PRIMARY KEY,
    direction TEXT,
    confidence DOUBLE PRECISION,
    price DOUBLE PRECISION
);

-- Replaces next_bar_predictions on the live widget (retired 2026-08-05: that
-- model's holdout AUC was ~0.51 across every variant tried -- full history,
-- balanced weights, recent-only data, deeper trees -- none of it moved the
-- needle, and it ran bearish on 14/15 live calls, 27% hit rate, straight
-- through a sustained rally). This one asks a more tractable question --
-- 3-class bearish/ranging/bullish over a 15-min/3-bar window with a min-move
-- threshold, same shape as the daily regime model -- rather than raw
-- single-candle direction.
CREATE TABLE IF NOT EXISTS near_term_predictions (
    ts BIGINT PRIMARY KEY,
    label TEXT,
    confidence DOUBLE PRECISION,
    price DOUBLE PRECISION
);

-- Single pinned row: last candles_m5.time the burst-assessment state machine
-- has already run next_event() on. Lets every tick pick up exactly where it
-- left off (routine 5-min step, or a multi-day gap after downtime) and
-- replay any missed bars through the same logic instead of just storing them.
CREATE TABLE IF NOT EXISTS intraday_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_processed_time BIGINT
);
"""


def get_connection():
    conn = psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT, dbname=config.PG_DBNAME,
        user=config.PG_USER, password=config.PG_PASSWORD,
    )
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    return conn


def upsert_daily_bar(conn, date, open_, high, low, close, source) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO daily_bars (date, open, high, low, close, source)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                high=GREATEST(daily_bars.high, excluded.high), low=LEAST(daily_bars.low, excluded.low),
                close=excluded.close, source=excluded.source
            """,
            (date, open_, high, low, close, source),
        )
    conn.commit()


def insert_prediction(conn, ts, date, label, probs) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO regime_predictions
            (ts, date, label, prob_bearish, prob_ranging, prob_bullish)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ts) DO UPDATE SET
                date=excluded.date, label=excluded.label, prob_bearish=excluded.prob_bearish,
                prob_ranging=excluded.prob_ranging, prob_bullish=excluded.prob_bullish
            """,
            (ts, date, label, float(probs[0]), float(probs[1]), float(probs[2])),
        )
    conn.commit()


def insert_signal(conn, ts, signal_type, regime, confidence, price, note) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signals (ts, signal_type, regime, confidence, price, note)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ts) DO UPDATE SET
                signal_type=excluded.signal_type, regime=excluded.regime,
                confidence=excluded.confidence, price=excluded.price, note=excluded.note
            """,
            (ts, signal_type, regime, confidence, price, note),
        )
    conn.commit()


def get_last_signal(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, signal_type, regime, confidence, price, note FROM signals ORDER BY ts DESC LIMIT 1"
        )
        return cur.fetchone()


def upsert_m5_candles(conn, rows, page_size: int = 1000) -> None:
    """rows: iterable of (time, open, high, low, close, tick_volume, spread).
    No pruning, ever -- kept forever so the company can review the full
    history for any trend burst that might have affected client trades.
    Uses execute_values (multi-row VALUES per round-trip, batched by
    page_size) since this also carries the full CSV seed -- 1M+ rows -- on
    every intraday_loop startup; the one-time sqlite migration passes a
    larger page_size to cut round-trips further for that bulk load."""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO candles_m5 (time, open, high, low, close, tick_volume, spread)
            VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, tick_volume=excluded.tick_volume, spread=excluded.spread
            """,
            rows,
            page_size=page_size,
        )
    conn.commit()


def get_last_m5_time(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(time) FROM candles_m5")
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def get_last_closed_m5_time(conn, closed_before_epoch: int):
    """MAX(time) among bars that have actually finished forming as of
    closed_before_epoch (typically server_now - bar_duration). candles_m5's
    newest row is continuously re-upserted while its bar is still forming, so
    plain MAX(time) always reflects that in-progress bar, never a closed one."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(time) FROM candles_m5 WHERE time <= %s", (closed_before_epoch,))
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def get_intraday_cursor(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT last_processed_time FROM intraday_cursor WHERE id = 1")
        row = cur.fetchone()
        return row[0] if row else None


def set_intraday_cursor(conn, last_processed_time: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intraday_cursor (id, last_processed_time) VALUES (1, %s)
            ON CONFLICT (id) DO UPDATE SET last_processed_time = excluded.last_processed_time
            """,
            (last_processed_time,),
        )
    conn.commit()


def insert_intraday_alert(
    conn, ts, event, direction, magnitude_atr, horizon_min,
    continuation_prob, in_window, window_name, price, note, commit: bool = True,
) -> None:
    """commit=False lets a bulk historical replay (thousands of events in one
    tick) batch everything into a single transaction instead of fsync'ing
    per-row -- callers doing that must commit() once themselves afterward."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intraday_alerts
            (ts, event, direction, magnitude_atr, horizon_min, continuation_prob, in_window, window_name, price, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ts) DO UPDATE SET
                event=excluded.event, direction=excluded.direction, magnitude_atr=excluded.magnitude_atr,
                horizon_min=excluded.horizon_min, continuation_prob=excluded.continuation_prob,
                in_window=excluded.in_window, window_name=excluded.window_name,
                price=excluded.price, note=excluded.note
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


def get_last_intraday_alert(conn):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT ts, event, direction, magnitude_atr, horizon_min, continuation_prob,
                      in_window, window_name, price, note
               FROM intraday_alerts ORDER BY ts DESC LIMIT 1"""
        )
        return cur.fetchone()


def insert_near_term_prediction(conn, ts, label, confidence, price, commit: bool = True) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO near_term_predictions (ts, label, confidence, price)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ts) DO UPDATE SET
                label=excluded.label, confidence=excluded.confidence, price=excluded.price
            """,
            (ts, label, float(confidence), float(price)),
        )
    if commit:
        conn.commit()


def get_last_near_term_prediction(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT ts, label, confidence, price FROM near_term_predictions ORDER BY ts DESC LIMIT 1")
        return cur.fetchone()


def get_current_burst_start_ts(conn):
    """ts of the BURST_START that began the currently-active (not yet ended) burst, or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts FROM intraday_alerts
            WHERE event = 'BURST_START'
              AND ts > COALESCE((SELECT MAX(ts) FROM intraday_alerts WHERE event = 'BURST_END'), 0)
            ORDER BY ts DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else None
