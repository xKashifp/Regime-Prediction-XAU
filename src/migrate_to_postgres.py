"""One-time migration: copies every table out of the old SQLite data/test.db
into the new Postgres DB (config.PG_*). Safe to re-run -- every write here
goes through the same ON CONFLICT upserts the live loops use (or an
equivalent for the two retired tables below), so re-running just re-applies
the same rows.

candles_m1 and next_bar_predictions are dead tables (nothing in the live
pipeline reads or writes them anymore -- see db.py's schema comments), but
whatever rows they still hold get carried over too rather than silently
dropped, same "never throw away history" rule the rest of this system follows
for candles_m5/intraday_alerts.

Run once, after Postgres is reachable:
    python -m src.migrate_to_postgres
"""

import sqlite3

import pandas as pd
import psycopg2.extras

from . import config, db

SQLITE_PATH = config.ROOT / "data" / "test.db"


def _none(x):
    return None if pd.isna(x) else x


def _migrate_daily_bars(sconn, pconn) -> int:
    df = pd.read_sql_query("SELECT date, open, high, low, close, source FROM daily_bars", sconn)
    for r in df.itertuples(index=False):
        db.upsert_daily_bar(pconn, r.date, r.open, r.high, r.low, r.close, _none(r.source))
    return len(df)


def _migrate_regime_predictions(sconn, pconn) -> int:
    df = pd.read_sql_query(
        "SELECT ts, date, label, prob_bearish, prob_ranging, prob_bullish FROM regime_predictions", sconn,
    )
    for r in df.itertuples(index=False):
        db.insert_prediction(pconn, int(r.ts), r.date, r.label, (r.prob_bearish, r.prob_ranging, r.prob_bullish))
    return len(df)


def _migrate_signals(sconn, pconn) -> int:
    df = pd.read_sql_query("SELECT ts, signal_type, regime, confidence, price, note FROM signals", sconn)
    for r in df.itertuples(index=False):
        db.insert_signal(pconn, int(r.ts), r.signal_type, r.regime, r.confidence, r.price, _none(r.note))
    return len(df)


def _migrate_intraday_alerts(sconn, pconn) -> int:
    df = pd.read_sql_query(
        """SELECT ts, event, direction, magnitude_atr, horizon_min, continuation_prob,
                  in_window, window_name, price, note FROM intraday_alerts""",
        sconn,
    )
    for i, r in enumerate(df.itertuples(index=False)):
        db.insert_intraday_alert(
            pconn, int(r.ts), r.event, r.direction, _none(r.magnitude_atr), _none(r.horizon_min),
            _none(r.continuation_prob), r.in_window, _none(r.window_name), r.price, _none(r.note),
            commit=False,
        )
        if (i + 1) % 500 == 0:
            pconn.commit()
    pconn.commit()
    return len(df)


def _migrate_near_term_predictions(sconn, pconn) -> int:
    df = pd.read_sql_query("SELECT ts, label, confidence, price FROM near_term_predictions", sconn)
    for r in df.itertuples(index=False):
        db.insert_near_term_prediction(pconn, int(r.ts), r.label, r.confidence, r.price, commit=False)
    pconn.commit()
    return len(df)


def _migrate_intraday_cursor(sconn, pconn) -> int:
    df = pd.read_sql_query("SELECT last_processed_time FROM intraday_cursor WHERE id = 1", sconn)
    if df.empty:
        return 0
    db.set_intraday_cursor(pconn, int(df.iloc[0]["last_processed_time"]))
    return 1


def _migrate_candles_m5(sconn, pconn) -> int:
    df = pd.read_sql_query(
        "SELECT time, open, high, low, close, tick_volume, spread FROM candles_m5 ORDER BY time", sconn,
    )
    rows = list(df.itertuples(index=False, name=None))
    db.upsert_m5_candles(pconn, rows, page_size=10000)
    return len(rows)


def _migrate_candles_m1(sconn, pconn) -> int:
    """No live writer left for this table -- upserted directly here rather
    than adding a permanent db.py helper nothing else would ever call."""
    df = pd.read_sql_query(
        "SELECT time, open, high, low, close, tick_volume, spread FROM candles_m1 ORDER BY time", sconn,
    )
    if df.empty:
        return 0
    with pconn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO candles_m1 (time, open, high, low, close, tick_volume, spread)
            VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, tick_volume=excluded.tick_volume, spread=excluded.spread
            """,
            list(df.itertuples(index=False, name=None)),
        )
    pconn.commit()
    return len(df)


def _migrate_next_bar_predictions(sconn, pconn) -> int:
    """Retired 2026-08-05 (superseded by near_term_predictions) -- carried
    over for the historical record, same reasoning as candles_m1 above."""
    df = pd.read_sql_query("SELECT ts, direction, confidence, price FROM next_bar_predictions ORDER BY ts", sconn)
    if df.empty:
        return 0
    with pconn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO next_bar_predictions (ts, direction, confidence, price)
            VALUES %s
            ON CONFLICT (ts) DO UPDATE SET
                direction=excluded.direction, confidence=excluded.confidence, price=excluded.price
            """,
            list(df.itertuples(index=False, name=None)),
        )
    pconn.commit()
    return len(df)


def main():
    if not SQLITE_PATH.exists():
        raise SystemExit(f"No SQLite DB found at {SQLITE_PATH} -- nothing to migrate.")

    sconn = sqlite3.connect(str(SQLITE_PATH))
    pconn = db.get_connection()

    steps = [
        ("daily_bars", _migrate_daily_bars),
        ("regime_predictions", _migrate_regime_predictions),
        ("signals", _migrate_signals),
        ("candles_m1", _migrate_candles_m1),
        ("candles_m5", _migrate_candles_m5),
        ("intraday_alerts", _migrate_intraday_alerts),
        ("next_bar_predictions", _migrate_next_bar_predictions),
        ("near_term_predictions", _migrate_near_term_predictions),
        ("intraday_cursor", _migrate_intraday_cursor),
    ]
    for name, fn in steps:
        n = fn(sconn, pconn)
        print(f"{name}: migrated {n} row(s).")

    sconn.close()
    pconn.close()
    print("Done.")


if __name__ == "__main__":
    main()
