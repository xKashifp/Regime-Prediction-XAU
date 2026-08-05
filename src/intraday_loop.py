"""Live 5-minute session-open burst watch.

Switched from 1-minute to 5-minute bars: the broker's M1 feed was too thin/
noisy to train on reliably. 5m bars are seeded once from a bulk CSV export
(data/raw/XAU_5m_data.csv, 2004 -> present) and grown forever afterward --
candles_m5 is never pruned, so the company can always look back and assess
whether a burst happened that could've affected client trades.

Event-driven, not a wall-clock timer: every ~10s (config.INTRADAY_CHECK_
INTERVAL_SECONDS) it asks the broker's own clock (MT5Connection.server_now())
whether a bar has actually finished closing since we last looked, and only
does the expensive reload+recompute when one genuinely has -- this machine's
own clock isn't guaranteed to line up with the broker's (confirmed off by
~3h at one point), so waiting for an assumed wall-clock boundary was the
wrong idea; detecting the real close directly from MT5 fixes that regardless
of any local clock drift.

Every time there's new data: figure out the last bar already saved, pull
anything MT5 has since then (covers both the routine "one new bar" case and
a multi-day gap after downtime/restart in the exact same code path), then
replay the burst-detection state machine over every bar not yet processed
(tracked via intraday_cursor) -- so a missed weekend or a crashed process
doesn't leave a silent hole in the alert log, it gets assessed retroactively
the moment this comes back up. Toast notifications only fire for bars close
to real "now"; historical catch-up is logged, not popped up.

No auto-trading, ever -- alert-only, so a human can decide when to close a
client trade by hand.

Usage:
    python -m src.intraday_loop            # runs forever
    python -m src.intraday_loop --once     # single iteration, for testing
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import xgboost as xgb

from . import config, db, session_windows
from .csv_seed import append_m5_rows, seed_candles_m5
from .intraday_features import build_intraday_feature_frame, get_feature_columns
from .intraday_signals import EVENT_CONTINUE, EVENT_END, EVENT_START, next_event
from .mt5_client import MT5Connection

try:
    from win11toast import notify as toast_notify
except ImportError:
    toast_notify = None

# A replayed (catch-up) bar older than this is logged only, never popped up as
# a toast -- otherwise a multi-day gap replay would fire a flood of desktop
# notifications for things that already happened.
LIVE_BAR_STALENESS_SEC = 2 * config.INTRADAY_BAR_MINUTES * 60


def setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intraday")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(config.LOG_DIR / "intraday.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_model():
    if not config.INTRADAY_MODEL_PATH.exists():
        return None, None
    booster = xgb.Booster()
    booster.load_model(str(config.INTRADAY_MODEL_PATH))
    feature_cols = json.loads(config.INTRADAY_FEATURE_COLUMNS_PATH.read_text())
    return booster, feature_cols


def load_near_term_model():
    if not config.NEAR_TERM_MODEL_PATH.exists():
        return None, None
    booster = xgb.Booster()
    booster.load_model(str(config.NEAR_TERM_MODEL_PATH))
    feature_cols = json.loads(config.NEAR_TERM_FEATURE_COLUMNS_PATH.read_text())
    return booster, feature_cols


def get_previous_direction(conn):
    last = db.get_last_intraday_alert(conn)
    if last is None:
        return None
    event, direction = last[1], last[2]
    if event == EVENT_END:
        return None
    return direction


def is_long_hike(magnitude: float) -> bool:
    """'LONG' here means a big hike worth worrying about (company-loss risk),
    within this same <=2h burst window -- NOT a multi-day/long-term thing."""
    return magnitude >= config.BURST_LONG_MAGNITUDE_ATR


def build_note(event, direction, magnitude, horizon_min, status, continuation_prob) -> str:
    window_txt = f"in the {status['window_name']} session window" if status["in_window"] else "outside the usual watch windows"

    if event == EVENT_END:
        return f"{direction} burst has faded -- no longer meeting the volatility threshold, {window_txt}"

    tier_txt = "LONG (big hike) " if is_long_hike(magnitude) else ""
    base = f"{tier_txt}{direction} move, {magnitude:.1f}x normal volatility over the last {horizon_min} min, {window_txt}"
    if continuation_prob is not None:
        base += f"; model continuation confidence {continuation_prob:.0%} (experimental -- weak signal, don't over-trust it)"
    return base


def fill_gap(logger: logging.Logger) -> int:
    """Fetches whatever candles_m5 is missing, from the last saved bar up to
    now, in one shot -- the same path whether that gap is 5 minutes (routine
    tick) or several days (weekend/restart/crash).

    Returns the broker's own current server time (epoch seconds) -- NOT this
    machine's clock. Those can be hours apart (confirmed on this machine: the
    system UTC clock ran ~3h behind this broker's server clock), and every
    bar epoch here is a broker-server-time epoch. Using the wrong "now" here
    was silently truncating every live fetch to a stale cutoff -- the chart
    and every alert were stuck ~3h in the past, never actually reaching the
    live edge."""
    conn = db.get_connection()
    last_time = db.get_last_m5_time(conn)
    conn.close()

    with MT5Connection() as mt5c:
        end = mt5c.server_now()
        start = end - timedelta(days=7) if last_time is None else datetime.fromtimestamp(last_time, tz=timezone.utc)
        rates = mt5c.fetch_m5_range(start, end)

    if rates is None or len(rates) == 0:
        return int(end.timestamp())

    rows = [
        (int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]),
         float(r["close"]), int(r["tick_volume"]), int(r["spread"]))
        for r in rates
    ]
    conn = db.get_connection()
    db.upsert_m5_candles(conn, rows)
    conn.close()
    # A routine tick only ever touches the still-forming bar (plus, right at a
    # boundary, the bar that just closed) -- 2 rows at most. Anything bigger
    # is a genuine catch-up (restart/weekend gap) worth a log line; logging
    # every single 10s tick otherwise just fills the file with noise, one line
    # per poll, forever.
    if len(rows) > 2:
        logger.info("Gap-fill: upserted %d M5 bars (%s -> %s).", len(rows), start, end)
    return int(end.timestamp())


def run_once(booster, feature_cols, near_term_booster, near_term_feature_cols, logger: logging.Logger) -> None:
    server_now_epoch = fill_gap(logger)
    bar_seconds = config.INTRADAY_BAR_MINUTES * 60

    conn = db.get_connection()

    # Cheap short-circuit, checked every ~10s: has a bar actually finished
    # closing (per the broker's own clock) since we last processed one? This
    # is the "detect a candle when it opens" check -- driven by real MT5 data,
    # not a wall-clock guess -- and it's just two small lookups, so polling
    # this often costs nothing. The expensive reload+feature-recompute below
    # only runs once there's genuinely new closed data to look at.
    #
    # candles_m5's newest row is the *currently forming* bar -- it gets
    # re-upserted (OHLC updated) on every tick until the next boundary, at
    # which point MAX(time) jumps straight to the new forming bar in the same
    # update. So MAX(time) is never the bar that just closed, and comparing
    # against it directly here always reads as "still forming" -- this used
    # to compare last_m5_time+bar_seconds<=server_now_epoch and, as a result,
    # never fired once in production (get_last_closed_m5_time below looks for
    # the newest bar older than the closing cutoff instead).
    cursor = db.get_intraday_cursor(conn)
    last_closed_time = db.get_last_closed_m5_time(conn, server_now_epoch - bar_seconds)
    a_new_bar_has_closed = cursor is None or (
        last_closed_time is not None and last_closed_time > cursor
    )
    if not a_new_bar_has_closed:
        conn.close()
        return

    m5 = pd.read_sql_query(
        """SELECT time, open, high, low, close, tick_volume FROM candles_m5
           ORDER BY time DESC LIMIT ?""",
        conn, params=(config.INTRADAY_MAX_BARS_LOADED,),
    )
    if len(m5) < 60:
        logger.warning("Not enough M5 history yet (%d bars) -- skipping this tick.", len(m5))
        conn.close()
        return
    m5 = m5.sort_values("time").reset_index(drop=True)
    m5["time_dt"] = pd.to_datetime(m5["time"], unit="s")

    bars = m5[["time_dt", "open", "high", "low", "close", "tick_volume"]].rename(columns={"time_dt": "time"})
    feats = build_intraday_feature_frame(bars)
    feature_check_cols = get_feature_columns(feats)
    feats["epoch"] = m5["time"].values
    feats["open"] = m5["open"].values
    feats["high"] = m5["high"].values
    feats["low"] = m5["low"].values
    feats["close"] = m5["close"].values
    feats["volume"] = m5["tick_volume"].values

    pending = feats[feats["epoch"] > cursor] if cursor is not None else feats

    # Never run the burst state machine on a bar that hasn't actually
    # finished forming yet -- its OHLC is still changing. candles_m5 already
    # stores/updates it live (so the dashboard chart shows current price),
    # but the cursor must not advance past it: once it's genuinely closed, a
    # later tick needs to see it again and process its final values. Without
    # this, the still-forming bar gets judged on a half-built candle and then
    # silently skipped forever the moment it actually closes.
    pending = pending[pending["epoch"] + bar_seconds <= server_now_epoch]

    if pending.empty:
        conn.close()
        return

    csv_rows = list(zip(
        pending["epoch"].astype(int), pending["open"], pending["high"],
        pending["low"], pending["close"], pending["volume"].astype(int),
    ))
    appended = append_m5_rows(csv_rows)
    if appended:
        logger.info("CSV: appended %d new bar(s) to %s.", appended, config.INTRADAY_CSV_SEED_PATH.name)

    previous_direction = get_previous_direction(conn)
    new_alerts = 0
    last_processed_time = cursor

    for _, row in pending.iterrows():
        bar_ts = int(row["epoch"])
        last_processed_time = bar_ts

        logger.info(
            "CANDLE %s O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
            pd.to_datetime(bar_ts, unit="s"), row["open"], row["high"], row["low"], row["close"], int(row["volume"]),
        )

        if row[feature_check_cols].isna().any():
            continue  # warm-up not complete yet for this bar (only happens near the very start of history)

        # Scored on every closed bar, independent of burst state -- unlike
        # continuation_prob below (which only exists once a burst already
        # fired), this is meant to always have a live read on whether a real
        # move is brewing over the next few bars.
        if near_term_booster is not None:
            X_nt = row[near_term_feature_cols].to_numpy(dtype=float).reshape(1, -1)
            dmat_nt = xgb.DMatrix(X_nt, feature_names=near_term_feature_cols)
            probs_nt = near_term_booster.predict(dmat_nt)[0]
            label_idx = int(probs_nt.argmax())
            near_term_label = config.REGIME_NAMES[label_idx]
            near_term_confidence = float(probs_nt[label_idx])
            db.insert_near_term_prediction(
                conn, bar_ts, near_term_label, near_term_confidence, float(row["close"]), commit=False,
            )

        event, direction, magnitude, horizon = next_event(row, previous_direction)

        if event in (EVENT_START, EVENT_CONTINUE):
            burst_start_ts = db.get_current_burst_start_ts(conn)
            reference_ts = burst_start_ts if burst_start_ts is not None else bar_ts
            duration_min = (bar_ts - reference_ts) / 60
            if duration_min >= config.MAX_BURST_DURATION_MINUTES:
                event = EVENT_END

        if event is None:
            continue

        horizon_min = None if horizon is None else horizon * config.INTRADAY_BAR_MINUTES
        price = float(row["close"])
        in_window, window_name = session_windows.active_window(int(row["hour"]))
        status = {"in_window": in_window, "window_name": window_name}

        continuation_prob = None
        if booster is not None and event in (EVENT_START, EVENT_CONTINUE):
            X = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
            dmat = xgb.DMatrix(X, feature_names=feature_cols)
            continuation_prob = float(booster.predict(dmat)[0])

        note = build_note(event, direction, magnitude, horizon_min, status, continuation_prob)
        db.insert_intraday_alert(
            conn, bar_ts, event, direction, magnitude, horizon_min, continuation_prob,
            in_window, window_name, price, note, commit=False,
        )
        new_alerts += 1
        previous_direction = None if event == EVENT_END else direction

        # server_now_epoch and bar_ts are both broker-server-time epochs --
        # comparing against time.time() (true system UTC) here used to widen
        # the intended ~10-minute live window to ~3h10m on this account
        # (confirmed broker clock runs ~3h ahead of true UTC), so bars up to
        # ~3h stale after a restart/gap were wrongly treated as live and
        # could fire a toast the catch-up path is supposed to suppress.
        is_live = (server_now_epoch - bar_ts) <= LIVE_BAR_STALENESS_SEC
        tag = "" if is_live else " [catch-up]"
        logger.info("ALERT%s %s %s | %s", tag, event, direction, note)

        if is_live and event in (EVENT_START, EVENT_END) and toast_notify is not None:
            if event == EVENT_START:
                tier_tag = "LONG " if is_long_hike(magnitude) else ""
                title = f"{tier_tag}{direction.upper()} burst starting"
            else:
                title = f"{direction.upper()} burst fading -- consider closing"
            try:
                toast_notify(title, note, duration="short")
            except Exception:
                logger.exception("Toast notification failed (non-fatal).")

    db.set_intraday_cursor(conn, last_processed_time)
    if new_alerts == 0:
        last_row = pending.iloc[-1]
        _, last_window = session_windows.active_window(int(last_row["hour"]))
        logger.info(
            "calm | price=%.2f window=%s (processed %d bar(s), no new alerts)",
            float(last_row["close"]), last_window or "none", len(pending),
        )
    else:
        logger.info("Processed %d bar(s), %d new alert(s), cursor now at %s.", len(pending), new_alerts, last_processed_time)
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single iteration and exit")
    args = parser.parse_args()

    logger = setup_logger()

    conn = db.get_connection()
    seeded = seed_candles_m5(conn)
    conn.close()
    if seeded:
        logger.info("CSV seed: upserted %d rows from %s into candles_m5.", seeded, config.INTRADAY_CSV_SEED_PATH.name)

    booster, feature_cols = load_model()
    if booster is None:
        logger.warning("No intraday model found -- running on the rule-based detector only (no continuation %%).")

    near_term_booster, near_term_feature_cols = load_near_term_model()
    if near_term_booster is None:
        logger.warning("No near-term model found -- run `python -m src.train_near_term_model` to enable it.")

    logger.info(
        "Starting intraday burst-watch loop for %s (%d-min bars, checking every %ds for a newly-closed bar).",
        config.MT5_SYMBOL, config.INTRADAY_BAR_MINUTES, config.INTRADAY_CHECK_INTERVAL_SECONDS,
    )

    if args.once:
        run_once(booster, feature_cols, near_term_booster, near_term_feature_cols, logger)
        return

    while True:
        time.sleep(config.INTRADAY_CHECK_INTERVAL_SECONDS)
        try:
            run_once(booster, feature_cols, near_term_booster, near_term_feature_cols, logger)
        except Exception:
            logger.exception("Error during intraday tick; will retry shortly.")


if __name__ == "__main__":
    main()