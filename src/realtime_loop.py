"""Live loop: every 5 minutes, pull today's D1 daily bar (still-forming,
live-updating) from the MT5 terminal, persist it into daily_bars, then
recompute regime features + XGBoost prediction over daily_bars' FULL
accumulated history (2007 -> present) -- not just the fresh MT5 pull, which
by itself only covers this account's own D1 window (~88 trading days, nowhere
near enough history for the model's 200-day SMA/MACD features to ever warm up).

No auto-trading -- this only prints/logs signals for the user to act on
manually.

Usage:
    python -m src.realtime_loop            # runs forever, aligned to 5-min boundaries
    python -m src.realtime_loop --once      # single iteration, for testing
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, db, signals
from .features import build_feature_frame
from .mt5_client import MT5Connection


def setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("regime")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(config.LOG_PATH)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def fetch_daily_series() -> pd.DataFrame:
    """Pulls fresh from the live MT5 terminal every call -- last row is today,
    still-forming and live-updating."""
    with MT5Connection() as conn:
        rates = conn.fetch_d1_including_today(count=config.DAILY_HISTORY_COUNT)
    daily = pd.DataFrame(rates)
    daily["date"] = pd.to_datetime(daily["time"], unit="s").dt.normalize()
    return daily[["date", "open", "high", "low", "close"]]


def persist_daily_bars(conn, daily: pd.DataFrame) -> None:
    for row in daily.itertuples(index=False):
        db.upsert_daily_bar(
            conn, row.date.strftime("%Y-%m-%d"), row.open, row.high, row.low, row.close, "live"
        )


def sleep_until_next_boundary(poll_seconds: int = config.POLL_SECONDS, buffer_seconds: int = 10) -> None:
    now = time.time()
    next_boundary = (int(now // poll_seconds) + 1) * poll_seconds + buffer_seconds
    time.sleep(max(0.0, next_boundary - now))


def run_once(booster, feature_cols, logger) -> None:
    conn = db.get_connection()

    live_tail = fetch_daily_series()
    persist_daily_bars(conn, live_tail)

    # Recompute over the full accumulated history, not just live_tail -- the
    # model's longer MA/MACD features need far more warm-up than this
    # account's own ~88-day MT5 window can provide on its own.
    daily = pd.read_sql_query(
        "SELECT date, open, high, low, close FROM daily_bars ORDER BY date",
        conn, parse_dates=["date"],
    )
    feats = build_feature_frame(daily)
    last = feats.iloc[-1]

    if last[feature_cols].isna().any():
        logger.warning("Feature warm-up incomplete for %s -- skipping this tick.", last["date"])
        return

    X = last[feature_cols].to_numpy(dtype=float).reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    probs = booster.predict(dmat)[0]

    regime_idx = int(np.argmax(probs))
    regime_name = config.REGIME_NAMES[regime_idx]
    confidence = float(probs[regime_idx])
    price = float(daily.iloc[-1]["close"])
    ts_now = int(time.time())

    db.insert_prediction(conn, ts_now, str(last["date"].date()), regime_name, probs)

    last_signal = db.get_last_signal(conn)
    position_state = signals.position_state_from_last_signal(last_signal)
    signal_type, _, _, note = signals.evaluate(last, probs, position_state)

    if signal_type:
        db.insert_signal(conn, ts_now, signal_type, regime_name, confidence, price, note)
        logger.info(
            "ALERT %s | regime=%s confidence=%.2f price=%.2f | %s",
            signal_type, regime_name, confidence, price, note,
        )
    else:
        logger.info(
            "date=%s regime=%s confidence=%.2f price=%.2f position=%s (no change)",
            last["date"].date(), regime_name, confidence, price, position_state,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single iteration and exit")
    args = parser.parse_args()

    logger = setup_logger()

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {config.MODEL_PATH} -- run `python -m src.train_model` first.")

    booster = xgb.Booster()
    booster.load_model(str(config.MODEL_PATH))
    feature_cols = json.loads(config.FEATURE_COLUMNS_PATH.read_text())

    logger.info("Starting live regime loop for %s (poll every %ds, live MT5 D1 data only).",
                config.MT5_SYMBOL, config.POLL_SECONDS)

    if args.once:
        run_once(booster, feature_cols, logger)
        return

    while True:
        sleep_until_next_boundary()
        try:
            run_once(booster, feature_cols, logger)
        except Exception:
            logger.exception("Error during tick; will retry on next 5-minute boundary.")


if __name__ == "__main__":
    main()
