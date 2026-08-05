"""Train the optional intraday continuation-probability model.

Reads the full accumulated 5-minute history straight from data/test.db's
candles_m5 table -- seeded once from a bulk CSV export (data/raw/
XAU_5m_data.csv, 2004 -> present) plus everything collected live by
intraday_loop.py since. Replays the burst rule (intraday_signals.detect_burst,
vectorized here for speed -- see _detect_burst_vectorized) over every bar,
and trains a small XGBoost binary classifier on just the bars where a burst
condition fired: "did price move at least INTRADAY_CONTINUATION_MOVE_ATR
further in the same direction over the next INTRADAY_CONTINUATION_HORIZON_MIN
minutes?"

The rule-based detector in intraday_signals.py is the real-time source of
truth and needs no training data. This model only adds a confidence score on
top once a burst is already flagged. If there aren't enough qualifying
burst-bars yet, this exits without writing a model, and the live loop falls
back to running on the rule alone (continuation_prob comes back None).

Rerun this periodically (e.g. weekly) as more live 5m history accumulates.

Run: python -m src.train_intraday_model
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

from . import config, db
from .csv_seed import seed_candles_m5
from .intraday_features import build_intraday_feature_frame, get_feature_columns

MIN_TRAINING_SAMPLES = 40


def load_m5_history() -> pd.DataFrame:
    conn = db.get_connection()
    seeded = seed_candles_m5(conn)
    if seeded:
        print(f"CSV seed: upserted {seeded} rows from {config.INTRADAY_CSV_SEED_PATH.name} into candles_m5.")
    df = pd.read_sql_query(
        "SELECT time, open, high, low, close, tick_volume FROM candles_m5 ORDER BY time", conn,
    )
    conn.close()
    if df.empty:
        raise SystemExit(
            "No 5-minute history in candles_m5 and no CSV seed found at "
            f"{config.INTRADAY_CSV_SEED_PATH} -- nothing to train on."
        )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def _detect_burst_vectorized(feats: pd.DataFrame):
    """Vectorized replica of intraday_signals.detect_burst applied to the whole
    feature frame at once. detect_burst itself stays row-at-a-time (that's all
    the live loop ever calls it with, once per closed bar) -- but replaying it
    via a Python for-loop over history stopped scaling once the seed grew from
    ~100k bars to ~1.5M (2004-> CSV backfill): that loop never even finished a
    background run. Same threshold/consistency/tie-break logic, just done with
    array ops instead of a per-row Python call.

    Returns (fired, direction, magnitude) arrays aligned to feats' rows.
    direction is only meaningful where fired is True.
    """
    threshold = np.where(
        feats["in_window"].to_numpy().astype(bool),
        config.BURST_ATR_MULT_IN_WINDOW, config.BURST_ATR_MULT_OUT_WINDOW,
    )
    consistency = feats["directional_consistency"].to_numpy()
    consistency_ok = consistency >= config.BURST_MIN_CONSISTENCY  # NaN-safe: NaN >= x is False

    best_abs = np.zeros(len(feats))
    best_val = np.zeros(len(feats))
    for h in config.INTRADAY_MOVE_WINDOWS:
        val = feats[f"move_{h}_atr"].to_numpy()
        clears = ~np.isnan(val) & (np.abs(val) >= threshold)
        better = clears & (np.abs(val) > best_abs)  # first horizon wins ties, same as the row-wise loop
        best_abs = np.where(better, np.abs(val), best_abs)
        best_val = np.where(better, val, best_val)

    fired_via_consistency = consistency_ok & (best_abs > 0)

    # Single-candle spike path, independent of consistency -- mirrors
    # detect_burst's early-return-on-spike (see its docstring); where a spike
    # fires it takes priority over whatever the consistency path found.
    body = feats["candle_body_atr"].to_numpy()
    spike = ~np.isnan(body) & (np.abs(body) >= threshold)
    fired = spike | fired_via_consistency
    best_abs = np.where(spike, np.abs(body), best_abs)
    best_val = np.where(spike, body, best_val)

    direction = np.where(best_val > 0, "bullish", "bearish")
    return fired, direction, best_abs


def build_training_set(m5: pd.DataFrame, feats: pd.DataFrame):
    horizon = config.INTRADAY_CONTINUATION_HORIZON_MIN // config.INTRADAY_BAR_MINUTES
    close = m5["close"].to_numpy()
    atr = feats["atr"].to_numpy()
    n = len(m5)

    feature_cols = get_feature_columns(feats)
    warm = ~feats[feature_cols].isna().any(axis=1).to_numpy()
    fired, direction, _magnitude = _detect_burst_vectorized(feats)
    valid_atr = ~np.isnan(atr) & (atr != 0)
    has_future = np.arange(n) < (n - horizon)

    idx = np.where(warm & fired & valid_atr & has_future)[0]

    future_move = (close[idx + horizon] - close[idx]) / atr[idx]
    signed_move = np.where(direction[idx] == "bullish", future_move, -future_move)
    labels = (signed_move >= config.INTRADAY_CONTINUATION_MOVE_ATR).astype(int)

    X = feats[feature_cols].iloc[idx].reset_index(drop=True)
    y = pd.Series(labels)
    return X, y, feature_cols


def main():
    print("Loading 5-minute history from candles_m5 (CSV seed + live-collected)...")
    m5 = load_m5_history()
    print(f"Got {len(m5)} M5 bars: {m5['time'].min()} -> {m5['time'].max()}")

    feats = build_intraday_feature_frame(m5)
    X, y, feature_cols = build_training_set(m5, feats)

    print(f"Burst-triggering bars found: {len(X)} (label distribution: {y.value_counts().to_dict() if len(y) else {}})")

    if len(X) < MIN_TRAINING_SAMPLES:
        print(
            f"Only {len(X)} qualifying samples (< {MIN_TRAINING_SAMPLES}). Not enough to train reliably yet -- "
            "skipping model save. The live loop will run on the rule alone (no continuation probability). "
            "Rerun this once more session-open bursts have accumulated in candles_m5."
        )
        return

    split_i = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_i], X.iloc[split_i:]
    y_train, y_test = y.iloc[:split_i], y.iloc[split_i:]

    model = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=100, max_depth=3,
        learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nClassification report (holdout):")
    print(classification_report(y_test, preds, zero_division=0))
    if y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        print(f"AUC: {roc_auc_score(y_test, proba):.3f}")

    config.INTRADAY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(config.INTRADAY_MODEL_PATH))
    with open(config.INTRADAY_FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model -> {config.INTRADAY_MODEL_PATH}")
    print(f"Saved feature columns -> {config.INTRADAY_FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
