"""Train the near-term strong-move model.

Different question from train_next_bar_model.py's "which way does the very
next candle go" (that one holdout-tested at 0.515 AUC -- barely better than
a coin flip, and live it ran 3/13 = 23%, called bearish almost every single
tick straight through a sustained rally). This one asks something more
tractable: "is a real move brewing over the next few bars, and which way" --
3-class (bearish/ranging/bullish) on whether price moves at least
config.INTRADAY_CONTINUATION_MOVE_ATR further in one direction over the next
config.NEAR_TERM_HORIZON_BARS bars (15 min at 5m). Most bars simply won't
have a strong move coming -- "ranging" is expected to dominate the label
distribution, same as it does for the daily regime model.

Same feature set as the burst detector and the next-bar model. Scored on
every bar (not gated on a burst already firing, unlike xgb_intraday).

Run: python -m src.train_near_term_model
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from . import config, db
from .csv_seed import seed_candles_m5
from .intraday_features import build_intraday_feature_frame, get_feature_columns

MIN_TRAINING_SAMPLES = 200


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


def build_training_set(m5: pd.DataFrame, feats: pd.DataFrame):
    horizon = config.NEAR_TERM_HORIZON_BARS
    close = m5["close"].to_numpy()
    atr = feats["atr"].to_numpy()
    n = len(m5)

    feature_cols = get_feature_columns(feats)
    warm = ~feats[feature_cols].isna().any(axis=1).to_numpy()
    valid_atr = ~np.isnan(atr) & (atr != 0)
    has_future = np.arange(n) < (n - horizon)

    idx = np.where(warm & valid_atr & has_future)[0]

    future_move = (close[idx + horizon] - close[idx]) / atr[idx]
    labels = np.where(
        future_move >= config.INTRADAY_CONTINUATION_MOVE_ATR, config.REGIME_BULLISH,
        np.where(future_move <= -config.INTRADAY_CONTINUATION_MOVE_ATR, config.REGIME_BEARISH, config.REGIME_RANGING),
    )

    X = feats[feature_cols].iloc[idx].reset_index(drop=True)
    y = pd.Series(labels)
    return X, y, feature_cols


def main():
    print("Loading 5-minute history from candles_m5 (CSV seed + live-collected)...")
    m5 = load_m5_history()
    print(f"Got {len(m5)} M5 bars: {m5['time'].min()} -> {m5['time'].max()}")

    feats = build_intraday_feature_frame(m5)
    X, y, feature_cols = build_training_set(m5, feats)

    print(f"Training rows: {len(X)} (label distribution: {y.value_counts().to_dict()})")

    if len(X) < MIN_TRAINING_SAMPLES:
        print(f"Only {len(X)} samples (< {MIN_TRAINING_SAMPLES}). Not enough history yet -- skipping model save.")
        return

    split_i = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_i], X.iloc[split_i:]
    y_train, y_test = y.iloc[:split_i], y.iloc[split_i:]
    sw_train = compute_sample_weight("balanced", y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "multi:softprob", "num_class": 3, "max_depth": 3,
        "eta": 0.08, "subsample": 0.9, "colsample_bytree": 0.9, "eval_metric": "mlogloss",
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=150,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10, verbose_eval=False,
    )

    proba = booster.predict(dtest)
    preds = proba.argmax(axis=1)
    labels = [config.REGIME_BEARISH, config.REGIME_RANGING, config.REGIME_BULLISH]
    print(f"\nClassification report (holdout, chronological last 20%, horizon={config.NEAR_TERM_HORIZON_BARS} bars):")
    print(classification_report(y_test, preds, labels=labels, target_names=["bearish", "ranging", "bullish"], zero_division=0))

    config.NEAR_TERM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.NEAR_TERM_MODEL_PATH))
    with open(config.NEAR_TERM_FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model -> {config.NEAR_TERM_MODEL_PATH}")
    print(f"Saved feature columns -> {config.NEAR_TERM_FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
