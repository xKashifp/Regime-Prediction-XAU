"""Train the next-single-bar direction model.

Separate from train_intraday_model.py on purpose: that one only scores
continuation *once a burst is already flagged*. This one scores every bar,
all the time -- "will the next 5-minute bar close above or below this bar's
close?" -- which is what powers the widget's NEXT BAR panel (a live read on
the bar that's about to open).

Same feature set as the burst detector (intraday_features.py), same
XGBoost setup as train_intraday_model.py, just a different label and no
burst-gating on the training rows.

Note upfront: predicting a single next 5-minute candle's direction is close
to a coin flip in a liquid market -- expect holdout accuracy/AUC not far
above 50/0.5. The printed classification report below is the real number,
not a sales pitch.

Run: python -m src.train_next_bar_model
"""

import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

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
    close = m5["close"].to_numpy()
    n = len(m5)

    feature_cols = get_feature_columns(feats)
    warm = ~feats[feature_cols].isna().any(axis=1).to_numpy()

    idx = warm.nonzero()[0]
    idx = idx[idx < n - 1]  # need a next bar to label against

    labels = (close[idx + 1] > close[idx]).astype(int)

    X = feats[feature_cols].iloc[idx].reset_index(drop=True)
    y = pd.Series(labels)
    return X, y, feature_cols


def main():
    print("Loading 5-minute history from candles_m5 (CSV seed + live-collected)...")
    m5 = load_m5_history()
    print(f"Got {len(m5)} M5 bars: {m5['time'].min()} -> {m5['time'].max()}")

    feats = build_intraday_feature_frame(m5)
    X, y, feature_cols = build_training_set(m5, feats)

    print(f"Training rows: {len(X)} (label distribution: {y.value_counts().to_dict() if len(y) else {}})")

    if len(X) < MIN_TRAINING_SAMPLES:
        print(f"Only {len(X)} samples (< {MIN_TRAINING_SAMPLES}). Not enough history yet -- skipping model save.")
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
    print("\nClassification report (holdout, chronological last 20%):")
    print(classification_report(y_test, preds, target_names=["bearish", "bullish"], zero_division=0))
    if y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        print(f"AUC: {roc_auc_score(y_test, proba):.3f}  (0.5 = coin flip, 1.0 = perfect)")

    config.NEXT_BAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(config.NEXT_BAR_MODEL_PATH))
    with open(config.NEXT_BAR_FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model -> {config.NEXT_BAR_MODEL_PATH}")
    print(f"Saved feature columns -> {config.NEXT_BAR_FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
