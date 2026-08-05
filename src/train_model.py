"""Build the daily dataset from daily_bars' full accumulated history, label
regimes, engineer features, and train an XGBoost multiclass classifier
(bearish=0 / ranging=1 / bullish=2).

daily_bars holds ~5000 days (2007 -> present) -- historical CSV-seeded data
plus everything realtime_loop.py has collected live since, read straight from
the DB rather than a fresh MT5 pull (this account's own D1 history via MT5
only goes back to 2026-03-31, ~88 trading days -- nowhere near enough to
learn genuine multi-year regime dynamics from).

Run: python -m src.train_model
"""

import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from . import config, db
from .features import build_feature_frame
from .labeling import build_labels


def load_daily_history() -> pd.DataFrame:
    conn = db.get_connection()
    daily = pd.read_sql_query(
        "SELECT date, open, high, low, close FROM daily_bars ORDER BY date",
        conn, parse_dates=["date"],
    )
    conn.close()
    return daily


def build_dataset():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily_history()
    labels, swings = build_labels(daily)
    labels.to_csv(config.REGIME_LABELS_CSV, index=False)

    feats = build_feature_frame(daily)

    data = feats.merge(labels[["date", "regime", "regime_name"]], on="date", how="inner")
    data = data.dropna().reset_index(drop=True)
    return data, swings


def main():
    data, swings = build_dataset()
    feature_cols = [c for c in data.columns if c not in ("date", "regime", "regime_name")]

    print(f"Dataset: {len(data)} labeled daily rows (daily_bars: historical + live), {len(swings)} confirmed swings")
    if len(data) < 60:
        print(
            f"NOTE: only {len(data)} usable rows after feature warm-up -- treat this model as "
            "provisional; it'll get more reliable as more days accumulate. Rerun this script periodically."
        )
    print("Regime distribution:\n", data["regime_name"].value_counts())

    split_i = int(len(data) * (1 - config.TEST_HOLDOUT_FRACTION))
    train, test = data.iloc[:split_i], data.iloc[split_i:]
    print(f"Train: {train['date'].min()} -> {train['date'].max()} ({len(train)} rows)")
    print(f"Test:  {test['date'].min()} -> {test['date'].max()} ({len(test)} rows)")

    X_train, y_train = train[feature_cols], train["regime"]
    X_test, y_test = test[feature_cols], test["regime"]
    sw_train = compute_sample_weight("balanced", y_train)

    # Low-level train()/DMatrix API rather than XGBClassifier: with this little live
    # history, some regimes (e.g. bullish) may never appear in a given window at all,
    # and XGBClassifier's classes_ auto-detection breaks (mismatched shapes) when the
    # data doesn't contain every class the fixed num_class=3 objective expects.
    # xgb.train() just uses num_class directly and always returns a plain (n,3)
    # softprob array, sidestepping that -- same approach the live loop already uses.
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=100,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    proba = booster.predict(dtest)
    preds = proba.argmax(axis=1)
    labels = [config.REGIME_BEARISH, config.REGIME_RANGING, config.REGIME_BULLISH]
    print("\nClassification report (holdout):")
    print(classification_report(
        y_test, preds, labels=labels,
        target_names=["bearish", "ranging", "bullish"], zero_division=0,
    ))
    print("Confusion matrix (rows=actual, cols=predicted, order=bearish/ranging/bullish):")
    print(confusion_matrix(y_test, preds, labels=labels))

    importance = booster.get_score(importance_type="gain")
    importances = sorted(
        ((name, importance.get(name, 0.0)) for name in feature_cols), key=lambda x: -x[1]
    )
    print("\nTop feature importances (gain):")
    for name, imp in importances[:15]:
        print(f"  {name}: {imp:.4f}")

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.MODEL_PATH))
    with open(config.FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved feature columns -> {config.FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
