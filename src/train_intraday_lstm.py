"""Train an LSTM alternative to the XGBoost continuation-probability model.

train_intraday_model.py feeds XGBoost a single snapshot feature row per
burst-bar and, even with the full 2004-> history seeded (103k qualifying
burst-bars), tops out at AUC ~0.52 -- no better than chance. That rules out
"not enough data" as the bottleneck. This script tests the other lever: give
the model the actual bar-by-bar sequence leading up to the burst instead of
one flattened snapshot, in case the continuation signal lives in the shape of
the recent path rather than in a single point-in-time reading.

Same label definition as train_intraday_model.py (did price move at least
INTRADAY_CONTINUATION_MOVE_ATR further in the burst direction over the next
INTRADAY_CONTINUATION_HORIZON_MIN minutes), same chronological 80/20 holdout,
so its AUC is directly comparable to the XGBoost run.

Sequence features are framed relative to the burst's own direction (bullish
bursts and bearish bursts share one model instead of learning two mirrored
copies of the same pattern): price-based columns are sign-flipped by burst
direction, and RSI is mirrored (100-RSI) for bearish bursts, so "positive"
always means "moving further in the burst's own direction."

Run: python -m src.train_intraday_lstm
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn

from . import config
from .intraday_features import build_intraday_feature_frame
from .train_intraday_model import _detect_burst_vectorized, load_m5_history

SEQUENCE_LENGTH = 24  # bars of history feeding the LSTM (=2h at 5m)
SEQ_FEATURE_COLS = [
    "move_5_atr", "move_10_atr", "move_15_atr", "bar_return", "bar_range",
    "directional_consistency", "rsi_signed", "vol_zscore", "in_window",
]
STATIC_FEATURE_COLS = ["magnitude"]
HIDDEN_SIZE = 32
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MIN_TRAINING_SAMPLES = 40


class ContinuationLSTM(nn.Module):
    def __init__(self, n_seq_features, n_static_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_seq_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq, static):
        _, (h_n, _) = self.lstm(seq)
        combined = torch.cat([h_n[-1], static], dim=1)
        return self.head(combined).squeeze(-1)


def build_sequence_dataset(m5: pd.DataFrame, feats: pd.DataFrame):
    """Returns (seq_X, static_X, y, warmup_idx) where seq_X is
    (n_samples, SEQUENCE_LENGTH, len(SEQ_FEATURE_COLS)) and static_X is
    (n_samples, len(STATIC_FEATURE_COLS))."""
    horizon = config.INTRADAY_CONTINUATION_HORIZON_MIN // config.INTRADAY_BAR_MINUTES
    close = m5["close"].to_numpy()
    atr = feats["atr"].to_numpy()
    n = len(m5)

    fired, direction, magnitude = _detect_burst_vectorized(feats)
    direction_sign = np.where(direction == "bullish", 1.0, -1.0)

    per_bar = pd.DataFrame({
        "move_5_atr": feats["move_5_atr"] * direction_sign,
        "move_10_atr": feats["move_10_atr"] * direction_sign,
        "move_15_atr": feats["move_15_atr"] * direction_sign,
        "bar_return": ((m5["close"] - m5["open"]).to_numpy() / atr) * direction_sign,
        "bar_range": (m5["high"] - m5["low"]).to_numpy() / atr,
        "directional_consistency": feats["directional_consistency"],
        "rsi_signed": np.where(direction_sign > 0, feats["rsi"], 100.0 - feats["rsi"]),
        "vol_zscore": feats["vol_zscore"],
        "in_window": feats["in_window"].astype(float),
    })
    seq_matrix = per_bar[SEQ_FEATURE_COLS].to_numpy(dtype=np.float64)

    warm = ~feats[[c for c in feats.columns if c != "time"]].isna().any(axis=1).to_numpy()
    warmup_idx = int(np.argmax(warm)) if warm.any() else n
    valid_atr = ~np.isnan(atr) & (atr != 0)
    has_future = np.arange(n) < (n - horizon)
    has_history = np.arange(n) >= (warmup_idx + SEQUENCE_LENGTH - 1)

    mask = fired & warm & valid_atr & has_future & has_history
    idx = np.where(mask)[0]

    windows = np.lib.stride_tricks.sliding_window_view(
        np.ascontiguousarray(seq_matrix), window_shape=SEQUENCE_LENGTH, axis=0
    )  # shape: (n - SEQUENCE_LENGTH + 1, n_features, SEQUENCE_LENGTH)
    windows = np.transpose(windows, (0, 2, 1))  # -> (n_windows, SEQUENCE_LENGTH, n_features)
    seq_X = windows[idx - SEQUENCE_LENGTH + 1]

    static_X = magnitude[idx].reshape(-1, 1)

    future_move = (close[idx + horizon] - close[idx]) / atr[idx]
    signed_move = future_move * direction_sign[idx]
    y = (signed_move >= config.INTRADAY_CONTINUATION_MOVE_ATR).astype(np.float32)

    finite = np.isfinite(seq_X).all(axis=(1, 2)) & np.isfinite(static_X).all(axis=1)
    return seq_X[finite].astype(np.float32), static_X[finite].astype(np.float32), y[finite]


def main():
    print("Loading 5-minute history from candles_m5 (CSV seed + live-collected)...")
    m5 = load_m5_history()
    print(f"Got {len(m5)} M5 bars: {m5['time'].min()} -> {m5['time'].max()}")

    feats = build_intraday_feature_frame(m5)
    seq_X, static_X, y = build_sequence_dataset(m5, feats)
    print(f"Burst-triggering sequences: {len(seq_X)} (label distribution: {dict(zip(*np.unique(y, return_counts=True)))})")

    if len(seq_X) < MIN_TRAINING_SAMPLES:
        print(f"Only {len(seq_X)} qualifying samples (< {MIN_TRAINING_SAMPLES}). Skipping.")
        return

    split_i = int(len(seq_X) * 0.8)
    val_i = int(split_i * 0.9)  # last 10% of the train portion, for epoch monitoring only -- holdout stays untouched until final report

    seq_train, seq_val, seq_test = seq_X[:val_i], seq_X[val_i:split_i], seq_X[split_i:]
    static_train, static_val, static_test = static_X[:val_i], static_X[val_i:split_i], static_X[split_i:]
    y_train, y_val, y_test = y[:val_i], y[val_i:split_i], y[split_i:]

    seq_mean = seq_train.reshape(-1, seq_train.shape[-1]).mean(axis=0)
    seq_std = seq_train.reshape(-1, seq_train.shape[-1]).std(axis=0) + 1e-8
    static_mean = static_train.mean(axis=0)
    static_std = static_train.std(axis=0) + 1e-8

    def normalize(seq, static):
        # seq/static are float32; seq_mean/std are float64 (numpy .mean() default) --
        # dividing promotes to float64, which then mismatches the model's float32
        # weights at the first matmul. Cast back explicitly.
        norm_seq = ((seq - seq_mean) / seq_std).astype(np.float32)
        norm_static = ((static - static_mean) / static_std).astype(np.float32)
        return norm_seq, norm_static

    seq_train, static_train = normalize(seq_train, static_train)
    seq_val, static_val = normalize(seq_val, static_val)
    seq_test, static_test = normalize(seq_test, static_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContinuationLSTM(len(SEQ_FEATURE_COLS), len(STATIC_FEATURE_COLS)).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    seq_train_t = torch.from_numpy(seq_train).to(device)
    static_train_t = torch.from_numpy(static_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    seq_val_t = torch.from_numpy(seq_val).to(device)
    static_val_t = torch.from_numpy(static_val).to(device)

    n_train = len(seq_train_t)
    best_val_auc = -1
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(seq_train_t[b], static_train_t[b])
            loss = criterion(logits, y_train_t[b])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            val_logits = model(seq_val_t, static_val_t).cpu().numpy()
        val_proba = 1 / (1 + np.exp(-val_logits))
        val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else float("nan")
        print(f"epoch {epoch + 1}/{EPOCHS}  train_loss={total_loss / n_train:.4f}  val_auc={val_auc:.3f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(
            torch.from_numpy(seq_test).to(device), torch.from_numpy(static_test).to(device)
        ).cpu().numpy()
    test_proba = 1 / (1 + np.exp(-test_logits))
    test_pred = (test_proba >= 0.5).astype(int)

    print(f"\nBest val AUC: {best_val_auc:.3f}")
    print("\nClassification report (holdout):")
    print(classification_report(y_test, test_pred, zero_division=0))
    if len(np.unique(y_test)) > 1:
        print(f"Holdout AUC: {roc_auc_score(y_test, test_proba):.3f}")

    config.INTRADAY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    lstm_path = config.INTRADAY_MODEL_PATH.parent / "lstm_intraday.pt"
    meta_path = config.INTRADAY_MODEL_PATH.parent / "lstm_intraday_meta.json"
    torch.save(model.state_dict(), lstm_path)
    with open(meta_path, "w") as f:
        json.dump({
            "sequence_length": SEQUENCE_LENGTH,
            "seq_feature_cols": SEQ_FEATURE_COLS,
            "static_feature_cols": ["magnitude"],
            "hidden_size": HIDDEN_SIZE,
            "seq_mean": seq_mean.tolist(), "seq_std": seq_std.tolist(),
            "static_mean": static_mean.tolist(), "static_std": static_std.tolist(),
        }, f, indent=2)
    print(f"\nSaved model -> {lstm_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
