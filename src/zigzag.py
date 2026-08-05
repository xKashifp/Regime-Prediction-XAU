"""ZigZag swing-point detection on a daily OHLC series.

A swing is only confirmed once price reverses from the running extreme by at
least a threshold (max of a flat % and an ATR-based amount). Swings are
appended to the result at the index where they are *confirmed*, not at the
pivot's own index -- this keeps the series lookahead-free: on any given day,
only swings confirmed by that day should be treated as "known".
"""

import numpy as np
import pandas as pd

from . import config


def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """Wilder-style ATR from daily high/low/close. Backward-looking only."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def find_swings(
    df: pd.DataFrame,
    min_pct: float = config.ZIGZAG_MIN_PCT,
    atr_mult: float = config.ZIGZAG_ATR_MULT,
    atr_period: int = config.ATR_PERIOD,
) -> pd.DataFrame:
    """Return confirmed swing points as a DataFrame with columns:
    pivot_i, confirm_i, date, price, kind ('H' or 'L').

    Both pivot_i and confirm_i are integer positions into `df` (reset index).
    """
    df = df.reset_index(drop=True)
    atr = compute_atr(df, atr_period)
    # threshold as a fraction of price; before ATR warms up, fall back to min_pct
    frac_threshold = np.maximum(min_pct, (atr_mult * atr / df["close"]).fillna(0.0))

    n = len(df)
    swings = []
    trend = None  # None, 'up', 'down'
    extreme_i = 0
    extreme_price = df["close"].iloc[0]

    for i in range(1, n):
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        thresh = frac_threshold.iloc[i]

        if trend is None:
            if high_i >= extreme_price * (1 + thresh):
                trend = "up"
                extreme_i, extreme_price = i, high_i
            elif low_i <= extreme_price * (1 - thresh):
                trend = "down"
                extreme_i, extreme_price = i, low_i
            continue

        if trend == "up":
            if high_i > extreme_price:
                extreme_i, extreme_price = i, high_i
            elif low_i <= extreme_price * (1 - thresh):
                swings.append(
                    {
                        "pivot_i": extreme_i,
                        "confirm_i": i,
                        "date": df["date"].iloc[extreme_i],
                        "price": extreme_price,
                        "kind": "H",
                    }
                )
                trend = "down"
                extreme_i, extreme_price = i, low_i
        else:  # trend == 'down'
            if low_i < extreme_price:
                extreme_i, extreme_price = i, low_i
            elif high_i >= extreme_price * (1 + thresh):
                swings.append(
                    {
                        "pivot_i": extreme_i,
                        "confirm_i": i,
                        "date": df["date"].iloc[extreme_i],
                        "price": extreme_price,
                        "kind": "L",
                    }
                )
                trend = "up"
                extreme_i, extreme_price = i, high_i

    return pd.DataFrame(swings, columns=["pivot_i", "confirm_i", "date", "price", "kind"])


def confirmed_up_to(swings: pd.DataFrame, day_idx: int) -> pd.DataFrame:
    """Swings that were already confirmed by (i.e. at or before) `day_idx`."""
    return swings[swings["confirm_i"] <= day_idx]
