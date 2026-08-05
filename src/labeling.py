"""Turn ZigZag swing structure + MA-slope confirmation into long-term regime labels.

Follows Pattern.md: HH+HL sequences (confirmed by rising, upward-sloping
50/200-day SMAs) = bullish; LH+LL sequences (confirmed by falling,
downward-sloping SMAs) = bearish; everything else = ranging. A minimum
run-length filter then strips out any bullish/bearish run too short to count
as "long-term", folding it back into ranging.
"""

import numpy as np
import pandas as pd

from . import config
from .zigzag import find_swings


def _classify_structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return None
    bullish = highs[-1] > highs[-2] and lows[-1] > lows[-2]
    bearish = highs[-1] < highs[-2] and lows[-1] < lows[-2]
    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return None


def _raw_regimes(daily_df: pd.DataFrame, swings: pd.DataFrame) -> np.ndarray:
    n = len(daily_df)
    close = daily_df["close"]

    sma_fast = close.rolling(config.TREND_CONFIRM_FAST).mean()
    sma_slow = close.rolling(config.TREND_CONFIRM_SLOW).mean()
    fast_slope = sma_fast - sma_fast.shift(config.MA_SLOPE_LOOKBACK)
    slow_slope = sma_slow - sma_slow.shift(config.MA_SLOPE_LOOKBACK)

    swings_sorted = swings.sort_values("confirm_i").reset_index(drop=True)
    sw_i, sw_n = 0, len(swings_sorted)
    highs, lows = [], []

    raw = np.full(n, config.REGIME_RANGING, dtype=int)

    for d in range(n):
        while sw_i < sw_n and swings_sorted.loc[sw_i, "confirm_i"] <= d:
            kind = swings_sorted.loc[sw_i, "kind"]
            price = swings_sorted.loc[sw_i, "price"]
            (highs if kind == "H" else lows).append(price)
            sw_i += 1

        structure = _classify_structure(highs, lows)

        f, s = sma_fast.iloc[d], sma_slow.iloc[d]
        fs, ss = fast_slope.iloc[d], slow_slope.iloc[d]
        ma_ready = pd.notna(f) and pd.notna(s) and pd.notna(fs) and pd.notna(ss)
        ma_bull = ma_ready and f > s and fs > 0 and ss > 0
        ma_bear = ma_ready and f < s and fs < 0 and ss < 0

        if structure == "bullish" and ma_bull:
            raw[d] = config.REGIME_BULLISH
        elif structure == "bearish" and ma_bear:
            raw[d] = config.REGIME_BEARISH
        else:
            raw[d] = config.REGIME_RANGING

    return raw


def _apply_min_run_filter(raw: np.ndarray, min_days: int) -> np.ndarray:
    """Causal confirmation delay: a run only gets promoted to bullish/bearish
    once it has ALREADY lasted min_days consecutive days as observed so far --
    never based on how long it eventually turns out to run.

    The previous version measured each run's total eventual length (looking
    forward to wherever raw[i] first changes) and applied that same verdict
    to every day in the run, including its first day -- which means, e.g., day
    2 of a new trend got labeled "bullish" today using knowledge of whether
    the trend was still intact 18 days from now. That's not knowable in real
    time; it's why the model looked far more confident than it should on
    trend days early in a still-unconfirmed run (confirmed: today sits 6 days
    into a new run, which the old filter would've already been calling
    "bearish" using 14 days of hindsight that don't exist live)."""
    out = raw.copy()
    n = len(raw)
    run_len_so_far = 0
    prev = None
    for d in range(n):
        run_len_so_far = run_len_so_far + 1 if raw[d] == prev else 1
        prev = raw[d]
        if raw[d] in (config.REGIME_BULLISH, config.REGIME_BEARISH) and run_len_so_far < min_days:
            out[d] = config.REGIME_RANGING
    return out


def build_labels(daily_df: pd.DataFrame):
    """Returns (labels_df, swings_df). labels_df has columns: date, regime, regime_name."""
    daily_df = daily_df.reset_index(drop=True)
    swings = find_swings(daily_df)
    raw = _raw_regimes(daily_df, swings)
    final = _apply_min_run_filter(raw, config.MIN_REGIME_RUN_DAYS)

    out = daily_df[["date"]].copy()
    out["regime"] = final
    out["regime_name"] = out["regime"].map(config.REGIME_NAMES)
    return out, swings
