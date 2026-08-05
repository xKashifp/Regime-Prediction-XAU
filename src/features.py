"""Feature engineering shared identically by training and the live loop.

Every feature is computed using only data up to and including the current
row (rolling/shift windows look backward only, and swing-derived features use
`confirm_i <= d`), so recomputing this over the full history each time new
live data arrives is safe and always consistent with how the model was
trained.
"""

import numpy as np
import pandas as pd

from . import config
from .zigzag import compute_atr, find_swings
from .labeling import _classify_structure


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.reset_index(drop=True)
    n = len(daily_df)
    close = daily_df["close"]

    feats = pd.DataFrame({"date": daily_df["date"]})

    for p in config.MA_PERIODS:
        sma = close.rolling(p, min_periods=p).mean()
        feats[f"sma_{p}"] = sma
        feats[f"price_to_sma_{p}"] = close / sma - 1.0
        # % slope, not a raw price diff -- a raw diff means something
        # completely different at $400 gold (2007) than at $4000+ gold (2026),
        # and this dataset now spans exactly that range.
        feats[f"sma_{p}_slope_pct"] = sma / sma.shift(config.MA_SLOPE_LOOKBACK) - 1.0

    atr = compute_atr(daily_df, config.ATR_PERIOD)
    feats["atr_pct"] = atr / close

    feats["rsi_14"] = _rsi(close, config.RSI_PERIOD)

    # MACD (12/26/9 EMA, the standard setup), normalized by price for the same
    # reason as the SMA slope above -- raw EMA differences aren't comparable
    # across a price range that spans roughly 10x over this history.
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_pct"] = macd / close
    feats["macd_signal_pct"] = macd_signal / close
    feats["macd_hist_pct"] = (macd - macd_signal) / close

    for w in config.RETURN_WINDOWS:
        feats[f"return_{w}d"] = close.pct_change(w)

    # --- swing-derived features (no lookahead: only swings confirmed by day d) ---
    swings = find_swings(daily_df)
    swings_sorted = swings.sort_values("confirm_i").reset_index(drop=True)
    sw_i, sw_n = 0, len(swings_sorted)
    highs, lows = [], []

    bars_since = np.full(n, np.nan)
    dist_high = np.full(n, np.nan)
    dist_low = np.full(n, np.nan)
    structure_streak = np.zeros(n, dtype=int)

    last_confirmed_i = -1
    streak = 0
    for d in range(n):
        while sw_i < sw_n and swings_sorted.loc[sw_i, "confirm_i"] <= d:
            kind = swings_sorted.loc[sw_i, "kind"]
            price = swings_sorted.loc[sw_i, "price"]
            (highs if kind == "H" else lows).append(price)
            last_confirmed_i = d

            structure = _classify_structure(highs, lows)
            if structure == "bullish":
                streak = streak + 1 if streak >= 0 else 1
            elif structure == "bearish":
                streak = streak - 1 if streak <= 0 else -1
            else:
                streak = 0
            sw_i += 1

        bars_since[d] = (d - last_confirmed_i) if last_confirmed_i >= 0 else np.nan
        if highs:
            dist_high[d] = close.iloc[d] / highs[-1] - 1.0
        if lows:
            dist_low[d] = close.iloc[d] / lows[-1] - 1.0
        structure_streak[d] = streak

    feats["bars_since_last_swing"] = bars_since
    feats["pct_dist_last_swing_high"] = dist_high
    feats["pct_dist_last_swing_low"] = dist_low
    feats["structure_streak"] = structure_streak

    return feats


def get_feature_columns(feats: pd.DataFrame) -> list:
    return [c for c in feats.columns if c != "date"]
