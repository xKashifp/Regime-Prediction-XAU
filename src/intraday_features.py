"""Rolling 5-minute feature computation for session-open burst detection.

Same design principle as features.py for the daily model: one pure function,
called identically by training (replayed bar-by-bar over history) and the
live loop (recomputed on the latest rolling window each tick). Every value
only looks backward from the current bar.
"""

import numpy as np
import pandas as pd

from . import config
from .zigzag import compute_atr


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_intraday_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: 5-minute OHLCV, columns time(datetime, server time), open, high,
    low, close, tick_volume. Returns a frame aligned 1:1 with bars' rows (NaN
    until each feature's warmup passes).
    """
    bars = bars.reset_index(drop=True)
    close = bars["close"]

    feats = pd.DataFrame({"time": bars["time"]})

    atr = compute_atr(bars, config.INTRADAY_ATR_PERIOD)
    feats["atr"] = atr

    # This bar's own body size, in ATR units -- separate from the move_w_atr
    # windows below (which all look back several bars). A single sudden spike
    # candle can clear the burst magnitude threshold on its own while still
    # reading as low "directional_consistency" (that's a trailing average
    # over the last INTRADAY_CONSISTENCY_WINDOW bars, and a lone fresh spike
    # is only one bar's worth of "votes" for the new direction) -- confirmed
    # missed in practice (2026-08-04 16:20: a 2.6x-ATR single candle after a
    # quiet run that the consistency-gated path never fired on). detect_burst
    # uses this to catch that case independently of the consistency gate.
    feats["candle_body_atr"] = (close - bars["open"]) / atr.replace(0, np.nan)

    for w in config.INTRADAY_MOVE_WINDOWS:
        move = close - close.shift(w)
        feats[f"move_{w}"] = move
        feats[f"move_{w}_atr"] = move / atr.replace(0, np.nan)

    consist_w = config.INTRADAY_CONSISTENCY_WINDOW
    candle_dir = np.sign(close - bars["open"])
    net_dir = np.sign(close - close.shift(consist_w))
    same_sign = (candle_dir == net_dir.reindex(candle_dir.index)).astype(float)
    feats["directional_consistency"] = same_sign.rolling(consist_w, min_periods=consist_w).mean()

    feats["rsi"] = _rsi(close, config.INTRADAY_ATR_PERIOD)

    vol_w = config.INTRADAY_VOL_ZSCORE_WINDOW
    vol = bars["tick_volume"].astype(float)
    vol_mean = vol.rolling(vol_w, min_periods=vol_w // 2).mean()
    vol_std = vol.rolling(vol_w, min_periods=vol_w // 2).std()
    feats["vol_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    feats["hour"] = bars["time"].dt.hour
    feats["in_window"] = feats["hour"].apply(lambda h: _in_any_window(h)).astype(int)

    return feats


def _in_any_window(hour: int) -> bool:
    return any(start <= hour < end for start, end in config.SESSION_WINDOWS.values())


def get_feature_columns(feats: pd.DataFrame) -> list:
    return [c for c in feats.columns if c not in ("time",)]
