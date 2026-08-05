"""Manual trigger-in / trigger-out signal logic.

This never places trades -- it only decides what to log/alert so the user can
act by hand. State (whether we consider ourselves "in" a long/short regime
call) is derived from the last row of the `signals` table, so the process is
safe to restart at any time.
"""

import numpy as np

from . import config

SIGNAL_IN_LONG = "TRIGGER_IN_LONG"
SIGNAL_OUT_LONG = "TRIGGER_OUT_LONG"
SIGNAL_IN_SHORT = "TRIGGER_IN_SHORT"
SIGNAL_OUT_SHORT = "TRIGGER_OUT_SHORT"


def position_state_from_last_signal(last_signal_row):
    """last_signal_row: (ts, signal_type, regime, confidence, price, note) or None."""
    if last_signal_row is None:
        return None
    signal_type = last_signal_row[1]
    if signal_type == SIGNAL_IN_LONG:
        return "LONG"
    if signal_type == SIGNAL_IN_SHORT:
        return "SHORT"
    return None


def evaluate(feats_row, probs, position_state):
    """Returns (signal_type_or_None, regime_name, confidence, note_or_None)."""
    regime_idx = int(np.argmax(probs))
    regime_name = config.REGIME_NAMES[regime_idx]
    confidence = float(probs[regime_idx])

    sma_fast = feats_row[f"sma_{config.TREND_CONFIRM_FAST}"]
    sma_slow = feats_row[f"sma_{config.TREND_CONFIRM_SLOW}"]
    streak = feats_row["structure_streak"]

    bull_structure = bool(sma_fast > sma_slow and streak > 0)
    bear_structure = bool(sma_fast < sma_slow and streak < 0)

    if position_state is None:
        if regime_name == "bullish" and confidence >= config.TRIGGER_CONFIDENCE_IN and bull_structure:
            return SIGNAL_IN_LONG, regime_name, confidence, f"bullish regime confirmed (confidence={confidence:.2f})"
        if regime_name == "bearish" and confidence >= config.TRIGGER_CONFIDENCE_IN and bear_structure:
            return SIGNAL_IN_SHORT, regime_name, confidence, f"bearish regime confirmed (confidence={confidence:.2f})"
        return None, regime_name, confidence, None

    if position_state == "LONG":
        if confidence < config.TRIGGER_CONFIDENCE_OUT or not bull_structure or regime_name == "bearish":
            return SIGNAL_OUT_LONG, regime_name, confidence, f"long regime weakening (confidence={confidence:.2f})"
        return None, regime_name, confidence, None

    if position_state == "SHORT":
        if confidence < config.TRIGGER_CONFIDENCE_OUT or not bear_structure or regime_name == "bullish":
            return SIGNAL_OUT_SHORT, regime_name, confidence, f"short regime weakening (confidence={confidence:.2f})"
        return None, regime_name, confidence, None

    return None, regime_name, confidence, None
