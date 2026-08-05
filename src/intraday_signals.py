"""Rule-based session-open burst detector.

With only ~30 days of M1 history available from the broker, a rule (rolling
move vs ATR, checked against a lower bar inside Tokyo/New York session
windows) is the primary, always-on detector -- it needs no training data and
works from day one. `train_intraday_model.py` adds an optional ML layer on
top that scores *continuation probability* once a burst is already flagged;
the live loop works fine without it (continuation_prob just comes back None).
"""

from . import config

EVENT_START = "BURST_START"
EVENT_CONTINUE = "BURST_CONTINUE"
EVENT_END = "BURST_END"


def detect_burst(feat_row):
    """Returns (direction, magnitude_atr, horizon_min) or (None, 0.0, None)."""
    in_window = bool(feat_row["in_window"])
    threshold = config.BURST_ATR_MULT_IN_WINDOW if in_window else config.BURST_ATR_MULT_OUT_WINDOW

    # Single-candle spike check, independent of directional_consistency below.
    # A lone dramatic candle can clear the magnitude threshold entirely on its
    # own while consistency still reads low, simply because consistency is a
    # trailing average over the last several bars and a fresh spike is only
    # one bar's worth of "votes" for the new direction -- confirmed missed in
    # practice (2026-08-04 16:20: a 2.6x-ATR single candle after a quiet run
    # that the consistency-gated path below never fired on). This check
    # exists specifically to catch that case; horizon=1 bar.
    body = feat_row.get("candle_body_atr")
    if body is not None and body == body and abs(body) >= threshold:
        return ("bullish" if body > 0 else "bearish"), abs(body), 1

    consistency = feat_row["directional_consistency"]

    if consistency is None or consistency != consistency:  # NaN check without importing numpy here
        return None, 0.0, None
    if consistency < config.BURST_MIN_CONSISTENCY:
        return None, 0.0, None

    best = None
    for horizon in config.INTRADAY_MOVE_WINDOWS:
        val = feat_row.get(f"move_{horizon}_atr")
        if val is None or val != val:
            continue
        if abs(val) >= threshold and (best is None or abs(val) > abs(best[1])):
            direction = "bullish" if val > 0 else "bearish"
            best = (direction, val, horizon)

    if best is None:
        return None, 0.0, None
    direction, val, horizon = best
    return direction, abs(val), horizon


def next_event(feat_row, previous_direction):
    """State transition given the previous alert's direction (or None if calm).

    Returns (event_or_None, direction, magnitude_atr, horizon_min).
    """
    direction, magnitude, horizon = detect_burst(feat_row)

    if previous_direction is None:
        if direction is not None:
            return EVENT_START, direction, magnitude, horizon
        return None, None, 0.0, None

    # previously in a burst
    if direction == previous_direction:
        return EVENT_CONTINUE, direction, magnitude, horizon
    # burst faded or reversed -> exit event
    return EVENT_END, previous_direction, magnitude, horizon
