"""Session-open watch windows, in broker *server* time.

MT5 candle timestamps are in the broker's server time, not UTC -- on this
account the server runs ~3h ahead of UTC (checked against symbol_info().time
vs true UTC). Rather than hardcode that offset (it can shift with DST), we
just work directly in server time throughout: the watch windows below were
measured from the broker's own trailing-30-day M1 volatility profile, in the
same server-time clock the live candles arrive in, so no conversion is
needed for the actual comparison. `broker_utc_offset_hours()` is provided
only to *display* the offset to the user, not for any decision logic.
"""

from datetime import datetime, timezone

import MetaTrader5 as mt5

from . import config


def broker_utc_offset_hours() -> float:
    info = mt5.symbol_info(config.MT5_SYMBOL)
    if info is None:
        return float("nan")
    server_dt = datetime.fromtimestamp(info.time, tz=timezone.utc)
    true_utc = datetime.now(timezone.utc)
    return round((server_dt - true_utc).total_seconds() / 3600, 1)


def active_window(server_hour: int):
    """Returns (in_window: bool, window_name: str | None) for a given server-time hour."""
    for name, (start, end) in config.SESSION_WINDOWS.items():
        if start <= server_hour < end:
            return True, name
    return False, None


def status_for_time(server_dt: datetime):
    """Returns dict: in_window, window_name, and minutes to the next window's start."""
    from datetime import timedelta

    hour = server_dt.hour
    in_window, name = active_window(hour)
    if in_window:
        return {"in_window": True, "window_name": name, "minutes_to_next": 0, "next_window_name": name}

    # find the soonest upcoming window start (today or tomorrow), in minutes
    best = None
    for wname, (start, _end) in config.SESSION_WINDOWS.items():
        candidate = server_dt.replace(hour=start, minute=0, second=0, microsecond=0)
        if candidate <= server_dt:
            candidate = candidate + timedelta(days=1)
        delta_min = (candidate - server_dt).total_seconds() / 60
        if best is None or delta_min < best[0]:
            best = (delta_min, wname)

    return {"in_window": False, "window_name": None, "minutes_to_next": round(best[0]) if best else None,
            "next_window_name": best[1] if best else None}
