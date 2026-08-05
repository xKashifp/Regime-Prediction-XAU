"""Thin wrapper around the MetaTrader5 python package."""

from datetime import datetime, timezone

import MetaTrader5 as mt5

from . import config


class MT5Connection:
    def __enter__(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        if not mt5.symbol_select(config.MT5_SYMBOL, True):
            raise RuntimeError(f"MT5 symbol_select({config.MT5_SYMBOL}) failed: {mt5.last_error()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        mt5.shutdown()

    def server_now(self) -> datetime:
        """The broker's own current server time, from the latest tick -- NOT
        this machine's system clock. The two can be hours apart (confirmed:
        this machine's UTC clock ran ~3h behind this broker's server clock),
        and every bar epoch in this codebase is a broker-server-time epoch.
        Using the wrong clock as "now" here silently truncates every live
        fetch to a stale cutoff that never catches up to the real live edge."""
        tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({config.MT5_SYMBOL}) failed: {mt5.last_error()}")
        return datetime.fromtimestamp(tick.time, tz=timezone.utc)

    def fetch_closed_d1_candles(self, count: int = 5000):
        """Return the last `count` *closed* D1 candles (excludes today's still-forming
        day). MT5 returns whatever it actually has if less than `count` exists --
        this account's D1 history only goes back to 2026-03-31."""
        rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, mt5.TIMEFRAME_D1, 1, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
        return rates

    def fetch_d1_including_today(self, count: int = 5000):
        """Return the last `count` D1 candles, with the last row being today's
        still-forming day (the broker keeps it live-updating as the session runs)."""
        rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, mt5.TIMEFRAME_D1, 0, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
        return rates

    def fetch_closed_m1_candles(self, count: int = 10):
        """Return the last `count` *closed* M1 candles (excludes the still-forming bar)."""
        rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, mt5.TIMEFRAME_M1, 1, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
        return rates

    def fetch_m1_range(self, start, end):
        """Return M1 candles between start and end (datetime objects)."""
        return mt5.copy_rates_range(config.MT5_SYMBOL, mt5.TIMEFRAME_M1, start, end)

    def fetch_closed_m5_candles(self, count: int = 50):
        """Return the last `count` *closed* M5 candles (excludes the still-forming bar)."""
        rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, mt5.TIMEFRAME_M5, 1, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
        return rates

    def fetch_m5_range(self, start, end):
        """Return M5 candles between start and end (datetime objects) -- used to
        backfill any gap left by downtime (weekend, restart, crash) in one shot."""
        return mt5.copy_rates_range(config.MT5_SYMBOL, mt5.TIMEFRAME_M5, start, end)
