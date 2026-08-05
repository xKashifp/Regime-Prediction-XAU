"""Terminal live status view for XAUUSD -- the same DAILY TREND / BURST WATCH /
POSITION / RECENT panels as widget.py, printed to the terminal instead of a
floating Tk window.

Meant to be run directly in a terminal you keep open, not supervised/hidden --
a CLI view with no visible console would defeat the point of it.

Unlike widget.py (which deliberately swallows read errors so a transient DB
glitch never crashes a background panel -- see widget.py's refresh()), this
prints any error it hits: if the underlying data is actually broken, you see
that immediately instead of a panel that quietly stops updating and looks
fine anyway.

Run:
    python -m src.cli_dashboard
    python -m src.cli_dashboard --interval 2   # override the refresh rate

Ctrl+C to exit.
"""

import argparse
import os
import sys
import time
import traceback

from .widget import (
    CALM_COLOR, REFRESH_MS, REGIME_COLOR,
    _format_age, get_burst_status, get_daily_trend, get_live_bar, get_position_state, get_recent_log,
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_GRAY = "\033[90m"
ANSI_YELLOW = "\033[33m"

HEX_TO_ANSI = {
    REGIME_COLOR["bullish"]: ANSI_GREEN,
    REGIME_COLOR["bearish"]: ANSI_RED,
    REGIME_COLOR["ranging"]: ANSI_GRAY,
    CALM_COLOR: ANSI_GRAY,
}


def _colorize(text, hex_color):
    return f"{HEX_TO_ANSI.get(hex_color, ANSI_GRAY)}{text}{RESET}"


def _colorize_name(text, regime_name):
    return _colorize(text, REGIME_COLOR.get(regime_name, CALM_COLOR))


def render() -> str:
    lines = [f"{BOLD}XAUUSD{RESET}", ""]

    daily = get_daily_trend()
    lines.append(f"{DIM}DAILY TREND{RESET}")
    if daily is None:
        lines.append("  no data yet")
    else:
        conf_txt = f"{daily['confidence']:.0%} confidence" if daily["confidence"] is not None else "no confidence data"
        lines.append(f"  {_colorize_name(daily['regime'].upper(), daily['regime'])}  ({conf_txt})  @ {daily['price']:,.2f}")
        if daily["age_sec"] is not None:
            lines.append(f"  {DIM}model updated {_format_age(daily['age_sec'])}{RESET}")
    lines.append("")

    # Raw OHLC of the current still-forming 5m bar -- placed here, not near
    # DAILY TREND, on purpose: it's 5-minute data with no bearing on the
    # daily/multi-month regime call above. Putting it near DAILY TREND made
    # it look related and caused exactly that confusion in practice.
    burst = get_burst_status()
    lines.append(f"{DIM}BURST WATCH{RESET}")
    live_bar = get_live_bar()
    if live_bar is not None:
        lines.append(
            f"  {DIM}{live_bar['opened_at']:%H:%M} MT5, {live_bar['bar_minutes']}m bar: "
            f"O {live_bar['open']:,.2f}  H {live_bar['high']:,.2f}  "
            f"L {live_bar['low']:,.2f}  C {live_bar['close']:,.2f}  vol {live_bar['volume']}{RESET}"
        )
    if burst is None:
        lines.append("  not running -- start intraday_loop")
    elif burst["calm"]:
        lines.append(f"  {_colorize('CALM', CALM_COLOR)}  @ {burst['price']:,.2f}")
        lines.append(f"  {DIM}{burst['detail']}{RESET}")
    else:
        tier = "LONG " if burst["is_long"] else ""
        lines.append(f"  {_colorize_name(tier + burst['direction'].upper(), burst['direction'])}  @ {burst['price']:,.2f}")
        mag_txt = f"{burst['magnitude']:.1f}x ATR" if burst["magnitude"] is not None else ""
        detail_txt = f"{mag_txt} - {burst['detail']}" if burst["detail"] else mag_txt
        lines.append(f"  {DIM}{detail_txt}{RESET}")
    lines.append("")

    pos = get_position_state()
    lines.append(f"{DIM}POSITION{RESET}")
    lines.append(f"  {_colorize(pos['text'], pos['color'])}")
    lines.append("")

    lines.append(f"{DIM}RECENT{RESET}")
    log_rows = get_recent_log()
    if not log_rows:
        lines.append("  (nothing yet)")
    for row in log_rows:
        lines.append(f"  {_colorize(row['text'], row['color'])}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=REFRESH_MS / 1000, help="seconds between refreshes")
    args = parser.parse_args()

    os.system("")  # gets legacy Windows consoles to honor ANSI codes instead of printing them literally
    print(f"Refreshing every {args.interval:.0f}s. Ctrl+C to quit.\n")
    time.sleep(1)

    try:
        while True:
            sys.stdout.write("\033[2J\033[H")  # clear + cursor home
            try:
                sys.stdout.write(render())
            except Exception:
                sys.stdout.write(f"{ANSI_RED}Error reading live data:{RESET}\n{traceback.format_exc()}")
            sys.stdout.write(f"\n\n{DIM}refreshing every {args.interval:.0f}s -- Ctrl+C to quit{RESET}\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
