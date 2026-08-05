"""Prints the full TRIGGER IN / TRIGGER OUT reconciliation history -- same
logic as widget.py's get_position_state(), replayed over the whole
intraday_alerts table instead of just the latest row, so past entries/exits
can be reconciled against actual times instead of just seeing "now."

Mirrors get_position_state() exactly:
  - BURST_START/CONTINUE below BURST_LONG_MAGNITUDE_ATR -> TRIGGER IN
  - the moment magnitude crosses that threshold             -> TRIGGER OUT - LONG
  - BURST_END, episode never reached LONG                    -> TRIGGER OUT - RANGING
  - BURST_END, episode had reached LONG at some point         -> TRIGGER IN (faded)

Timestamps are broker-server-time, printed literally (same convention as
everything else derived from candles_m5/intraday_alerts -- matches what
MT5's own terminal shows, not this machine's local clock).

Run: python -m src.trigger_log
"""

import pandas as pd

from . import config, db


def _fmt_ts(ts) -> str:
    dt = pd.to_datetime(int(ts), unit="s")
    return f"{dt.strftime('%I:%M %p')} - {dt.day}{dt.strftime('%b').upper()}{dt.year}"


def build_reconciliation_log() -> list:
    conn = db.get_connection()
    alerts = pd.read_sql_query(
        "SELECT ts, event, direction, magnitude_atr, price FROM intraday_alerts ORDER BY ts", conn,
    )
    conn.close()

    rows = []
    already_long = False
    for _, a in alerts.iterrows():
        mag = a["magnitude_atr"]
        is_long = pd.notna(mag) and mag >= config.BURST_LONG_MAGNITUDE_ATR

        if a["event"] == "BURST_START":
            already_long = is_long
            label = f"Trigger out - LONG {a['direction']}" if is_long else f"Trigger in - {a['direction']}"
            rows.append({"ts": a["ts"], "label": label, "price": a["price"]})
        elif a["event"] == "BURST_CONTINUE":
            if not already_long and is_long:
                already_long = True
                rows.append({"ts": a["ts"], "label": f"Trigger out - LONG {a['direction']}", "price": a["price"]})
        elif a["event"] == "BURST_END":
            # A burst can cool back below the LONG line before it's actually
            # logged as ended, so whether this episode counts as "had gone
            # LONG" comes from already_long (set the moment any row in this
            # episode crossed the threshold), not this row's own magnitude.
            label = f"Trigger in - {a['direction']} faded" if already_long else f"Trigger out - RANGING ({a['direction']} faded)"
            already_long = False
            rows.append({"ts": a["ts"], "label": label, "price": a["price"]})

    return rows


def main():
    rows = build_reconciliation_log()
    if not rows:
        print("No trigger events recorded yet.")
        return
    for r in rows:
        print(f"{r['label']:<28} {_fmt_ts(r['ts'])}  @ {r['price']:,.2f}")


if __name__ == "__main__":
    main()
