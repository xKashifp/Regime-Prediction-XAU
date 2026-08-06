"""Streamlit dashboard: visualize the long-term regime read on XAUUSD.

Manual/optional -- not auto-started by scripts\\start_all.ps1. src/widget.py
is the always-on desktop overlay for a quick glance; run this by hand when
you want the deeper drill-down (chart, full history) it deliberately omits.

Run:
    streamlit run src/dashboard.py

Two things, at a glance, and nothing else competing for attention:
  - LONG-TERM card: the daily-bar regime model (weeks/months view).
  - SHORT-TERM card: the 5-minute burst watch (last few hours view).
Everything else (the intraday chart, full signal history, raw burst log,
color legend) is tucked into collapsed expanders below -- there if you want
to dig in, out of the way if you don't.

Reads directly from the live Postgres DB, which realtime_loop.py /
intraday_loop.py populate entirely from the live MT5 terminal (no static file).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

from src import config, db, session_windows
from src.labeling import build_labels
from src.trigger_log import build_reconciliation_log

st.set_page_config(page_title="XAUUSD Regime Monitor", layout="wide")

REGIME_COLOR = {"bullish": "#22c55e", "bearish": "#ef4444", "ranging": "#94a3b8"}
CALM_COLOR = "#94a3b8"

# Plain-language explanations, in case the reader has never looked at a chart before.
REGIME_PLAIN = {
    "bullish": ("📈", "BULLISH", "Higher highs and higher lows for a while now -- long-term, price looks like it's climbing."),
    "bearish": ("📉", "BEARISH", "Lower highs and lower lows for a while now -- long-term, price looks like it's falling."),
    "ranging": ("➖", "SIDEWAYS", "No clear long-term direction right now. Best to just wait."),
}

SIGNAL_PLAIN = {
    "TRIGGER_IN_LONG": ("🟢", "BUY SIGNAL", "A new long-term uptrend was just confirmed."),
    "TRIGGER_OUT_LONG": ("⚪", "EXIT SIGNAL (close the buy)", "The uptrend looks like it's weakening or ending."),
    "TRIGGER_IN_SHORT": ("🔴", "SELL SIGNAL", "A new long-term downtrend was just confirmed."),
    "TRIGGER_OUT_SHORT": ("⚪", "EXIT SIGNAL (close the sell)", "The downtrend looks like it's weakening or ending."),
}

BURST_STALE_MINUTES = 15  # if the last alert is older than this, treat it as "calm" even if it was never formally closed (3 bars at 5m)


def status_card(column, label: str, emoji: str, title: str, color: str, detail: str) -> None:
    with column:
        st.markdown(
            f"""
            <div style='padding:18px 20px;border-radius:12px;background:{color}22;border:2px solid {color};min-height:118px;'>
                <div style='font-size:13px;color:#888;font-weight:700;letter-spacing:0.5px;'>{label}</div>
                <div style='font-size:28px;font-weight:800;color:{color};margin-top:2px;'>{emoji} {title}</div>
                <div style='font-size:13px;margin-top:6px;color:#888;'>{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_short_term_status():
    """One combined read of the burst-watch state, reused for the card, the
    action line, and the "recent burst events" expander -- a single query
    instead of three. Returns None if intraday_loop hasn't produced any data yet."""
    try:
        conn = db.get_connection()
    except psycopg2.OperationalError:
        return None

    last_candle = pd.read_sql_query("SELECT time, close FROM candles_m5 ORDER BY time DESC LIMIT 1", conn)
    alerts = pd.read_sql_query("SELECT * FROM intraday_alerts ORDER BY ts DESC LIMIT 20", conn)
    conn.close()

    if last_candle.empty:
        return None

    server_dt = pd.to_datetime(last_candle.iloc[0]["time"], unit="s")
    price = float(last_candle.iloc[0]["close"])
    window_status = session_windows.status_for_time(server_dt.to_pydatetime())

    last = alerts.iloc[0] if not alerts.empty else None
    # server_dt (above) and last["ts"] are both broker-server-time epochs --
    # pd.Timestamp.now() is this machine's local wall clock, a different
    # domain that can be hours off from the broker's (confirmed ~3h on this
    # account), which was making every burst look stale/miscalculated
    # regardless of when it actually happened.
    age_min = (
        (server_dt - pd.to_datetime(last["ts"], unit="s")).total_seconds() / 60
        if last is not None else None
    )

    if alerts.empty or last["event"] == "BURST_END" or age_min > BURST_STALE_MINUTES:
        calm, direction, magnitude = True, None, None
        detail = (
            f"Last burst ({last['direction']}) ended {age_min:.0f} min ago."
            if last is not None and last["event"] == "BURST_END"
            else "No burst right now."
        )
    else:
        calm, direction = False, last["direction"]
        magnitude = float(last["magnitude_atr"]) if pd.notna(last["magnitude_atr"]) else None
        detail = last["note"]

    is_long = calm is False and magnitude is not None and magnitude >= config.BURST_LONG_MAGNITUDE_ATR

    return {
        "calm": calm, "direction": direction, "detail": detail, "magnitude": magnitude, "is_long": is_long,
        "price": price, "server_dt": server_dt, "window_status": window_status,
        "alerts": alerts,
    }


@st.cache_data(ttl=15)
def load_data():
    """Reads whatever realtime_loop.py has written to data/test.db -- all of it
    sourced live from MT5, nothing from a static file."""
    try:
        conn = db.get_connection()
    except psycopg2.OperationalError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily = pd.read_sql_query(
        "SELECT date, open, high, low, close FROM daily_bars ORDER BY date", conn, parse_dates=["date"]
    )
    preds = pd.read_sql_query("SELECT * FROM regime_predictions ORDER BY ts", conn)
    sigs = pd.read_sql_query("SELECT * FROM signals ORDER BY ts", conn)
    conn.close()

    if daily.empty:
        return daily, pd.DataFrame(), pd.DataFrame(), preds, sigs

    labels, swings = build_labels(daily)
    return daily, labels, swings, preds, sigs


@st.cache_data(ttl=15)
def load_today_intraday():
    """Today's 5-minute candles + today's burst alerts, from data/test.db only."""
    try:
        conn = db.get_connection()
    except psycopg2.OperationalError:
        return pd.DataFrame(), pd.DataFrame()

    m5 = pd.read_sql_query("SELECT time, open, high, low, close FROM candles_m5 ORDER BY time", conn)
    alerts = pd.read_sql_query("SELECT * FROM intraday_alerts ORDER BY ts", conn)
    conn.close()

    if m5.empty:
        return m5, alerts

    m5["time"] = pd.to_datetime(m5["time"], unit="s")
    today_start = m5["time"].iloc[-1].normalize()
    m5_today = m5[m5["time"] >= today_start].reset_index(drop=True)

    if not alerts.empty:
        alerts["time"] = pd.to_datetime(alerts["ts"], unit="s")
        alerts = alerts[alerts["time"] >= today_start].reset_index(drop=True)

    return m5_today, alerts


def build_today_chart(m5_today: pd.DataFrame, alerts_today: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=m5_today["time"], open=m5_today["open"], high=m5_today["high"],
        low=m5_today["low"], close=m5_today["close"],
        name="Gold price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ))

    if not alerts_today.empty:
        starts = alerts_today[alerts_today["event"] == "BURST_START"]
        ends = alerts_today[alerts_today["event"] == "BURST_END"]
        bull_starts = starts[starts["direction"] == "bullish"]
        bear_starts = starts[starts["direction"] == "bearish"]

        if not bull_starts.empty:
            fig.add_trace(go.Scatter(
                x=bull_starts["time"], y=bull_starts["price"], mode="markers", name="🟢 Burst starts (up)",
                marker=dict(symbol="triangle-up", color="#16a34a", size=13, line=dict(width=1, color="white")),
            ))
        if not bear_starts.empty:
            fig.add_trace(go.Scatter(
                x=bear_starts["time"], y=bear_starts["price"], mode="markers", name="🔴 Burst starts (down)",
                marker=dict(symbol="triangle-down", color="#dc2626", size=13, line=dict(width=1, color="white")),
            ))
        if not ends.empty:
            fig.add_trace(go.Scatter(
                x=ends["time"], y=ends["price"], mode="markers", name="⚪ Burst fades",
                marker=dict(symbol="x", color="#94a3b8", size=10, line=dict(width=1, color="white")),
            ))

    fig.update_layout(
        height=420, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        xaxis=dict(tickformat="%H:%M"),
    )
    return fig


def render():
    st.title("🪙 Gold (XAUUSD) Monitor")

    daily, labels, _swings, preds, sigs = load_data()
    short = get_short_term_status()

    if daily.empty:
        st.warning("No data yet. Run `python -m src.train_model` and/or `python -m src.realtime_loop` first.")
        return

    latest_price = float(daily.iloc[-1]["close"])
    latest_date = daily.iloc[-1]["date"]

    if not preds.empty:
        latest_pred = preds.iloc[-1]
        regime = latest_pred["label"]
        confidence = float(latest_pred[f"prob_{regime}"]) if f"prob_{regime}" in latest_pred else None
        last_updated = pd.to_datetime(latest_pred["ts"], unit="s")
    else:
        regime = labels.iloc[-1]["regime_name"] if not labels.empty else "ranging"
        confidence = None
        last_updated = None

    lt_emoji, lt_title, _lt_desc = REGIME_PLAIN.get(regime, REGIME_PLAIN["ranging"])
    lt_color = REGIME_COLOR.get(regime, CALM_COLOR)

    # --- two status cards, side by side: this IS the whole dashboard at a glance ---
    # "LONG" is reserved everywhere on this page for burst MAGNITUDE (a big hike,
    # within this same <=2h window, real company-loss risk) -- never for time
    # horizon. So the daily model is "DAILY TREND", not "long-term".
    col1, col2 = st.columns(2)
    conf_txt = f"{confidence:.0%} confidence" if confidence is not None else "no confidence data yet"
    status_card(col1, "DAILY TREND  ·  overall direction", lt_emoji, lt_title, lt_color, conf_txt)

    if short is None:
        status_card(col2, "BURST WATCH  ·  next few hours", "⏳", "NOT RUNNING", CALM_COLOR, "Start `intraday_loop` to begin.")
    elif short["calm"]:
        status_card(col2, "BURST WATCH  ·  next few hours", "✅", "CALM", CALM_COLOR, short["detail"])
    else:
        st_color = REGIME_COLOR.get(short["direction"], CALM_COLOR)
        st_emoji = "🟢" if short["direction"] == "bullish" else "🔴"
        tier = "LONG " if short["is_long"] else ""
        status_card(col2, "BURST WATCH  ·  next few hours", st_emoji, f"{tier}{short['direction'].upper()} BURST", st_color, short["detail"])

    st.write("")

    # --- one line: what should I actually do right now ---
    position_state = "nothing open"
    last_sig = sigs.iloc[-1] if not sigs.empty else None
    if last_sig is not None:
        if last_sig["signal_type"] == "TRIGGER_IN_LONG":
            position_state = "a BUY (long)"
        elif last_sig["signal_type"] == "TRIGGER_IN_SHORT":
            position_state = "a SELL (short)"

    if short is not None and not short["calm"]:
        action_emoji = "🟢" if short["direction"] == "bullish" else "🔴"
        tier = "LONG " if short["is_long"] else ""
        urgency = " — big enough to matter, worth a look now." if short["is_long"] else ""
        action_title = f"{action_emoji} {tier}{short['direction'].upper()} BURST ACTIVE"
        action_desc = f"{short['detail']}{urgency} Daily trend is still **{lt_title}**."
        action_color = REGIME_COLOR.get(short["direction"], CALM_COLOR)
    elif last_sig is not None:
        a_emoji, a_title, a_desc = SIGNAL_PLAIN.get(last_sig["signal_type"], ("⏳", "NO SIGNAL YET", ""))
        action_title = f"{a_emoji} {a_title}"
        action_desc = f"{a_desc} (fired at price {last_sig['price']:.2f})"
        action_color = REGIME_COLOR["bullish"] if "IN_LONG" in last_sig["signal_type"] else (
            REGIME_COLOR["bearish"] if "IN_SHORT" in last_sig["signal_type"] else CALM_COLOR
        )
    else:
        action_title, action_desc, action_color = "⏳ NO SIGNAL YET", "Nothing has fired yet — just keep watching.", CALM_COLOR

    st.markdown(
        f"""
        <div style='padding:18px 22px;border-radius:12px;background:{action_color}15;border:2px dashed {action_color};'>
            <div style='font-size:13px;color:#888;font-weight:700;'>WHAT SHOULD I DO?</div>
            <div style='font-size:24px;font-weight:800;margin-top:2px;'>{action_title}</div>
            <div style='font-size:14px;margin-top:6px;'>{action_desc}</div>
            <div style='font-size:12px;margin-top:8px;color:#888;'>
                System currently thinks you have <b>{position_state}</b> open. This app never trades for you.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Gold price: **{latest_price:,.2f}** · Daily-trend data through {latest_date.date()}"
        + (f" · last check {last_updated:%H:%M:%S}" if last_updated is not None else "")
    )

    st.sidebar.caption("Auto-refreshes every 30 seconds.")
    if st.sidebar.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    # --- everything else: collapsed by default, there if you want to dig in ---
    with st.expander("📈 Today's 5-minute chart"):
        m5_today, alerts_today = load_today_intraday()
        if m5_today.empty:
            st.info("No 5-minute data yet today.")
        else:
            fig = build_today_chart(m5_today, alerts_today)
            st.plotly_chart(fig, width="stretch")

    with st.expander("📋 Full signal history (long-term buy / sell / exit)"):
        if sigs.empty:
            st.write("Nothing has fired yet.")
        else:
            show = sigs.copy()
            show["When"] = pd.to_datetime(show["ts"], unit="s")
            show["What happened"] = show["signal_type"].map(
                lambda t: f"{SIGNAL_PLAIN.get(t, ('', t, ''))[0]} {SIGNAL_PLAIN.get(t, ('', t, ''))[1]}"
            )
            show["Gold trend at the time"] = show["regime"].map(lambda r: REGIME_PLAIN.get(r, ('', r, ''))[1])
            show["How sure"] = show["confidence"].map(lambda c: f"{c:.0%}")
            show["Price"] = show["price"].map(lambda p: f"{p:,.2f}")
            show["Why"] = show["note"]
            st.dataframe(
                show[["When", "What happened", "Gold trend at the time", "How sure", "Price", "Why"]]
                .sort_values("When", ascending=False),
                width="stretch", hide_index=True,
            )

    with st.expander("🔁 Trigger in/out reconciliation log (5-minute burst-driven)"):
        rows = build_reconciliation_log()
        if not rows:
            st.write("No trigger events recorded yet.")
        else:
            show = pd.DataFrame(rows)
            show["When"] = pd.to_datetime(show["ts"], unit="s")
            show["Price"] = show["price"].map(lambda p: f"{p:,.2f}")
            st.dataframe(
                show[["When", "label", "Price"]].rename(columns={"label": "Event"}).sort_values("When", ascending=False),
                width="stretch", hide_index=True,
            )

    with st.expander("⚡ Recent burst events (short-term)"):
        if short is None or short["alerts"].empty:
            st.write("Nothing yet.")
        else:
            alerts = short["alerts"]
            show = alerts.copy()
            show["When"] = pd.to_datetime(show["ts"], unit="s")
            show["Strength (x normal)"] = show["magnitude_atr"].round(2)
            show["Confidence"] = show["continuation_prob"].apply(lambda p: f"{p:.0%}" if pd.notna(p) else "n/a")
            show["Price"] = show["price"].map(lambda p: f"{p:,.2f}")
            st.dataframe(
                show[["When", "event", "direction", "Strength (x normal)", "Confidence", "window_name", "Price"]]
                .rename(columns={"event": "Event", "direction": "Direction", "window_name": "Window"}),
                width="stretch", hide_index=True,
            )

    with st.expander("❓ What do the colors mean?"):
        st.markdown(
            f"""
- **DAILY TREND** card = the daily-bar trend model (weeks/months view).
- **BURST WATCH** card = the 5-minute watch for a hike in the next few minutes to max 2 hours.
- **LONG** (as in "LONG BULLISH BURST") = a *big* hike, {config.BURST_LONG_MAGNITUDE_ATR:.1f}x normal volatility or more --
  real company-loss risk, not just a minor wiggle. No "LONG" tag = it fired but isn't that big.
  This "LONG" is about size, not time -- it has nothing to do with days or weeks.
- 🟢 green = bullish/buy · 🔴 red = bearish/sell · ⚪ grey = calm/no signal.
- This is a decision-support tool, not an auto-trader. It never places or closes trades for you.
            """
        )


render = st.fragment(run_every="30s")(render)
render()