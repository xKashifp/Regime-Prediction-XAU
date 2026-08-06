"""Always-on-top desktop widget: a small, semi-transparent status panel
for XAUUSD, so you don't need a browser tab open just to glance at the trend.

Shows everything the old Streamlit dashboard did, condensed into one panel:
DAILY TREND (daily-bar regime + confidence), NEAR TERM (train_near_term_model.py's
live bearish/ranging/bullish call over the next few bars -- expect this to
disagree with DAILY TREND often, they're answering different questions on
different horizons), BURST WATCH (5-minute detector, tagged LONG when it's a big hike >=
config.BURST_LONG_MAGNITUDE_ATR, plus magnitude/note), POSITION (last
long-term signal fired), and a RECENT log of the last few signals/burst
events. Polls data/test.db directly, read-only, every few seconds.

Run:
    python -m src.widget                     (standalone: closing it is final)
    scripts\\start_all.ps1 / Start Monitoring.bat  (supervised alongside
                                              realtime_loop/intraday_loop: a
                                              closed window respawns in ~5s)

Drag anywhere on the panel to move it. Click the small _ to minimize it to
the taskbar, or the x to close it.

The old Streamlit dashboard (src/dashboard.py) is no longer auto-started --
it's kept as an optional manual deep-dive (`streamlit run src/dashboard.py`)
for the intraday chart, full signal history, and raw burst log this widget
deliberately leaves out.
"""

import sqlite3
import time
import tkinter as tk

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk

from . import config
from .labeling import build_labels

REGIME_COLOR = {"bullish": "#22c55e", "bearish": "#ef4444", "ranging": "#94a3b8"}
CALM_COLOR = "#94a3b8"
BG = "#14161b"
FG_DIM = "#7a8290"
FG_BRIGHT = "#e6e8eb"

BURST_STALE_MINUTES = 15
REFRESH_MS = 5000
LOG_LIMIT = 5
WIDTH, HEIGHT = 420, 556
BG_ALPHA = 0.85  # whole-panel transparency: slightly see-through, not fully clear

_FONT_FILES = {False: "segoeui.ttf", True: "segoeuib.ttf"}
_FONT_CACHE = {}


def _load_font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(_FONT_FILES[bold], size)
        except OSError:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

SIGNAL_LOG = {
    "TRIGGER_IN_LONG": "BUY",
    "TRIGGER_OUT_LONG": "EXIT BUY",
    "TRIGGER_IN_SHORT": "SELL",
    "TRIGGER_OUT_SHORT": "EXIT SELL",
}


def _format_age(age_sec):
    """Formats a same-domain seconds-elapsed value as a short relative label
    ("just now" / "3m ago" / "2.1h ago"). Used everywhere instead of an
    absolute clock time: this DB mixes two clock domains (broker-server-time
    in candles_m5/intraday_alerts vs true UTC in regime_predictions/signals,
    confirmed ~3h apart on this account), and printing two absolute times
    from different domains side by side is exactly what made the panel look
    wrong at a glance. "X ago" only ever needs now-minus-then within ONE
    domain, so it sidesteps reconciling the domains for display entirely."""
    if age_sec is None:
        return "?"
    age_sec = max(0.0, age_sec)
    if age_sec < 5:
        return "just now"
    if age_sec < 60:
        return f"{int(age_sec)}s ago"
    if age_sec < 3600:
        return f"{int(age_sec / 60)}m ago"
    return f"{age_sec / 3600:.1f}h ago"


def _format_datetime(ts):
    """Formats a broker-server-time epoch (candles_m5/intraday_alerts' domain)
    as e.g. "10:00 AM - 4AUG2026" -- formatted directly with no timezone
    conversion, same convention as get_live_bar's opened_at, so it lines up
    with what MT5's own terminal would show for that moment."""
    if ts is None:
        return ""
    dt = pd.to_datetime(int(ts), unit="s")
    return f"{dt:%I:%M %p} - {dt.day}{dt:%b}{dt.year}".upper()


def get_daily_trend():
    """Returns {regime, confidence, price, age_sec} or None if no daily data yet.
    age_sec is seconds since the model last ran, both sides true UTC
    (time.time() vs regime_predictions.ts) -- same domain, safe to subtract."""
    if not config.DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(config.DB_PATH))
    daily = pd.read_sql_query("SELECT date, close FROM daily_bars ORDER BY date", conn, parse_dates=["date"])
    preds = pd.read_sql_query("SELECT * FROM regime_predictions ORDER BY ts", conn)
    conn.close()
    if daily.empty:
        return None

    if not preds.empty:
        p = preds.iloc[-1]
        regime = p["label"]
        confidence = float(p[f"prob_{regime}"]) if f"prob_{regime}" in p else None
        age_sec = time.time() - float(p["ts"])
    else:
        labels, _swings = build_labels(daily)
        regime = labels.iloc[-1]["regime_name"] if not labels.empty else "ranging"
        confidence = None
        age_sec = None

    return {
        "regime": regime, "confidence": confidence,
        "price": float(daily.iloc[-1]["close"]), "age_sec": age_sec,
    }


def get_near_term_outlook():
    """Returns {label, confidence, price} from the near-term model (trained
    by train_near_term_model.py, scored live by intraday_loop.py on every
    closed bar): bearish/ranging/bullish over the next few bars. Replaces the
    single-next-candle model (retired 2026-08-05: ~0.51 AUC across every
    variant tried, and it ran bearish on 14/15 live calls -- 27% hit rate --
    straight through a sustained rally). Returns None if nothing's been
    produced yet (model not trained, or intraday_loop not running)."""
    if not config.DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(config.DB_PATH))
    row = pd.read_sql_query(
        "SELECT label, confidence, price FROM near_term_predictions ORDER BY ts DESC LIMIT 1", conn,
    )
    conn.close()
    if row.empty:
        return None
    r = row.iloc[0]
    return {"label": r["label"], "confidence": float(r["confidence"]), "price": float(r["price"])}


def _walk_episode_start(alerts, start_idx):
    """Walks back through alerts[start_idx:] across this episode's
    BURST_START/BURST_CONTINUE rows (stopping at the first BURST_START,
    inclusive, or the first row that belongs to an earlier episode). Returns
    (peak_magnitude, start_ts) -- a burst can cool back below the LONG line
    before it's actually logged as over (observed in practice: peaked at
    5.5x ATR, faded to 4.7x, then ended), so the peak has to come from this
    walk, not just the row right before END; start_ts is this episode's own
    BURST_START time, for "since when" display."""
    peak_magnitude = 0.0
    start_ts = None
    for _, row in alerts.iloc[start_idx:].iterrows():
        if row["event"] not in ("BURST_START", "BURST_CONTINUE"):
            break  # ran past this episode's start into an earlier one
        peak_magnitude = max(peak_magnitude, float(row["magnitude_atr"]) if pd.notna(row["magnitude_atr"]) else 0.0)
        start_ts = float(row["ts"])
        if row["event"] == "BURST_START":
            break
    return peak_magnitude, start_ts


def get_burst_status():
    """Returns {calm, direction, is_long, was_long, magnitude, detail, price,
    since_ts} or None if intraday_loop hasn't produced data yet. was_long only
    matters while calm: whether the burst that just stopped had already
    reached the LONG/extended tier (>= BURST_LONG_MAGNITUDE_ATR) before it
    stopped, vs. fizzling out while still fresh -- used by get_position_state
    to tell "a big move just ended" apart from "nothing was really happening".
    since_ts (broker-server-time epoch, same domain as alerts.ts) is when the
    *current* position state began: the active episode's BURST_START while a
    burst is running, or the moment it ended/went stale while calm."""
    if not config.DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(config.DB_PATH))
    last_candle = pd.read_sql_query("SELECT time, close FROM candles_m5 ORDER BY time DESC LIMIT 1", conn)
    # LIMIT 50 gives plenty of headroom to scan a whole episode back to its
    # BURST_START if the latest row is a BURST_END (see below) -- observed
    # episodes top out around 18 rows, and this table stays small (~100 rows
    # total), so this is a cheap query either way.
    alerts = pd.read_sql_query("SELECT * FROM intraday_alerts ORDER BY ts DESC LIMIT 50", conn)
    conn.close()
    if last_candle.empty:
        return None

    price = float(last_candle.iloc[0]["close"])
    if alerts.empty:
        return {"calm": True, "direction": None, "is_long": False, "was_long": False, "magnitude": None, "detail": "No burst right now.", "price": price, "since_ts": None}

    last = alerts.iloc[0]
    # "now" here must be broker-server time, same clock domain as ts -- this
    # machine's own wall clock (pd.Timestamp.now()) can be an hour-plus off
    # from the broker's, which was making every burst look stale immediately
    # (age_min always blew past BURST_STALE_MINUTES) regardless of whether it
    # was actually still active. last_candle's own time is continuously
    # updated by intraday_loop.py from real broker-server time, so it's
    # already the right clock to compare against -- no new MT5 call needed.
    now_broker = pd.to_datetime(int(last_candle.iloc[0]["time"]), unit="s")
    age_min = (now_broker - pd.to_datetime(last["ts"], unit="s")).total_seconds() / 60
    if last["event"] == "BURST_END" or age_min > BURST_STALE_MINUTES:
        # BURST_END rows are logged with magnitude_atr zeroed out (there's no
        # "current" magnitude once a burst is over), so whether THIS episode
        # ever reached the LONG tier has to come from the rows before it --
        # walk back through its BURST_START/CONTINUE history and take the
        # peak, not just the row right before END (see _walk_episode_start).
        # A burst that went stale without an explicit END has no such
        # follow-up row, so `last` itself is already the last active reading
        # and needs no walk-back.
        if last["event"] == "BURST_END":
            peak_magnitude, _ = _walk_episode_start(alerts, 1)
        else:
            peak_magnitude = float(last["magnitude_atr"]) if pd.notna(last["magnitude_atr"]) else 0.0
        was_long = peak_magnitude >= config.BURST_LONG_MAGNITUDE_ATR
        detail = f"Last burst ({last['direction']}) ended {age_min:.0f} min ago." if last["event"] == "BURST_END" else "No burst right now."
        # since_ts = when it ended/went stale (last's own ts), not when the
        # episode started -- this state ("faded"/"ranging") began at the end.
        return {"calm": True, "direction": last["direction"], "is_long": False, "was_long": was_long, "magnitude": None, "detail": detail, "price": price, "since_ts": float(last["ts"])}

    magnitude = float(last["magnitude_atr"]) if pd.notna(last["magnitude_atr"]) else 0.0
    # since_ts here IS the episode start -- this state ("in"/"long") began
    # when the burst itself started, not at this latest CONTINUE reading.
    _, since_ts = _walk_episode_start(alerts, 0)
    return {
        "calm": False, "direction": last["direction"],
        "is_long": magnitude >= config.BURST_LONG_MAGNITUDE_ATR, "was_long": False, "magnitude": magnitude,
        "detail": last["note"] if pd.notna(last["note"]) else "", "price": price, "since_ts": since_ts,
    }


def get_live_bar():
    """Returns {open, high, low, close, volume, bar_minutes, opened_at} for
    the current still-forming 5m candle -- candles_m5's newest row,
    continuously re-upserted by intraday_loop.py until this bar closes -- or
    None if no data yet. Meant to let you eyeball this against your own MT5
    terminal to confirm the pipeline is actually tracking the live market.

    opened_at is the bar's own broker-server-time epoch, formatted directly
    (no conversion) -- that's deliberate: candles_m5.time is stored as the
    broker's own clock reading treated as literal UTC (see csv_seed.py /
    mt5_client.py's docstrings), so formatting it straight reproduces exactly
    the HH:MM MT5's own terminal shows for this candle. Converting it to true
    UTC or this machine's local time first would make it *stop* matching
    what's on your MT5 screen, which defeats the point of showing it."""
    if not config.DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(config.DB_PATH))
    row = pd.read_sql_query(
        "SELECT time, open, high, low, close, tick_volume FROM candles_m5 ORDER BY time DESC LIMIT 1", conn,
    )
    conn.close()
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]),
        "volume": int(r["tick_volume"]), "bar_minutes": config.INTRADAY_BAR_MINUTES,
        "opened_at": pd.to_datetime(int(r["time"]), unit="s"),
    }


def get_position_state():
    """Returns {text, color, since_text} for the position readout: TRIGGER IN
    while a strong directional intraday burst is active, TRIGGER OUT the
    moment it fades back to normal/ranging. Sourced directly from the same
    burst state as BURST WATCH (get_burst_status) -- NOT the daily swing
    model. That was the previous version of this panel (signals/
    regime_predictions-driven), and it showed a bearish "position" entered
    days ago on the multi-month model while the 5-minute chart was in the
    middle of a huge rally -- a different question entirely from "is a
    strong move happening right now," which is what this panel is actually
    for. since_text is when the CURRENT state began (formatted, e.g.
    "10:00 AM - 4AUG2026"), not just this refresh's timestamp."""
    burst = get_burst_status()
    if burst is None:
        return {"text": "NO SIGNAL YET", "color": CALM_COLOR, "since_text": ""}
    since_text = _format_datetime(burst.get("since_ts"))
    if burst["calm"]:
        if burst["was_long"]:
            # The move that just stopped had already gone LONG/extended --
            # that's the "get out, don't chase it" state fading back to
            # normal, i.e. exactly the point this panel exists to flag as a
            # fresh entry again (mirrors TRIGGER IN below for an active-but-
            # not-yet-extended burst).
            color = REGIME_COLOR.get(burst["direction"], CALM_COLOR)
            return {"text": f"TRIGGER IN - {burst['direction'].upper()} FADED @ {burst['price']:,.2f}", "color": color, "since_text": since_text}
        return {"text": "TRIGGER OUT - RANGING", "color": CALM_COLOR, "since_text": since_text}
    color = REGIME_COLOR.get(burst["direction"], CALM_COLOR)
    if burst["is_long"]:
        # Already a big/extended hike (>= BURST_LONG_MAGNITUDE_ATR) -- past
        # the point of a fresh entry. This is the "get out, don't chase it"
        # moment, not a signal to newly get in.
        return {"text": f"TRIGGER OUT - LONG {burst['direction'].upper()} @ {burst['price']:,.2f}", "color": color, "since_text": since_text}
    return {"text": f"TRIGGER IN - {burst['direction'].upper()} @ {burst['price']:,.2f}", "color": color, "since_text": since_text}


def get_recent_log(limit=LOG_LIMIT):
    """Last few signals + burst events, merged and sorted newest-first --
    the same events the old dashboard's expanders showed, condensed to
    one line each.

    signals.ts is a true-UTC epoch (realtime_loop.py's time.time());
    intraday_alerts.ts is a broker-server-time epoch (candles_m5.time's
    domain, confirmed ~3h ahead of true UTC on this account). Sorting them
    together by raw ts, or printing both as absolute clock times, silently
    gets both the merge order and the displayed time wrong -- fixed two ways:
    (1) each row's displayed age is same-domain (true UTC "now" for signals,
    the latest candle's broker time for alerts, never mixed); (2) for MERGE
    ORDER ONLY, alerts.ts is shifted into a true-UTC-equivalent using the live
    offset between those same two "now" values -- no extra MT5 call needed,
    both are already being read."""
    if not config.DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(config.DB_PATH))
    sigs = pd.read_sql_query("SELECT ts, signal_type, price FROM signals ORDER BY ts DESC LIMIT ?", conn, params=(limit,))
    alerts = pd.read_sql_query(
        "SELECT ts, event, direction, magnitude_atr, price FROM intraday_alerts ORDER BY ts DESC LIMIT ?",
        conn, params=(limit,),
    )
    last_candle = pd.read_sql_query("SELECT time FROM candles_m5 ORDER BY time DESC LIMIT 1", conn)
    conn.close()

    now_true_utc = time.time()
    broker_now = float(last_candle.iloc[0]["time"]) if not last_candle.empty else None
    broker_offset = (broker_now - now_true_utc) if broker_now is not None else 0.0

    rows = []
    for _, s in sigs.iterrows():
        title = SIGNAL_LOG.get(s["signal_type"], s["signal_type"])
        color = REGIME_COLOR["bullish"] if "IN_LONG" in s["signal_type"] else (
            REGIME_COLOR["bearish"] if "IN_SHORT" in s["signal_type"] else CALM_COLOR
        )
        rows.append({
            "sort_ts": float(s["ts"]), "age_sec": now_true_utc - float(s["ts"]),
            "color": color, "text": f"{title} @ {s['price']:,.2f}",
        })

    for _, a in alerts.iterrows():
        if a["event"] in ("BURST_START", "BURST_CONTINUE"):
            is_long = pd.notna(a["magnitude_atr"]) and a["magnitude_atr"] >= config.BURST_LONG_MAGNITUDE_ATR
            tier = "LONG " if is_long else ""
            verb = "burst" if a["event"] == "BURST_START" else "continuing"
            color = REGIME_COLOR.get(a["direction"], CALM_COLOR)
            text = f"{tier}{a['direction']} {verb} @ {a['price']:,.2f}"
        else:
            color, text = CALM_COLOR, f"burst faded @ {a['price']:,.2f}"
        age_sec = (broker_now - float(a["ts"])) if broker_now is not None else None
        rows.append({
            "sort_ts": float(a["ts"]) - broker_offset, "age_sec": age_sec,
            "color": color, "text": text,
        })

    rows.sort(key=lambda r: r["sort_ts"], reverse=True)
    return [
        {"text": f"{_format_age(r['age_sec'])}  {r['text']}", "color": r["color"]}
        for r in rows[:limit]
    ]


class BitmapLabel:
    """A Label whose text is a pre-rendered PIL bitmap instead of Tk's native
    (ClearType) text path.

    -alpha blends this whole window against the desktop a second time to get
    the "slightly see-through" look. ClearType antialiasing bakes in colored
    sub-pixel fringing calibrated against one known background; blend that a
    second time against a different, unknown background (the real desktop)
    and the fringing math no longer holds -- confirmed in practice as bold
    text visibly slicing apart. Plain grayscale antialiasing (what PIL does)
    has no per-subpixel color assumption, so it survives a second blend
    cleanly. Composited here against BG first (this widget's own opaque
    fill), then the whole finished window gets the single -alpha pass.
    """

    def __init__(self, parent, size, bold=False, fg=FG_BRIGHT, bg=BG):
        self.bg = bg
        self.fg = fg
        self.font = _load_font(size, bold)
        self.label = tk.Label(parent, bg=bg, bd=0, highlightthickness=0)
        self._photo = None
        self.text = ""

    def place(self, **kwargs):
        self.label.place(**kwargs)
        return self

    def config(self, text=None, fg=None):
        if text is None and fg is None:
            return
        if text is not None:
            self.text = text
        if fg is not None:
            self.fg = fg
        self._render()

    def _render(self):
        if not self.text:
            self.label.config(image="")
            self._photo = None
            return
        dummy = Image.new("RGBA", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), self.text, font=self.font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img = Image.new("RGBA", (max(w, 1) + 2, max(h, 1) + 2), self.bg)
        ImageDraw.Draw(img).text((-bbox[0] + 1, -bbox[1] + 1), self.text, font=self.font, fill=self.fg)
        self._photo = ImageTk.PhotoImage(img)
        self.label.config(image=self._photo)


class Widget:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("XAUUSD")
        self.root.overrideredirect(True)  # no title bar/border
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", BG_ALPHA)
        self.root.configure(bg=BG)

        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{sw - WIDTH - 28}+28")

        self._drag = {"x": 0, "y": 0}
        for widget in (self.root,):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._do_drag)

        self._titles = []  # keep BitmapLabel (and its PhotoImage) alive -- a
        # locals-only reference gets garbage-collected once __init__/_title
        # returns, which blanks the label even though the Tk widget itself
        # is still on screen (confirmed: this is exactly what was happening).

        self.close_btn = BitmapLabel(self.root, 12, fg="#555")
        self.close_btn.config(text="x")
        self.close_btn.place(x=WIDTH - 25, y=8)
        self.close_btn.label.config(cursor="hand2")
        self.close_btn.label.bind("<Button-1>", lambda e: self.root.destroy())

        # overrideredirect(True) windows have no OS-drawn minimize control and
        # iconify() alone is a no-op for them on Windows -- the window just
        # vanishes with no taskbar entry to bring it back. The fix is to drop
        # overrideredirect right before iconifying (so Windows treats it as a
        # normal window and gives it a taskbar entry), then reapply it once
        # the taskbar click restores the window (caught via <Map>, since
        # that's what fires on restore) so it goes back to borderless.
        self._minimized = False
        self.minimize_btn = BitmapLabel(self.root, 12, fg="#555")
        self.minimize_btn.config(text="_")
        self.minimize_btn.place(x=WIDTH - 46, y=8)
        self.minimize_btn.label.config(cursor="hand2")
        self.minimize_btn.label.bind("<Button-1>", lambda e: self._minimize())
        self.root.bind("<Map>", self._on_map)

        self.price_lbl = BitmapLabel(self.root, 14, bold=True)
        self.price_lbl.place(x=14, y=8)

        self._title("DAILY TREND", 32)
        self.daily_val = BitmapLabel(self.root, 20, bold=True)
        self.daily_val.place(x=14, y=52)
        self.daily_conf = BitmapLabel(self.root, 11, fg=FG_DIM)
        self.daily_conf.place(x=14, y=80)

        # Separate from DAILY TREND on purpose (same reasoning as BURST WATCH
        # below being its own panel): this is a live, always-scored read on
        # the next few bars, not the multi-month swing call above -- the two
        # will often disagree and that's expected, not a bug.
        self._title("NEAR TERM", 113)
        self.near_term_val = BitmapLabel(self.root, 20, bold=True)
        self.near_term_val.place(x=14, y=133)
        self.near_term_conf = BitmapLabel(self.root, 11, fg=FG_DIM)
        self.near_term_conf.place(x=14, y=161)

        self._title("BURST WATCH", 194)
        # Raw OHLC of the current still-forming 5m bar -- placed here, not
        # near DAILY TREND, on purpose: it's 5-minute data and has no bearing
        # on the daily/multi-month regime call above (that's the whole point
        # of the two being separate panels). Putting it right above DAILY
        # TREND made it look related and caused exactly that confusion in
        # practice -- moved next to the panel it actually describes.
        self.live_bar_lbl = BitmapLabel(self.root, 11, fg=FG_DIM)
        self.live_bar_lbl.place(x=14, y=217)
        self.burst_val = BitmapLabel(self.root, 20, bold=True)
        self.burst_val.place(x=14, y=239)
        self.burst_detail = BitmapLabel(self.root, 11, fg=FG_DIM)
        self.burst_detail.place(x=14, y=267)

        self._title("POSITION", 300)
        self.position_val = BitmapLabel(self.root, 17, bold=True)
        self.position_val.place(x=14, y=320)
        self.position_ts = BitmapLabel(self.root, 11, fg=FG_DIM)
        self.position_ts.place(x=14, y=348)

        self._title("RECENT", 381)
        self.log_lbls = []
        for i in range(LOG_LIMIT):
            lbl = BitmapLabel(self.root, 11, fg=FG_DIM)
            lbl.place(x=14, y=404 + i * 21)
            self.log_lbls.append(lbl)

        self.updated_lbl = BitmapLabel(self.root, 10, fg="#555")
        self.updated_lbl.place(x=14, y=HEIGHT - 25)

        self.refresh()

    def _title(self, text, y):
        t = BitmapLabel(self.root, 11, fg=FG_DIM)
        t.config(text=text)
        t.place(x=14, y=y)
        self._titles.append(t)

    def _start_drag(self, event):
        self._drag["x"], self._drag["y"] = event.x, event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag["x"])
        y = self.root.winfo_y() + (event.y - self._drag["y"])
        self.root.geometry(f"+{x}+{y}")

    def _minimize(self):
        self._minimized = True
        self.root.overrideredirect(False)
        self.root.iconify()

    def _on_map(self, event):
        if self._minimized and self.root.state() == "normal":
            self._minimized = False
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)

    def refresh(self):
        try:
            daily = get_daily_trend()
            burst = get_burst_status()
            price = None
            age_sec = None

            if daily is None:
                self.daily_val.config(text="no data", fg="#555")
                self.daily_conf.config(text="")
            else:
                price = daily["price"]
                age_sec = daily["age_sec"]
                color = REGIME_COLOR.get(daily["regime"], CALM_COLOR)
                self.daily_val.config(text=daily["regime"].upper(), fg=color)
                conf_txt = f"{daily['confidence']:.0%} confidence" if daily["confidence"] is not None else "no confidence data"
                self.daily_conf.config(text=conf_txt)

            near_term = get_near_term_outlook()
            if near_term is None:
                self.near_term_val.config(text="not running", fg="#555")
                self.near_term_conf.config(text="Run train_near_term_model.py to enable.")
            else:
                color = REGIME_COLOR.get(near_term["label"], CALM_COLOR)
                self.near_term_val.config(text=near_term["label"].upper(), fg=color)
                self.near_term_conf.config(text=f"{near_term['confidence']:.0%} confidence")

            if burst is None:
                self.burst_val.config(text="not running", fg="#555")
                self.burst_detail.config(text="Start intraday_loop to begin.")
            elif burst["calm"]:
                self.burst_val.config(text="CALM", fg=CALM_COLOR)
                self.burst_detail.config(text=burst["detail"][:62])
                price = burst["price"]
            else:
                price = burst["price"]
                tier = "LONG " if burst["is_long"] else ""
                color = REGIME_COLOR.get(burst["direction"], CALM_COLOR)
                self.burst_val.config(text=f"{tier}{burst['direction'].upper()}", fg=color)
                mag_txt = f"{burst['magnitude']:.1f}x ATR" if burst["magnitude"] is not None else ""
                detail_txt = f"{mag_txt} · {burst['detail']}" if burst["detail"] else mag_txt
                self.burst_detail.config(text=detail_txt[:62])

            if price is not None:
                self.price_lbl.config(text=f"XAUUSD   {price:,.2f}")

            live_bar = get_live_bar()
            if live_bar is None:
                self.live_bar_lbl.config(text="")
            else:
                self.live_bar_lbl.config(
                    text=f"{live_bar['opened_at']:%H:%M} MT5  O{live_bar['open']:.1f} "
                         f"H{live_bar['high']:.1f} L{live_bar['low']:.1f} C{live_bar['close']:.1f}"
                )

            pos = get_position_state()
            self.position_val.config(text=pos["text"], fg=pos["color"])
            self.position_ts.config(text=pos["since_text"])

            log_rows = get_recent_log()
            for i, lbl in enumerate(self.log_lbls):
                if i < len(log_rows):
                    lbl.config(text=log_rows[i]["text"], fg=log_rows[i]["color"])
                else:
                    lbl.config(text="")

            if age_sec is not None:
                self.updated_lbl.config(text=f"model updated {_format_age(age_sec)}")
            else:
                self.updated_lbl.config(text="")
        except Exception:
            pass  # a transient DB read glitch should never crash the widget

        self.root.after(REFRESH_MS, self.refresh)

    def run(self):
        self.root.mainloop()


def main():
    Widget().run()


if __name__ == "__main__":
    main()
