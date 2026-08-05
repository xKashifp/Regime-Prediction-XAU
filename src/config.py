"""Central configuration for the regime detection + live monitoring system."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- paths ---
PROCESSED_DIR = ROOT / "data" / "processed"
REGIME_LABELS_CSV = PROCESSED_DIR / "regime_labels.csv"  # output only, for inspection -- never read back in
DB_PATH = ROOT / "data" / "test.db"
MODEL_PATH = ROOT / "models" / "xgb_regime.json"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.json"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "regime.log"

# --- MT5 ---
MT5_SYMBOL = "XAUUSD"

# --- ZigZag swing detection ---
# A swing reverses only once price moves back by at least this fraction (3%)
# or this many ATR(14) multiples, whichever is larger. Filters daily noise so
# only genuine swings register.
ZIGZAG_MIN_PCT = 0.03
ZIGZAG_ATR_MULT = 3.0
ATR_PERIOD = 14

# --- Moving averages / regime confirmation ---
# train_model.py now trains on daily_bars' full accumulated history (2007 ->
# present, ~5000 days -- historical CSV-seeded plus everything realtime_loop.py
# has collected live since) rather than a fresh MT5 pull limited to this
# account's own ~88-day D1 window. With that much real history available,
# these run at the originally-intended "years of data" scale instead of the
# scaled-down 5/10/20/40 used while training was still bottlenecked on ~88 days.
MA_PERIODS = (20, 50, 100, 200)
TREND_CONFIRM_FAST = 50
TREND_CONFIRM_SLOW = 200
MA_SLOPE_LOOKBACK = 10  # days back to measure slope direction

# --- Regime labeling ---
MIN_REGIME_RUN_DAYS = 20  # minimum consecutive days for a run to count as "long-term" at this scale

REGIME_BEARISH = 0
REGIME_RANGING = 1
REGIME_BULLISH = 2
REGIME_NAMES = {REGIME_BEARISH: "bearish", REGIME_RANGING: "ranging", REGIME_BULLISH: "bullish"}

# --- Feature engineering ---
RSI_PERIOD = 14
# 1d added (2026-08): confirmed the live model's output was flat across a
# ~50-point range of today's still-forming price -- the only price-sensitive
# features (RSI/MACD/ATR%) don't cross a tree split in that range, and
# return_5d/10d/20d all dilute today's move across a multi-day window
# instead of isolating it. return_1d gives the model a feature that isolates
# just today's move so it has *something* that can react within the day.
RETURN_WINDOWS = (1, 5, 10, 20)
DAILY_HISTORY_COUNT = 5000  # request this many D1 bars from MT5; it just returns whatever actually exists

# --- Train/test split ---
TEST_HOLDOUT_FRACTION = 0.3

# --- Live trigger thresholds ---
TRIGGER_CONFIDENCE_IN = 0.60
TRIGGER_CONFIDENCE_OUT = 0.45

# Above TRIGGER_CONFIDENCE_IN (0.60) is enough to log a TRIGGER_IN signal at
# all; this is the higher bar the widget's POSITION line uses to decide
# whether a trend is strong enough to actually surface -- an entry that only
# just cleared 0.60 is still real (it's in signals/regime_predictions either
# way) but isn't flagged as a call to pay attention to.
#
# Picked against train_model.py's chronological 70/30 holdout over the full
# daily_bars history (2007 -> present) -- revisit if the holdout confidence
# distribution shifts materially after a retrain (e.g. once MA_PERIODS'
# longer windows and MACD have been in production for a while).
TRIGGER_STRONG_CONFIDENCE = 0.80

# --- Live loop ---
POLL_SECONDS = 300  # 5 minutes

# --- Intraday burst detection (session-open trigger warning) ---
# Runs on 5-minute bars, not 1-minute -- the broker's M1 feed proved too thin/
# noisy to train on (~30 days rolling window only). 5m bars are seeded from a
# real ~4-month CSV export (data/raw/xau5M.csv) plus everything collected live
# since, giving a much deeper and steadier training/assessment history.
INTRADAY_BAR_MINUTES = 5
# XAU_5m_data.csv is the same broker feed as the original xau5M.csv seed
# (confirmed identical OHLCV on overlapping dates) but exported back to 2004
# instead of just ~4 months -- strictly more history for the same symbol.
INTRADAY_CSV_SEED_PATH = ROOT / "data" / "raw" / "XAU_5m_data.csv"

# Server-time hour windows measured from the broker's own M1 volatility
# profile: these two blocks showed roughly 2x the average range of quiet
# hours (Tokyo open ~03:00-05:00, New York open ~16:00-18:00 server time).
# Detection itself always runs -- these windows only lower the alert
# threshold (more sensitive) and get flagged in the UI, they don't gate it.
SESSION_WINDOWS = {
    "tokyo": (3, 5),
    "new_york": (16, 18),
}
INTRADAY_MODEL_PATH = ROOT / "models" / "xgb_intraday.json"
INTRADAY_FEATURE_COLUMNS_PATH = ROOT / "models" / "intraday_feature_columns.json"

# Separate model, trained by train_next_bar_model.py: plain "will the next
# 5-minute bar close above or below this bar's close" on every bar (not
# gated on a burst being flagged, unlike xgb_intraday above). Powers the
# widget's NEXT BAR panel.
NEXT_BAR_MODEL_PATH = ROOT / "models" / "xgb_next_bar.json"
NEXT_BAR_FEATURE_COLUMNS_PATH = ROOT / "models" / "next_bar_feature_columns.json"

# Separate model, trained by train_near_term_model.py: 3-class (bearish/
# ranging/bullish) call on whether price moves at least
# INTRADAY_CONTINUATION_MOVE_ATR further in one direction over the next
# NEAR_TERM_HORIZON_BARS bars -- scored on every bar (not gated on a burst
# already firing, unlike xgb_intraday). The "is a real move brewing" answer
# to next_bar_model's noisy single-candle-direction question.
NEAR_TERM_MODEL_PATH = ROOT / "models" / "xgb_near_term.json"
NEAR_TERM_FEATURE_COLUMNS_PATH = ROOT / "models" / "near_term_feature_columns.json"
NEAR_TERM_HORIZON_BARS = 3  # 15 min at 5m bars
INTRADAY_MAX_BARS_LOADED = 200000  # safety ceiling on history reloaded each tick -- comfortably covers the ~17-month CSV seed plus over a year of live growth; not a validity window, just a memory cap
INTRADAY_ATR_PERIOD = 14  # bars (=70 min at 5m)
INTRADAY_MOVE_WINDOWS = (5, 10, 15)  # bars (=25/50/75 min at 5m)
INTRADAY_CONSISTENCY_WINDOW = 10  # bars (=50 min at 5m)
INTRADAY_VOL_ZSCORE_WINDOW = 24  # bars (=2h at 5m, matches MAX_BURST_DURATION_MINUTES)
BURST_ATR_MULT_IN_WINDOW = 2.5
BURST_ATR_MULT_OUT_WINDOW = 4.0
BURST_MIN_CONSISTENCY = 0.65

# Among bursts that already clear the thresholds above, this further splits
# "LONG" (a big hike, real company-loss risk, within this same <=2h window --
# NOT a multi-day thing) from an ordinary/weak burst. Picked as roughly the
# 80th percentile of historically observed burst magnitudes (~5.4x normal
# volatility) -- confirmed with the user 2026-08-03, adjust here if the
# alert rate feels wrong in practice.
BURST_LONG_MAGNITUDE_ATR = 5.4
INTRADAY_CONTINUATION_HORIZON_MIN = 30  # minutes ahead used as the training label (=6 bars at 5m)
INTRADAY_CONTINUATION_MOVE_ATR = 0.5  # min further move (in ATR) to count as "continued"

# Event-driven, not a wall-clock timer: check this often whether a new bar has
# actually closed on the broker's own clock (via MT5Connection.server_now()),
# and only pay for the full reload+feature-recompute when one genuinely has.
# Deliberately NOT tied to bar length/wall-clock boundaries -- this machine's
# clock isn't guaranteed to line up with the broker's (confirmed off by ~3h),
# so waiting for an assumed wall-clock boundary is exactly the wrong idea.
INTRADAY_CHECK_INTERVAL_SECONDS = 10

# This kind of session-open move plays out within a single day, at most ~2 hours.
# If a burst is still flagged as ongoing past this, it's no longer "the same trend"
# by definition -- auto-close the episode so a fresh START is required afterward.
MAX_BURST_DURATION_MINUTES = 120
