"""One-time (but safe-to-rerun-forever) seed of candles_m5 from a bulk MT5
CSV export, so the burst-watch subsystem starts with real history instead of
an empty table. Called automatically at the top of intraday_loop.py and
train_intraday_model.py -- upserts are idempotent (ON CONFLICT DO UPDATE), so
re-running costs nothing once the file's rows are already in the DB.

Two export formats are recognized, auto-detected from the header line:

Tab-separated MT5 terminal export (original ~4-month seed, xau5M.csv):
    <DATE>      <TIME>    <OPEN>  <HIGH>  <LOW>   <CLOSE>  <TICKVOL> <VOL> <SPREAD>
    2026.03.31  09:15:00  4560.08 4561.01 4554.05 4555.02  1701      0     3

Semicolon-separated bulk export (XAU_5m_data.csv, same feed back to 2004 --
confirmed identical OHLCV on the dates the two files overlap):
    Date;Open;High;Low;Close;Volume
    2026.03.31 09:15;4560.08;4561.01;4554.05;4555.02;1701

The semicolon format has no seconds and no separate spread column -- spread
defaults to 0 for those rows (not present in the export, and not used by any
feature/threshold downstream).

DATE/TIME are broker server time -- converted to epoch seconds the same way
the live MT5 API's own timestamps work (server-time clock read as if it were
UTC, no local-timezone conversion), so they line up exactly with candles
fetched live via mt5_client / compared in session_windows.py.
"""

import calendar
import csv
from datetime import datetime, timezone

from . import config, db


def _parse_tab_rows(reader):
    rows = []
    for line in reader:
        if not line or len(line) < 7:
            continue
        date_s, time_s, open_s, high_s, low_s, close_s, tickvol_s = line[:7]
        spread_s = line[8] if len(line) > 8 else "0"
        dt = datetime.strptime(f"{date_s} {time_s}", "%Y.%m.%d %H:%M:%S")
        ts = calendar.timegm(dt.timetuple())
        rows.append((
            ts, float(open_s), float(high_s), float(low_s), float(close_s),
            int(tickvol_s), int(spread_s),
        ))
    return rows


def _parse_semicolon_rows(reader):
    rows = []
    for line in reader:
        if not line or len(line) < 6:
            continue
        datetime_s, open_s, high_s, low_s, close_s, volume_s = line[:6]
        dt = datetime.strptime(datetime_s, "%Y.%m.%d %H:%M")
        ts = calendar.timegm(dt.timetuple())
        rows.append((
            ts, float(open_s), float(high_s), float(low_s), float(close_s),
            int(volume_s), 0,
        ))
    return rows


def _parse_rows(path):
    with open(path, newline="") as f:
        first_line = f.readline()
        delimiter = ";" if ";" in first_line else "\t"
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)  # header
        if delimiter == ";":
            return _parse_semicolon_rows(reader)
        return _parse_tab_rows(reader)


def seed_candles_m5(conn) -> int:
    """Loads config.INTRADAY_CSV_SEED_PATH into candles_m5 if present. Returns
    the number of rows read (0 if the file doesn't exist -- not an error, just
    means no seed was provided)."""
    path = config.INTRADAY_CSV_SEED_PATH
    if not path.exists():
        return 0
    rows = _parse_rows(path)
    if rows:
        db.upsert_m5_candles(conn, rows)
    return len(rows)


def _fmt_price(x) -> str:
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def _last_csv_epoch(path):
    """Epoch (seconds, broker-server-time-as-UTC) of the last data row already
    in the semicolon CSV, or None if the file is missing/empty/header-only.
    Reads only the trailing bytes -- cheap regardless of how large the file
    has grown -- so this can run every tick without scanning the whole
    history."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(-min(size, 1024), 2)
        tail = f.read()
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None
    last_line = lines[-1].decode()
    if last_line.startswith("Date;") or last_line.startswith("Date "):
        return None
    date_s = last_line.split(";")[0]
    dt = datetime.strptime(date_s, "%Y.%m.%d %H:%M")
    return calendar.timegm(dt.timetuple())


def append_m5_rows(rows) -> int:
    """rows: iterable of (epoch, open, high, low, close, volume), ascending by
    epoch. Appends whichever are newer than the CSV's own last row to
    config.INTRADAY_CSV_SEED_PATH, in the same semicolon format as the bulk
    export (Date;Open;High;Low;Close;Volume) -- so the raw historical file
    keeps growing in lockstep with candles_m5, never pruned, same as the DB
    table.

    Dedup reads the file's actual last line rather than trusting an in-memory
    or DB cursor, so calling this again after a crash mid-write just resumes
    from wherever the file really left off instead of double-appending."""
    path = config.INTRADAY_CSV_SEED_PATH
    last_epoch = _last_csv_epoch(path)
    new_rows = [r for r in rows if last_epoch is None or r[0] > last_epoch]
    if not new_rows:
        return 0

    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        if is_new_file:
            f.write("Date;Open;High;Low;Close;Volume\n")
        for ts, open_, high, low, close, volume in new_rows:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            f.write(
                f"{dt:%Y.%m.%d %H:%M};{_fmt_price(open_)};{_fmt_price(high)};"
                f"{_fmt_price(low)};{_fmt_price(close)};{int(volume)}\n"
            )
    return len(new_rows)
