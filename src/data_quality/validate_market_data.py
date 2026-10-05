"""Audit existing daily/5m NIFTY files locally. No downloads or price filling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Bounded calendar snapshot for the user's current intraday research window.
# Older/newer dates are deliberately NOT certified by this snapshot.
START, END = '2026-09-07', '2026-10-05'
HOLIDAYS = {'2026-09-14': 'Ganesh Chaturthi', '2026-10-02': 'Mahatma Gandhi Jayanti'}
SOURCES = ['https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf',
           'https://www.nseindia.com/static/market-data/market-timings']
PRICE = ['open', 'high', 'low', 'close']
TZ = 'Asia/Kolkata'


def grid(day, interval):
    if interval == 'daily':
        return pd.DatetimeIndex([pd.Timestamp(day, tz=TZ)])
    return pd.date_range(day+' 09:15', day+' 15:25', freq='5min', tz=TZ)


def audit(raw, interval):
    if raw.empty:
        raise ValueError('Empty input dataset.')
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError('Expected timezone-aware index.')
    if raw.index.hasnans:
        raise ValueError('Missing timestamps must be resolved first.')
    if not set(PRICE).issubset(raw.columns):
        raise ValueError('Missing OHLC columns.')
    df = raw.copy().sort_index()
    df.index = df.index.tz_convert(TZ)
    df.index.name = 'timestamp'
    prices = df[PRICE].apply(pd.to_numeric, errors='coerce')
    bad = (~np.isfinite(prices.to_numpy()).all(axis=1) | prices.le(0).any(axis=1)
           | (prices.high < prices.max(axis=1)) | (prices.low > prices.min(axis=1)))
    duplicates = df.index.duplicated(keep=False)
    dates = df.index.strftime('%Y-%m-%d')
    df['session_date'] = dates
    df['invalid_ohlc'] = np.asarray(bad)
    df['duplicate_timestamp'] = duplicates
    df['flat_candle'] = prices.nunique(axis=1).eq(1).to_numpy() & ~np.asarray(bad)
    # This is a review flag, not proof of a stale quote. Do not automatically delete it.
    adjacent = pd.Series(df.index).diff().eq(pd.Timedelta(minutes=5)).to_numpy()
    same_prices = prices.eq(prices.shift()).all(axis=1).to_numpy()
    repeated = df.flat_candle.to_numpy() & df.flat_candle.shift(fill_value=False).to_numpy() & same_prices & adjacent
    repeated_pair = repeated | np.r_[repeated[1:], False]
    if interval == 'daily':
        repeated_pair[:] = False
    df['repeated_flat_review'] = repeated_pair
    df['calendar_status'] = 'unverified'
    df['on_expected_grid'] = False
    df['session_complete'] = False
    records, gaps = [], []
    for dt in pd.date_range(START, END, freq='D'):
        day = dt.strftime('%Y-%m-%d')
        mask = dates == day
        part = df.loc[mask]
        closed = day in HOLIDAYS or dt.weekday() >= 5
        df.loc[mask, 'calendar_status'] = 'closed' if closed else 'scheduled'
        if closed:
            records.append({'date': day, 'status': 'unexpected_closed_day_rows' if len(part) else 'closed',
                            'expected': 0, 'observed_valid': 0, 'missing': 0,
                            'unexpected_rows': len(part), 'flat_review_rows': int(part.repeated_flat_review.sum())})
            continue
        expected = grid(day, interval)
        on_grid = part.index.isin(expected)
        valid = ~part.invalid_ohlc & ~part.duplicate_timestamp
        actual = part.index[on_grid & valid.to_numpy()]
        missing = expected.difference(actual)
        unexpected = int((~on_grid).sum())
        complete = not len(missing) and not unexpected and bool(valid.all())
        df.loc[mask, 'on_expected_grid'] = on_grid
        df.loc[mask, 'session_complete'] = complete
        status = 'complete' if complete else ('absent' if part.empty else 'incomplete')
        records.append({'date': day, 'status': status, 'expected': len(expected),
                        'observed_valid': len(actual), 'missing': len(missing),
                        'unexpected_rows': unexpected, 'flat_review_rows': int(part.repeated_flat_review.sum())})
        for ts in missing:
            gaps.append({'date': day, 'timestamp': ts.isoformat(), 'reason': 'missing_or_invalid_candle'})
    # Full-session study eligibility is conservative and limited to this calendar window.
    reviewed_dates = set(df.loc[df.repeated_flat_review, 'session_date'])
    df['session_review_required'] = df.session_date.isin(reviewed_dates)
    df['eligible_full_session'] = (df.session_complete & df.calendar_status.eq('scheduled')
                                   & ~df.session_review_required & ~df.invalid_ohlc & ~df.duplicate_timestamp)
    # Each block contains consecutive valid intraday bars only; never bridge sessions/gaps.
    if interval == '5m':
        usable = (~df.invalid_ohlc & ~df.duplicate_timestamp & df.on_expected_grid
                  & df.calendar_status.eq('scheduled') & ~df.repeated_flat_review)
        breaks = (~adjacent | df.session_date.ne(df.session_date.shift()).to_numpy()
                  | ~usable.to_numpy() | ~usable.shift(fill_value=False).to_numpy())
        df['continuous_block_id'] = pd.Series(np.cumsum(breaks), index=df.index).where(usable).astype('Int64')
    else:
        df['continuous_block_id'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    sessions = pd.DataFrame(records)
    gap_table = pd.DataFrame(gaps, columns=['date', 'timestamp', 'reason'])
    summary = {
        'interval': interval, 'input_rows': len(df), 'calendar_start': START, 'calendar_end': END,
        'calendar_scope': 'Weekdays except listed NSE holidays in the bounded window; no automatic future calendar updates.',
        'sources': SOURCES, 'invalid_ohlc_rows': int(df.invalid_ohlc.sum()),
        'duplicate_rows': int(df.duplicate_timestamp.sum()),
        'calendar_unverified_rows': int(df.calendar_status.eq('unverified').sum()),
        'complete_sessions': int(sessions.status.eq('complete').sum()),
        'incomplete_or_absent_sessions': int(sessions.status.isin(['incomplete', 'absent']).sum()),
        'missing_expected_candles': len(gaps), 'repeated_flat_review_rows': int(df.repeated_flat_review.sum()),
        'eligible_full_sessions': int(df.loc[df.eligible_full_session, 'session_date'].nunique()),
        'limitations': ['No independent price verification; complete does not mean economically accurate.',
                       'Older daily coverage is unverified; no automatic holiday assumptions outside the snapshot.',
                       'Flat-price flags require review and can be genuine; no prices are filled or deleted.',
                       'Index volume may be zero/absent and is not treated as proof of invalid prices.'],
    }
    return df, sessions, gap_table, summary


def atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        writer(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    failures = 0
    for interval in ['daily', '5m']:
        source = root / 'data/raw/historical' / interval / 'nifty50.parquet'
        if not source.exists():
            print(f'{interval}: input missing: {source}'); failures += 1; continue
        try:
            data, sessions, gaps, summary = audit(pd.read_parquet(source), interval)
            out = root / 'data/processed/candles' / interval
            report = root / 'reports/data_quality' / interval
            atomic(out/'nifty50_flagged.parquet', lambda f: data.to_parquet(f, engine='pyarrow'))
            atomic(out/'nifty50_flagged.csv', lambda f: data.to_csv(f))
            atomic(report/'sessions.csv', lambda f: sessions.to_csv(f, index=False))
            atomic(report/'missing_candles.csv', lambda f: gaps.to_csv(f, index=False))
            flags = data.loc[data.invalid_ohlc | data.duplicate_timestamp | data.repeated_flat_review]
            atomic(report/'flagged_candles.csv', lambda f: flags.to_csv(f))
            atomic(report/'summary.json', lambda f: f.write_text(json.dumps(summary, indent=2)+'\n'))
            print(f'\n=== {interval}: QUALITY REPORT ===')
            print(f"Rows: {len(data):,} | Calendar audit: {START} to {END}")
            print(f"Complete sessions: {summary['complete_sessions']} | Incomplete/absent: {summary['incomplete_or_absent_sessions']}")
            print(f"Missing candles: {len(gaps)} | Repeated-flat review rows: {summary['repeated_flat_review_rows']}")
            print(f"Eligible full sessions after review flags: {summary['eligible_full_sessions']}")
            print(f"Rows outside calendar coverage: {summary['calendar_unverified_rows']}")
            issues = sessions.loc[~sessions.status.isin(['complete', 'closed'])]
            if not issues.empty: print(issues.to_string(index=False))
            print(f'Reports: {report}')
        except Exception as exc:
            failures += 1
            print(f'{interval}: FAILED: {exc}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
