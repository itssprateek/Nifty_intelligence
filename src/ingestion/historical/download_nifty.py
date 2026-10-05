"""Local NIFTY 50 historical ingestion. Python 3.10+; run --help for usage."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SYMBOL = '^NSEI'
IST = ZoneInfo('Asia/Kolkata')
# Conservative request windows, not guarantees of provider coverage.
JOBS = {'daily': ('1d', '5y'), '1m': ('1m', '5d'),
        '5m': ('5m', '1mo'), '15m': ('15m', '1mo'), '1h': ('1h', '1mo')}
PRICE = ['open', 'high', 'low', 'close']
COLUMNS = PRICE + ['volume']
ROOT = Path(__file__).resolve().parents[3]


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError('No completed-session candles available.')
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError('Expected timezone-aware timestamps.')
    if df.index.hasnans or df.index.has_duplicates:
        raise ValueError('Missing or duplicate timestamps; data was not saved.')
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    p = df[PRICE].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not (p > 0).all():
        raise ValueError('OHLC contains missing, infinite or nonpositive prices.')
    if ((df.high < df[PRICE].max(axis=1)) | (df.low > df[PRICE].min(axis=1))).any():
        raise ValueError('Invalid high/low relationships.')
    volume = df.volume.dropna().to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('Invalid volume. Missing/zero index volume is allowed.')


def normalize(raw: pd.DataFrame, today=None, quarantine: Path | None = None) -> pd.DataFrame:
    if raw.empty:
        raise ValueError('Yahoo returned no data; no synthetic fallback is used.')
    df = raw.copy()
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError('Provider timestamps lack a timezone; refusing to guess.')
    df.index = df.index.tz_convert(IST)
    df.index.name = 'timestamp'
    if df.index.hasnans or df.index.has_duplicates:
        raise ValueError('Missing or duplicate timestamps; refusing ambiguous data.')
    # Keep previous sessions only, even after the close, for conservative ingestion.
    df = df.loc[df.index.date < (today or datetime.now(IST).date())]
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    original = df.copy()
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')
    reasons = pd.Series('', index=df.index)
    checks = {
        'missing_or_nonnumeric_ohlc': df[PRICE].isna().any(axis=1),
        'infinite_ohlc': pd.Series(np.isinf(df[PRICE].to_numpy()).any(axis=1), index=df.index),
        'nonpositive_ohlc': df[PRICE].le(0).any(axis=1),
        'inconsistent_high_low': (df.high < df[PRICE].max(axis=1)) | (df.low > df[PRICE].min(axis=1)),
        'invalid_volume': (df.volume < 0) | np.isinf(df.volume) | (original.volume.notna() & df.volume.isna()),
    }
    for reason, mask in checks.items():
        reasons.loc[mask] += reason + ';'
    bad = reasons.ne('')
    rejected = int(bad.sum())
    report = None
    if rejected:
        logging.warning('Rejected %s/%s completed-session rows; reasons=%s; first timestamps=%s',
                        rejected, len(df), {k: int(v.sum()) for k, v in checks.items() if v.any()},
                        [str(x) for x in df.index[bad][:5]])
        if quarantine is None:
            validate(df)  # Preserve strict validation for callers not explicitly quarantining.
            raise ValueError('Rejected rows require a quarantine destination.')
        quarantine.mkdir(parents=True, exist_ok=True)
        report = quarantine / ('rejected_' + datetime.now(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8] + '.csv')
        rejected_rows = original.loc[bad].copy()
        rejected_rows['rejection_reason'] = reasons.loc[bad]
        rejected_rows.to_csv(report, index=True)
        logging.warning('Rejected rows preserved at %s. No prices filled or fabricated.', report)
    clean = df.loc[~bad, COLUMNS].sort_index()
    validate(clean)  # All-invalid responses still fail without replacing existing data.
    clean.attrs['quality'] = {'completed_rows_received': len(df), 'rejected_rows': rejected,
                              'status': 'partial' if rejected else 'passed',
                              'quarantine_file': str(report) if report else None}
    return clean


def merge_history(old: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    validate(fresh)
    if old is None:
        return fresh.sort_index()
    validate(old)
    old = old.copy()
    old.index = old.index.tz_convert(IST)
    # Refreshed provider candles replace prior observations at matching timestamps.
    merged = pd.concat([old[COLUMNS], fresh[COLUMNS]])
    merged = merged.loc[~merged.index.duplicated(keep='last')].sort_index()
    validate(merged)
    return merged


def fetch(interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    config = getattr(yf, 'config', None)
    if config is not None and hasattr(config, 'debug'):
        yf.config.debug.hide_exceptions = False
    for attempt in range(1, 4):
        try:
            raw = yf.Ticker(SYMBOL).history(
                interval=interval, period=period, auto_adjust=False,
                back_adjust=False, repair=False, actions=False, keepna=True,
                prepost=False, timeout=30,
            )
            if raw.empty:
                raise ValueError('Empty provider response')
            return raw
        except Exception:
            if attempt == 3:
                raise
            logging.warning('Request attempt %s failed; retrying in %ss.', attempt, 2**attempt)
            time.sleep(2**attempt)
    raise RuntimeError('Unreachable')


def save(root: Path, name: str, fresh: pd.DataFrame, interval: str, period: str):
    import yfinance as yf
    folder = root / 'data' / 'raw' / 'historical' / name
    folder.mkdir(parents=True, exist_ok=True)
    parquet = folder / 'nifty50.parquet'
    old = pd.read_parquet(parquet) if parquet.exists() else None
    merged = merge_history(old, fresh)
    export = merged.reset_index()
    if name == 'daily':
        export.insert(0, 'date', export.pop('timestamp').dt.strftime('%Y-%m-%d'))
    metadata = {
        'symbol': SYMBOL, 'source': 'Yahoo Finance via yfinance',
        'yfinance_version': yf.__version__, 'timezone': 'Asia/Kolkata',
        'interval': interval, 'requested_period_this_run': period,
        'downloaded_at_utc': datetime.now(ZoneInfo('UTC')).isoformat(),
        'total_rows': len(merged), 'fetched_completed_rows': len(fresh),
        'net_new_rows': len(merged) - (0 if old is None else len(old)),
        'first_timestamp': merged.index.min().isoformat(),
        'last_timestamp': merged.index.max().isoformat(),
        'auto_adjust': False, 'repair': False, 'current_session_excluded': True,
        'quality_this_run': fresh.attrs.get('quality', {}),
        'notes': ['Index volume may be zero or absent; do not infer volume signals.',
                  'Missing exchange sessions and provider omissions are not validated.',
                  'Parquet is canonical; CSV and metadata are regenerable exports.'],
    }
    # Stage all outputs before changing any existing files. Each replacement is atomic.
    token = uuid4().hex
    targets = [parquet, folder / 'nifty50.csv', folder / 'metadata.json']
    temps = [p.with_name(f'.{p.name}.{token}.tmp') for p in targets]
    try:
        merged.to_parquet(temps[0], engine='pyarrow', index=True)
        export.to_csv(temps[1], index=False)
        temps[2].write_text(json.dumps(metadata, indent=2) + '\n')
        for temp, target in zip(temps, targets):
            temp.replace(target)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)
    logging.info('%s: %s rows, %s → %s | %s', name, len(merged),
                 merged.index.min(), merged.index.max(), folder)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', nargs='+', choices=list(JOBS), default=['daily', '5m'])
    parser.add_argument('--root', type=Path, default=ROOT, help='Default: project containing this script')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'logs' / 'download_nifty.log')])
    # Exclusive file lock is released by the OS even if the process crashes (macOS/Linux).
    import fcntl
    with (root / 'logs' / '.download_nifty.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.error('Another downloader is running for this project.')
            return 1
        failed = []
        for name in dict.fromkeys(args.intervals):
            interval, period = JOBS[name]
            try:
                logging.info('Requesting %s %s (%s); current session excluded.', SYMBOL, interval, period)
                fresh = normalize(fetch(interval, period), quarantine=root / 'data' / 'raw' / 'quarantine' / name)
                save(root, name, fresh, interval, period)
            except Exception as exc:
                failed.append(name)
                logging.error('%s failed: %s. Check log and rerun this interval.', name, exc)
        if failed:
            logging.error('Failed intervals: %s', ', '.join(failed))
            return 1
        logging.info('All requested intervals saved successfully.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
