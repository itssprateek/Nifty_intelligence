"""Build offline research features from quality-flagged daily/1m/5m candles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_quality.validate_market_data import atomic

FEATURES = ['return_1_pct', 'candle_range_pct', 'sma_20', 'distance_sma20_pct',
            'rsi_14', 'volatility_20_pct']


def wilder_rsi(close, period=14):
    values = np.asarray(close, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) <= period:
        return result
    delta = np.diff(values)
    gain = np.maximum(delta[:period], 0).mean()
    loss = np.maximum(-delta[:period], 0).mean()
    def value():
        if gain == 0 and loss == 0:
            return 50.
        return 100. if loss == 0 else 100. - 100. / (1. + gain / loss)
    result[period] = value()
    for i in range(period + 1, len(values)):
        gain = (gain * (period - 1) + max(delta[i - 1], 0)) / period
        loss = (loss * (period - 1) + max(-delta[i - 1], 0)) / period
        result[i] = value()
    return result


def build(frame, interval, sessions):
    if interval not in ['daily', '1m', '5m']:
        raise ValueError('Unsupported interval')
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('Expected nonempty timezone-aware candle data.')
    # Duplicates are ambiguous for time-indexed feature assignment. Stop rather than collapse.
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError('Resolve duplicate/missing timestamps before building features.')
    required = ['open', 'high', 'low', 'close', 'invalid_ohlc', 'duplicate_timestamp',
                'calendar_status', 'on_expected_grid', 'repeated_flat_review', 'continuous_block_id']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Run the updated validator first. Missing fields: {sorted(missing)}')
    df = frame.sort_index().copy()
    df.index = df.index.tz_convert('Asia/Kolkata')
    df.index.name = 'timestamp'
    prices = df[['open','high','low','close']].apply(pd.to_numeric, errors='coerce')
    finite = pd.Series(np.isfinite(prices.to_numpy()).all(axis=1), index=df.index)
    base = (finite & prices.gt(0).all(axis=1) & ~df.invalid_ohlc & ~df.duplicate_timestamp
            & (prices.high >= prices.max(axis=1)) & (prices.low <= prices.min(axis=1))
            & ~df.repeated_flat_review)
    if interval != 'daily':
        usable = base & df.calendar_status.eq('scheduled') & df.on_expected_grid & df.continuous_block_id.notna()
        adjacent = pd.Series(df.index, index=df.index).diff().eq(pd.Timedelta(minutes={'1m':1,'5m':5}[interval]))
        dates = pd.Series(df.index.date, index=df.index)
        breaks = (~adjacent | dates.ne(dates.shift()) | ~usable | ~usable.shift(fill_value=False)
                  | df.continuous_block_id.ne(df.continuous_block_id.shift()).fillna(True))
    else:
        # Older candles are explicitly exploratory: no claim of calendar continuity.
        usable = base & ~df.calendar_status.eq('closed')
        midnight = (df.index.hour == 0) & (df.index.minute == 0) & (df.index.second == 0)
        usable &= midnight
        missing_days = pd.to_datetime(sessions.loc[sessions['missing'].gt(0), 'date'])
        missed_between = np.zeros(len(df), dtype=bool)
        dates = df.index.tz_localize(None).normalize()
        for i in range(1, len(df)):
            missed_between[i] = bool(((missing_days > dates[i-1]) & (missing_days <= dates[i])).any())
        breaks = (~usable | ~usable.shift(fill_value=False) | missed_between)
    groups = breaks.astype(int).cumsum().where(usable).astype('Int64')
    df['feature_block_id'] = groups
    df['feature_input_usable'] = usable
    for name in FEATURES:
        df[name] = np.nan
    df['bars_in_feature_block'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df['feature_calendar_verified'] = False
    for _, block in df.loc[usable].groupby('feature_block_id', sort=False):
        idx = block.index
        p = prices.loc[idx]
        c = p.close
        returns = c.pct_change(fill_method=None) * 100.
        sma = c.rolling(20, min_periods=20).mean()
        df.loc[idx, 'return_1_pct'] = returns
        df.loc[idx, 'candle_range_pct'] = (p.high - p.low) / p.open * 100.
        df.loc[idx, 'sma_20'] = sma
        df.loc[idx, 'distance_sma20_pct'] = (c / sma - 1.) * 100.
        df.loc[idx, 'rsi_14'] = wilder_rsi(c)
        # Standard deviation of 20 percentage returns; no annualization.
        df.loc[idx, 'volatility_20_pct'] = returns.rolling(20, min_periods=20).std(ddof=1)
        df.loc[idx, 'bars_in_feature_block'] = np.arange(1, len(idx)+1)
        # RSI recursively depends on all earlier prices in its block.
        verified = block.calendar_status.eq('scheduled') & block.on_expected_grid
        df.loc[idx, 'feature_calendar_verified'] = verified.cummin().to_numpy()
    df['feature_ready'] = df[FEATURES].notna().all(axis=1) & usable
    df['feature_status'] = np.select(
        [~usable, usable & ~df.feature_ready, df.feature_ready & ~df.feature_calendar_verified],
        ['excluded_input', 'warmup', 'ready_calendar_unverified'], default='ready')
    summary = {
        'interval': interval, 'rows': len(df), 'feature_ready_rows': int(df.feature_ready.sum()),
        'excluded_input_rows': int((~usable).sum()),
        'warmup_rows': int(df.feature_status.eq('warmup').sum()),
        'ready_calendar_unverified_rows': int(df.feature_status.eq('ready_calendar_unverified').sum()),
        'feature_blocks': int(groups.nunique()),
        'definitions': {
            'return_1_pct': '100 * (close / previous close - 1), within the feature block',
            'candle_range_pct': '100 * (high - low) / open',
            'sma_20': 'Arithmetic mean of 20 closing prices, including the current completed candle',
            'distance_sma20_pct': '100 * (close / SMA20 - 1)',
            'rsi_14': 'Wilder RSI: mean of first 14 gains/losses, then recursive smoothing; flat=50',
            'volatility_20_pct': 'Sample std (ddof=1) of 20 percentage returns; NOT annualized',
        },
        'policies': [
            'No synthetic prices, interpolation, forward-fill or backward-fill.',
            'Intraday blocks reset each session, at missing bars, and at review/excluded rows.',
            'Daily calculations use observed valid rows; older calendar continuity is unverified.',
            'All features become ready after at least 21 usable closes within one block.',
            'Quality flags are retrospective: repeated-flat detection may use the following bar.',
            'These files are offline research inputs, not point-in-time trading/backtest certification.',
            'No alpha/beta: no aligned independent benchmark is connected.',
        ],
    }
    return df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--intervals', nargs='+', choices=['daily','1m','5m'], default=['daily','1m','5m'])
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    failures = 0
    for interval in dict.fromkeys(args.intervals):
        try:
            source = root / 'data/processed/candles' / interval / 'nifty50_flagged.parquet'
            sessions_file = root / 'reports/data_quality' / interval / 'sessions.csv'
            frame = pd.read_parquet(source)
            output, summary = build(frame, interval, pd.read_csv(sessions_file))
            summary['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            summary['sessions_sha256'] = hashlib.sha256(sessions_file.read_bytes()).hexdigest()
            out = root / 'data/processed/features' / interval
            report = root / 'reports/features' / interval
            atomic(out/'nifty50_features.parquet', lambda f: output.to_parquet(f, engine='pyarrow'))
            atomic(out/'nifty50_features.csv', lambda f: output.to_csv(f))
            atomic(report/'summary.json', lambda f: f.write_text(json.dumps(summary, indent=2)+'\n'))
            print(f'\n=== {interval}: FEATURES ===')
            print(f"Rows: {len(output):,} | Ready: {summary['feature_ready_rows']:,} | Warm-up: {summary['warmup_rows']:,}")
            print(f"Excluded inputs: {summary['excluded_input_rows']} | Calendar-unverified ready rows: {summary['ready_calendar_unverified_rows']}")
            print(output[['close','rsi_14','sma_20','volatility_20_pct','feature_status']].tail(5).to_string())
            print(f'Saved: {out}')
        except Exception as exc:
            failures += 1
            print(f'{interval}: FAILED: {exc}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
