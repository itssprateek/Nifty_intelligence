# Offline local feature engine

## Run
From the project root in your activated environment:

```bash
python -m unittest discover -s tests -p 'test_features.py' -v
python src/data_quality/validate_market_data.py
python src/features/build_features.py
```

Both scripts default to daily, 1m, and 5m. Select a subset with `--intervals daily 1m` if needed. The existing pandas, numpy and pyarrow dependencies are sufficient. No internet requests occur in either script.

## New 1-minute validation
Normal session grid: 09:15 through 15:29 IST, 375 one-minute candle-start timestamps. The 5-minute grid remains 09:15 through 15:25, 75 bars. The bounded calendar snapshot is 2026-09-07 through 2026-10-05, excluding weekends and Sept 14 / Oct 2 holidays. Each interval's audit starts at the later of its first saved date and the snapshot start. For the current 1m download that is September 29, so earlier dates are not falsely reported as missing.

The audit still ends at October 5 to detect absent trailing sessions, including the missing daily record on that date. Rows outside the snapshot are retained and marked unverified. This is deliberately not an automatically updating production exchange calendar. Dates after the snapshot are not certified or used for intraday features. Boundary-day gaps may reflect a partial download; the report flags missing timestamps without assuming their cause.

## Features and units
- `return_1_pct`: percentage close-to-close change within the feature block.
- `candle_range_pct`: 100 × (high − low) / open.
- `sma_20`: mean of 20 closes, expressed in index points.
- `distance_sma20_pct`: percentage distance from SMA20.
- `rsi_14`: Wilder smoothing, initialized from 14 changes (15 closes). Flat=50, only gains=100, only losses=0.
- `volatility_20_pct`: sample standard deviation of 20 percentage returns. Per-bar volatility, not annualized. Do not directly compare daily and minute values as equivalent horizons.

All use the current completed candle and earlier usable candles. `feature_ready` requires all metrics to be available, therefore at least 21 consecutive usable closes. Insufficient history remains NaN; no prices/features are filled.

## Data-quality policies
Intraday calculations restart at session boundaries, missing timestamps, invalid/repeated-flat review rows, or existing quality-block changes. Unflagged portions of sessions remain available: `eligible_full_session` does not discard the whole session from local intraday feature calculations.

Daily metrics use observed valid daily candles, splitting at known calendar gaps or invalid rows. Older calendar coverage is unverified. `feature_calendar_verified` is false if any input in the recursive RSI block has unverified calendar status. Such ready rows use `feature_status=ready_calendar_unverified`; they are exploratory and not certified as a complete sequence of trading sessions.

This update does not adjudicate or remove the provider's suspicious flat candles. The validator's retrospective repeated-flat flag can inspect a following candle; consequently the entire pipeline is NOT a point-in-time backtest/live-trading pipeline. Feature mathematics is past/current-only given fixed input quality flags. A future backtester must make quality decisions with event-time availability and retain source revisions.

No benchmark is connected, so alpha and beta are not fabricated. No models, signals, orders or live connections are added.

## Outputs
- `data/processed/candles/<interval>/nifty50_flagged.parquet` and CSV: refreshed quality annotations.
- `reports/data_quality/<interval>/`: session/missing/flagged tables and summary.
- `data/processed/features/<interval>/nifty50_features.parquet` and CSV: all original annotated rows plus metrics and readiness fields.
- `reports/features/<interval>/summary.json`: counts, feature definitions, limitations and SHA-256 hashes of the input Parquet/session report.

Raw files are never written. Output files are atomically replaced individually; output sets are not multi-file transactions. On an interrupted run, rerun validation followed by features.

## Commit after reviewing results
```bash
git add src/data_quality/validate_market_data.py src/features/build_features.py tests/test_features.py docs/FEATURES.md
git commit -m "Add 1m quality validation and offline research features"
git push -u origin feature/local-feature-engine
```
