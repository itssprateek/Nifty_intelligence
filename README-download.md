# Local NIFTY 50 historical downloader

Requires Python 3.10+ on macOS/Linux. This package merges into the folder structure already created for `~/Projects/nifty-intelligence`. Only the downloader, requirements, this guide and a test file are supplied; no market data, API credentials or virtual environment is bundled.

## Setup
Save the ZIP to Downloads, then run:

```bash
cd ~/Projects/nifty-intelligence
unzip -n ~/Downloads/nifty-downloader.zip -d .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-download.txt
python src/ingestion/historical/download_nifty.py
```

If the browser extracted the ZIP automatically, copy its contents into `~/Projects/nifty-intelligence` first and omit the unzip command. The archive has no wrapper folder. `unzip -n` preserves any pre-existing files with matching names.

Default: daily plus 5-minute data. All intervals:

```bash
python src/ingestion/historical/download_nifty.py --intervals daily 1m 5m 15m 1h
```

Single-interval retry:

```bash
python src/ingestion/historical/download_nifty.py --intervals 5m
```

Validation tests:

```bash
python -m unittest discover -s tests -p 'test_download_nifty.py' -v
```

## Data and behavior

| Folder | Yahoo interval | Requested history |
|---|---|---|
| daily | 1d | 5 years |
| 1m | 1m | 5 days |
| 5m | 5m | 1 month |
| 15m | 15m | 1 month |
| 1h | 1h | 1 month |

Provider coverage and intraday limits can change; these are conservative requests, not guaranteed completeness. This downloads the NIFTY 50 **index** (^NSEI), not its constituent stocks. Yahoo/yfinance is an unofficial research source, not an exchange-authorized live execution feed. See Yahoo terms for data use; yfinance describes personal research use.

Each selected `data/raw/historical/<interval>/` folder contains:
- `nifty50.parquet`: canonical accumulated candles, timezone-aware Asia/Kolkata index.
- `nifty50.csv`: readable export; daily rows have a `date` column compatible with the prototype.
- `metadata.json`: source, symbol, request window, library version, fetched time, coverage and row counts.

`logs/download_nifty.log` records run status. Data is local; nothing is sent to the hosted research Site. No live socket or agent runs.

All current-day rows are excluded, even after market close; they become eligible the next day. No price adjustment, repair, gap filling or synthetic fallback is performed. Invalid OHLC rows are preserved with rejection reasons under data/raw/quarantine/<interval>/ and excluded from the saved candles. Metadata marks the run partial and reports rejected counts. An all-invalid response or duplicate provider timestamps fails that interval. Zero or absent index volume is allowed and is unsuitable as a volume trading signal. Exchange calendars, missing sessions and completeness are not checked. A benchmark is not downloaded; alpha/beta need a separately aligned series.

Reruns fetch the request window again, refresh overlapping timestamps with provider values, and preserve older stored history. They do not fetch older revisions outside that window or recover minute bars that aged out of Yahoo retention. Run frequently enough if you want to accumulate intraday history.

Each file replacement is atomic; the three files are not a transactional group. A machine crash during replacement may leave CSV/metadata behind Parquet; rerun to regenerate them. One process per project is enforced by a macOS/Linux file lock. Successful intervals stay saved if another fails; the program exits with status 1 for any failure. Invalid downloads do not replace existing data.

## Verification boundary
Offline tests cover OHLC validation, timezone normalization, incomplete-session exclusion, deduplication, provider revisions, and file round trips. Internet retrieval must be confirmed by running on your Mac; no claim is made that your local files already contain market data.

## Provider documentation
https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
https://github.com/ranaroussi/yfinance

## Fix for missing/invalid OHLC errors
The updated downloader retains `keepna=True` for auditability, reports invalid rows, and saves them separately. No interpolation, zero-filling or synthetic prices are used. Inspect quarantine files before research: retained data can have gaps. The deprecated `raise_errors` argument is removed in favor of modern yfinance config when available.

Your log may show Python 3.9 with LibreSSL. This warning did not cause the OHLC exception; the supported setup remains Python 3.10+ using an OpenSSL-backed Python installation. Do not disable certificate verification.
