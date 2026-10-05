import importlib.util
from datetime import date
from pathlib import Path
import unittest
import tempfile
import json
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'src/ingestion/historical/download_nifty.py'
spec = importlib.util.spec_from_file_location('downloader', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def frame():
    return pd.DataFrame({'Open': [100., 101., 102.], 'High': [103., 104., 105.],
                         'Low': [98., 99., 100.], 'Close': [102., 103., 104.],
                         'Volume': [0., 0., 0.]},
                        index=pd.date_range('2026-10-01 09:15', periods=3, freq='D', tz='Asia/Kolkata'))


class DownloaderTests(unittest.TestCase):
    def test_current_day_removed(self):
        out = m.normalize(frame(), date(2026, 10, 3))
        self.assertEqual(len(out), 2)
        self.assertEqual(str(out.index.tz), 'Asia/Kolkata')

    def test_utc_converted(self):
        raw = frame().tz_convert('UTC')
        self.assertEqual(m.normalize(raw, date(2026, 10, 4)).index[0].hour, 9)

    def test_missing_timezone_rejected(self):
        with self.assertRaisesRegex(ValueError, 'timezone'):
            m.normalize(frame().tz_localize(None))

    def test_invalid_prices_rejected(self):
        for value in [0, float('nan'), float('inf')]:
            raw = frame()
            raw.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                m.normalize(raw, date(2026, 10, 4))

    def test_invalid_high_rejected(self):
        raw = frame(); raw.iloc[0, 1] = 99
        with self.assertRaisesRegex(ValueError, 'high/low'):
            m.normalize(raw, date(2026, 10, 4))

    def test_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            m.normalize(pd.concat([frame(), frame()]), date(2026, 10, 4))

    def test_repeat_run_idempotent(self):
        fresh = m.normalize(frame(), date(2026, 10, 4))
        pd.testing.assert_frame_equal(m.merge_history(fresh, fresh), fresh, check_freq=False)

    def test_provider_revision_and_old_history_preserved(self):
        old = m.normalize(frame(), date(2026, 10, 4))
        fresh = old.iloc[-1:].copy(); fresh['close'] = 103.5
        merged = m.merge_history(old, fresh)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.iloc[-1].close, 103.5)
        self.assertEqual(merged.iloc[0].close, old.iloc[0].close)

    def test_file_round_trip_and_repeat_save(self):
        fresh = m.normalize(frame(), date(2026, 10, 4))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m.save(root, 'daily', fresh, '1d', '5y')
            meta = m.save(root, 'daily', fresh, '1d', '5y')
            folder = root / 'data/raw/historical/daily'
            restored = pd.read_parquet(folder / 'nifty50.parquet')
            restored.index = restored.index.tz_convert(m.IST)
            pd.testing.assert_frame_equal(restored, fresh, check_freq=False)
            export = pd.read_csv(folder / 'nifty50.csv')
            self.assertEqual(export.date.iloc[0], '2026-10-01')
            self.assertEqual(meta['net_new_rows'], 0)
            self.assertEqual(json.loads((folder / 'metadata.json').read_text())['total_rows'], 3)

    def test_quarantine_saves_bad_rows_and_valid_candles(self):
        raw = frame(); raw.iloc[0, 0] = float('nan')
        with tempfile.TemporaryDirectory() as temp:
            clean = m.normalize(raw, date(2026, 10, 4), Path(temp))
            self.assertEqual(len(clean), 2)
            self.assertEqual(clean.attrs['quality']['rejected_rows'], 1)
            report = pd.read_csv(next(Path(temp).glob('rejected_*.csv')))
            self.assertIn('missing_or_nonnumeric_ohlc', report.rejection_reason.iloc[0])
            self.assertEqual(len(report), 1)

    def test_all_invalid_still_fails_and_preserves_report(self):
        raw = frame(); raw['Close'] = 0
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'No completed'):
                m.normalize(raw, date(2026, 10, 4), Path(temp))
            self.assertEqual(len(list(Path(temp).glob('rejected_*.csv'))), 1)

    def test_empty_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no data'):
            m.normalize(pd.DataFrame())


if __name__ == '__main__':
    unittest.main()
