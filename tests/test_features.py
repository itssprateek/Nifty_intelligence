import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_quality.validate_market_data import audit, grid
from src.features.build_features import build, wilder_rsi, atomic


def prices(index):
    close = np.arange(len(index), dtype=float) + 100.
    return pd.DataFrame({'open': close-.2, 'high':close+.5, 'low':close-.5,
                         'close':close,'volume':0.}, index=index)


class ResearchTests(unittest.TestCase):
    def test_1m_grid_and_scope(self):
        idx = grid('2026-09-29','1m')
        self.assertEqual(len(idx),375)
        self.assertEqual(idx[-1].strftime('%H:%M'),'15:29')
        _,sessions,gaps,summary=audit(prices(idx),'1m')
        self.assertEqual(summary['calendar_start'],'2026-09-29')
        self.assertFalse(sessions.date.lt('2026-09-29').any())
        self.assertEqual(sessions.set_index('date').loc['2026-10-02','expected'],0)

    def test_1m_gap_and_flags(self):
        raw=prices(grid('2026-09-29','1m').delete(30))
        raw.iloc[5:7,:4]=200.
        flagged,sessions,_,_=audit(raw,'1m')
        self.assertEqual(flagged.repeated_flat_review.sum(),2)
        self.assertEqual(sessions.set_index('date').loc['2026-09-29','missing'],1)
        out,_=build(flagged,'1m',sessions)
        self.assertTrue(out.iloc[5:7].rsi_14.isna().all())
        self.assertTrue(np.isnan(out.iloc[7].return_1_pct))
        self.assertTrue(np.isnan(out.iloc[30].return_1_pct))
        self.assertFalse(out.iloc[49].feature_ready)
        self.assertTrue(out.iloc[50].feature_ready)

    def test_5m_regression(self):
        flagged,sessions,gaps,_=audit(prices(grid('2026-09-21','5m')[:-2]),'5m')
        row=sessions.set_index('date').loc['2026-09-21']
        self.assertEqual(row['missing'],2)
        self.assertEqual(list(gaps[gaps.date.eq('2026-09-21')].timestamp.str[11:16]),['15:20','15:25'])

    def test_known_wilder_value(self):
        c=[44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28]
        r=wilder_rsi(c)
        self.assertTrue(np.isnan(r[:14]).all())
        self.assertAlmostEqual(r[14],70.4641350211,places=8)

    def test_rsi_edge_cases(self):
        self.assertEqual(wilder_rsi(np.arange(30))[14],100.)
        self.assertEqual(wilder_rsi(-np.arange(30))[14],0.)
        self.assertEqual(wilder_rsi(np.ones(30))[14],50.)

    def test_sma_return_and_warmup(self):
        raw=prices(grid('2026-09-29','1m'))
        flagged,s,_,_=audit(raw,'1m');out,_=build(flagged,'1m',s)
        self.assertAlmostEqual(out.sma_20.iloc[19],109.5)
        self.assertAlmostEqual(out.return_1_pct.iloc[1],1.)
        self.assertTrue(np.isnan(out.volatility_20_pct.iloc[19]))
        expected=np.std(np.diff(raw.close.iloc[:21])/raw.close.iloc[:20].to_numpy()*100,ddof=1)
        self.assertAlmostEqual(out.volatility_20_pct.iloc[20],expected)
        self.assertFalse(out.feature_ready.iloc[19]);self.assertTrue(out.feature_ready.iloc[20])

    def test_sessions_reset(self):
        raw=prices(grid('2026-09-29','5m').append(grid('2026-09-30','5m')))
        flagged,s,_,_=audit(raw,'5m');out,_=build(flagged,'5m',s)
        self.assertTrue(np.isnan(out.return_1_pct.iloc[75]))
        self.assertTrue(np.isnan(out.rsi_14.iloc[75]))

    def test_daily_exploratory_labels(self):
        idx=pd.date_range('2026-07-01',periods=40,freq='B',tz='Asia/Kolkata')
        flagged,s,_,_=audit(prices(idx),'daily');out,_=build(flagged,'daily',s)
        self.assertEqual(out.feature_status.iloc[-1],'ready_calendar_unverified')
        self.assertFalse(out.feature_calendar_verified.any())

    def test_known_daily_gap_resets(self):
        idx=pd.DatetimeIndex([pd.Timestamp(d,tz='Asia/Kolkata') for d in ['2026-09-07','2026-09-09','2026-09-10']])
        flagged,s,_,_=audit(prices(idx),'daily');out,_=build(flagged,'daily',s)
        self.assertTrue(np.isnan(out.return_1_pct.iloc[1]))
        self.assertFalse(np.isnan(out.return_1_pct.iloc[2]))

    def test_features_do_not_read_future_prices_given_fixed_quality(self):
        flagged,s,_,_=audit(prices(grid('2026-09-29','1m')),'1m')
        full,_=build(flagged,'1m',s);prefix,_=build(flagged.iloc[:50],'1m',s)
        pd.testing.assert_frame_equal(full.iloc[:50],prefix)

    def test_future_calendar_not_certified(self):
        flagged,s,_,meta=audit(prices(grid('2026-10-06','1m')),'1m')
        self.assertFalse(meta['calendar_audit_applied'])
        out,_=build(flagged,'1m',s)
        self.assertFalse(out.feature_ready.any())

    def test_roundtrip(self):
        flagged,s,_,_=audit(prices(grid('2026-09-29','1m')),'1m')
        out,_=build(flagged,'1m',s)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'features.parquet';atomic(p,lambda f:out.to_parquet(f))
            pd.testing.assert_frame_equal(out,pd.read_parquet(p),check_freq=False)


if __name__=='__main__':
    unittest.main()
