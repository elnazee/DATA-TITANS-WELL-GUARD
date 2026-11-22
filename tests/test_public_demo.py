"""Deterministic public showcase checks; does not access private source data."""
import unittest
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_data import make_demo, schedule_demo


class PublicDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fleet, cls.history, cls.jobs = make_demo()

    def test_fictional_ids_only(self):
        self.assertTrue(self.fleet.well_id.str.fullmatch(r"DEMO-\d{4}").all())
        self.assertEqual(self.history.well_id.nunique(),len(self.fleet))
        self.assertFalse(self.fleet.well_id.duplicated().any())

    def test_reproducibility(self):
        _,again,_=make_demo()
        self.assertTrue(self.history.equals(again))

    def test_capacity_scenarios(self):
        latest=self.history.week_end.max()
        for capacity in (25,50,100):
            for cooldown in (0,1,3):
                queue=schedule_demo(self.history,latest,capacity,cooldown=cooldown)
                self.assertEqual(len(queue),capacity)
                self.assertEqual(queue.well_id.nunique(),capacity)
                self.assertTrue(queue.week_end.eq(latest).all())
                self.assertTrue(queue.relative_risk_score.is_monotonic_decreasing)

    def test_sector_filter(self):
        q=schedule_demo(self.history,self.history.week_end.max(),50,zone="North Sector")
        self.assertTrue(q.zone.eq("North Sector").all())

    def test_reference_is_not_synthetic(self):
        benchmark=json.loads((ROOT/'public_data'/'benchmark.json').read_text())
        self.assertEqual(benchmark['comparison']['relative_risk_score']['event_coverage']['25']['caught_events'],116)
        self.assertEqual(benchmark['comparison']['relative_risk_score']['event_coverage']['100']['caught_events'],289)


if __name__=='__main__':
    unittest.main()
