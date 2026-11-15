import unittest

import numpy as np

from planetarium.modeling.range_occurrence_benchmark import _fit_window, _scores


class DateRangeOccurrenceTest(unittest.TestCase):
    def setUp(self):
        self.dates = np.arange("2024-01-01", "2025-02-01", dtype="datetime64[D]")

    def test_fit_window_keeps_calibration_and_targets_separate(self):
        origin = int(self.dates.searchsorted(np.datetime64("2025-01-16")))
        first, calibration, stop = _fit_window(
            self.dates, origin, np.datetime64("2024-03-01"), None, 90
        )
        self.assertEqual(self.dates[first], np.datetime64("2024-03-01"))
        self.assertEqual(self.dates[stop - 1], np.datetime64("2025-01-16"))
        self.assertEqual(stop - calibration, 90)
        self.assertEqual(self.dates[stop], np.datetime64("2025-01-17"))
        with self.assertRaisesRegex(ValueError, "observation origin"):
            _fit_window(self.dates, origin, None, self.dates[stop], 90)

    def test_score_masks_incomplete_dates_and_handles_a_single_class(self):
        metrics = _scores(np.array([0, -1, 0]), np.array([0.1, 0.9, 0.3]))
        self.assertEqual(metrics["rows"], 2)
        self.assertIsNone(metrics["pr_auc"])
        self.assertAlmostEqual(metrics["brier"], 0.05)


if __name__ == "__main__":
    unittest.main()
