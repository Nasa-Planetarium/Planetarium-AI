import unittest
from types import SimpleNamespace

import numpy as np

from planetarium.modeling.daily_occurrence import (
    SEASONS,
    STATE_COUNT,
    Series,
    add_transition_counts,
    history_stats,
    smooth_transitions,
)
from planetarium.modeling.monthly_occurrence_benchmark import (
    folds_for_months,
    pack_history,
    targets,
)


class MonthlyOccurrenceTest(unittest.TestCase):
    def test_entire_calendar_month_and_cutoff_advance(self):
        dates = np.arange("2023-01-01", "2024-04-01", dtype="datetime64[D]")
        coverage = np.full(len(dates), 3, dtype=np.uint8)
        coverage[
            np.isin(dates, np.array(["2023-12-30", "2023-12-31"], dtype=dates.dtype))
        ] = 0
        jan, feb = folds_for_months(
            dates, coverage, np.datetime64("2024-01"), np.datetime64("2024-02")
        )
        self.assertEqual(str(dates[jan.origin]), "2023-12-29")
        self.assertEqual(str(dates[feb.origin]), "2024-01-31")
        self.assertEqual((jan.stop - jan.start, jan.horizon), (31, 33))
        self.assertEqual((feb.stop - feb.start, feb.horizon), (29, 29))

    def test_history_and_labels_keep_observation_masks(self):
        sides = np.array([1, 2], dtype=np.uint8)
        events = np.zeros((66, 2), dtype=np.uint8)
        events[62:65] = [[1, 0], [0, 1], [0, 1]]
        coverage = np.full(len(events), 3, dtype=np.uint8)
        coverage[63] = 2
        coverage[65] = 1
        series = SimpleNamespace(events=events, sides=sides)
        panel = SimpleNamespace(coverage=coverage)
        days = np.arange(len(events))
        known = {
            side: np.maximum.accumulate(np.where((coverage & side) == side, days, -1))
            for side in sides
        }
        baseline = pack_history(panel, series, 64, known)
        unpacked = np.unpackbits(baseline.view(np.uint8).reshape(-1, 8), axis=1)
        np.testing.assert_array_equal(unpacked[:, -3:], [[1, 1, 0], [0, 1, 1]])
        events[65] = 1
        np.testing.assert_array_equal(pack_history(panel, series, 64, known), baseline)
        np.testing.assert_array_equal(targets(panel, series, 65, 66), [[1, -1]])

    def test_incremental_transitions_match_full_history_at_each_month(self):
        dates = np.arange("2025-01-01", "2025-04-01", dtype="datetime64[D]")
        rng = np.random.default_rng(7)
        series = Series(
            rng.integers(0, 2, size=(len(dates), 3), dtype=np.uint8),
            np.array([1, 2, 3], dtype=np.uint8),
            np.zeros(3),
            np.ones(3),
        )
        coverage = rng.choice(np.array([1, 2, 3], dtype=np.uint8), size=len(dates))
        n = np.zeros((SEASONS, 3, STATE_COUNT), dtype=np.float32)
        y = np.zeros_like(n)
        start = 2
        for stop in (31, 59, 90):
            add_transition_counts(series, dates, coverage, start, stop, n, y)
            incremental = smooth_transitions(n, y)
            full = history_stats(series, dates, coverage, stop)
            np.testing.assert_array_equal(incremental.two, full.two)
            np.testing.assert_array_equal(incremental.support, full.support)
            start = stop


if __name__ == "__main__":
    unittest.main()
