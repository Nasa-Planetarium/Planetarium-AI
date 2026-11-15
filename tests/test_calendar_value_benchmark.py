from unittest import TestCase

import numpy as np

from planetarium.modeling.calendar_value_benchmark import (
    calendar_window,
    make_month_features,
)


class TestCalendarValueBenchmark(TestCase):
    def test_february_window_uses_latest_available_day_and_all_calendar_days(self):
        observed = np.asarray(
            ["2024-01-11", "2024-01-30", "2024-02-01"], dtype="datetime64[D]"
        )
        month = np.datetime64("2024-02", "M")
        for issue_day, expected_origin in [(12, 0), (31, 1)]:
            origin, window = calendar_window(observed, month, issue_day)
            self.assertEqual(origin, expected_origin)
            self.assertEqual(
                observed[origin] + np.timedelta64(window.start + 1, "D"),
                np.datetime64("2024-02-01"),
            )
            self.assertEqual(
                observed[origin] + np.timedelta64(window.stop, "D"),
                np.datetime64("2024-02-29"),
            )
            self.assertEqual(window.stop - window.start, 29)

    def test_finalized_values_from_previous_month_onward_do_not_affect_features(self):
        months = np.arange(64, dtype=float)[:, None]
        date_features = np.ones((1, 30, 4), dtype=np.float32)
        before = make_month_features(date_features, months, 60)
        months[59:] = 100_000
        after = make_month_features(date_features, months, 60)
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(old, new)
