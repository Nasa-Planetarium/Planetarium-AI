from unittest import TestCase

import numpy as np

from planetarium.modeling.calendar_value_ensemble import AREAS, fit_and_score, fit_blend


class TestCalendarValueEnsemble(TestCase):
    def test_mix_meets_calibration_bias_with_minimum_absolute_error(self):
        actual = np.array([[10.0, 20.0]])
        lightgbm = np.array([[6.0, 16.0]])
        planner = np.array([[20.0, 17.0]])
        alpha = fit_blend(
            actual, lightgbm, planner, np.full(2, 1 / 30), np.zeros((1, 2), dtype=int)
        )
        np.testing.assert_allclose(alpha, [1 / 3], atol=1e-8)

    def test_holdout_labels_do_not_influence_weights(self):
        months = np.array(["2025-01", "2026-01"], dtype="datetime64[M]")
        actual = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        lightgbm = actual * 0.8
        planner = np.array([[12.0, 23.0, 34.0], [7.0, 14.0, 21.0]])
        scales = np.array(AREAS)
        _, weights = fit_and_score(lightgbm, planner, actual, months, scales, 12)
        changed = actual.copy()
        changed[1] *= 3
        _, next_weights = fit_and_score(lightgbm, planner, changed, months, scales, 12)
        self.assertEqual(weights, next_weights)

    def test_identical_predictions_leave_the_mix_at_zero(self):
        actual = np.array([[10.0]])
        prediction = np.array([[8.0]])
        alpha = fit_blend(
            actual, prediction, prediction, np.array([0.1]), np.zeros((1, 1), dtype=int)
        )
        np.testing.assert_array_equal(alpha, [0.0])
