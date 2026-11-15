from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import polars as pl

from planetarium.modeling.daily_occurrence import _region_mask, forecast, roll
from planetarium.modeling.daily_occurrence_data import DailyPanel


class DailyOccurrenceTest(unittest.TestCase):
    def fixture(self, poison=False):
        dates = np.arange("2024-01-01", "2024-01-23", dtype="datetime64[D]")
        lon = np.array([-0.25, 0.25, 0.75])
        grid = 180 * 720 + np.floor((lon + 180) * 2).astype(int)
        cells = pl.DataFrame({"g": grid, "lon": lon, "lat": [0.25] * 3})
        events = (np.arange(len(dates))[:, None] % np.arange(2, 5) == 0).astype(
            np.uint8
        )
        if poison:
            events[15:] ^= 1
        coverage = np.full(len(dates), 3, dtype=np.uint8)
        coverage[20:] = 1
        return DailyPanel(cells, dates, events, coverage, "fixture")

    def test_probability_state_is_propagated(self):
        initial = np.array([[1.0, 0.0, 0.0, 0.0]])
        conditional = np.array([[0.2, 0.8, 0.2, 0.8]])
        path = roll(initial, np.zeros(3, dtype=int), lambda _: conditional)
        np.testing.assert_allclose(path[:, 0], [0.2, 0.32, 0.392])

    def test_forecast_skips_output_days_without_restarting_and_ignores_future(self):
        regions = [
            {"id": "union", "bboxes": [[0, 0, 1, 0.5], [0.4, 0, 1, 0.5]]},
            {"id": "west", "point": [-0.25, 0.25]},
        ]
        with TemporaryDirectory() as folder:
            with patch(
                "planetarium.modeling.daily_occurrence.load_panel",
                return_value=self.fixture(),
            ):
                original = forecast(
                    regions,
                    "2024-01-16",
                    "2024-01-20",
                    as_of="2024-01-15",
                    work=Path(folder),
                )
            with patch(
                "planetarium.modeling.daily_occurrence.load_panel",
                return_value=self.fixture(poison=True),
            ):
                window = forecast(
                    regions,
                    "2024-01-18",
                    "2024-01-20",
                    as_of="2024-01-15",
                    work=Path(folder),
                )
        self.assertEqual(window["step"].to_list(), [3, 4, 5, 3, 4, 5])
        self.assertEqual(window["n_cells"].to_list(), [2, 2, 2, 1, 1, 1])
        self.assertTrue(original.filter(pl.col("step") >= 3).equals(window))
        np.testing.assert_allclose(
            window["p_any"].to_numpy() + window["p_none"].to_numpy(), 1.0
        )

    def test_batch_uses_last_day_complete_for_both_sides(self):
        regions = [
            {"id": "east", "point": [0.25, 0.25]},
            {"id": "west", "point": [-0.25, 0.25]},
        ]
        with (
            TemporaryDirectory() as folder,
            patch(
                "planetarium.modeling.daily_occurrence.load_panel",
                return_value=self.fixture(),
            ),
        ):
            result = forecast(regions, "2024-01-22", "2024-01-22", work=Path(folder))
            east = forecast(regions[:1], "2024-01-22", "2024-01-22", work=Path(folder))
        self.assertEqual(result["step"].to_list(), [2, 2])
        self.assertEqual(east["step"].to_list(), [1])

    def test_dateline_bbox_keeps_both_edges(self):
        frame = pl.DataFrame(
            {"g": [0, 1, 2], "lon": [-179.75, 0.25, 179.75], "lat": [0.25] * 3}
        )
        mask = _region_mask({"bbox": [179.6, 0, -179.6, 0.5]}, frame)
        np.testing.assert_array_equal(mask, [True, False, True])
