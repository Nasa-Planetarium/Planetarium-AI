from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import polars as pl
from joblib import dump

from planetarium import store
from planetarium.modeling.value_benchmark import errors
from planetarium.modeling.value_features import FEATURES, make_batch
from planetarium.modeling.value_forecast import (
    DEFAULT_MODEL,
    location_history,
    region_boxes,
    region_geometry,
)
from planetarium.modeling.value_models import restore_probability


class RecentModel:
    def predict(self, x, scale):
        mean = np.expm1(x[..., FEATURES.index("log_mean30")])
        return mean * scale[:, None]


class TestValueForecast(TestCase):
    def test_bounds_union_and_point(self):
        west = region_boxes({"id": "west", "point": (180, 0)})
        self.assertEqual(west, [(0, 900, 1, 901)])
        bounds = (179.9, 0, -179.9, 0.1)
        boxes = region_boxes({"id": "two", "bboxes": [bounds, bounds]})
        geometry = region_geometry(boxes)
        self.assertEqual(geometry["n_cells"], 2)
        self.assertEqual(geometry["sides"], 3)
        self.assertAlmostEqual(abs(geometry["lon"]), 180)

    def test_live_entry_point_preserves_value_and_observation_mask(self):
        first = date(2024, 1, 1)
        latest = date(2025, 1, 5)
        days = [
            first + timedelta(days=step) for step in range((latest - first).days + 1)
        ]
        regions = [
            {"id": "area", "bboxes": [(0, 0, 0.2, 0.1), (0, 0, 0.1, 0.1)]},
            {"id": "point", "point": (0.01, 0.01)},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl.DataFrame(
                {
                    "satellite": ["S-NPP"] * len(days),
                    "date_utc": days,
                    "status": [
                        "missing" if d == date(2025, 1, 2) else "ok" for d in days
                    ],
                    "coverage": ["global"] * len(days),
                }
            ).write_parquet(root / "satellite_day_status.parquet")
            sample_days = [date(2025, 1, day) for day in [1, 1, 2, 5]]
            pl.DataFrame(
                {
                    "cell_id": [3241800, 3241801, 3241800, 3241800],
                    "date_local": sample_days,
                    "year": [2025] * 4,
                    "value": [1.25, 2.5, 100.0, 10_000.0],
                    "value_source": ["snpp_observed"] * 4,
                    "residual_month": [400, 400, 0, 0],
                }
            ).write_parquet(root / "harmonized_cell_day.parquet")
            as_of = date(2025, 1, 4)
            panel = location_history(regions, root, as_of)
            np.testing.assert_allclose(panel.values[-4], [3.75, 1.25])
            self.assertTrue(np.isnan(panel.values[-3:-1]).all())
            np.testing.assert_array_equal(panel.values[-1], [0, 0])
            self.assertLess(make_batch(panel, len(panel.dates) - 1).context.max(), 4)

            dump(
                {
                    "model": DEFAULT_MODEL,
                    "regressor": RecentModel(),
                    "fit_end": "2024-12-31",
                    "features": list(FEATURES),
                },
                root / f"{DEFAULT_MODEL}.joblib",
            )
            result = store.value_forecast(
                regions, as_of=as_of, root=root, model_dir=root
            )
            self.assertEqual(result.height, 60)
            self.assertEqual(result["date_local"].min(), date(2025, 1, 5))
            self.assertEqual(result["date_local"].max(), date(2025, 2, 3))
            self.assertEqual(
                result["n_cells"].value_counts()["n_cells"].sort().to_list(), [1, 2]
            )
            self.assertAlmostEqual(result["value"][0], 3.75 / 28, places=6)
            with self.assertRaisesRegex(ValueError, "predates"):
                store.value_forecast(
                    regions, as_of="2024-12-30", root=root, model_dir=root
                )

    def test_zero_subsampling_probability_restores_original_prevalence(self):
        # A 5% prevalence becomes 5 / (5 + 19) after retaining 20% of negatives.
        np.testing.assert_allclose(
            restore_probability(np.array([0, 5 / 24, 1]), 0.2), [0, 0.05, 1]
        )

    def test_daily_errors_exclude_missing_but_include_observed_zeros(self):
        result = errors(np.array([0, 1.5, np.nan]), np.array([2, 1, 400]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["actual_sum"], 1.5)
        self.assertEqual(result["abs_error"], 2.5)
