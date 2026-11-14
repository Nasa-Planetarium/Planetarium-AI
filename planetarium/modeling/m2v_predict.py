"""새 달 MODIS → VIIRS(S-NPP) 기준 배율: 입력 MODIS 점만으로 칸·월 특성을 만들고 m2v_xgb(+m2v_q10_q90)로 예측.

통합 표(harmonized_cell_month)에 m2v 값이 없는 달(2012-02 이후, 실시간 등)의 MODIS 점에 가중치를 주기 위해 쓴다.
특성은 학습(train_table)과 같은 정의:
  - 공통 정상일(Aqua·Terra 모두 status=ok; 상태표에 없는 최근 날짜는 자료가 있으면 ok로 봄)의 감지만 칸 × 월로 집계
  - 합계 값(n_det, n_det_conf, active_days, frp_sum, 주변 3×3 합)은 days_in_month / common_days로 월 환산
  - frp_max, night_ratio, obs_days(진행 중인 달은 경과 일수 기준을 월로 환산), month_sin/cos, cell_lat/lon,
    terra_drift, 국가·UN 하위지역·대륙(범주형)
  - Aqua 자료가 없으면 m2v_xgb_terra
결과: 칸마다 m2v 배율 ratio = 예측(달 전체 VIIRS 기준) / 환산된 MODIS 감지 수
      → 점 가중치 = ratio (센서 변환만; Σ 점 가중치 ≈ 관측한 날까지의 VIIRS 기준 감지량)
      → value_projected(달 전체 추정) = 예측값
주의: 입력이 좁은 지역이면 가장자리 칸의 주변 3×3 합이 실제보다 작다 (summary.json에 경고).
"""

from __future__ import annotations

import calendar
import math
from datetime import date

import numpy as np
import polars as pl

N_LON = 3600


class M2VPredictor:
    def __init__(self) -> None:
        import xgboost as xgb

        from planetarium.modeling.m1_xgb import M1_FEATURES, M1_TERRA_FEATURES, category_maps
        from planetarium.modeling.names import M2V_PATH, M2V_Q_PATH, M2V_TERRA_PATH
        from planetarium.monthly import STATUS_PATH
        from planetarium.world import WORLD_CELLS_PATH
        self.feats, self.tfeats, self.cats = M1_FEATURES, M1_TERRA_FEATURES, category_maps()
        self.m, self.q, self.t = xgb.Booster(), xgb.Booster(), xgb.Booster()
        self.m.load_model(M2V_PATH)
        self.q.load_model(M2V_Q_PATH)
        self.t.load_model(M2V_TERRA_PATH)
        st = pl.read_parquet(STATUS_PATH, columns=["satellite", "date_utc", "status"])
        self.status = {(s, d): v for s, d, v in st.iter_rows()}
        self.status_last = st["date_utc"].max()
        self.attrs = pl.read_parquet(WORLD_CELLS_PATH, columns=["cell_id", "adm0_a3", "subregion", "continent"])

    def ok(self, sat: str, d: date, has_data: bool) -> bool:
        if d > self.status_last:  # 상태표보다 최근: 자료가 있으면 정상으로 본다
            return has_data
        return self.status.get((sat, d)) == "ok"

    def predict_month(self, modis: pl.DataFrame, year: int, month: int, as_of: date | None = None) -> pl.DataFrame:
        """modis: harmonize_detections의 prepare를 거친 MODIS 점 (cell_id, date_local, satellite, confidence, daynight, frp).
        반환: cell_id, ratio, ratio_lo, ratio_hi, value_projected, value_projected_lo, value_projected_hi, common_days"""
        from planetarium.modeling.m1_xgb import xgb_predict
        dim = calendar.monthrange(year, month)[1]
        last = as_of or date(year, month, dim)
        elapsed = min(dim, (last - date(year, month, 1)).days + 1)
        days = [date(year, month, d) for d in range(1, elapsed + 1)]
        have = {(s, d) for s, d in modis.select("satellite", "date_local").unique().iter_rows()}
        aqua_era = date(year, month, 1) >= date(2002, 7, 4)
        aqua_ok = [d for d in days if aqua_era and self.ok("Aqua", d, ("Aqua", d) in have)]
        terra_ok = [d for d in days if self.ok("Terra", d, ("Terra", d) in have)]
        terra_only = not aqua_ok
        common = terra_ok if terra_only else sorted(set(aqua_ok) & set(terra_ok))
        if not common:
            return pl.DataFrame()
        scale = dim / len(common)
        c = modis.filter(pl.col("date_local").is_in(common))
        agg = (c.group_by("cell_id", "satellite", "date_local")
               .agg(n=pl.len(), nc=(pl.col("confidence") != "l").sum(), nn=(pl.col("daynight") == "N").sum(),
                    fs=pl.col("frp").cast(pl.Float64).sum(), fm=pl.col("frp").cast(pl.Float64).max())
               .group_by("cell_id", "satellite")
               .agg(n_det=pl.col("n").sum(), n_det_conf=pl.col("nc").sum(), n_night=pl.col("nn").sum(),
                    frp_sum=pl.col("fs").sum(), frp_max=pl.col("fm").max(), active_days=pl.len()))
        cells = modis.select("cell_id").unique()
        wide = cells
        for sat, p in (("Aqua", "aqua"), ("Terra", "terra")):
            s = agg.filter(pl.col("satellite") == sat).drop("satellite")
            wide = wide.join(s.rename({k: f"{p}_{k}" for k in s.columns if k != "cell_id"}), on="cell_id", how="left")
        for p in ("aqua", "terra"):
            for k in ("n_det", "n_det_conf", "n_night", "frp_sum", "frp_max", "active_days"):
                if f"{p}_{k}" not in wide.columns:
                    wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{p}_{k}"))
        wide = wide.with_columns(pl.col("^(aqua|terra)_.*$").cast(pl.Float64).fill_null(0))
        # 주변 3×3 합 (입력에 들어 있는 칸만)
        base = wide.select("cell_id", a="aqua_n_det", t="terra_n_det")
        li, lj = pl.col("cell_id") // N_LON, pl.col("cell_id") % N_LON
        nb = pl.concat([base.with_columns(cell_id=(li + di) * N_LON + (lj + dj) % N_LON) for di in (-1, 0, 1) for dj in (-1, 0, 1)]) \
            .group_by("cell_id").agg(aqua_n_det_nb=pl.col("a").sum(), terra_n_det_nb=pl.col("t").sum())
        f = (wide.join(nb, on="cell_id", how="left").join(self.attrs, on="cell_id", how="left")
             .with_columns(
                 aqua_night_ratio=pl.when(pl.col("aqua_n_det") > 0).then(pl.col("aqua_n_night") / pl.col("aqua_n_det")),
                 terra_night_ratio=pl.when(pl.col("terra_n_det") > 0).then(pl.col("terra_n_night") / pl.col("terra_n_det")),
                 *[(pl.col(k) * scale).alias(k) for k in ("aqua_n_det", "terra_n_det", "aqua_n_det_conf", "terra_n_det_conf",
                                                           "aqua_active_days", "terra_active_days", "aqua_frp_sum",
                                                           "terra_frp_sum", "aqua_n_det_nb", "terra_n_det_nb")],
                 month_sin=pl.lit(math.sin(2 * math.pi * month / 12)), month_cos=pl.lit(math.cos(2 * math.pi * month / 12)),
                 cell_lat=((pl.col("cell_id") // N_LON + 0.5) * 0.1 - 90).cast(pl.Float32),
                 cell_lon=((pl.col("cell_id") % N_LON + 0.5) * 0.1 - 180).cast(pl.Float32),
                 terra_drift=pl.lit(int(year >= 2021), dtype=pl.Int8),
                 aqua_obs_days=pl.lit(len(aqua_ok) * dim / elapsed), terra_obs_days=pl.lit(len(terra_ok) * dim / elapsed),
                 adm0_a3=pl.col("adm0_a3").fill_null("none"), subregion=pl.col("subregion").fill_null("none"),
                 continent=pl.col("continent").fill_null("none"),
             ))
        if terra_only:
            pred = xgb_predict(self.t, f, self.tfeats, self.cats)
            lo = hi = np.full(f.height, np.nan)
            modis_total = f["terra_n_det"].to_numpy()
        else:
            pred = xgb_predict(self.m, f, self.feats, self.cats)
            q = xgb_predict(self.q, f, self.feats, self.cats)
            lo, hi = np.minimum(q[:, 0], q[:, 1]).clip(0), np.maximum(q[:, 0], q[:, 1]).clip(0)
            modis_total = (f["aqua_n_det"] + f["terra_n_det"]).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(modis_total > 0, pred / modis_total, np.nan)
            rlo = np.where(modis_total > 0, lo / modis_total, np.nan)
            rhi = np.where(modis_total > 0, hi / modis_total, np.nan)
        return f.select("cell_id").with_columns(
            ratio=pl.Series(ratio).fill_nan(None), ratio_lo=pl.Series(rlo).fill_nan(None), ratio_hi=pl.Series(rhi).fill_nan(None),
            value_projected=pl.Series(pred, dtype=pl.Float64), value_projected_lo=pl.Series(lo).fill_nan(None),
            value_projected_hi=pl.Series(hi).fill_nan(None), common_days=pl.lit(len(common), dtype=pl.Int16),
            terra_only=pl.lit(terra_only))
