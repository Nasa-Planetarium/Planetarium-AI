"""v1.2 통합 데이터 조회 도우미 (백엔드·노트북용). 모두 polars DataFrame을 돌려준다.

규칙은 v1.2 그대로: 값은 S-NPP 기준(value), 고정 열원 점은 합계에서 이미 빠져 있고 점 조회에서도 기본으로 뺀다.
큰 표는 scan + 필터로 필요한 부분만 읽는다 (서버 시작 때 전체를 메모리에 올리지 않음).
설계 참고: 팀원 Jonghun Shin(ozingmw)의 firms-1km-backend API 조회 구조.

    from planetarium import store
    store.points("2025-03-22", "2025-03-27", bbox=(128.4, 36.0, 129.6, 36.8))   # 감지 점 (기준 점, 고정 열원 제외)
    store.cells("2025-03", bbox=(124.5, 33.0, 132.0, 38.7))                     # 0.1° 칸 × 월
    store.cell_days("2025-03-20", "2025-03-31", bbox=...)                       # 0.1° 칸 × 일
    store.aoi("SGG_47730", "2025-03")                                           # 지역 × 월 (+ 예년 비교)
    store.aoi("KR", start="2024-01", end="2025-12")                             # 기간
    store.aoi_list(level="sgg", q="의성")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from planetarium.config import PROCESSED_DATA_DIR

ROOT = PROCESSED_DATA_DIR
BBox = tuple[float, float, float, float]   # (서, 남, 동, 북)
PT_KEY = ["satellite", "latitude", "longitude", "acq_datetime_utc"]


def _d(x: str | date) -> date:
    return x if isinstance(x, date) else date.fromisoformat(x)


def _ym(x: str) -> tuple[int, int]:
    y, m = x.split("-")[:2]
    return int(y), int(m)


def _bbox(b: BBox | None, lat: str, lon: str) -> pl.Expr:
    if b is None:
        return pl.lit(True)
    w, s, e, n = b
    lon_ok = pl.col(lon).is_between(w, e) if w <= e else (pl.col(lon) >= w) | (pl.col(lon) <= e)   # 날짜 변경선을 넘는 상자
    return pl.col(lat).is_between(s, n) & lon_ok


def _cell_center(prefix: str = "") -> list[pl.Expr]:
    return [((pl.col("cell_id") // 3600 + 0.5) * 0.1 - 90).alias(f"{prefix}lat"),
            ((pl.col("cell_id") % 3600 + 0.5) * 0.1 - 180).alias(f"{prefix}lon")]


def frp_ratios(root: Path | None = None) -> pl.DataFrame:
    """FRP S-NPP 환산 비율표 (satellite 코드 × daynight → ratio). Release 설치본(root/reports/)이 있으면 그것, 없으면 저장소 reports/.
    frp_snpp_equiv = frp ÷ ratio (S-NPP는 1)."""
    from planetarium.config import REPORTS_DIR
    path = next(p for p in (Path(root or ROOT) / "reports" / "frp_snpp_ratios.csv", REPORTS_DIR / "frp_snpp_ratios.csv") if p.exists())
    t = pl.read_csv(path).filter(pl.col("year") == "all").select("satellite", "daynight", "ratio")
    return pl.concat([t, pl.DataFrame({"satellite": ["N", "N"], "daynight": ["D", "N"], "ratio": [1.0, 1.0]})])


def points(start: str | date, end: str | date | None = None, bbox: BBox | None = None, layer: str = "reference",
           exclude_static: bool = True, frp_equiv: bool = False, columns: list[str] | None = None, limit: int | None = None,
           root: Path | None = None) -> pl.DataFrame:
    """통합 핫스팟 점 (현지 날짜 start~end). layer: reference(합계용 기준 점) | other(그 밖의 VIIRS) | all.
    exclude_static: 고정 열원 점(harmonized_hotspots_static_flags)을 뺀다 → 기준 점 가중치 합 = 칸 값.
    frp_equiv: frp_snpp_equiv = frp ÷ 비율(위성·낮밤) 열을 붙인다 (reports/frp_snpp_ratios.csv)."""
    root = Path(root or ROOT)
    d0 = _d(start)
    d1 = _d(end) if end is not None else d0
    months = [(y, m) for y in range(d0.year, d1.year + 1) for m in range(1, 13) if date(y, m, 1) <= d1 and (y, m) >= (d0.year, d0.month)]
    layers = ["reference", "other"] if layer == "all" else [layer]
    files = [f for ly in layers for (y, m) in months for f in (root / "harmonized_hotspots" / ly / f"year={y}" / f"month={m:02d}").glob("*.parquet")]
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files).filter(pl.col("date_local").is_between(d0, d1), _bbox(bbox, "latitude", "longitude"))
    if exclude_static:
        flags = [root / "harmonized_hotspots_static_flags" / f"year={y}.parquet" for y in range(d0.year, d1.year + 1)]
        flags = [f for f in flags if f.exists()]
        if flags:
            lf = lf.join(pl.scan_parquet(flags).select(PT_KEY), on=PT_KEY, how="anti")
    if frp_equiv:
        lf = (lf.join(frp_ratios(root).lazy(), on=["satellite", "daynight"], how="left")
              .with_columns(frp_snpp_equiv=(pl.col("frp") / pl.col("ratio")).cast(pl.Float32)).drop("ratio"))
    if columns:
        lf = lf.select(columns)
    if limit:
        lf = lf.head(limit)
    return lf.collect()


def cells(month: str, bbox: BBox | None = None, columns: list[str] | None = None, root: Path | None = None) -> pl.DataFrame:
    """0.1° 칸 × 월 (harmonized_cell_month). month = "YYYY-MM". 칸 중심 lat·lon을 붙인다."""
    y, m = _ym(month)
    lf = (pl.scan_parquet(Path(root or ROOT) / "harmonized_cell_month.parquet").filter(pl.col("year") == y, pl.col("month") == m)
          .with_columns(_cell_center()).filter(_bbox(bbox, "lat", "lon")))
    return (lf.select(["cell_id", "lat", "lon", *columns]) if columns else lf).collect()


def cell_days(start: str | date, end: str | date | None = None, bbox: BBox | None = None, root: Path | None = None) -> pl.DataFrame:
    """0.1° 칸 × 현지 날짜 (harmonized_cell_day). residual_month는 그 달 1일 행에만 있는 월 합계라 기간 합에는 따로 다룬다 (api_contract /summary)."""
    d0 = _d(start)
    d1 = _d(end) if end is not None else d0
    return (pl.scan_parquet(Path(root or ROOT) / "harmonized_cell_day.parquet")
            .filter(pl.col("year").is_between(d0.year, d1.year), pl.col("date_local").is_between(d0, d1))
            .with_columns(_cell_center()).filter(_bbox(bbox, "lat", "lon")).collect())


def value_forecast(
    regions: list[dict],
    *,
    as_of: str | date | None = None,
    model: str | None = None,
    root: Path | None = None,
    model_dir: Path | None = None,
) -> pl.DataFrame:
    """Predict the next 30 daily S-NPP-equivalent values for each region."""
    from planetarium.modeling.value_forecast import DEFAULT_MODEL, forecast

    return forecast(
        regions, as_of=as_of, model=DEFAULT_MODEL if model is None else model,
        root=root, model_dir=model_dir,
    )


def aoi(aoi_id: str, month: str | None = None, start: str | None = None, end: str | None = None, compare: bool = True,
        root: Path | None = None) -> pl.DataFrame:
    """지역 × 월 (aoi_month, 면적 가중 합계·status·as_of_date) + 예년 비교(aoi_month_compare). month 하나 또는 start~end ("YYYY-MM")."""
    root = Path(root or ROOT)
    t = pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32)
    lf = pl.scan_parquet(root / "aoi_month.parquet").filter(pl.col("aoi_id") == aoi_id)
    if month:
        y, m = _ym(month)
        lf = lf.filter(pl.col("year") == y, pl.col("month") == m)
    else:
        if start:
            y, m = _ym(start)
            lf = lf.filter(t >= y * 12 + m)
        if end:
            y, m = _ym(end)
            lf = lf.filter(t <= y * 12 + m)
    if compare:
        cmp = pl.scan_parquet(root / "aoi_month_compare.parquet").select(
            "aoi_id", "year", "month", "baseline_median", "baseline_p25", "baseline_p75", "percentile", "ratio_to_median", "verdict")
        lf = lf.join(cmp, on=["aoi_id", "year", "month"], how="left")
    return lf.sort("year", "month").collect()


def aoi_list(level: str | None = None, q: str | None = None, parent: str | None = None, root: Path | None = None) -> pl.DataFrame:
    """지역 목록 (level: world·continent·country·adm1·kr·sido·sgg). q: 이름 일부, parent: 상위 aoi_id. 숨김 지역 제외."""
    lf = pl.scan_parquet(Path(root or ROOT) / "aoi_list.parquet").filter(~pl.col("hidden"))
    if level:
        lf = lf.filter(pl.col("level") == level)
    if parent:
        lf = lf.filter(pl.col("parent_aoi") == parent)
    if q:
        lf = lf.filter(pl.col("name").str.contains(q, literal=True))
    return lf.sort("aoi_id").collect()
