"""FireOutlook Planner — 원하는 지역 + 원하는 기간의 열원(위성 탐지) 예측.

    from planetarium.forecast import forecast
    forecast("SIDO_51", "2026-11-15", "2026-11-21")                 # aoi_id (aoi_list 5,051개)
    forecast((-66.0, -35.0, -61.8, -29.5), "2026-12-01", "2026-12-31")   # bbox (서, 남, 동, 북)
    forecast({"type": "Polygon", "coordinates": [...]}, "2027-01-01", "2027-03-31")  # GeoJSON 폴리곤
    uv run python -m planetarium.forecast --region SIDO_51 --start 2026-11-15 --end 2026-11-21

미리 계산한 격자 예측을 모아서 답한다 (매 갱신 때 planner_monthly final / planner_daily final 이 다시 만든다).
  월 단위: data/processed/planner_monthly_grid.parquet (0.5° 격자 × 1~12개월 뒤; 기준 달 2026-11)
  일 단위: data/processed/planner_daily_grid.parquet (0.5° 격자 × 1~7일 뒤; 있으면 사용)
기간 조각 규칙 (겹치는 부분)
  1) 기준일(as_of) 다음 날부터 7일 안의 날짜는 일 단위 예측을 쓴다 (있을 때).
  2) 나머지 날짜는 달별로 묶어 월 단위 예측을 쓴다. 달의 일부만 포함되면 일별 기후값 가중치(planner_dayweights)로
     건수 f_c(기후 탐지 수 모양)·확률 f_p(기후 탐지일 모양)를 정해 p = 1 − (1 − p_달)^f_p, 건수 × f_c.
     일 단위가 쓴 날짜는 월 단위 조각에서 빼고 계산한다 (두 번 세지 않음).
  3) 기준 달(진행 중인 달)의 남은 날짜 중 일 단위가 덮지 못한 날은 월 예측이 없으므로 기준선(예년 같은 달)을 비율로 쓴다(skill none).
  4) 조각을 합칠 때: p_any = 1 − Π(1 − p_조각) (조각 사이 독립 가정 → 긴 기간은 다소 높게 나올 수 있음),
     expected_count = Σ, q10/q90 = 조각 구간의 합 (보수적).
지역 확률: 격자 hazard 합 → 지역 크기 구간별 isotonic 보정 (planner_regions; 독립 곱보다 과대평가가 적다).
라이선스: 서비스 모델은 XGBoost·LightGBM·로지스틱·통계 모델만 (TimesFM 3.0 등 비상업 라이선스 모델은 비교에만 썼고 쓰지 않음).
"""

from __future__ import annotations

import argparse
import calendar
import json
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import polars as pl

from planetarium.config import PROCESSED_DATA_DIR

MONTHLY = PROCESSED_DATA_DIR / "planner_monthly_grid.parquet"
DAILY = PROCESSED_DATA_DIR / "planner_daily_grid.parquet"
WORK = PROCESSED_DATA_DIR.parent / "interim" / "planner"   # 학습·검증 작업 폴더 (서비스는 읽지 않음)
# 서비스 파일: 모두 data/processed/ (Release data-v1.2 에 포함; planner_publish 가 작업 결과에서 만든다)
SVC = {"aoi_weights": PROCESSED_DATA_DIR / "planner_aoi_weights.npz", "region_cal": PROCESSED_DATA_DIR / "planner_region_calibration.json",
       "region_skill": PROCESSED_DATA_DIR / "planner_region_skill.json", "count_choice": PROCESSED_DATA_DIR / "planner_count_choice.json",
       "region_counts_sf": PROCESSED_DATA_DIR / "planner_region_counts_sf.parquet", "sgg_choice": PROCESSED_DATA_DIR / "planner_sgg_choice.json",
       "day_weights": PROCESSED_DATA_DIR / "planner_day_weights.npz", "skill_30d": PROCESSED_DATA_DIR / "planner_30d_skill.json",
       "day_any": PROCESSED_DATA_DIR / "planner_day_any.npz"}
NEAR_CERTAIN = 0.98     # 월 조각 확률이 이 이상이면 날짜별 확률은 탐지일 기후값으로 (planner_dayany)
UNIT_30D = {"sgg": "Korea sgg", "sido": "Korea sido", "adm1": "adm1", "country": "countries", "continent": "continents", "world": "world"}
NOTICE = "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다. (Probability of satellite fire detections; it does not guarantee that a wildfire will occur.)"
SIZE_BINS = [0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 1e9]
SIZE_LABELS = ["<0.5", "0.5-2", "2-8", "8-32", "32-128", ">=128"]


@lru_cache(maxsize=1)
def _monthly() -> pl.DataFrame:
    return pl.read_parquet(MONTHLY)


@lru_cache(maxsize=1)
def _daily() -> pl.DataFrame | None:
    return pl.read_parquet(DAILY) if DAILY.exists() else None


@lru_cache(maxsize=1)
def _grid() -> pl.DataFrame:
    return _monthly().select("g", "lat", "lon", "persistent_source").unique("g").sort("g")


@lru_cache(maxsize=1)
def _aoi_w():
    import scipy.sparse as sp
    z = np.load(SVC["aoi_weights"])          # 피클 없이 읽음 (ids는 문자열 배열)
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return W, list(z["ids"]), z["gids"]


@lru_cache(maxsize=1)
def _cal() -> dict:
    return json.loads(SVC["region_cal"].read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _skill_table() -> dict:
    """{(시점, 지역 등급): high | low | none} — 검증(해 하나씩 뺀 보정)과 확인 구간 BSS 중 작은 값 기준 (planner_publish가 만든 파일)."""
    if not SVC["region_skill"].exists():
        return {}
    return {(int(h), g): v for h, gs in json.loads(SVC["region_skill"].read_text(encoding="utf-8")).items() for g, v in gs.items()}


def _near(keys, h: int) -> int:
    """검증표에 없는 시점은 가장 가까운(같으면 더 먼) 시점 값을 쓴다."""
    ks = sorted({int(k) for k in keys})
    return min(ks, key=lambda k: (abs(k - h), -k)) if ks else h


def _group(region) -> str:
    if isinstance(region, str):
        from planetarium.planner_regions import LEVEL_GROUP
        lv = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "level"]).filter(pl.col("aoi_id") == region)["level"]
        return LEVEL_GROUP.get(lv[0] if lv.len() else "", "all regions")
    return "all regions"


def save_aoi_weights(path=None) -> None:
    """aoi_list 지역 × 0.5° 격자 면적 비율 행렬을 저장 (갱신 때 한 번). ids는 고정 길이 문자열 배열(피클 없음)."""
    from planetarium.planner_regions import aoi_weights, grid_index
    W, ids, _ = aoi_weights()
    gids = grid_index()["g"].to_numpy()
    np.savez(path or SVC["aoi_weights"], data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape), ids=np.array(ids, dtype=str), gids=gids)


@lru_cache(maxsize=1)
def _cell_tree():
    """0.5° 격자 사각형과 공간 색인 (한 번만 만든다; bbox·폴리곤 질의 속도)."""
    from shapely.geometry import box
    from shapely.strtree import STRtree
    g = _grid()
    cells = [box(lo - 0.25, la - 0.25, lo + 0.25, la + 0.25) for la, lo in zip(g["lat"].to_list(), g["lon"].to_list())]
    return cells, STRtree(cells)


def warmup() -> None:
    """서버 시작 때 한 번: 파일·색인을 미리 읽어 첫 요청도 1초 안에."""
    forecast("SIDO_51", horizon_days=30)
    _cell_tree()
    _dayany()


def region_weights(region) -> tuple[pl.DataFrame, str]:
    """→ (g, w) 표와 지역 설명. aoi_id | (서,남,동,북) | GeoJSON."""
    if isinstance(region, str):
        W, ids, gids = _aoi_w()
        if region not in ids:
            raise ValueError(f"모르는 aoi_id: {region}")
        row = W.getrow(ids.index(region))
        return pl.DataFrame({"g": gids[row.indices], "w": row.data}), f"aoi_id {region}"
    from shapely.geometry import box, shape
    geom = box(*region) if isinstance(region, (tuple, list)) else shape(region.get("geometry", region))
    g = _grid()
    cells, tree = _cell_tree()
    hit = tree.query(geom, predicate="intersects")
    w = np.array([cells[i].intersection(geom).area / 0.25 for i in hit])
    return pl.DataFrame({"g": g["g"].to_numpy()[hit], "w": w}).filter(pl.col("w") > 0), "bbox" if isinstance(region, (tuple, list)) else "polygon"


def _iso(x: float, pts: dict, base: bool = False) -> float:
    xs, ys = (pts["base_x"], pts["base_y"]) if base else (pts["x"], pts["y"])
    return float(np.interp(x, xs, ys)) if xs else x


def _size_label(n: float) -> str:
    return SIZE_LABELS[int(np.clip(np.searchsorted(SIZE_BINS, n, side="right") - 1, 0, len(SIZE_LABELS) - 1))]


@lru_cache(maxsize=1)
def _district_next() -> dict:
    """한국 시군구 다음 달 = FireOutlook Monthly – District (aoi_occurrence) 값 그대로 (/occurrence와 같은 숫자)."""
    d = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_occurrence_nextmonth_latest.parquet")
    return {a: (p, b, mo, tm) for a, p, b, mo, tm in d.select("aoi_id", "p_any", "baseline_p", "model", "target_month").iter_rows()}


@lru_cache(maxsize=1)
def _sgg_direct() -> dict:
    """한국 시군구 2·3개월 뒤 (planner_sgg: 시군구 직접 학습 vs 격자 합산 중 검증에서 나은 쪽)."""
    f = PROCESSED_DATA_DIR / "planner_sgg_monthly.parquet"
    ch = SVC["sgg_choice"]
    if not f.exists() or not ch.exists():
        return {}
    choice = json.loads(ch.read_text())
    d = pl.read_parquet(f)
    return {(a, int(h)): (p, b, mo, sk, choice[str(h)]["model"]) for a, h, p, b, mo, sk in
            d.select("aoi_id", "h", "p_any", "baseline_p", "model", "skill").iter_rows()}


@lru_cache(maxsize=1)
def _grid_next() -> dict:
    """0.5° 격자 다음 달 = /occurrence?unit=grid 파일 (grid_occurrence_nextmonth_latest) 그대로."""
    d = pl.read_parquet(PROCESSED_DATA_DIR / "grid_occurrence_nextmonth_latest.parquet")
    return {g: (p, b, mo) for g, p, b, mo in d.select("cell_id", "p_any", "baseline_p", "model").iter_rows()}


@lru_cache(maxsize=1)
def _count_choice() -> tuple[dict, dict]:
    """큰 지역(대륙·전 세계·한국·아르헨티나) 월 건수: 검증 WAPE로 고른 후보 + AutoETS/AutoTheta 서비스 예측."""
    ch, sv = SVC["count_choice"], SVC["region_counts_sf"]
    if not ch.exists() or not sv.exists():
        return {}, {}
    s = pl.read_parquet(sv)
    return json.loads(ch.read_text()), {(a, t): (e, th) for a, t, e, th in s.select("aoi_id", "t", "AutoETS", "AutoTheta").iter_rows()}


def _count_group(region) -> str | None:
    from planetarium.planner_regioncount import group_of
    return group_of(region) if isinstance(region, str) else None


@lru_cache(maxsize=1)
def _dayw():
    """일별 기후 가중치 (planner_dayweights build): C = 건수 모양, CA = 탐지일 모양, g 순서."""
    f = SVC["day_weights"]
    if not f.exists():
        return None
    z = np.load(f)
    return z["C"], z["CA"], {int(g): i for i, g in enumerate(z["g"])}


def _fractions(mo: pl.DataFrame, days: list[date]) -> tuple[float, float, str]:
    """달 일부 기간의 비율 → (건수용 f, 확률용 f, 방식). 전체 달이면 1, 1.
    칸별 f = 그 날짜들의 기후값 합 / 그 달 전체 기후값 합 (planner_dayweights: ±7일 평균, 자료 적은 칸은 나라·대륙 모양).
    지역은 건수 f를 예년 중앙값 건수로, 확률 f를 격자 hazard로 가중 평균한다."""
    y, m = days[0].year, days[0].month
    dim = calendar.monthrange(y, m)[1]
    uni = len(days) / dim
    dw = _dayw()
    if len(days) == dim:
        return 1.0, 1.0, "full month"
    if dw is None:                 # 가중치 파일이 없으면 날짜 수 비율
        return uni, uni, "uniform_fallback"
    from planetarium.planner_dayweights import frac
    C, CA, pos = dw
    gi = np.array([pos.get(int(g), -1) for g in mo["g"].to_list()])
    ok = gi >= 0
    if not ok.any():
        return uni, uni, "uniform_fallback"
    fc, fp = np.full(len(gi), uni), np.full(len(gi), uni)
    fc[ok], fp[ok] = frac(C[gi[ok]], days), frac(CA[gi[ok]], days)
    w = mo["w"].to_numpy()
    wc = w * mo["base_count_median"].fill_null(0).to_numpy()
    wp = w * -np.log1p(-np.clip(mo["p_any"].to_numpy(), 0, 0.999999))
    f_c = float((wc * fc).sum() / wc.sum()) if wc.sum() > 0 else float((w * fc).sum() / w.sum())
    f_p = float((wp * fp).sum() / wp.sum()) if wp.sum() > 0 else float((w * fp).sum() / w.sum())
    return f_c, f_p, "daily climatology weights"


@lru_cache(maxsize=1)
def _dayany():
    """지역·격자별 "과거 같은 날짜(±7일)에 탐지가 있던 날의 비율" (2012–2023)."""
    f = SVC["day_any"]
    if not f.exists():
        return None
    z = np.load(f)
    return z["region"], {a: i for i, a in enumerate(z["region_ids"].tolist())}, z["cell"], {int(g): i for i, g in enumerate(z["g"])}


def _day_any_rate(region, mo: pl.DataFrame, days: list[date]) -> np.ndarray | None:
    """날짜별 탐지일 기후값. aoi_id는 지역 자체 기록, bbox·폴리곤은 격자 값을 독립 가정으로 합침."""
    from planetarium.planner_dayweights import SLOT
    da = _dayany()
    if da is None:
        return None
    R, rix, Cc, cix = da
    sl = [SLOT[(d.month, d.day)] for d in days]
    if isinstance(region, str) and region in rix:
        return R[rix[region], sl].astype(float)
    gi = np.array([cix.get(int(g), -1) for g in mo["g"].to_list()])
    ok = gi >= 0
    if not ok.any():
        return None
    w = mo["w"].to_numpy()[ok][:, None]
    return 1 - np.exp((w * np.log1p(-np.clip(Cc[gi[ok]][:, sl].astype(float), 0, 0.999999))).sum(0))


def _fractions_by_day(mo: pl.DataFrame, days: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """날짜 하나씩의 (건수용 f, 확률용 f) — _fractions와 같은 가중이라 날짜별 합 = 조각 전체 f."""
    dim = calendar.monthrange(days[0].year, days[0].month)[1]
    dw = _dayw()
    if dw is None:
        u = np.full(len(days), 1 / dim)
        return u, u
    from planetarium.planner_dayweights import SLOT
    C, CA, pos = dw
    gi = np.array([pos.get(int(g), -1) for g in mo["g"].to_list()])
    ok = gi >= 0
    if not ok.any():
        u = np.full(len(days), 1 / dim)
        return u, u
    all_s = [SLOT[(days[0].month, d)] for d in range(1, dim + 1)]
    sel = [SLOT[(d.month, d.day)] for d in days]
    w = mo["w"].to_numpy()[ok]
    out = []
    for M, wv in ((C, w * mo["base_count_median"].fill_null(0).to_numpy()[ok]),
                  (CA, w * -np.log1p(-np.clip(mo["p_any"].to_numpy()[ok], 0, 0.999999)))):
        sub = M[gi[ok]]
        per = sub[:, sel] / sub[:, all_s].sum(1, keepdims=True)          # 칸 × 날짜
        wv = wv if wv.sum() > 0 else w
        out.append((wv[:, None] * per).sum(0) / wv.sum())
    return out[0], out[1]


_REGION_CTX: dict = {}


def _month_piece(rw: pl.DataFrame, y: int, m: int, days: list[date], h: int, group: str = "all regions", region=None) -> dict:
    mo = _monthly().filter(pl.col("year") == y, pl.col("month") == m).join(rw, on="g")
    frac = len(days) / calendar.monthrange(days[0].year, days[0].month)[1]
    n_eff = float(rw["w"].sum())
    if mo.height == 0:
        return {"kind": "monthly", "month": f"{y}-{m:02d}", "fraction": frac, "p_any": 0.0, "base_p": 0.0, "expected": 0.0, "q10": 0.0,
                "q90": 0.0, "base_count": 0.0, "model": "none", "horizon_months": h, "skill": "none"}
    w = mo["w"].to_numpy()
    single_cell = rw.height == 1 and float(rw["w"][0]) >= 0.999
    raw = 1 - np.exp(-(w * -np.log1p(-np.clip(mo["p_any"].to_numpy(), 0, 0.999999))).sum())
    rawb = 1 - np.exp(-(w * -np.log1p(-np.clip(mo["base_p"].to_numpy(), 0, 0.999999))).sum())
    cal = _cal()
    ch = cal.get(str(_near(cal.keys(), h)), {})
    pts = ch.get(_size_label(n_eff))
    sk = _skill_table()
    skill = sk.get((_near([k for k, g in sk if g == group], h), group), "none")
    model = mo["model"][0]
    route = "grid hazard sum + region isotonic"
    if single_cell:
        # 격자 한 칸: 지역 보정 없이 격자 값 (다음 달은 /occurrence?unit=grid 파일 값)
        p, pb, skill, route = raw, rawb, mo["skill"][0], "single 0.5° cell"
        if h == 1 and int(mo["g"][0]) in _grid_next():
            p, pb, model = _grid_next()[int(mo["g"][0])]
            route = "single 0.5° cell = /occurrence?unit=grid"
    else:
        p = _iso(raw, pts) if pts else raw
        pb = _iso(rawb, pts, base=True) if pts else rawb
        if isinstance(region, str) and region.startswith("SGG_") and h == 1 and region in _district_next():
            p, pb, model, _ = _district_next()[region]
            skill, route = "low", "FireOutlook Monthly – District (= /occurrence)"   # 기존 검증 PR-AUC 0.422 vs 0.404 (Brier 개선 3.8%)
        elif isinstance(region, str) and region.startswith("SGG_") and (region, h) in _sgg_direct():
            pd_, pbd, md, skd, src = _sgg_direct()[(region, h)]
            if src == "direct":
                p, pb, model, skill, route = pd_, pbd, md, skd, "Korea sgg direct model (planner_sgg)"
            elif src == "base_rate":
                p, pb, model, skill, route = pbd, pbd, "baseline_same_month_rate", "none", "baseline (planner_sgg choice)"
        if skill == "none" and not route.startswith("baseline"):   # 지역으로 모았을 때 기준선보다 확실히 낫지 않음 → 기준선
            p, model = pb, "baseline_same_month_rate"
            route = "baseline (region skill none)"
    e_mod = float((w * mo["mu"].to_numpy()).sum())
    e_base = float((w * mo["base_count_median"].fill_null(0).to_numpy()).sum())
    # 건수 라우팅 (planner_regioncount.routing): 큰 지역은 자기 묶음(대륙·전 세계·한국·아르헨티나), 나머지는 확률과 같은 등급
    cchoice, csv = _count_choice()
    cg = _count_group(region) or group
    gk = [k for k, v in cchoice.items() if cg in v] or [k for k, v in cchoice.items() if "all regions" in v]
    ent = cchoice.get(str(_near(gk, h)), {}).get(cg) or cchoice.get(str(_near(gk, h)), {}).get("all regions", {})
    src, count_skill = ent.get("source", "baseline_median"), ent.get("count_skill", "none")
    if single_cell:
        src, count_skill = "model_mu", mo["skill"][0]
    if src in ("AutoETS", "AutoTheta") and (region, y * 12 + m - 1) not in csv:
        src, count_skill = "baseline_median", "none"
    if src in ("AutoETS", "AutoTheta"):
        e = float(csv[(region, y * 12 + m - 1)][0 if src == "AutoETS" else 1])
        nm = "base"   # 구간 폭은 같은 크기 구간의 예년 중앙값 비율을 빌려 씀
    else:
        e, nm = (e_mod, "model") if src == "model_mu" else (e_base, "base")
    if "ratio_q10" in ent and src == ent.get("source") and not single_cell:     # 큰 지역 묶음: 그 묶음·시점 검증 비율
        q10, q90 = ent["ratio_q10"] * e, ent["ratio_q90"] * e
    else:
        q10, q90 = (pts[f"ratio_{nm}_q10"] * e, pts[f"ratio_{nm}_q90"] * e) if pts else (0.0, 2 * e)
    f_c, f_p, how = _fractions(mo, days)
    # 날짜별 값: 같은 가중으로 하루씩 나눔 (Σ 하루 hazard = 조각 hazard)
    fc_d, fp_d = _fractions_by_day(mo, days)
    day_rows = [{"date": d, "p_day": 1 - (1 - float(p)) ** float(fp), "base_p_day": 1 - (1 - float(pb)) ** float(fp),
                 "expected_day": e * float(fc), "base_expected_day": e_base * float(fc)}
                for d, fc, fp in zip(days, fc_d, fp_d)]
    # 월 확률이 거의 1이면 1 − (1 − p)^f 도 하루마다 ≈ 1 → 예상 탐지일이 부풀려진다. 그 조각은 탐지일 기후값으로 (모델·평년 각각).
    if float(p) >= NEAR_CERTAIN or float(pb) >= NEAR_CERTAIN:
        rate = _day_any_rate(region if region is not None else _REGION_CTX.get("region"), mo, days)
        if rate is not None:
            for r_, rv in zip(day_rows, rate):
                if float(p) >= NEAR_CERTAIN:
                    r_["p_day"] = float(rv)
                if float(pb) >= NEAR_CERTAIN:
                    r_["base_p_day"] = float(rv)
                r_["day_prob_method"] = "detection-day climatology (month p >= 0.98)"
    return {"kind": "monthly", "month": f"{y}-{m:02d}", "days": len(days), "fraction_count": f_c, "fraction_prob": f_p, "fraction_method": how,
            "_daily": day_rows,
            "p_any": 1 - (1 - float(p)) ** f_p, "base_p": 1 - (1 - float(pb)) ** f_p,
            "expected": e * f_c, "q10": q10 * f_c, "q90": q90 * f_c, "base_count": e_base * f_c, "count_source": src, "count_skill": count_skill,
            "model": model, "route": route, "horizon_months": h, "skill": skill}


@lru_cache(maxsize=1)
def _skill30() -> dict:
    return json.loads(SVC["skill_30d"].read_text(encoding="utf-8")) if SVC["skill_30d"].exists() else {}


def _unit30(region, rw: pl.DataFrame) -> str:
    """30일 검증표의 단위: 격자 한 칸, aoi 등급, 그 밖(bbox·폴리곤·kr)은 크기가 가까운 등급."""
    if rw.height == 1 and float(rw["w"][0]) >= 0.999:
        return "0.5° grid cells"
    if isinstance(region, str):
        lv = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "level"]).filter(pl.col("aoi_id") == region)["level"]
        if lv.len() and lv[0] in UNIT_30D:
            return UNIT_30D[lv[0]]
    n = float(rw["w"].sum())        # 0.5° 격자 수로 가장 비슷한 등급 (시군구 ~0.5, 시도 ~4, adm1 ~수십, 나라 ~수백)
    return "Korea sgg" if n < 1.5 else "Korea sido" if n < 10 else "adm1" if n < 150 else "countries" if n < 3000 else "continents"


def today_kst() -> date:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def forecast(region, start: str | date | None = None, end: str | date | None = None, as_of: str | date | None = None,
             horizon_days: int | None = None, today: str | date | None = None) -> dict:
    """region: aoi_id | (서,남,동,북) | GeoJSON. 기간: start~end, 또는 horizon_days(예: 30) → 오늘(KST)부터 horizon_days일.
    as_of(마지막 완결 관측일)와 오늘 사이가 비어 있어도, 오늘은 as_of에서 lead_days_from_as_of일 앞선 예측으로 계산한다."""
    mo = _monthly()
    dly = _daily()
    if as_of:
        as_of = date.fromisoformat(str(as_of))
    else:   # 일 단위가 더 자주 갱신되므로 둘 중 늦은 기준일
        as_of = max([mo["as_of_date"][0]] + ([dly["as_of_date"][0]] if dly is not None else []))
    if horizon_days:
        t = date.fromisoformat(str(today)) if today else today_kst()
        d0 = max(date.fromisoformat(str(start)) if start else t, as_of + timedelta(1))
        d1 = d0 + timedelta(int(horizon_days) - 1)
    else:
        if start is None or end is None:
            raise ValueError("start·end 또는 horizon_days가 필요합니다")
        d0, d1 = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if d1 < d0 or d0 <= as_of:
        raise ValueError(f"기간은 기준일({as_of}) 다음 날부터, 시작 ≤ 끝이어야 합니다")
    origin_y, origin_m = map(int, mo["origin_month"][0].split("-"))
    rw, desc = region_weights(region)
    group = _group(region)
    _REGION_CTX["region"] = region
    pieces, used_days, daily_rows = [], set(), []
    # 1) 일 단위 (일 단위 파일에 있는 날짜 = 일 단위 기준일 + 1 ~ 7일)
    if dly is not None:
        have = set(dly["date"].unique().to_list())
        days = [d0 + timedelta(i) for i in range((d1 - d0).days + 1) if (d0 + timedelta(i)) in have]
        if days:
            dd = dly.filter(pl.col("date").is_in(days)).join(rw, on="g")
            for (dday,), g in sorted(dd.group_by("date"), key=lambda x: x[0][0]):
                w = g["w"].to_numpy()
                p = 1 - np.exp(-(w * -np.log1p(-np.clip(g["p_any"].to_numpy(), 0, 0.999999))).sum())
                pb = 1 - np.exp(-(w * -np.log1p(-np.clip(g["base_p"].to_numpy(), 0, 0.999999))).sum())
                e = float((w * g["mu"].to_numpy()).sum())
                eb = float((w * g["base_count"].to_numpy()).sum())
                # 일 단위는 격자에서만 검증했다 → 여러 칸을 모은 지역은 skill 상한 low
                sk = g["skill"][0] if (rw.height == 1 or g["skill"][0] == "none") else "low"
                pieces.append({"kind": "daily", "date": str(dday), "p_any": p, "base_p": pb, "expected": e,
                               "q10": float(g["count_ratio_q10"][0]) * e, "q90": float(g["count_ratio_q90"][0]) * e, "base_count": eb,
                               "model": g["model"][0], "horizon_days": int((dday - g["as_of_date"][0]).days), "skill": sk,
                               "route": "daily grid hazard sum (region not separately validated)" if rw.height > 1 else "daily single cell",
                               # 일 건수: 기후값 대비 홀드아웃 WAPE 23~31% 개선 (reports/planner_daily_count_baseline.csv)
                               "count_source": "daily_model_mu", "count_skill": "high" if rw.height == 1 else "low"})
                daily_rows.append({"date": dday, "p_day": p, "base_p_day": pb, "expected_day": e, "base_expected_day": eb, "skill": sk,
                                   "source": "daily model"})
                used_days.add(dday)
    # 2) 월 단위 (남은 날짜를 달별로)
    rest = [d0 + timedelta(i) for i in range((d1 - d0).days + 1) if d0 + timedelta(i) not in used_days]
    by_month: dict[tuple[int, int], list[date]] = {}
    for d in rest:
        by_month.setdefault((d.year, d.month), []).append(d)
    for (y, m), mdays in sorted(by_month.items()):
        h = (y * 12 + m - 1) - (origin_y * 12 + origin_m - 1)
        if h < 1:
            # 3) 진행 중인 달의 나머지: 기준선(예년 같은 달)으로
            #    같은 달력 월의 기준선은 12개월 뒤(내년 같은 달) 행의 base_p·base_count_median에 들어 있다
            pc = _month_piece(rw, origin_y + 1, origin_m, mdays, 12, group, None)
            for r_ in pc.get("_daily", []):
                r_.update(p_day=r_["base_p_day"], expected_day=r_["base_expected_day"])
            pc = {**pc, "month": f"{y}-{m:02d}", "p_any": pc["base_p"], "expected": pc["base_count"], "q10": 0.0, "q90": 2 * pc["base_count"],
                  "model": "baseline (current month, no forecast)", "horizon_months": 0, "skill": "none",
                  "note": "진행 중인 달의 나머지는 예년 같은 달 기준선을 비율로 사용"}
        elif h > 12:
            raise ValueError(f"{y}-{m:02d}은 기준 달({origin_y}-{origin_m:02d})에서 12개월을 넘습니다")
        else:
            pc = _month_piece(rw, y, m, mdays, h, group, region)
        for r_ in pc.pop("_daily", []):
            daily_rows.append({**r_, "skill": pc["skill"],
                               "source": "baseline (current month)" if h < 1 else f"monthly model h={h} x daily weights"})
        pieces.append(pc)
    p_any = 1 - float(np.prod([1 - x["p_any"] for x in pieces]))
    base_p = 1 - float(np.prod([1 - x["base_p"] for x in pieces]))
    skills = [x["skill"] for x in pieces]
    skill = "none" if "none" in skills else ("low" if "low" in skills else "high")
    count_skill = _worst([x.get("count_skill", "none") for x in pieces])
    n_days = (d1 - d0).days + 1
    skill_basis = "worst piece"
    value_source = "model"
    if 28 <= n_days <= 31 and (d0 - as_of).days <= 3 and _skill30():
        # 30일 질의(화면 기본): 조각별 최솟값 대신 "기준일 다음 날부터 30일" 홀드아웃 검증의 지역 등급별 skill
        u = _unit30(region, rw)
        ent = _skill30().get(u)
        if ent:
            skill, count_skill = ent["skill"], ent["count_skill"]
            skill_basis = f"30-day holdout validation 2024-2025 ({u}; reports/planner_30d_validation.csv)"
            # "평년을 못 이기면 평년을 보여 준다": skill none → 평년 값, high/low → 등급별 isotonic으로 보정한 모델 값.
            # 평년 값도 같은 등급의 평년용 isotonic으로 보정한다 (보정 전 합성은 지역에서 과대).
            iso = ent.get("iso") or {}
            cal_base = float(np.interp(base_p, iso["base_x"], iso["base_y"])) if iso.get("base_x") else base_p
            _scale_days(daily_rows, "base_p_day", cal_base)
            if skill == "none":
                for r_ in daily_rows:
                    r_["p_day"] = r_["base_p_day"]
                p_any, value_source = cal_base, "baseline"
            else:
                cal = float(np.interp(p_any, iso["x"], iso["y"])) if iso.get("x") else p_any
                _scale_days(daily_rows, "p_day", cal)
                p_any, value_source = cal, "model_calibrated"
            base_p = cal_base
    g = _grid().join(rw, on="g")
    pers = g.filter("persistent_source")
    daily_rows.sort(key=lambda r_: r_["date"])
    daily_out = [{"date": str(r_["date"]), "lead_days_from_as_of": (r_["date"] - as_of).days, "p_day": round(float(r_["p_day"]), 4),
                  "base_p_day": round(float(r_["base_p_day"]), 4), "expected_count_day": round(float(r_["expected_day"]), 3), "skill": r_["skill"],
                  "source": r_["source"]} for r_ in daily_rows]
    return {
        "product": "FireOutlook Planner", "region": desc, "region_group": group, "start": str(d0), "end": str(d1), "as_of": str(as_of),
        "lead_days_from_as_of": {"first": (d0 - as_of).days, "last": (d1 - as_of).days},
        "p_any": round(p_any, 4), "expected_count": round(sum(x["expected"] for x in pieces), 2),
        "expected_count_q10": round(sum(x["q10"] for x in pieces), 2), "expected_count_q90": round(sum(x["q90"] for x in pieces), 2),
        "expected_fire_days": round(float(sum(r_["p_day"] for r_ in daily_rows)), 2),
        "baseline": {"p_any": round(base_p, 4), "count": round(sum(x["base_count"] for x in pieces), 2),
                     "expected_fire_days": round(float(sum(r_["base_p_day"] for r_ in daily_rows)), 2),
                     "definition": "예년 같은 기간 (2012~ 같은 달 비율·중앙값)"},
        "skill": skill, "count_skill": count_skill, "skill_basis": skill_basis, "value_source": value_source, "models": sorted({x["model"] for x in pieces}),
        "horizons": [x.get("horizon_days", x.get("horizon_months")) for x in pieces],
        "daily": daily_out,
        "pieces": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pieces],
        "persistent_source_cells": {"count": pers.height, "area_share": round(float(pers["w"].sum() / max(g["w"].sum(), 1e-9)), 4),
                                    "note": "공장·가스 플레어 같은 지속 열원 의심 격자가 들어 있으면 값이 높게 나올 수 있음" if pers.height else ""},
        "display_hint": ("거의 확실히 탐지됨 — 예상 건수를 보세요" if p_any >= 0.98 else ""),
        "suspected_industrial_sgg": _industrial(region),
        "unit": "S-NPP 기준 탐지 수 (expected_count)", "notice": NOTICE,
    }


def _scale_days(rows: list[dict], key: str, target: float) -> None:
    """날짜별 확률을 hazard 비율로 함께 조정해 1 − Π(1 − p_day) = target 이 되게 한다 (날짜 사이 모양은 그대로)."""
    h = np.array([-np.log1p(-min(float(r_[key]), 0.999999)) for r_ in rows])
    comp = 1 - float(np.prod([1 - float(r_[key]) for r_ in rows]))
    if h.sum() <= 0 or (target >= 0.999 and comp >= 0.999):
        return          # 거의 확실한 경우(큰 지역)는 날짜별 값을 그대로 둔다 (hazard 상한 때문에 하루 값이 깎이지 않게)
    ht = -np.log1p(-min(float(target), 0.999999))
    f = ht / h.sum()
    for r_, hv in zip(rows, h):
        r_[key] = float(1 - np.exp(-hv * f))


def _worst(sk: list[str]) -> str:
    return "none" if "none" in sk else ("low" if "low" in sk else "high")


def _industrial(region) -> list[str]:
    """한국 시군구 산업 열원 의심(FireOutlook Monthly – District의 suspected_industrial) 중 이 aoi에 드는 것."""
    f = PROCESSED_DATA_DIR / "aoi_occurrence_nextmonth_latest.parquet"
    if not isinstance(region, str) or not f.exists():
        return []
    ind = pl.read_parquet(f, columns=["aoi_id", "suspected_industrial"]).filter("suspected_industrial")["aoi_id"].to_list()
    if region.startswith("SGG_"):
        return [a for a in ind if a == region]
    if region.startswith("SIDO_"):
        par = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "parent_aoi"])
        return par.filter(pl.col("aoi_id").is_in(ind), pl.col("parent_aoi") == region)["aoi_id"].to_list()
    return ind if region in ("KR", "ADM0_KOR") else []


def main(argv=None):
    import time
    ap = argparse.ArgumentParser(description="FireOutlook Planner: 지역 + 기간 예측")
    ap.add_argument("--region", required=True, help="aoi_id, 또는 서,남,동,북")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--horizon-days", type=int, help="예: 30 → 오늘(KST)부터 30일")
    ap.add_argument("--today", help="오늘 날짜 고정 (시험용)")
    ap.add_argument("--as-of")
    a = ap.parse_args(argv)
    reg = tuple(map(float, a.region.split(","))) if "," in a.region else a.region
    t0 = time.perf_counter()
    r = forecast(reg, a.start, a.end, a.as_of, horizon_days=a.horizon_days, today=a.today)
    r["elapsed_s"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
