"""Shared FIRMS platform constants and the monthly feature projection."""

BLOCK_KM = 250
PLATFORMS = ("Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21")


def monthly_features_query(source, block_km=BLOCK_KM):
    return f"""
        WITH cell_day AS (
            SELECT day, x, y, platform,
                   count(*)::INT AS pixels, avg(pixel_area) AS pixel_area,
                   avg(nrt) AS nrt_share, avg(daytime) AS day_share,
                   sum(frp) AS frp_sum, max(frp) AS frp_peak
            FROM {source} GROUP BY day,x,y,platform
        )
        SELECT date_trunc('month',day)::DATE AS month,
               floor(x / {block_km})::INT AS bx, floor(y / {block_km})::INT AS grid_by,
               platform, count(*) AS cell_days, sum(pixels)::BIGINT AS pixels,
               count(DISTINCT day)::INT AS active_days, avg(pixel_area) AS pixel_area,
               avg(nrt_share) AS nrt_share, avg(day_share) AS day_share,
               sum(frp_sum) AS frp_sum, avg(frp_peak) AS mean_cell_day_peak
        FROM cell_day GROUP BY ALL
    """
