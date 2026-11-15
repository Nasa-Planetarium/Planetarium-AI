# v1.2 고정 열원 점 제외 → AOI·비교·검증·정렬 (2026-11-15). 한 단계가 실패하면 멈춘다.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location C:\Planetarium-AI
$steps = @(
    @("-m", "planetarium.static_exclusion"),
    @("-m", "planetarium.harmonize", "--skip-cells", "--skip-boundaries"),
    @("-m", "planetarium.analysis"),
    @("-m", "planetarium.verify_hotspots"),
    @("-c", "from planetarium.fetch_nrt import _resort_outputs; _resort_outputs()")
)
foreach ($s in $steps) {
    Write-Host "===== $($s -join ' ') ====="
    & uv run python @s
    if ($LASTEXITCODE -ne 0) { Write-Host "[실패] $($s -join ' ')"; exit 1 }
}
Write-Host "[완료] static_fix_v12"
