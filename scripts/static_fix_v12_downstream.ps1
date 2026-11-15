# v1.2 고정 열원 제외 뒤 파생 산출물 다시 만들기 (모델 재학습 없음). 한 단계가 실패하면 멈춘다.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location C:\Planetarium-AI
$steps = @(
    @("-m", "planetarium.fire_events", "all"),
    @("-m", "planetarium.spread_nowcast_model", "latest"),
    @("-m", "planetarium.recurrence", "panel"),
    @("-m", "planetarium.recurrence", "latest"),
    @("-m", "planetarium.recurrence", "sites")
)
foreach ($s in $steps) {
    Write-Host "===== $($s -join ' ') ====="
    & uv run python @s
    if ($LASTEXITCODE -ne 0) { Write-Host "[실패] $($s -join ' ')"; exit 1 }
}
Write-Host "[done] static_fix_v12_downstream"
