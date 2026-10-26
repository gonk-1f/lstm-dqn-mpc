param([int]$TrainingProcessId, [string]$RunDirectory)
$ErrorActionPreference = 'Stop'
$v6Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $v6Root 'src'
$v6LastRound = -1
do {
    $v6Alive = $null -ne (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue)
    $v6ReportPath = Join-Path $RunDirectory 'report.json'
    if (Test-Path -LiteralPath $v6ReportPath) {
        $v6Report = Get-Content -LiteralPath $v6ReportPath -Raw | ConvertFrom-Json
        if ($v6Report.completed_rounds -ne $v6LastRound -or -not $v6Alive) {
            & 'D:\py\Python3\python.exe' (Join-Path $PSScriptRoot 'report_v6.py') $RunDirectory
            $v6LastRound = $v6Report.completed_rounds
        }
    }
    if ($v6Alive) { Start-Sleep -Seconds 30 }
} while ($v6Alive)
