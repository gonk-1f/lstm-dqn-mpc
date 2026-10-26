param([string]$OutputDir = '', [string]$ResumeCheckpoint = '', [int]$StopAfterRound = 0)
$ErrorActionPreference = 'Stop'
$v6Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $v6Root
$env:PYTHONPATH = Join-Path $v6Root 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
if (-not $OutputDir) {
    $OutputDir = if ($ResumeCheckpoint) {
        Split-Path -Parent (Split-Path -Parent $ResumeCheckpoint)
    } else {
        Join-Path $v6Root ('outputs\v6_dqn_1s_seed42_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
}
$v6Arguments = @('-u', '-m', 'v4.v6_training', '--output-dir', $OutputDir)
if ($ResumeCheckpoint) { $v6Arguments += @('--resume', $ResumeCheckpoint) }
if ($StopAfterRound) { $v6Arguments += @('--stop-after-round', $StopAfterRound) }
& 'D:\py\Python3\python.exe' @v6Arguments
if ($LASTEXITCODE -ne 0) { throw "V6 stopped with exit code $LASTEXITCODE" }
