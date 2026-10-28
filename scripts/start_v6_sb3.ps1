param([string]$OutputDir = '', [string]$ResumeCheckpoint = '')
$ErrorActionPreference = 'Stop'
$sb3Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $sb3Root
$env:PYTHONPATH = Join-Path $sb3Root 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$sb3Python = Join-Path $sb3Root '.venv\Scripts\python.exe'
$sb3Arguments = @('-u', '-m', 'v6_sb3.training')
if ($OutputDir) { $sb3Arguments += @('--output-dir', $OutputDir) }
if ($ResumeCheckpoint) { $sb3Arguments += @('--resume', $ResumeCheckpoint) }
& $sb3Python @sb3Arguments
if ($LASTEXITCODE -ne 0) { throw "SB3 training exited with code $LASTEXITCODE" }
