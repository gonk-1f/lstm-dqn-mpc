param(
    [string]$OutputDir = '',
    [string]$ResumeFrom = ''
)
$ErrorActionPreference = 'Stop'
$v5Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $v5Root
$env:PYTHONPATH = Join-Path $v5Root 'src'
$v5Python = 'D:\py\Python3\python.exe'
if (-not (Test-Path -LiteralPath $v5Python -PathType Leaf)) {
    throw "Python interpreter missing: $v5Python"
}
if ($ResumeFrom) {
    $ResumeFrom = (Resolve-Path -LiteralPath $ResumeFrom).Path
    if (-not $OutputDir) { $OutputDir = Split-Path -Parent $ResumeFrom }
} elseif (-not $OutputDir) {
    $OutputDir = Join-Path $v5Root ('outputs\v5_economic4_delta7_td1_per_interleaved_seed42_100r_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$v5Args = @(
    '--dataset-mode', 'independent',
    '--data-root', (Join-Path $v5Root 'data\processed\operating_dataset_independent_v2'),
    '--experiment-id', 'V5-ECON4-DELTA7-TD1-PER-INTERLEAVED-S42',
    '--state-mode', 'load_delta4', '--action-mode', 'increment7',
    '--reward-cost-mode', 'zhou_economic_only', '--beta-soc', '0',
    '--n-step', '1', '--rounds', '100', '--seed', '42',
    '--reward-feedback', 'original', '--reward-scale', '0.001',
    '--replay-sampling', 'prioritized', '--per-alpha', '0.6',
    '--per-beta', '1', '--per-epsilon', '1e-6',
    '--cadence', 'replay32', '--update-timing', 'interleaved',
    '--failure-terminal-quota', '2', '--failure-penalty-scale', '1',
    '--episode-credit-scope', 'voyage', '--unknown-replay-mode', 'observed_nstep',
    '--diagnostic-rounds', '1,10,20,30,40,50,60,70,80,90,100'
)
if ($ResumeFrom) { $v5Args += @('--resume-from', $ResumeFrom) }
Write-Host "V5 output: $OutputDir"
& $v5Python -u -m v4.feedback_study --output-dir $OutputDir @v5Args
if ($LASTEXITCODE -ne 0) { throw "V5 process exited with code $LASTEXITCODE" }
