# Configuration for the single baseline already launched as PID 22996.
# This file defines commands only. It does NOT start or resume on import.
$ddqnRoot = 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-increment7-zhou'
Set-Location -LiteralPath $ddqnRoot
$env:PYTHONPATH = Join-Path $ddqnRoot 'src'
$ddqnPython = 'D:\py\Python3\python.exe'
$ddqnRunDir = Join-Path $ddqnRoot 'outputs/v4_independent35_m3_delta7_zhou_td1_per_interleaved_seed42_100r_20261020'
$ddqnArgs = @(
    '--dataset-mode','independent','--data-root','data/processed/operating_dataset_independent_v2',
    '--state-mode','minimal3','--action-mode','increment7','--reward-cost-mode','zhou_joint',
    '--n-step','1','--rounds','100','--seed','42','--beta-soc','200',
    '--reward-scale','0.001','--reward-feedback','original',
    '--replay-sampling','prioritized','--per-alpha','0.6','--per-beta','1','--per-epsilon','1e-6',
    '--cadence','replay32','--update-timing','interleaved',
    '--failure-terminal-quota','2','--failure-penalty-scale','1','--episode-credit-scope','voyage',
    '--unknown-replay-mode','observed_nstep',
    '--experiment-id','V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-INTERLEAVED-S42-20261020',
    '--diagnostic-rounds','1,10,20,30,40,50,60,70,80,90,100'
)
function Start-NewBaseline {
    if (Test-Path -LiteralPath $ddqnRunDir) { throw 'Output already exists: do not duplicate or overwrite.' }
    & $ddqnPython -m v4.feedback_study --output-dir $ddqnRunDir @ddqnArgs
}
function Resume-Baseline {
    $active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*v4.feedback_study*' -and $_.CommandLine -like '*per_interleaved_seed42_100r_20261020*' }
    if ($active) { throw 'Baseline is still running. Do not launch a second writer.' }
    $checkpoint = Join-Path $ddqnRunDir 'training_state_latest.pt'
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw 'No completed-round checkpoint exists yet.' }
    & $ddqnPython -m v4.feedback_study --output-dir $ddqnRunDir @ddqnArgs --resume-from $checkpoint
}
# After stopping the original process, use Resume-Baseline only if needed.
# No B/C training commands are defined: their CLI interfaces are not implemented,
# and preparation is gated on completion/reconciliation of this baseline.
