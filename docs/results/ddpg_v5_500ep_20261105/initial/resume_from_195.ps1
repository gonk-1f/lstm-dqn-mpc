# Resume the checkpoint safely saved on 2026-11-05 at episode 195.
# The interrupted episode restarts; model/Adam/Replay/global counters continue.
$ddpgProject = 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
Set-Location -LiteralPath $ddpgProject
$env:PYTHONPATH = "$ddpgProject;$ddpgProject\src"

@'
from functools import partialmethod
from experiments.sb3_manual.ddpg_aggregate.huber_ddpg import HuberDDPG

# Preserve the saved step clock and avoid repeating the 5000-step warmup.
HuberDDPG.learn = partialmethod(HuberDDPG.learn, reset_num_timesteps=False)

from rl_zoo3.train import train
train()
'@ | & "$ddpgProject\.venv\Scripts\python.exe" -u - `
    --algo ddpg `
    --env AggregateShipDDPG-v0 `
    --gym-packages experiments.sb3_manual.ddpg_aggregate.zoo_register `
    --conf-file experiments/sb3_manual/ddpg_aggregate/zoo_ddpg.yml `
    --device cuda `
    --eval-freq -1 `
    --seed 42 `
    --trained-agent outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_4/AggregateShipDDPG-v0.zip `
    --n-timesteps 18193504 `
    --log-folder outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3_resume `
    --save-replay-buffer

if ($LASTEXITCODE -ne 0) {
    throw "DDPG resume exited with code $LASTEXITCODE"
}
