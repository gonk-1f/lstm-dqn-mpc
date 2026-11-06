# Local official source checkouts

These shallow checkouts are ignored by the parent repository. Their upstream
algorithm and training sources are unmodified.

| Directory | Upstream tag | Commit | Usage |
| --- | --- | --- | --- |
| `stable-baselines3/` | v2.9.2 | `0cdcd200deef06b9ad0af72f6ddfb07c06c6df39` | Source reference; installed SB3 wheel remains in use |
| `rl-baselines3-zoo/` | v2.9.2 | `61956d90d9f8ac126043ed4024efa12a462d97e1` | Editable install in the existing `.venv` |

```powershell
git clone --depth 1 --branch v2.9.2 https://github.com/DLR-RM/stable-baselines3.git third_party/stable-baselines3
git clone --depth 1 --branch v2.9.2 https://github.com/DLR-RM/rl-baselines3-zoo.git third_party/rl-baselines3-zoo
.\.venv\Scripts\python.exe -m pip install -e third_party/rl-baselines3-zoo
```

Use the TD3 experiment README for the tested installation versions and commands.
