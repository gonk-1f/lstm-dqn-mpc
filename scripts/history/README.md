# 历史 V4 PER 离线研究脚本

`v4_per_frozen_probe.py` 与 `v4_per_jointfit_state_probe.py` 保留原始内容，归档用于复查
既有 V4 PER 拟合/状态研究。它们不是当前 V6 SB3 训练入口。两者均没有 `__file__`、
`ROOT` 或相对其所在目录加载其他脚本的逻辑；参数仍相对调用者的工作目录解释。

从仓库根目录手动调用时继续设置原来的 `PYTHONPATH=src`：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/history/v4_per_frozen_probe.py --help
python scripts/history/v4_per_jointfit_state_probe.py --help
```

`--help` 之外的调用可能加载历史 Train 资料/模型并进行离线拟合，需要显式提供原有参数。
本次归档没有运行这两个脚本，没有读取 checkpoint、Replay 或数据内容，也没有删除它们的依赖。
完整路径映射及恢复步骤见 `docs/audits/v6_optuna_20261028/cleanup_manifest.json`。
