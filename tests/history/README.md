# 历史 DQN 专用测试

仅将确认属于历史 H1/H2 DQN 训练研究的 CLI 测试移至此目录；原有测试内容保留，
只把 `ROOT` 的父目录层数从 1 改为 2，以适配归档路径。默认 `pytest` 不收集 `history`。

需要复查历史实现时，从仓库根目录显式执行：

```powershell
python -m pytest tests/history/test_v2_history_study_cli.py -q
```

共享物理、经济、数据完整性、checkpoint、合法动作 mask 和 TD 回归继续放在 `tests/`。
完整前后清单及恢复映射见 `docs/audits/v6_optuna_20261028/cleanup_manifest.json`。
