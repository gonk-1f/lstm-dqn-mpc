# 终端命令与执行边界

以下使用本机隔离工作树及已有虚拟环境。正式命令只供后续明确授权后执行；本次没有执行正式 500 episode、长搜索或 Test。

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

## 本次已执行的测试

```powershell
.venv-optuna\Scripts\python.exe scripts\audit_small_train_loads.py --self-check
$testFiles=@(Get-ChildItem tests -Filter 'test_v6*.py' | Sort-Object Name | ForEach-Object FullName)
.venv-optuna\Scripts\python.exe -m pytest @testFiles tests/test_one_plus_reward.py tests/test_inverse_pre_audit.py tests/test_realtime_value_design.py tests/test_inverse_reward_analysis.py tests/test_v2_supervisory_rules.py tests/test_v4_independent_economics.py -q --tb=line
```

来源审计另执行了 `.venv-optuna\Scripts\python.exe scripts\audit_small_train_loads.py`，读取全部 35 Train 并执行至多 5382 个独立一秒反事实核算。没有学习、完整新策略航段或真实 Validation/Test rollout。实际 CLI Smoke 的每个参数、输出、退出码和耗时保存在 [cli_final.json](cli_final.json)，核对结果见 [cli_final_summary.json](cli_final_summary.json)。

## 复现同样的有界 Smoke

下面的 `smoke_next_*` 都必须使用新的空目录；本次实际产物位于 `.tmp/v6_one_plus_cli_20261029_final/`，不要覆盖。所有航段均由程序合成，不读真实 Validation/Test。

```powershell
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --run-dir .tmp/smoke_next_baseline --smoke --reward-mode inverse_one_plus_economic --stop-after-steps 97
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --run-dir .tmp/smoke_next_baseline --resume --reward-mode inverse_one_plus_economic
.venv-optuna\Scripts\python.exe -m v6_sb3.tuning --episode-search --study-dir .tmp/smoke_next_study --smoke --reward-mode inverse_one_plus_economic --trials 1 --episodes-per-trial 2 --learning-starts 4 --validation-every 1 --pruner none --stop-after-steps 37
.venv-optuna\Scripts\python.exe -m v6_sb3.tuning --episode-search --study-dir .tmp/smoke_next_study --smoke --reward-mode inverse_one_plus_economic --trials 1 --episodes-per-trial 2 --learning-starts 4 --validation-every 1 --pruner none --resume
.venv-optuna\Scripts\python.exe -m v6_sb3.tuning --episode-search --study-dir .tmp/smoke_next_study --export-best-parameters .tmp/smoke_next_parameters.json
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --run-dir .tmp/smoke_next_retrained --smoke --reward-mode inverse_one_plus_economic --parameters-json .tmp/smoke_next_parameters.json
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --compare-runs .tmp/smoke_next_baseline .tmp/smoke_next_retrained
```

新模式 Smoke 导入参数必须符合小模型上限；不会静默把导出参数改成别的值。新模式正式训练不要求旧 F 的 acknowledge 标志。

## 后续正式命令：未执行，需另行授权

全新 500 episode 基线：

```powershell
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --run-dir outputs/v6_one_plus_baseline500_new --reward-mode inverse_one_plus_economic --episodes 500 --learning-starts 50000 --train-freq 32 --gradient-steps 1 --target-update-interval 10000 --validation-every 50
```

正式 Optuna 示例预算是 10 Trials、每 Trial 100 episodes；这只是后续命令模板，不表示该预算已获授权。所有 Trial 均使用独立目录和新的 Study。

```powershell
.venv-optuna\Scripts\python.exe -m v6_sb3.tuning --episode-search --study-dir outputs/v6_one_plus_study_new --reward-mode inverse_one_plus_economic --trials 10 --episodes-per-trial 100 --learning-starts 50000 --validation-every 25
.venv-optuna\Scripts\python.exe -m v6_sb3.tuning --episode-search --study-dir outputs/v6_one_plus_study_new --export-best-parameters outputs/v6_one_plus_best_new.json
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --run-dir outputs/v6_one_plus_tuned500_new --reward-mode inverse_one_plus_economic --parameters-json outputs/v6_one_plus_best_new.json --episodes 500 --learning-starts 50000 --validation-every 50
.venv-optuna\Scripts\python.exe -m v6_sb3.episode_training --compare-runs outputs/v6_one_plus_baseline500_new outputs/v6_one_plus_tuned500_new
```

同一源码、依赖、数据、配置下续训使用 `--resume`，搜索续训还须保留原 Trials、episode 预算、seed 和其他身份参数。读取历史旧模式 checkpoint 时，回到它的原提交和依赖；不要更改旧 `identity.json` 或给旧 Replay 改标签。参数导出与导入也会验证源码身份。没有提供或执行 Test 命令。
