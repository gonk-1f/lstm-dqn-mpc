# 旧 V6 训练安全暂停记录

按用户要求写入既有 `pause.request`，等待当前100000步分段正常保存并退出；没有强杀、重启、额外训练或测试。控制器最终为 `paused_by_request`，训练为 `paused`，Python退出码0；训练进程和自动续跑控制器均已退出。

- 已完成477/500个正式Episode；warmup50000步、formal3150064步、总3200064步、Adam98439次。
- 当前恢复入口仍是原实验 `outputs/v6_one_plus_baseline500_seed42/latest.json`，指向 `checkpoint_000043`。最后物理前缀、模型、优化器、Replay和RNG由原训练器按既有逻辑保存；本次没有运行恢复测试。
- 暂停请求时现存的 `checkpoint_000041`、`checkpoint_000042` 已先复制，最终新生成的 `checkpoint_000043` 也已复制。备份共3代、12个文件、240761620字节；原运行目录按既有轮换规则保留42/43，41仍保存在本次独立备份中。更早已经轮换删除的历史checkpoint不在本次现存备份范围内。
- 全部备份位于 `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\outputs\v6_one_plus_pause_backup_20261030_143440`。`before/`、`after/`另存根目录元数据、日志和best_model；checkpoint副本与其源文件SHA256一致。
- 未修改旧训练源码、配置、身份或依赖，未动CartPole/Zoo用户文件。暂停标记保留，不会自动开始下一段。
- 本提交仅含轻量暂停记录和文件哈希。模型、Replay及备份二进制留在本机，不推送到GitHub。此次请求运行测试数为0。

完整时间、状态与文件清单见 [pause_record.json](pause_record.json)。
