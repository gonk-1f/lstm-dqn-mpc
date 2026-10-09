# V4 方案A阶段一：Train/Greedy统计勘误核查（待完整归档执行）

## 当前状态

起点 `8a4022b9d1e1af61fa94c759193c34eff8c013a1`。仅添加只读可复算脚本 `scripts/audit_v4_round40_train_alignment.py`，没有更改控制、奖励、经济账本或数据，没有启动训练或读取Test。**阶段二未获通过，不得实施保守门控。**

## 已核查并需解释的差异

冻结Train的30个模式CSV文件，以 `mode=onboard` 和原始 `load_total_kw` 严格判断，汇总为18,448个ONBOARD点、其中30个严格等于零、另有457个落在开区间(0,10) kW。30个零负载点分别出现在每个Train样本起点；这与P1“第40轮高SOC、FC=0时有2057个零负载步”的表述不能同时成立。前述汇总经GitHub模式文件读取核算，**尚未在当前执行环境运行正式`FormalTrainingDataset.load_train()`交叉验证**。

P0/P1归档另称：高SOC且FC=0共2607步，2372步只有0合法，235步允许正功率；启动326次，其中航段首次62次、同段重启264次。完整第40轮贪婪轨迹保存在`docs/results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/round_040_train_trajectories.json.gz`。当前可用GitHub连接器无法解码该gzip文件，**所以尚不能独立复现这些归档数值，也未定位根因**。此时不能断言为哪一种字段混用、索引错位或真实数据问题。

## 复核入口

从仓库根目录在完整、已安装项目依赖且包含原归档的本地worktree运行：

```bash
python scripts/audit_v4_round40_train_alignment.py --output /tmp/v4_round40_train_alignment.json
```

脚本先核对3个冻结manifest的SHA-256，只调用`FormalTrainingDataset.open(...).load_train()`；对照第40轮Greedy每个样本的已执行transition，记录源数据行号、轨迹索引、时间、总负载、FC与电池功率、SOC，按正式`feasible_fc_actions()`重新判别高SOC零出力掩码，并核对326/62/264启动统计。源负载与归档不匹配或P0/P1数字不一致时返回退出码2。所有输出写到另设JSON，不覆盖冻结数据或归档。Test payload打开数需要维持0。

**注意**：请先审查结果中的`source_archive_load_mismatch_examples`并检查行号/航段对齐。脚本采用每个样本已执行transition与该样本ONBOARD源行依次一一对应的假设；若存在整段跳过、失败恢复或其他采样索引变化，应以正式执行日志校正映射后再得结论，切勿机械解释为数据损坏。

## 阻断条件与下一步

1. 本地运行脚本并保存控制台结果与JSON，核对30个样本及20个具体对应行。
2. 如果源数据与归档不一致，定位字段映射、行号或反事实重建的实际原因；针对确证错误撰写对P0/P1报告的勘误，逐项说明2372/235/2057和326/62/264是否保留。
3. 只有输入字段、当前物理掩码及启停事件统计复核可靠，才依据已审批方案A实现当前/后继策略候选掩码一致的保守门控，并运行正式回归。
4. 本次**未进入阶段二**，没有进行正式测试或训练。不能将脚本通过语法检查当作完成原始数据/轨迹比对。

