# 模型与数据依据

这些文档来自 v2，但物理、退化与经济实现仍被当前 v4 使用；旧版本号不代表可删除。文档数值以当前正式模型代码为准。

## 物理、经济与退化

- [v2_battery_degradation_model](models/v2_battery_degradation_model.md)
- [v2_economic_parameters](models/v2_economic_parameters.md)
- [v2_fc_degradation_model](models/v2_fc_degradation_model.md)
- [v2_plant_configuration](models/v2_plant_configuration.md)
## 数据来源

- [v2_data_provenance](data/v2_data_provenance.md)
- [v2_raw_excel_inventory](data/v2_raw_excel_inventory.md)

## 保留原路径的兼容资料

- [FC 效率来源记录](../v2_fc_efficiency_model.md)：源码 provenance 引用。
- [旧 DQN 状态审核](../v2_dqn_state_audit.md)：旧生成入口固定路径，不是 v4 状态定义。
- [v2 preflight 报告](../v2_preflight_report.md)：必要回归测试直接读取。
- [原清理 manifest](../v2_cleanup_manifest.md)：历史恢复依据，保持原字节。
