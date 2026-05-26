# TencentUniRec-TAAC2026

[English README](README.md)

TAAC x KDD Cup 2026 腾讯广告算法大赛工业组的非官方方案记录与参考实现。

> 这是一个从私有参赛工作区整理出来的干净开源版本。仓库包含代码和技术文档，但**不包含官方数据、checkpoint、私有日志、平台路径、最终提交的精确 recipe 或任何非公开材料**。

## 成绩

| 项目 | 数值 |
|---|---:|
| 赛道 | 工业组 |
| 队伍排名 | 35/689 |
| 百分位 | Top 5.1% |
| 最佳 public AUC | 0.851365 |
| 最终精确 recipe | 不在公开仓库披露 |
| 任务 | 大规模广告 pCVR 预估 |

## 项目亮点

- 从 cleaned baseline 出发，完整记录从 baseline 到最终版本的演进过程。
- 面向工业广告推荐数据的序列建模和 pCVR 预估。
- 使用 RankMixer 风格的非序列 sparse/dense feature tokenization。
- 引入 time-aware sequence bucket 和 public-tail-oriented validation。
- 使用 click/conversion multi-task learning 缓解转化标签稀疏问题。
- 使用 auxiliary validation windows 和 leaderboard-correlation analysis 辅助模型选择。
- 记录关键实验主线、负结果和验证体系校准经验。

## 从 Baseline 到最终版本

| 阶段 | 主要变化 |
|---|---|
| Early cleaned baseline | 初始 HyFormer 风格 baseline，快速序列编码器 |
| Time bucket correction | 修正 per-domain sequence recency 建模 |
| Stronger baseline | 更可靠的时间验证和 auxiliary diagnostics |
| Fresh-tail family | 更接近 public-adjacent tail 的训练/验证设置 |
| MTL family | click/conversion multi-task regularization |
| Final selected family | 基于验证证据和有限 public eval 选择的最终家族 |

核心经验是：public 分数的提升更多来自**验证体系对齐和训练目标校准**，而不是简单堆更大的模型或更复杂的模块。

## 仓库结构

```text
.
├── src/                       # 训练、推理、数据集、模型、trainer、EDA 工具
├── configs/                   # 关键里程碑配置
├── scripts/                   # 本地示例命令
├── docs/                      # 技术报告、验证体系和复盘文档
├── experiments/               # 脱敏后的实验结果摘要
└── examples/                  # 示例说明；不包含官方数据
```

## 包含内容

- 清洗后的比赛模型代码。
- 脱敏后的公开参考配置。最终提交的精确运行参数不在公开仓库披露。
- 关于 temporal validation、model selection、final sprint 的技术文档。
- 简洁的时间线和脱敏负结果总结。

## 不包含内容

- 官方训练 / 测试数据。
- Checkpoints 或模型输出。
- 私有平台日志、leaderboard 截图、用户 ID、平台路径。
- 任何账号、凭证或平台运行时状态。

如果要运行代码，请按照比赛规则获取或挂载官方数据，并通过 `--data_dir /path/to/data` 指定路径。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/train_reference_example.sh /path/to/official/data
```

该命令是可读的参考模板，不是最终提交的精确 recipe。精确 public 分数依赖官方环境、完整数据集、比赛评估服务和私有运行记录。

## 推荐阅读顺序

1. [赛题概览](docs/01_competition_overview.md)
2. [方案报告](docs/02_solution_report.md)
3. [时间验证体系](docs/03_temporal_validation.md)
4. [实验摘要](docs/04_experiment_summary.md)
5. [项目时间线](docs/05_timeline.md)
6. [英文技术报告](docs/06_technical_report.md)
7. [中文复盘文章](docs/07_chinese_retrospective.md)

## 如何引用

如果引用本仓库，可以写成：

```text
TencentUniRec-TAAC2026: TAAC x KDD Cup 2026 工业组非官方方案记录与参考实现。
Rank 35/689, Top 5.1%, Public AUC 0.851365. 最终精确 recipe 不在公开仓库披露。
```

## 免责声明

本项目不是 Tencent、TAAC 或 KDD Cup 官方仓库。所有比赛名称归对应主办方所有。代码和文档仅用于学习、复盘和作品集展示。源码、许可和数据处理说明见 [`NOTICE.md`](NOTICE.md)。
