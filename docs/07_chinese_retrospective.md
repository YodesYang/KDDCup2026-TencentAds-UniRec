# 一次工业推荐竞赛复盘

这是一篇关于 TAAC x KDD Cup 2026 腾讯广告算法大赛工业组的非官方复盘。我们最终在工业组取得 **35/689，Top 5.1%**，最佳 public AUC 为 **0.851365**。

最终提交参数、最终验证窗口和 checkpoint 选择记录不在公开仓库披露。

这不是冠军方案，也不会把故事包装成“只差一点就成功”的鸡血文。更真实的情况是：从工程和研究角度看，这次比赛非常有价值，因为它集中暴露了工业推荐任务里最核心的问题之一：

> 模型本身很重要，但在强时间分布偏移下，验证体系往往更重要。

## 1. 赛题是什么

工业组任务是广告推荐中的 pCVR 预估：

```text
pCVR = P(conversion = 1 | ad, user, context)
```

输入包括用户特征、广告/物品特征、用户 dense 特征，以及四个域的用户行为序列。指标是 AUC。

这类任务非常接近真实广告推荐系统：样本稀疏、特征高基数、序列很长、标签有延迟、训练和测试天然存在时间差。

## 2. 从 baseline 开始

我们最早的 cleaned baseline 是一个 HyFormer 风格的序列推荐模型，使用快速序列编码器。

它的结构大致是：

```text
user/item sparse + dense features -> 非序列 token
四个行为序列域                    -> 序列编码器
序列输出 + 非序列 token            -> query decoding / token mixing
最终表示                           -> pCVR 预测
```

这个 baseline 有两个优点：

- 推理速度快，平台上比较稳；
- 架构方向合理，能承接后续的序列建模和特征交互。

但它也有一个很明显的问题：**本地验证集和 public leaderboard 的关系不稳定**。很多看起来 local AUC 提升的改动，到了 public 上并没有收益，甚至明显掉分。

## 3. 第一个重要转折：不要盲信本地 AUC

比赛早期，我们也会自然地看 local validation AUC。但很快发现，这个指标不够可靠。

原因是 public/test 是训练集之后的未来时间窗口，而普通验证切分可能和训练时间高度重叠，或者无法代表 public 分布。

这导致了一个典型问题：

```text
local validation AUC 上升
public AUC 下降
```

于是我们的策略从：

```text
谁 local AUC 高就评谁
```

变成：

```text
local AUC
+ clean window
+ auxiliary validation
+ public 已验证家族的一致性
共同决定是否值得消耗评估额度
```

这也是整个比赛里最重要的认知变化。

## 4. 第二个转折：修正时间桶

在序列推荐里，行为发生的时间和目标广告之间的时间差非常关键。我们修正了 per-domain sequence time bucket，让不同序列域的时间桶更符合各自分布。

这一阶段的启发是：**时间信息有价值，但绝对时间特征很危险**。修正序列 recency 建模是正向的；直接堆绝对时间特征则容易过拟合 public-adjacent 的局部模式。

## 5. 第三个转折：fresh-tail + MTL

真正稳定向前推进的是 fresh-tail 训练/验证设置：训练和选择更靠近 public-adjacent tail 的候选。

MTL 的作用是用 click 任务辅助 conversion 任务，缓解转化标签稀疏带来的训练不稳定。它不是一个巨大跃迁，但在 public-positive 的 fresh-tail 家族里确实带来了稳定收益。

## 6. 为什么要写失败路线

很多复盘只写成功实验，但这次我认为失败实验必须写。因为在这个比赛里，失败路线不是噪音，而是验证体系的一部分。

几个重要负结果：

| 方向 | 经验 |
|---|---|
| 绝对时间特征 | 时间特征很容易拟合局部窗口 |
| History-CVR | 历史统计特征没有按预期泛化 |
| UE / item interaction variants | 高辅助验证分不等于真实收益 |
| Checkpoint averaging | 探索性副产物，没有用于最终选择 checkpoint |
| seed / logloss-only selection | 熟悉 seed 或低 logloss 不能单独作为选择依据 |

这些失败让我们逐渐意识到：不能只看一个漂亮指标，也不能只因为某个方向“听起来像工业经验”就相信它。

## 7. 对工业推荐任务的几点体会

### 7.1 验证集不是形式，它决定你看到的世界

如果验证集和目标分布不一致，模型越强，可能越会过拟合错误方向。

这次最重要的技术资产不是某个模块，而是我们对验证体系的反复校准。

### 7.2 负结果要被认真记录

时间特征、history-CVR、UE interaction、checkpoint averaging、seed rerun，这些方向听起来都合理，但很多没有 public 收益。

把它们记录下来，可以防止团队在最后阶段重复踩坑。

### 7.3 有限评估额度下，比赛变成决策问题

每天只有有限 public eval。因此问题不是“还有什么能试”，而是：

```text
哪个候选最值得消耗一次不可逆的 public evaluation？
```

这更接近真实业务里的上线决策。

## 8. 如果重来一次

如果有更多时间，我会优先做：

- 更系统的 rolling time-window validation；
- delayed-feedback 建模；
- 对 public-like window 的分布距离建模；
- 更稳健的 tiny validation uncertainty estimate；
- 更原则化的 target-aware sequence interaction；
- 更早把失败路线纳入决策表，而不是最后阶段才系统化。

## 9. 结语

这次比赛留下了一个有价值的工程和研究过程：

- 怎么识别验证失效；
- 怎么在复杂特征空间里做科学实验；
- 怎么面对看似合理但不涨分的方向；
- 怎么在有限评估额度下做理性决策；
- 怎么把一次竞赛整理成可复用的技术资产。

这也是这个开源仓库想保留下来的东西。
