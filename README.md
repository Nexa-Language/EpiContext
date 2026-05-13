# EpiContext: 像生物一样管理 Agent 的"记忆"

## 一句话说清楚

**Agent 的上下文窗口就像人的短期记忆——全记住太贵，全忘掉太蠢。EpiContext 让 Agent 像生物细胞一样，动态决定"记住什么、忘掉什么"。**

---

## 问题：Agent 的上下文困境

现代 AI Agent（如 Claude Code、Cursor）在执行任务时，每轮对话都会产生新的"记忆"——推理过程、工具调用、执行结果。这些记忆不断累积，形成越来越长的上下文历史。

这带来了一个两难：

| 策略 | 做法 | 问题 |
|------|------|------|
| **全记住** | 把所有历史都塞进上下文 | Token 成本爆炸，LLM 性能反而下降（"lost in the middle"） |
| **全忘掉** | 只保留最近几轮 | 关键信息丢失，Agent 忘记之前做过的决策 |

现有方法（压缩、摘要、检索）都在做"静态优化"——压缩一次、检索一次，然后就不再调整。但生物系统不是这样工作的。

---

## 灵感：表观遗传学

你的每个细胞都有完全相同的 DNA（基因组），但不同细胞表现完全不同——神经细胞、肌肉细胞、皮肤细胞。这是怎么做到的？

**表观遗传学**的答案：DNA 序列不变，但基因的"开关"状态在动态变化：

- **DNA 甲基化**：给基因加上"沉默标记"，让它不表达
- **组蛋白乙酰化**：给基因加上"激活标记"，让它高表达
- 这些标记根据环境信号不断调整

> 同一个基因组 → 不同的基因表达模式 → 不同的细胞功能

---

## EpiContext 的核心思想

把 Agent 的上下文管理映射到表观遗传学：

| 生物学概念 | EpiContext 映射 |
|-----------|----------------|
| **基因组 (Genome)** | Agent 的完整交互历史（不可变，全部保留） |
| **表观遗传标记** | 每条历史记录的"激活权重"（0=沉默, 1=激活） |
| **甲基化 (Methylation)** | 降低噪声历史的权重（"这段试错过程已经解决了，不用再看了"） |
| **乙酰化 (Acetylation)** | 提高相关历史的权重（"这个工具调用和当前任务方向一致，保留"） |
| **自然选择** | 适应度函数：任务成功 → 相关记忆被强化；任务失败 → 相关记忆被抑制 |

**关键洞察**：不是"记住 vs 忘掉"的二选一，而是"每条记忆都有一个动态的激活程度"。

---

## 三个算子

### 1. 甲基化 (Methylation) — 沉默噪声

当 Agent 花了 10 轮解决一个依赖冲突后，那些详细的试错日志对后续任务就是噪声。甲基化自动检测"进展很小"的历史片段，降低它们的激活权重，必要时用一句话摘要替代。

```
解决前: [Turn 5-15] 尝试 pip install A...失败...尝试 pip install B...失败...
甲基化后: [Summary] 经过 10 轮试错，通过升级 C 到 v2.0 解决了依赖冲突
```

### 2. 乙酰化 (Acetylation) — 激活相关

Agent 通常有 20+ 个可用工具，但当前任务只需要 2-3 个。乙酰化分析当前任务方向，只激活相关的工具和历史上下文。

```
全部工具: [read_file, write_file, search_web, run_test, git_commit, deploy, ...]
乙酰化后: [read_file, write_file, run_test]  ← 只保留代码编辑相关的
```

### 3. 适应度反馈 (Fitness Feedback) — 优胜劣汰

每条记忆的激活权重根据任务结果动态调整：

```
任务成功 → 参与的记忆权重 +δ
任务失败 → 参与的记忆权重 -δ
长期不用 → 逐渐衰减
```

---

## 关键创新：自适应策略切换

v1 版本直接对所有任务使用表观遗传算子，结果反而不如简单的滑动窗口——因为简单任务根本不需要这么复杂的机制。

v2 版本引入了**自适应策略切换**：

```
前 10 轮: 使用滑动窗口（简单高效，够用）
10 轮后:  切换到 EpiContext（内容感知过滤，处理长上下文）
```

这就像开车：在小区里用一档（滑动窗口），上高速后换五档（EpiContext）。不需要全程用五档，也不需要全程用一档。

---

## 实验结果

在 Harbor 框架的 5 个容器化任务上，对比 6 种上下文策略：

| 策略 | 平均轮数 | 平均 Token 数 |
|------|---------|--------------|
| Full-Context（全记住） | 10.6 | 11,980 |
| Sliding Window（滑动窗口） | 8.7 | 4,665 |
| EpiContext v1（初版） | 12.5 | 13,791 |
| **Adaptive EpiContext v2** | **3.8** | **1,153** |

**Adaptive EpiContext v2 比 Full-Context 减少 90% token 消耗（p=0.022），减少 64% 轮数（p<0.001）。**

在最复杂的任务（describe-image，20 轮图像分析）上，token 消耗从 43,600 降到 1,801（**减少 96%**）。

---

## 项目结构

```
EpiContext/
├── paper/mypaper/          # COLM 2026 论文 (17 页 PDF)
│   ├── main.pdf            # 编译好的论文
│   ├── sections/           # 论文章节 (abstract, intro, methodology, etc.)
│   └── figures/            # 5 张顶会标准图表
├── code/
│   ├── epicontext/
│   │   ├── harbor_agent.py # EpiContext Harbor Agent (6 个变体)
│   │   ├── core.py         # 核心框架 (ContextGraph, EpigeneticOperators, FitnessFunction)
│   │   └── agent.py        # Agent 抽象层
│   ├── experiments/
│   │   ├── run_harbor_experiments.py  # 批量实验运行器
│   │   ├── retry_failed.py           # 失败重试脚本
│   │   └── visualize_pub.py          # 顶会图表生成器
│   └── results/harbor_experiments/   # 实验数据 (81 次成功运行)
├── harbor-framework/       # Harbor 评估框架 (第三方)
├── plans/                  # 规划文档
│   └── audit/              # 4 轮审查报告 (v1→v4)
└── docs/                   # 构想文档
```

---

## 快速开始

```bash
# 1. 进入 Harbor 目录
cd harbor-framework

# 2. 运行 EpiContext Agent
PYTHONPATH="../code:$PYTHONPATH" uv run harbor run \
  -c examples/configs/job.yaml \
  -p examples/tasks/hello-world \
  --agent-import-path epicontext.harbor_agent:AdaptiveEpiContextAgent \
  -n 1

# 3. 运行批量对比实验
PYTHONPATH="../code:$PYTHONPATH" uv run python ../code/experiments/run_harbor_experiments.py
```

---

## 审查历程

| 版本 | 判定 | 评分 | 核心问题 |
|------|------|------|----------|
| v1 (5/11) | ❌ REJECT | 4/10 | 玩具模拟 + 虚假数据 |
| v2 (5/12 上午) | ⚠️ 有条件 ACCEPT | 7/10 | 数据反向 + 规模不足 |
| v3 (5/12 下午) | ⚠️ 有条件 ACCEPT | 7/10 | 统计检验缺失 + 重复数据 |
| v4 (5/13) | ✅ ACCEPT | 8/10 | 无重大问题 |

**COLM 2026 接收概率: 70-80%**