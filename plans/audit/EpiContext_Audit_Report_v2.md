# EpiContext 项目审查报告 v2 (更新审查)

**审查时间**: 2026-05-12  
**审查范围**: Harbor 框架集成、新实验数据、更新后论文内容  
**审查结论**: ⚠️ **有条件的 ACCEPT — 需要中等修订**

---

## 1. 执行摘要

相比 v1 审查报告（2026-05-11），项目经历了重大改进：

1. **✅ 抛弃了玩具模拟实验**，转向 Harbor 框架进行真实容器化实验
2. **✅ 论文大幅重写**，诚实呈现了实验结果和限制
3. **✅ 代码中移除了硬编码 API Key**，改为环境变量
4. **✅ 实验使用真实 LLM API 调用**（mimo-v2.5-pro），通过 Harbor 框架编排
5. **✅ 论文不再虚假声称**，而是诚实地报告了 EpiContext 当前不优于简单策略的事实

**核心改进**: 从"REJECT - 需要重大修订"提升为"有条件的 ACCEPT"。

---

## 2. Harbor 框架集成审查

### 2.1 Agent 实现 ([`code/epicontext/harbor_agent.py`](code/epicontext/harbor_agent.py))

**评分: 7/10**

| 维度 | 状态 | 说明 |
|------|------|------|
| 架构设计 | ✅ 优秀 | 继承 Harbor `BaseAgent`，实现 `setup()`/`run()` 生命周期 |
| LLM 集成 | ✅ 真实 | 使用 OpenAI-compatible API，从环境变量读取配置 |
| 上下文策略 | ✅ 完整 | 实现了 FullContext、SlidingWindow、EpiContext 三种策略 |
| 表观遗传算子 | ✅ 实现 | 甲基化（loss 变化小的节点）、乙酰化（梯度方向一致）、适应度反馈 |
| 结果记录 | ✅ 完善 | 写入 `epicontext_result.json` 和 `reward.txt` |
| 消融 Agent | ✅ 合理 | MethylationOnly、AcetylationOnly 通过调整 α/β/γ 参数实现 |
| 代码质量 | ⚠️ 一般 | 缺少类型注解、单元测试 |

**关键亮点**:
- [`EpiContextAgent._generate_turn()`](code/epicontext/harbor_agent.py:436) 合并思考+动作为单次 LLM 调用，减少 API 开销
- [`_execute_action()`](code/epicontext/harbor_agent.py:475) 通过 `environment.exec()` 执行真实 shell 命令
- [`_check_done()`](code/epicontext/harbor_agent.py:506) 实现了多种终止检测（连续失败、Agent 声明完成、重复命令）

**问题**:
- [`_check_done()`](code/epicontext/harbor_agent.py:506) 的"连续 3 轮无进展"终止条件可能导致提前退出，影响任务完成率
- Token 估算仍使用 `len(text) // 4`，而非真实 tokenizer
- `loss_delta` 和 `grad_norm` 的映射过于简化（`reward > 0 → grad_norm = 1.0`）

### 2.2 实验编排 ([`code/experiments/run_harbor_experiments.py`](code/experiments/run_harbor_experiments.py))

**评分: 7/10**

| 维度 | 状态 | 说明 |
|------|------|------|
| 实验设计 | ✅ 合理 | 5 个 agent × 8 个 task × 3 次重复 = 120 次运行 |
| 结果解析 | ✅ 正确 | 从 Harbor job 目录解析 `epicontext_result.json` |
| 中间保存 | ✅ 每 5 次保存 | 支持断点续跑 |
| 统计检验 | ✅ 配对 t 检验 | EpiContext vs FullContext 的 turns 和 input_tokens |
| 错误处理 | ✅ 完善 | 失败重试脚本 ([`retry_failed.py`](code/experiments/retry_failed.py)) |

### 2.3 Harbor 框架本身 ([`harbor-framework/`](harbor-framework/))

**评分: N/A（第三方依赖，不在审查范围）**

- Harbor 是一个成熟的 agent 评估框架，提供容器化环境、任务编排、结果验证
- 项目正确地将 EpiContext 实现为 Harbor custom agent

---

## 3. 实验数据审查

### 3.1 实验规模与成功率

**总运行数**: 120 次（5 agents × 8 tasks × 3 reps）  
**成功运行**: 69 次（57.5%）  
**失败运行**: 51 次（42.5%）

**按 Agent 统计**:

| Agent | 成功数 | 成功率 | 说明 |
|-------|--------|--------|------|
| EpiContext | 15 | 50% | 5 tasks × 3 reps |
| FullContext | 15 | 50% | 5 tasks × 3 reps |
| SlidingWindow | 15 | 50% | 5 tasks × 3 reps |
| MethylationOnly | 9 | 30% | 3 tasks × 3 reps |
| AcetylationOnly | 15 | 50% | 5 tasks × 3 reps |

**失败原因分析**:
- `hello-healthcheck`: **所有 agent 全部失败** — Docker Hub registry 访问问题（TLS handshake timeout）
- `llm-judge-example`: **所有 agent 全部失败** — 无结果文件
- `hello-multi-step-advanced`: **仅 EpiContext 成功**（通过 retry），其他 agent 全部失败
- `describe-image`: MethylationOnly 全部失败（Docker 问题），其他成功
- `reward-kit-example`: MethylationOnly 全部失败（Docker 问题），其他成功

**关键发现**: 大量失败是 **Harbor 基础设施问题**（Docker Hub 访问、任务不存在），而非 agent 本身的问题。论文诚实承认了这一点。

### 3.2 成功运行的实验数据

论文 Table 1 声称 90 次成功运行，但实际数据需要验证。

**逐 Agent 成功数据**:

| Agent | 成功任务 | 平均 Turns | 平均 Input Tokens | 平均 Output Tokens |
|-------|----------|-----------|-------------------|-------------------|
| EpiContext | hello-world, hello-user, hello-workdir, describe-image, reward-kit (×3) | 12.5 | 13,791 | 3,351 |
| FullContext | hello-world, hello-user, hello-workdir, describe-image, reward-kit (×3) | 10.6 | 11,980 | 2,581 |
| SlidingWindow | hello-world, hello-user, hello-workdir, describe-image, reward-kit (×3) | 8.7 | 4,665 | 1,801 |
| MethylationOnly | hello-world, hello-user, hello-workdir (×3) | 10.0 | 5,919 | 1,854 |
| AcetylationOnly | hello-world, hello-user, hello-workdir, describe-image, reward-kit (×3) | 10.6 | 11,654 | 2,852 |

### 3.3 统计检验结果

| 检验 | t 值 | p 值 | 显著性 |
|------|------|------|--------|
| EpiContext vs FullContext (turns) | 1.833 | 0.088 | 不显著 |
| EpiContext vs FullContext (input tokens) | 0.909 | 0.379 | 不显著 |

---

## 4. 论文与数据一致性核对

### 4.1 Abstract 声称核对

| 论文声称 | 实际数据 | 一致性 | 说明 |
|----------|----------|--------|------|
| "controlled experiments comparing five context strategies" | ✅ 5 个 agent 确实在运行 | ✅ | |
| "multiple containerized tasks" | ✅ 6 个任务有成功结果 | ✅ | |
| "all context strategies perform equivalently on simple tasks" | ✅ hello-world/user/workdir 确实表现一致 | ✅ | |
| "context management overhead only becomes relevant beyond a minimum complexity threshold" | ✅ describe-image 上有差异 | ✅ | |

### 4.2 Experiments 部分核对

| 论文声称 | 实际数据 | 一致性 |
|----------|----------|--------|
| "90 successful experimental runs" | 4 agents × 3 simple tasks × 3 reps = 36 + EpiContext × multi-step × 3 = 3 + 各 agent × describe-image × 3 = 12 + 各 agent × reward-kit × 3 = 12 + SlidingWindow × describe-image × 3 = 3 + ... | ⚠️ 需要精确计算 |
| "5 strategies × 6 tasks × 3 repetitions = 90" | 实际: 5 × 6 × 3 = 90 设计，但 MethylationOnly 缺少 describe-image 和 reward-kit 的成功结果 | ⚠️ 部分不一致 |

**精确计算实际成功运行数**:
- EpiContext: hello-world(3) + hello-user(3) + hello-workdir(3) + describe-image(3) + reward-kit(3) = 15
- FullContext: 同上 = 15
- SlidingWindow: 同上 = 15
- MethylationOnly: hello-world(3) + hello-user(3) + hello-workdir(3) = 9
- AcetylationOnly: hello-world(3) + hello-user(3) + hello-workdir(3) + describe-image(3) + reward-kit(3) = 15
- **总计**: 15 + 15 + 15 + 9 + 15 = **69**

**论文声称 90 次成功运行，实际只有 69 次。** 这是一个 **数据不一致** 问题。

### 4.3 Table 1 数据核对

| 策略 | 论文 N | 实际 N | 论文 AvgTurns | 实际 AvgTurns | 论文 AvgInTok | 实际 AvgInTok |
|------|--------|--------|--------------|--------------|--------------|--------------|
| Full-Context | 18 | 15 | 12.3 | 10.6 | 11,912 | 11,980 |
| Sliding Window | 18 | 15 | 10.8 | 8.7 | 5,971 | 4,665 |
| Methylation-Only | 18 | 9 | 12.3 | 10.0 | 9,568 | 5,919 |
| Acetylation-Only | 18 | 15 | 12.0 | 10.6 | 11,647 | 11,654 |
| EpiContext | 18 | 15 | 14.4 | 12.5 | 14,195 | 13,791 |

**发现**: 论文 Table 1 的数据与实际 JSON 数据 **不一致**。论文声称每个 agent 有 N=18，但实际只有 15（MethylationOnly 只有 9）。数值也略有差异。

**可能原因**: 论文可能使用了不同的统计口径（例如排除了某些失败重试后的数据，或使用了 intermediate_results.json 而非 final_results.json）。

### 4.4 Per-Task Table 核对

论文 Table 2 声称:
- `describe-image`: EpiContext 20.0 turns, 35,872 tokens
- 实际 JSON: EpiContext describe-image 3 次运行的 input_tok 分别为 35,872, 35,872, 35,872 ✅（一致）
- `reward-kit-example`: EpiContext 12.3 turns, 15,196 tokens
- 实际 JSON: reward-kit 3 次运行的 turns 分别为 13, 14, 10，平均 = 12.3 ✅（一致）
- 实际 JSON: reward-kit 3 次运行的 input_tok 分别为 15,724, 20,164, 9,701，平均 = 15,196 ✅（一致）

**Per-Task Table 的数据与 JSON 一致。**

### 4.5 Discussion 声称核对

| 论文声称 | 实际数据 | 一致性 |
|----------|----------|--------|
| "EpiContext uses significantly more turns than Full-Context (p = 0.027)" | 实际 p = 0.088 | ❌ **不一致** |
| "17.7% token advantage on describe-image" | FullContext: (49114+38111+43575)/3 = 43,600; EpiContext: 35,872; 差异 = 17.7% | ✅ |
| "50% token reduction for Sliding Window vs Full-Context" | 需要精确计算 | ⚠️ |

**关键不一致**: 论文声称 p = 0.027（显著），但 JSON 分析结果为 p = 0.088（不显著）。这是一个 **严重的数据不一致**。

---

## 5. 与 v1 审查的对比

### v1 严重问题的解决情况

| v1 问题 | 解决状态 | 说明 |
|---------|----------|------|
| C1: Agent 基准实验使用玩具级模拟 | ✅ 已解决 | 转向 Harbor 真实容器化实验 |
| C2: 基线 Agent 使用硬编码 token 倍数 | ✅ 已解决 | 所有 agent 使用相同 LLM API |
| C3: 3/4 基准成功率 0% | ✅ 已解决 | Harbor 任务有合理的成功率 |
| C4: 论文声称与实验数据不一致 | ⚠️ 部分解决 | 主要声称已修正，但存在新的不一致 |
| C5: 真实 LLM 实验结果缺失 | ✅ 已解决 | Harbor 实验使用真实 LLM |
| M3: 代码中硬编码 API Key | ✅ 已解决 | 改为环境变量 |

---

## 6. 当前问题清单

### 🔴 严重问题 (Critical)

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| C1 | **论文声称 90 次成功运行，实际只有 69 次** | 数据不一致，降低可信度 | 修正为 69 次，或重新运行缺失的实验 |
| C2 | **论文声称 p = 0.027，实际 p = 0.088** | 统计显著性声称错误 | 修正 p 值，移除"statistically significant"标签 |
| C3 | **Table 1 的 N 值和均值与 JSON 不一致** | 数据不一致 | 重新计算 Table 1，确保与 JSON 一致 |
| C4 | **论文未充分说明 MethylationOnly 数据缺失** | 消融实验不完整 | 明确说明 MethylationOnly 仅有 3/6 任务的数据 |

### 🟡 中等问题 (Major)

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| M1 | **MethylationOnly 消融实验不完整** | 无法全面评估消融效果 | 补充运行或在论文中明确说明 |
| M2 | **hello-multi-step-advanced 仅 EpiContext 成功** | 无法公平对比 | 调查其他 agent 失败原因，或排除此任务 |
| M3 | **任务多样性不足** | 6 个任务中有 2 个全部失败 | 考虑增加更多 Harbor 原生任务 |
| M4 | **Methodology 中的 Crossover 算子未在实验中实现** | 理论与实验脱节 | 在论文中明确说明 crossover 未在当前实验中测试 |

### 🟢 次要问题 (Minor)

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| m1 | Token 估算使用 len(text)//4 | 不精确 | 使用 tiktoken |
| m2 | 缺少单元测试 | 代码质量 | 添加测试 |
| m3 | Appendix 中的 Overhead Table 数据是估计值 | 非实测 | 标注为估计或提供实测数据 |

---

## 7. 积极方面

### 7.1 学术诚信 ✅

论文最大的改进是 **诚实性**。v4/v6 版本的论文:
- 不再虚假声称"70.3% token reduction"
- 诚实地报告"EpiContext is not yet competitive in aggregate"
- 明确承认"Sliding Window is the most efficient strategy overall"
- 提供了详细的 Root Cause Analysis
- 给出了具体的 Engineering Improvements 路线

这种诚实的报告方式在学术界是值得尊敬的，比虚假声称更有价值。

### 7.2 实验框架可复现 ✅

- 所有实验通过 Harbor 框架编排，可复现
- 代码、配置、原始结果都已保存
- 提供了失败重试脚本

### 7.3 理论贡献 ✅

- 表观遗传学隐喻新颖且有理论基础
- 数学形式化严谨（Methodology 质量高）
- 收敛性定理和证明有价值

### 7.4 实验设计改进 ✅

- 使用真实 LLM API 而非模拟
- 5 个 agent 变体的公平对比
- 统计检验（尽管结果不显著）

---

## 8. 总体评估

| 维度 | v1 评分 | v2 评分 | 变化 |
|------|---------|---------|------|
| 创新性 | 8/10 | 8/10 | 不变 |
| 实验真实性 | 3/10 | 8/10 | +5 |
| 实验充足性 | 5/10 | 6/10 | +1 |
| 论文质量 | 6/10 | 8/10 | +2 |
| 代码质量 | 5/10 | 7/10 | +2 |
| **综合评分** | **4/10** | **7/10** | **+3** |

---

## 9. 修订建议

### 必须完成的修订 (Must-Fix)

1. **修正 Table 1 的 N 值**
   - Full-Context, SlidingWindow, AcetylationOnly, EpiContext: N = 15
   - MethylationOnly: N = 9
   - 或者补充运行 MethylationOnly 的 describe-image 和 reward-kit 实验

2. **修正统计检验的 p 值**
   - 当前 JSON 分析: p = 0.088（不显著）
   - 论文声称: p = 0.027（显著）
   - 必须统一为实际计算结果

3. **修正"90 successful runs"为"69 successful runs"**
   - 更新 Abstract、Experiments、Discussion 中的所有相关数字

4. **明确说明数据缺失**
   - MethylationOnly 仅有 3/6 任务的数据
   - hello-healthcheck 和 llm-judge-example 全部失败
   - hello-multi-step-advanced 仅 EpiContext 成功

### 建议完成的修订 (Should-Fix)

1. 补充 MethylationOnly 的 describe-image 和 reward-kit 实验（如果 Docker 问题已修复）
2. 调查 hello-multi-step-advanced 其他 agent 失败的原因
3. 在 Methodology 中明确说明 Crossover 算子未在当前实验中测试
4. 使用 tiktoken 替代 len(text)//4 的 token 估算
5. 为 Table 2 的 describe-image 数据添加置信区间（因为 SlidingWindow 的 3 次运行差异很大：5, 7, 20 turns）

### 可选修订 (Nice-to-Have)

1. 增加更多 Harbor 原生任务以提高实验覆盖面
2. 添加代码单元测试
3. 在 Appendix 中提供实测的计算开销数据

---

## 10. 最终判定

**⚠️ 有条件的 ACCEPT — 需要中等修订**

项目在 v1 审查后经历了显著改进：
- 从玩具模拟转向真实 Harbor 实验
- 论文大幅重写，诚实呈现结果
- 代码质量提升，移除了安全问题

剩余问题主要是 **数据一致性**（Table 1/2 的数字需要与 JSON 精确对齐）和 **统计声称的准确性**（p 值需要修正）。这些问题可以在不改变实验设计的情况下通过精确计算修复。

**建议路径**:
1. 立即修正 Table 1 和统计检验的数字
2. 明确说明数据缺失和实验限制
3. 考虑补充 MethylationOnly 的缺失实验
4. 论文的核心叙事（诚实报告、识别复杂度阈值、提供工程改进路线）是有价值的，应保持

---

## 11. 附录：原始数据摘要

### 11.1 成功运行的完整数据（来自 final_results.json）

**EpiContext (15 次成功)**:
- hello-world: turns=10,10,10 | input_tok=5613,5613,5613
- hello-user: turns=10,10,10 | input_tok=6173,6173,6173
- hello-workdir: turns=10,10,10 | input_tok=6100,6100,6100
- describe-image: turns=20,20,20 | input_tok=35872,35872,35872
- reward-kit: turns=13,14,10 | input_tok=15724,20164,9701

**FullContext (15 次成功)**:
- hello-world: turns=10,10,10 | input_tok=4662,4662,4662
- hello-user: turns=10,10,10 | input_tok=5202,5202,5202
- hello-workdir: turns=10,10,10 | input_tok=5134,5134,5134
- describe-image: turns=20,20,20 | input_tok=49114,38111,43575
- reward-kit: turns=3,3,3 | input_tok=1301,1302,1302

**SlidingWindow (15 次成功)**:
- hello-world: turns=10,10,10 | input_tok=3955,3955,3955
- hello-user: turns=10,10,10 | input_tok=4503,4503,4503
- hello-workdir: turns=10,10,10 | input_tok=4410,4410,4410
- describe-image: turns=5,7,20 | input_tok=1988,4884,20590
- reward-kit: turns=3,3,3 | input_tok=1301,1302,1300

**MethylationOnly (9 次成功)**:
- hello-world: turns=10,10,10 | input_tok=5571,5571,5571
- hello-user: turns=10,10,10 | input_tok=6128,6128,6128
- hello-workdir: turns=10,10,10 | input_tok=6057,6057,6057

**AcetylationOnly (15 次成功)**:
- hello-world: turns=10,10,10 | input_tok=5816,5816,5816
- hello-user: turns=10,10,10 | input_tok=6496,6496,6496
- hello-workdir: turns=10,10,10 | input_tok=5955,5955,5955
- describe-image: turns=20,20,20 | input_tok=35841,41724,38799
- reward-kit: turns=3,3,3 | input_tok=1215,1215,1215

---

*报告生成时间: 2026-05-12T12:23:00+08:00*
*审查人: Owen's AI Pair Programmer (Roo, Architect Mode)*
