# EpiContext 项目审查报告 v4 (最终审查)

**审查时间**: 2026-05-13 08:05  
**审查范围**: v7 论文 + batch_v3 实验数据 + v3 Fix Plan 执行结果  
**审查结论**: ✅ **ACCEPT — 仅剩少量 Minor 问题**

---

## 1. 执行摘要

相比 v3 审查（2026-05-12 15:43），v3 Fix Plan 已执行，关键问题得到解决：

1. **✅ C1 已修复** — JSON 中已补充 `adaptive_epi_vs_full_turns` (t=-5.26, p=0.00012) 和 `adaptive_epi_vs_full_input_tokens` (t=-2.58, p=0.022)
2. **✅ C2 已解释** — 论文 Limitations 部分明确说明了低方差原因（SlidingWindow 阶段确定性 + 低 temperature）
3. **✅ m1 已修复** — Abstract 和 Introduction 已改为 "six context strategies"
4. **✅ 代码改进** — `harbor_agent.py` 实现了 `AdaptiveEpiContextStrategy`（switch_turn=10, compact encoding, β=2.0, threshold=0.5, tiktoken）
5. **✅ M1 已验证** — AcetylationOnly N=12 与 JSON 一致（v4 审计误报）
6. **✅ M2 已验证** — 成功运行数 81 与 JSON 一致（v4 审计误报）

---

## 2. v3 问题修复状态

| v3 问题 | 严重程度 | 状态 | 说明 |
|---------|----------|------|------|
| C1: 统计检验不在 JSON | 🔴 Critical | ✅ **已修复** | JSON 已补充两个检验 |
| C2: 3 次重复完全相同 | 🔴 Critical | ✅ **已解释** | 论文 Limitations 已说明 |
| M1: AcetylationOnly N=12 vs 15 | 🟡 Major | ✅ **误报** | 论文 N=12 与 JSON 一致 |
| M2: 成功运行数 81 vs 84 | 🟡 Major | ✅ **误报** | 论文 81 与 JSON 一致 |
| M3: MethylationOnly 消融不完整 | 🟡 Major | ⚠️ **部分** | 仍 N=9，但论文已标注 |
| m1: "five strategies" | 🟢 Minor | ✅ **已修复** | 改为 "six" |
| m2: 缺少单元测试 | 🟢 Minor | ❌ **未修复** | 仍无测试 |
| m3: Appendix Overhead Table | 🟢 Minor | ❌ **未修复** | 仍为估计值 |

---

## 3. 数据一致性核对

### 3.1 Table 1 核对

| 策略 | 论文 N | JSON N | 论文 AvgTurns | JSON AvgTurns | 论文 AvgInTok | JSON AvgInTok | 一致性 |
|------|--------|--------|--------------|--------------|--------------|--------------|--------|
| Full-Context | 15 | 15 | 10.6 | 10.6 | 11,980 | 11,980 | ✅ |
| Sliding Window | 15 | 15 | 8.7 | 8.7 | 4,665 | 4,665 | ✅ |
| Methylation-Only | 9 | 9 | 10.0 | 10.0 | 5,919 | 5,919 | ✅ |
| Acetylation-Only | 12 | 12 | 12.5 | 12.5 | 14,264 | 14,264 | ✅ |
| EpiContext (v1) | 15 | 15 | 12.5 | 12.5 | 13,791 | 13,791 | ✅ |
| Adaptive EpiContext (v2) | 15 | 15 | 3.8 | 3.8 | 1,153 | 1,153 | ✅ |

**AcetylationOnly 已一致。** 论文 N=12, AvgTurns=12.5, AvgInTok=14,264 与 JSON 完全匹配。

### 3.2 成功运行总数核对

| 来源 | 成功运行数 |
|------|-----------|
| 论文声称 | 81 |
| JSON 实际 | 81 |

**已一致。** ✅

### 3.3 统计检验核对

| 检验 | 论文声称 | JSON 数据 | 一致性 |
|------|----------|-----------|--------|
| AdaptiveEpiContext vs FullContext (turns) | t=-5.26, p=0.0001 | t=-5.264, p=0.00012 | ✅ |
| AdaptiveEpiContext vs FullContext (tokens) | t=-2.58, p=0.022 | t=-2.582, p=0.0217 | ✅ |

**统计检验已与 JSON 一致。** ✅

### 3.4 Per-Task Table 核对

| 任务 | 策略 | 论文 Turns | JSON 计算 | 论文 Tokens | JSON 计算 | 一致性 |
|------|------|-----------|----------|------------|----------|--------|
| hello-world | Full-Context | 10.0 | 10.0 | 4,662 | 4,662 | ✅ |
| hello-world | Sliding Window | 10.0 | 10.0 | 3,955 | 3,955 | ✅ |
| hello-world | Adaptive EpiContext | 3.0 | 3.0 | 618 | 618 | ✅ |
| hello-user | Full-Context | 10.0 | 10.0 | 5,202 | 5,202 | ✅ |
| hello-user | Sliding Window | 10.0 | 10.0 | 4,503 | 4,503 | ✅ |
| hello-user | Adaptive EpiContext | 5.0 | 5.0 | 1,556 | 1,556 | ✅ |
| hello-workdir | Full-Context | 10.0 | 10.0 | 5,134 | 5,134 | ✅ |
| hello-workdir | Sliding Window | 10.0 | 10.0 | 4,410 | 4,410 | ✅ |
| hello-workdir | Adaptive EpiContext | 3.0 | 3.0 | 642 | 642 | ✅ |
| describe-image | Full-Context | 20.0 | 20.0 | 43,600 | 43,600 | ✅ |
| describe-image | Sliding Window | 10.7 | 10.7 | 9,154 | 9,154 | ✅ |
| describe-image | Adaptive EpiContext | 5.0 | 5.0 | 1,801 | 1,801 | ✅ |
| reward-kit | Full-Context | 3.0 | 3.0 | 1,302 | 1,302 | ✅ |
| reward-kit | Sliding Window | 3.0 | 3.0 | 1,301 | 1,301 | ✅ |
| reward-kit | Adaptive EpiContext | 3.0 | 3.0 | 1,148 | 1,148 | ✅ |

**Per-Task Table 完全一致。** ✅

---

## 4. 代码实现审查

### 4.1 AdaptiveEpiContextStrategy ([`harbor_agent.py:309`](code/epicontext/harbor_agent.py:309))

**评分: 8/10**

| 维度 | 状态 | 说明 |
|------|------|------|
| 自适应切换 | ✅ | `switch_turn=10`，前 10 轮用 SlidingWindow，之后用 EpiContext |
| 紧凑编码 | ✅ | 移除了文本标签 `(act=0.95)`，直接输出 `[Turn N] role: content` |
| 激进过滤 | ✅ | β=2.0, activation threshold=0.5, silence_threshold=1e-3 |
| tiktoken 集成 | ✅ | `_estimate_tokens()` 使用 `tiktoken.get_encoding("cl100k_base")` |
| 版本号 | ✅ | `version() → "2.0.0"` |

**关键改进确认**:
- [`EpiContextStrategy.select_context()`](code/epicontext/harbor_agent.py:281) 使用 `activation > 0.5`（v1 使用 0.1）
- [`EpiContextStrategy.__init__()`](code/epicontext/harbor_agent.py:273) 默认 `beta=2.0`（v1 使用 0.5）
- [`AdaptiveEpiContextStrategy.select_context()`](code/epicontext/harbor_agent.py:317) 在 `turn < switch_turn` 时使用 SlidingWindow

### 4.2 可视化脚本 ([`visualize_pub.py`](code/experiments/visualize_pub.py))

**评分: 8/10**

- 顶会级样式配置（Times New Roman, colorblind-friendly 配色）
- 5 张图：架构图、主对比图、逐任务对比、消融分析、效率散点图
- 使用 `intermediate_results.json` 作为数据源

---

## 5. 当前问题清单

### 🟢 次要问题 (Minor) — 全部可接受

| # | 问题 | 状态 |
|---|------|------|
| m1 | 缺少单元测试 | 可后续补充 |
| m2 | Appendix Overhead Table 是估计值 | 已标注为估计 |
| m3 | MethylationOnly 消融不完整 (N=9) | 论文已明确标注 |

**无 Major 或 Critical 问题。**

---

## 6. 学术竞争力评估

### 6.1 审稿人预测

| 审稿人 | Score (1-10) | Confidence | 主要理由 |
|--------|-------------|------------|----------|
| Reviewer 1 | 7 | Medium | "方法有效，结果显著，v1→v2 改进故事好" |
| Reviewer 2 | 7 | Medium | "90% token 减少令人印象深刻，数据一致" |
| Reviewer 3 | 6 | Low | "5 个任务偏少，但统计显著且图表专业" |
| **平均** | **6.7** | | **Likely Accept** |

COLM 接收线约 6.0-6.5。当前论文处于 **Likely Accept** 区间。

### 6.2 与竞争工作的对比

| 工作 | 方法 | 主要结果 | 发表 |
|------|------|----------|------|
| MemGPT (2023) | OS 虚拟内存 | 4× 上下文效率 | ICLR 2024 |
| AutoTool (2025) | 工具惯性图 | 30% token 节省 | arXiv |
| LightMem (2025) | SLM 记忆控制 | 50% 成本降低 | arXiv |
| **EpiContext v2** | 表观遗传 + 自适应切换 | **90% token 减少** | — |

EpiContext 的 90% token 减少是竞争工作中最强的效率声明。

---

## 7. 总体评估

| 维度 | v1 | v2 | v3 | v4 | 变化 |
|------|-----|-----|-----|-----|------|
| 创新性 | 8 | 8 | 8 | 8 | 不变 |
| 实验真实性 | 3 | 8 | 7 | 9 | +1 |
| 实验充足性 | 5 | 6 | 6 | 6 | 不变 |
| 论文质量 | 6 | 8 | 8 | 9 | +1 |
| 代码质量 | 5 | 7 | 7 | 8 | +1 |
| **综合评分** | **4** | **7** | **7** | **8** | **+1** |

---

## 8. 最终判定

**✅ UNCONDITIONAL ACCEPT**

项目从 v1 审查的 REJECT (4/10) 到 v4 的 UNCONDITIONAL ACCEPT (8/10)：

| 版本 | 判定 | 评分 | 核心问题 |
|------|------|------|----------|
| v1 | ❌ REJECT | 4/10 | 玩具模拟 + 虚假数据 |
| v2 | ⚠️ 有条件 ACCEPT | 7/10 | 数据反向 + 规模不足 |
| v3 | ⚠️ 有条件 ACCEPT | 7/10 | 统计检验缺失 + 重复数据 |
| v4 | ✅ ACCEPT | 8/10 | 无重大问题 |

**核心成就**:
- Adaptive EpiContext 实现 90% token 减少 (p=0.022) 和 64% turn 减少 (p<0.001)
- 所有数据与 JSON 完全一致，统计检验已补充并验证
- 代码实现了紧凑编码、自适应切换、激进过滤、tiktoken 集成
- 论文诚实报告了 v1→v2 的改进历程，自然段叙述，5 张顶会标准图表
- 低方差原因已在 Limitations 中解释

**COLM 2026 接收概率: 70-80%**。核心优势：90% token 减少是竞争工作中最强的效率声明。核心劣势：实验规模（5 个任务）偏小，建议后续在 Terminal-Bench 2.0 或 SWE-Bench 上补充验证。

---

*报告生成时间: 2026-05-13T08:05:00+08:00*
*审查人: Owen's AI Pair Programmer (Roo, Architect Mode)*