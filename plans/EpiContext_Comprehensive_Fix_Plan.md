# EpiContext 综合修复计划

**基于**: EpiContext_Audit_Report_v2.md + EpiContext_Academic_Competitiveness.md
**生成时间**: 2026-05-12
**目标**: 修复所有 Critical/Major 问题，使论文达到 COLM 2026 接收标准

---

## 0. 问题总览

### 🔴 Critical (4 个)
| # | 问题 | 根因 |
|---|------|------|
| C1 | 论文声称 90 次成功，实际 69 次 | 统计口径混淆 (v2 用 retry 后数据 vs v2 原始数据) |
| C2 | p=0.027 vs 实际 p=0.088 | 使用了不同数据集计算 |
| C3 | Table 1 的 N 和均值与 JSON 不一致 | 论文用 retry 后 intermediate_results，审计用原始 data |
| C4 | MethylationOnly 消融不完整 | Docker build 失败导致 3/6 任务缺失 |

### 🔴 学术竞争力问题 (1 个致命)
| 问题 | 根因 |
|------|------|
| EpiContext 在所有聚合指标上不如 SlidingWindow | 元数据开销 15-22% + 保守保留策略 |

### 🟡 Major (4 个)
| # | 问题 | 根因 |
|---|------|------|
| M1 | MethylationOnly 消融不完整 | Docker 网络问题 |
| M2 | hello-multi-step-advanced 仅 EpiContext 成功 | 其他 agent 未 retry |
| M3 | 任务多样性不足 | 仅 3 个简单 + 1 个中等任务有效 |
| M4 | Crossover 算子未在实验中实现 | 未实现 |

### 🟢 Minor (3 个)
| # | 问题 |
|---|------|
| m1 | Token 估算用 len(text)//4 |
| m2 | 缺少单元测试 |
| m3 | Appendix Overhead Table 是估计值 |

---

## 1. 修复路径：路径 A（做大实验）

选择路径 A 的理由：
- 用户明确要求"完美符合所有学术规范"
- 路径 A 是唯一能让论文被 COLM 接收的路径
- Harbor 框架已安装就绪，可以跑长程任务

---

## 2. 分阶段执行计划

### Phase 1: 立即修复数据一致性 (1-2 小时)

**目标**: 消除所有 Critical 数据不一致问题

1. **重新计算 Table 1**
   - 读取 `intermediate_results.json` (retry 后最终数据)
   - 精确统计每个 agent 的成功数、均值
   - 更新论文 Table 1，N 值按实际成功数填写

2. **重新计算统计检验**
   - 使用 retry 后的 intermediate_results.json
   - 配对 t 检验: EpiContext vs FullContext (按 task+rep 配对)
   - 更新论文中的 p 值和显著性标签

3. **修正 Abstract/Experiments/Discussion**
   - 将 "90 successful runs" 改为实际数字
   - 将所有 p 值统一为重新计算的结果
   - 移除任何不一致的声称

4. **明确说明数据缺失**
   - MethylationOnly 仅有 3/6 任务
   - hello-healthcheck 和 llm-judge-example 全部失败
   - hello-multi-step-advanced 仅部分 agent 成功

### Phase 2: 修复 EpiContext 核心实现 (2-3 小时)

**目标**: 解决 EpiContext 不优于 SlidingWindow 的根因

1. **紧凑激活编码**
   - 移除文本标签 `(act=0.95)`
   - 改为只在 context 选择时内部使用 activation，不附加到输出文本
   - 预期: 消除 15-22% 元数据开销

2. **自适应激活阈值**
   - 前 N 轮 (如 N=10) 使用 SlidingWindow
   - N 轮后切换到 EpiContext 策略
   - 预期: 在简单任务上等价 SlidingWindow，在复杂任务上优于 FullContext

3. **激进过滤参数**
   - 提高 activation threshold 到 0.5 (从 0.1)
   - 增大 β (token 效率权重) 到 2.0 (从 0.5)
   - 预期: 减少过度保留

4. **集成 tiktoken**
   - 替代 len(text)//4 估算
   - 提供精确的 token 统计

### Phase 3: 接入 Harbor 长程基准 (2-4 小时)

**目标**: 在 50-200+ turns 的真实任务上验证 EpiContext

1. **下载 Terminal-Bench 2.0 任务**
   ```bash
   cd harbor-framework
   uv run harbor dataset download terminal-bench-2
   ```

2. **选择 10-20 个适合的任务**
   - 优先选择 code-related 任务 (SWE-bench 类)
   - 确保 Docker 环境能正常构建
   - 目标: 每个任务 50-200+ turns

3. **运行 Pilot**
   - 先在 1-2 个任务上跑 5 个 agent 变体
   - 验证终止条件、token 统计、结果解析
   - 确认 EpiContext 的优势是否出现

4. **批量运行**
   - 目标: 10 个任务 × 5 agents × 3 reps = 150 runs
   - 使用 nohup 后台运行
   - 每 10 个保存中间结果

### Phase 4: 补充消融实验 (1-2 小时)

**目标**: 完善消融实验覆盖

1. **补充 MethylationOnly 的缺失任务**
   - describe-image 和 reward-kit-example
   - 各 3 次重复

2. **补充 hello-multi-step-advanced 的其他 agent**
   - FullContext、SlidingWindow、MethylationOnly、AcetylationOnly

3. **验证 Crossover 算子**
   - 在 long-horizon 任务上实现并测试
   - 或在论文中明确说明并作为 future work

### Phase 5: 论文最终修订 (2-3 小时)

**目标**: 基于新实验数据重写论文

1. **重写 Experiments 部分**
   - 基于 long-horizon 实验数据
   - 精确的 Table 1/2 数据
   - 正确的统计检验

2. **重写 Discussion 部分**
   - 如果 EpiContext 在长程任务上优于 SlidingWindow: 强调复杂度阈值假说的验证
   - 如果仍然不如: 诚实报告，调整论文定位

3. **更新 Abstract/Introduction**
   - 移除任何不一致的声称
   - 基于实际实验结果

4. **编译验证**
   - pdflatex + bibtex
   - 检查页数限制 (8-9 页正文)
   - 检查所有引用

### Phase 6: 第三方审查 (1-2 小时)

**目标**: 外部专家评审

1. **代码审查**: 检查实现与论文描述的一致性
2. **数据审查**: 检查 Table 数据与 JSON 的一致性
3. **文献审查**: 检查引用的真实性和相关性
4. **论文审查**: 检查 claim 与 evidence 的对应

---

## 3. 预期时间线

| 阶段 | 时间 | 产出 |
|------|------|------|
| Phase 1: 数据一致性修复 | 1-2h | 修正后的论文 |
| Phase 2: 核心实现修复 | 2-3h | 改进的 agent 代码 |
| Phase 3: 长程基准实验 | 4-8h | 新实验数据 |
| Phase 4: 消融实验补充 | 1-2h | 完整消融数据 |
| Phase 5: 论文最终修订 | 2-3h | 最终论文 |
| Phase 6: 第三方审查 | 1-2h | 审查报告 |
| **总计** | **11-20h** | |

---

## 4. 风险评估

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| Docker Hub TLS 超时持续 | 中 | 无法跑 Terminal-Bench 任务 | 配置 Docker 镜像代理 |
| EpiContext 长程任务上仍不如 SlidingWindow | 中 | 论文核心 claim 不成立 | 调整算子设计或定位为 position paper |
| Terminal-Bench 任务构建失败 | 中 | 无法获取足够任务 | 选择简单任务，或使用 Harbor 其他 adapter |
| 实验时间超过 deadline | 低 | 无法完成论文 | 优先 Phase 1-2，Phase 3-4 作为补充 |

---

## 5. 决策点

1. **Phase 3 Pilot 后**: 如果 EpiContext 在长程任务上显著优于 SlidingWindow → 继续路径 A
2. **Phase 3 Pilot 后**: 如果 EpiContext 仍然不优于 → 考虑路径 B (workshop paper) 或算子重新设计
3. **Phase 5 后**: 如果论文数据一致且 claim 有支撑 → 提交
