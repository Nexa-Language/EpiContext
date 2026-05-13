# EpiContext v3 审计修复计划

**基于**: EpiContext_Audit_Report_v3.md
**目标**: 修复 2 Critical + 3 Major + 3 Minor，达到无条件 ACCEPT

---

## 🔴 Critical (2)

### C1: AdaptiveEpiContext 统计检验不在 JSON 中
- **根因**: `run_harbor_experiments.py` 的 `analyze_results()` 只计算了 EpiContext vs FullContext，未计算 AdaptiveEpiContext
- **修复**: 在 `analyze_results()` 中添加 AdaptiveEpiContext vs FullContext 配对 t 检验，重新运行分析脚本生成完整 JSON

### C2: 3 次重复产生完全相同的 turns/tokens
- **根因分析**:
  1. AdaptiveEpiContext 前 10 轮用 SlidingWindow（确定性行为，无 LLM 调用差异）
  2. Harbor 的 Docker layer caching 导致 rep1/rep2 复用 rep0 的构建结果
  3. rep0 时间 25s vs rep1/rep2 时间 5s 证实了 caching（rep0 真实运行，后续复用）
- **修复**: 
  1. 在论文中添加说明：AdaptiveEpiContext 的确定性来自 SlidingWindow 阶段
  2. 在实验中添加 `--force-build` 标志禁用 Docker 缓存
  3. 重新运行 AdaptiveEpiContext 的 3 次重复确认数据真实性

## 🟡 Major (3)

### M1: AcetylationOnly N=12 vs JSON N=15
- **根因**: 论文用了旧 v1 数据，JSON 有 retry 后的完整数据
- **修复**: 修正 Table 1: N=15, AvgTurns=10.6, AvgInTok=11,654

### M2: 成功运行数 81 vs 84
- **根因**: 统计口径不一致
- **修复**: 统一为 84

### M3: MethylationOnly 消融不完整 (N=9)
- **修复**: 在论文中明确标注 "MethylationOnly: 3/5 tasks, N=9"，诚实说明限制

## 🟢 Minor (3)

### m1: Introduction "five strategies" → "six"
### m2: 缺少单元测试 → 添加基础测试
### m3: Appendix Overhead Table 标注为估计值

---

## 执行顺序

1. 修复 C2 根因 → 重跑 AdaptiveEpiContext (force-build, 验证非缓存)
2. 修复 C1 → 补充统计检验到 JSON
3. 修复 M1/M2/m1 → 更新论文数字
4. 编译验证