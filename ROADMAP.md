# EpiContext Harbor Rerun ROADMAP

## 版本: v4.0 (审计后重建)
## 启动时间: 2026-05-11
## 状态: 执行中

---

## 决策记录

### D001: 废弃旧模拟实验栈
- **时间**: 2026-05-11
- **原因**: 审计报告确认 [`run_main.py`](code/experiments/run_main.py)、[`environments.py`](code/epicontext/benchmarks/environments.py) 为玩具级模拟，3/4 基准成功率 0%，token 数据为硬编码倍数
- **影响**: 所有基于这些产物的论文声称作废
- **替代**: 使用 Harbor + 真实 benchmark 重建

### D002: 保留 Agent 方向
- **时间**: 2026-05-11
- **原因**: 用户明确选择严格扩展路线
- **影响**: 必须重建真实 Agent 实验栈，不能退化为纯优化论文

### D003: 使用 Harbor 作为统一评测框架
- **时间**: 2026-05-11
- **原因**: Harbor 是专门为 agent evaluation 设计的框架，支持多 agent、多 benchmark、标准留痕
- **影响**: 所有实验必须通过 Harbor 运行

### D004: 优先方案 A (Wrapper Agent)
- **时间**: 2026-05-11
- **原因**: 快速落地，公平对比直接
- **影响**: EpiContext 作为底座 agent 的上下文装配层

---

## 变更记录

| 日期 | 变更 | 详情 |
|------|------|------|
| 2026-05-11 | 初始化 | 创建 ROADMAP，标记旧证据失效 |
| 2026-05-11 | 密钥清理 | 移除 [`run_real_llm.py`](code/experiments/run_real_llm.py:51) 硬编码密钥 |
| 2026-05-11 | Harbor 安装 | 克隆并安装 Harbor framework |

---

## 已知风险

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| Harbor 无法接入用户 endpoint | 中 | 改用 Harbor 已支持的 agent/model 组合 |
| Docker 不可用 | 中 | 检查 Docker 状态，必要时使用 Daytona cloud |
| 实验时间不足 6h | 高 | 增加 benchmark 任务数、重复次数 |
| 统计不显著 | 中 | 增加样本量，诚实报告 |