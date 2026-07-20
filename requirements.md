# 需求：平圩电力市场 XGBoost 实时电价预测工程

## 输入

- 数据目录路径：默认 `/data/ztwen2/project_dir/pingwei_data_analyser/2025-2026市场情况`
  - 包含每月一对 Excel：`*市场价格趋势.xlsx` + `*市场供需情况.xlsx`
  - 工程**仅消费 2025 全年数据**，2026 数据完全忽略
- 切分配置：`outputs/split.json`（由 ② 数据划分模块自动生成）

## 工程结构

5 个功能模块 + 1 个共享工具 + 1 个入口脚本，单一职责、无功能冗余：

| 模块                        | 文件                   | 唯一 HTML                        |
| --------------------------- | ---------------------- | -------------------------------- |
| ① 数据清洗                 | `src/cleaning.py`    | `outputs/01_cleaning.html`     |
| ② 数据划分                 | `src/split.py`       | `outputs/02_split.html`        |
| ③ 数据相关性分析（含 EDA） | `src/correlation.py` | `outputs/03_correlation.html`  |
| ④ 模型训练                 | `src/training.py`    | `outputs/04_training.html`     |
| ⑤ 模型评测                 | `src/evaluation.py`  | `outputs/05_evaluation.html`   |
| 共享工具                    | `src/common.py`      | —                               |
| 一键入口                    | `run.sh`             | 生成 `outputs/index.html` 总览 |

## 各模块功能清单

### ① 数据清洗 `cleaning.py`

1. 加载原始 Excel：价格 96 点明细 + 供需日前/实际多 sheet
2. 完整清洗策略：异常电价过滤、负荷类缺失前向填充、电价缺失剔除、datetime 去重、无效列剔除（缺失率 >80% 或方差=0）
3. **仅保留 2025 年数据**
4. 持久化清洗缓存到 `outputs/cleaned_data.pkl`，下游模块复用
5. 输出清洗前后样本数对比、目标分布对比图、逐月样本数
6. HTML 报告：`outputs/01_cleaning.html`

### ② 数据划分 `split.py`

1. 对 2025 数据做 4 个候选切分方案诊断对比：
   - A：训 1-8 / 验 9-10 / 测 11-12
   - **B（推荐）**：训 1-10 / 验 11 / 测 12
   - C：训 1-9 / 验 10-11 / 测 12
   - D：训 1-10 / 验 12 / 测 11（错位对照，时序违法）
2. 评分准则（每条 1 分，满分 3 分）：
   - 训练占比 ≥ 80%
   - 时序严格 train < val < test
   - 训练 vs 测试均值漂移 < 20%
3. 计算 KS 距离 + 训-测均值漂移 + 时序合法性
4. 选定方案 B，持久化 `outputs/split.json`
5. 输出 2025 逐月分布箱线图、方案对比柱图
6. HTML 报告：`outputs/02_split.html`

### ③ 数据相关性分析（含 EDA） `correlation.py`

**职责**：整合原 EDA 与相关性分析的全部能力，单一模块单一 HTML 出口。

1. 数据整体可视化：缺失率、各数值字段分布直方图
2. 电价时序特征：全周期日均走势、月度均价柱状图、24 小时模式（按月分线）、小时×月热力图
3. 全因子 Pearson 相关性矩阵
4. 与目标 |r| 排序的因子条形图（Top 20）
5. 峰/平/谷分时段相关性差异
6. Top 4 主因子 vs 目标散点图（含一次拟合线）
7. day-ahead 可用性标注：泄漏列（实时节点电价）、`_实际`列、可用列三类区分
8. 电力业务语义解读（推升/抑制因子、日内峰谷、新能源挤压效应、月度波动等）
9. 持久化相关性矩阵：`outputs/correlation.pkl`
10. HTML 报告：`outputs/03_correlation.html`

### ④ 模型训练 `training.py`

1. 加载清洗缓存 + 切分配置
2. day-ahead 合法特征工程（详见 `common.build_features`）：
   - 周期编码：小时/月/星期 sin/cos
   - target lag：lag_{24, 48, 72, 120, 168}h（严格 ≥ 24h）
   - rolling 统计：基于 shift(24).rolling
   - 时段 one-hot
   - 业务衍生：新能源渗透率_日前、竞价空间紧张度_日前
   - 显式禁用：实时节点电价、所有 `_实际` 列、lag < 24h
3. **残差预测架构**：训 residual = y - 日前价，最终预测 ŷ = 日前价 + XGB(residual)
4. 抗过拟合配置：depth=4 / min_child_weight=30 / reg_lambda=15 / colsample=0.5
5. 线性月权（近月权重更高）
6. 三集分别计算 MAE/RMSE/MSE/MAPE + 残差 R² + 方向命中率
7. 持久化模型 `outputs/model.joblib` + 完整指标 `outputs/metrics.pkl`
8. 输出训练曲线 + 特征重要性 Top 20
9. HTML 报告：`outputs/04_training.html`

### ⑤ 模型评测 `evaluation.py`

1. 复用 `metrics.pkl` 的测试集预测序列（不重训）
2. 测试集可视化：真实 vs 预测时序对比、散点图、误差分布直方图
3. 分段诊断：按时段（峰/平/谷）、按电价四分位、按 24 小时
4. Naive baseline 对比（day-ahead 合法）：
   - B7' 直接采用日前价
   - B2' 前 1 天同时刻
   - B3' 前 1 周同时刻
   - B4 训练集均值
   - 本项目 XGB 生产模型
5. 诚实呈现：MAE/RMSE 改进率、是否跑赢 baseline、过拟合 R² gap
6. 生成总览导航页 `outputs/index.html`
7. HTML 报告：`outputs/05_evaluation.html`

## 输出清单

```
outputs/
├── index.html              # 总览导航（5 份 HTML 链接 + 关键指标摘要）
├── 01_cleaning.html
├── 02_split.html
├── 03_correlation.html
├── 04_training.html
├── 05_evaluation.html
├── cleaned_data.pkl        # 2025 全年清洗缓存
├── split.json              # 切分配置
├── correlation.pkl         # 相关性矩阵 + 分段相关性
├── model.joblib            # 生产模型
├── metrics.pkl             # 训练 + 评估完整指标
└── plots/*.png             # 所有可视化图表（base64 已内嵌 HTML，PNG 可独立查看）
```

## 约束

1. **无功能冗余**：同一类能力只在一处实现。EDA + 相关性分析合并到 ③，不保留独立的 `analyzer.py`、`report.txt`、`report.md`、`report.html`、`plots/` 顶层目录等过时产物。
2. **单一 HTML 出口**：每个模块只生成一个 HTML，不输出 txt/md，不做多格式追加。
3. **运行入口归一**：所有流程通过 `bash run.sh` 触发：
   - `--all`：全流程一键
   - `--module <name>`：单模块独立运行（用于调试）
   - `--clean`：清空 `outputs/`
   - `--help`：帮助
4. **时序严格**：train < val < test 按年月切分，禁止随机打乱样本。
5. **缓存优先**：清洗后必须持久化；下游模块加载缓存，不重复清洗。
6. **day-ahead 合法性**：禁止任何泄漏列、`_实际` 列、lag < 24h 进入模型。
7. **数据范围**：仅 2025 全年，2026 数据不参与任何环节。
