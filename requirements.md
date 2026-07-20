# 需求：安徽省电力市场 XGBoost 实时电价预测工程

## 输入
- **数据目录路径**：默认 `/data/ztwen2/project_dir/pingwei_data_analyser/安徽真实电网交易数据情况`
  - 5 个宽表（混合粒度 15min/1h/1day）+ 中英对照表 + 空值说明
  - 工程**默认仅使用 2025 全年数据**（含 2026 的对比实验放在 `experiments/`）
- **切分配置**：`outputs/split.json`（由 ② 数据划分模块自动生成）

## 工程结构

5 个功能模块 + 1 个共享工具 + 1 个入口脚本，单一职责、无功能冗余：

| 模块 | 文件 | 唯一 HTML |
|------|------|-----------|
| ① 数据清洗 | `src/cleaning.py` | `outputs/01_cleaning.html` |
| ② 数据划分 | `src/split.py` | `outputs/02_split.html` |
| ③ 数据相关性分析（含 EDA） | `src/correlation.py` | `outputs/03_correlation.html` |
| ④ 模型训练 | `src/training.py` | `outputs/04_training.html` |
| ⑤ 模型评测 | `src/evaluation.py` | `outputs/05_evaluation.html` |
| 共享工具 | `src/common.py` | — |
| 一键入口 | `run.sh` | 生成 `outputs/index.html` 总览 |

## 各模块功能清单

### ① 数据清洗 `cleaning.py`
1. 加载 5 个原始宽表（日前实时出清 96 点、统一结算点电价 24 点、日前申报 1day、负荷预测 96 点、负荷实际 96 点）
2. 混合粒度对齐：15min 主键，1h 和 1day 数据广播到 15min
3. 中英列名映射（表头是英文，项目沿用中文命名）
4. 完整清洗策略：异常电价过滤、整日污染剔除（有效样本 < 24 则整日删）、负荷类缺失前向填充、电价缺失剔除、datetime 去重、无效列剔除（缺失率 >80% 或方差=0）
5. **默认仅保留 2025 年数据**
6. 持久化清洗缓存到 `outputs/cleaned_data.pkl`（15min 粒度），下游模块复用
7. 输出 8 条清洗规则说明、逐步样本数对比、目标分布对比图、逐月样本数
8. HTML 报告：`outputs/01_cleaning.html`

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
7. day-ahead 可用性标注：泄漏列（实时出清电价）、`_实际`列、可用列三类区分
8. 电力业务语义解读（推升/抑制因子、日内峰谷、新能源挤压效应、月度波动等）
9. 持久化相关性矩阵：`outputs/correlation.pkl`
10. HTML 报告：`outputs/03_correlation.html`

### ④ 模型训练 `training.py`
1. 加载清洗缓存 + 切分配置
2. day-ahead 合法特征工程（59 个特征）：
   - **周期编码**：一阶 + 二阶谐波（小时/月/星期 sin/cos）
   - **季节 + 工作日**：`季节_{春夏秋冬}` one-hot + `是否工作日`
   - **target lag**：`lag_{24, 48, 72, 120, 168}h`（严格 ≥ 24h）
   - **rolling 统计**：基于 `shift(24).rolling(24/168)` 的均值/标准差
   - **差分**：`target_yest_vs_lastweek = lag_24 − lag_168`
   - **时段 one-hot**：峰 (8-11, 18-21) / 平 / 谷 (0-6)
   - **业务衍生**：新能源渗透率_日前、竞价空间紧张度_日前
   - **煤价 F组**（4）：`bspi_current/ma7/diff30d/yoy`
   - **天气 F组**（8）：`{ghi,wind_speed,t2m,tcc}_{lag1d,diff1d}`（仅 2025 有效）
   - **原始市场特征**：日前预测的省调负荷、新能源、风光水电、外送计划、发电总出力等
   - **显式禁用**：实时出清电价、所有 `_实际` 列、lag < 24h
3. **残差预测架构**：训 `residual = y − 日前价`，最终预测 `ŷ = 日前价 + α · ensemble_mean(XGB(residual))`
4. **5-seed Ensemble**：seeds=[42, 7, 137, 2024, 9527]，预测取均值降低方差
5. **α 加权融合**：验证集扫描 [0, 1] 学最优融合权重（向日前价方向收缩）
6. 抗过拟合配置：`depth=4 / min_child_weight=30 / reg_lambda=15 / colsample=0.5 / subsample=0.7`
7. 线性月权（近月权重更高）
8. 三集分别计算 MAE/RMSE/MSE/MAPE + 残差 R² + 方向命中率
9. **Ablation 消融对比**：single seed vs ensemble, α=1 vs α*
10. 持久化模型 `outputs/model.joblib` + 完整指标 `outputs/metrics.pkl`
11. 输出训练曲线 + 特征重要性 Top 20 + α 扫描曲线
12. HTML 报告：`outputs/04_training.html`（含 59 个特征的完整公式表）

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
6. **逐日对比图**（小时粒度 + 15min 粒度）：
   - `outputs/daily/` - 小时粒度逐日对比（355 张）
   - `outputs/daily_15min/` - 15min 粒度逐日对比（355 张）
7. 生成总览导航页 `outputs/index.html`
8. HTML 报告：`outputs/05_evaluation.html`

## 输出清单

```
outputs/
├── index.html              # 总览导航（5 份 HTML 链接 + 关键指标摘要）
├── 01_cleaning.html ~ 05_evaluation.html
├── cleaned_data.pkl        # 清洗缓存（15min 粒度）
├── split.json              # 切分配置（方案 B）
├── correlation.pkl         # 相关性矩阵 + 分段相关性
├── model.joblib            # 生产模型（5 seed ensemble + α）
├── metrics.pkl             # 训练 + 评估完整指标
├── plots/*.png             # HTML 内嵌图表（base64）
├── daily/                  # 小时粒度逐日对比图（355 张）
├── daily_15min/            # 15min 粒度逐日对比图（355 张）
└── data_comparison/        # 新旧数据集对比报告与图表
```

## 约束

1. **无功能冗余**：同一类能力只在一处实现。EDA + 相关性分析合并到 ③，不保留独立的 `analyzer.py`、`report.txt`、`report.md`、`report.html`、`plots/` 顶层目录等过时产物。
2. **单一 HTML 出口**：每个模块只生成一个 HTML，不输出 txt/md，不做多格式追加。
3. **运行入口归一**：所有流程通过 `bash run.sh` 触发：
   - `--all`：全流程一键
   - `--module <name>[,<name>...]`：单/多模块独立运行（用于调试）
   - `--clean`：清空 `outputs/`（保留 `data_comparison/`）
   - `--help`：帮助
4. **时序严格**：train < val < test 按年月切分，禁止随机打乱样本。
5. **缓存优先**：清洗后必须持久化；下游模块加载缓存，不重复清洗。
6. **day-ahead 合法性**：禁止任何泄漏列（实时出清电价）、`_实际` 列、lag < 24h 进入模型。
7. **数据范围**：默认仅 2025 全年，含 2026 的对比实验放在 `experiments/` 独立脚本。
8. **主流程不改**：探索性实验放到 `experiments/` 独立脚本，不修改 `src/`。
