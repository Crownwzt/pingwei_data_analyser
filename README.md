# 平圩电力市场 XGBoost 实时电价预测

针对省级电力市场逐月「市场价格趋势」与「市场供需情况」Excel 数据，完成 day-ahead 场景下的实时电价 (`实时统一结算点电价(元/MWh)`) 预测。

工程按 **5 个功能模块独立**，单一职责、无功能冗余；统一通过 `run.sh` 入口运行；每个模块独立输出**单一 HTML** 报告（含内嵌图表）。

---

## 1. 工程结构

```
pingwei_data_analyser/
├── run.sh                    # ★ 一键入口 (--all / --module / --clean / --help)
├── src/
│   ├── common.py             共享: 路径常量 / 字体 / HTML 模板 / 特征工程 / 切分 / 指标
│   ├── cleaning.py           ① 数据清洗 (含 load_data)  → outputs/01_cleaning.html
│   ├── split.py              ② 数据划分                 → outputs/02_split.html
│   ├── correlation.py        ③ 相关性分析 (含 EDA)      → outputs/03_correlation.html
│   ├── training.py           ④ 模型训练                 → outputs/04_training.html
│   └── evaluation.py         ⑤ 模型评测 + 总览页        → outputs/05_evaluation.html
├── outputs/                  所有产物 (HTML / 模型 / 缓存 / 图表)
│   ├── index.html             ★ 总览导航 (五份 HTML 链接 + 关键指标)
│   ├── 01_cleaning.html
│   ├── 02_split.html
│   ├── 03_correlation.html
│   ├── 04_training.html
│   ├── 05_evaluation.html
│   ├── cleaned_data.pkl       2025 全年清洗缓存
│   ├── split.json             切分配置 (方案 B)
│   ├── correlation.pkl        相关性矩阵 + 分段相关性
│   ├── model.joblib           生产模型 (XGB 残差预测)
│   ├── metrics.pkl            训练 + 评估完整指标
│   └── plots/                 所有可视化 PNG (HTML 已内嵌 base64)
├── 2025-2026市场情况/         原始数据 (逐月 Excel)
├── CLAUDE.md                 项目角色 + 代码规范
├── requirements.md           需求文档
└── README.md                 本文档
```

无 `analyzer.py`、`plots/`、`report.{txt,md,html}` 等冗余文件 — EDA 能力已整合进 `correlation.py`，所有报告统一为模块对应的 HTML。

---

## 2. 快速开始

### 全流程一键运行

```bash
bash run.sh --all
```

依次执行：清洗 → 划分 → 相关性（含 EDA）→ 训练 → 评测。耗时约 90 秒。
完成后打开 `outputs/index.html` 进入总览。

### 单模块运行（调试用）

```bash
bash run.sh --module cleaning      # ① 数据清洗
bash run.sh --module split         # ② 数据划分
bash run.sh --module correlation   # ③ 相关性分析（含 EDA）
bash run.sh --module training      # ④ 模型训练
bash run.sh --module evaluation    # ⑤ 模型评测
```

单模块依赖前序模块产物：训练依赖 `cleaned_data.pkl` + `split.json`；评测依赖 `metrics.pkl`。

### 其他命令

```bash
bash run.sh --clean    # 清空 outputs/ 重新开始
bash run.sh --help     # 查看帮助
```

---

## 3. 五份 HTML 报告内容

| 报告                              | 内容                                                                                                                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`01_cleaning.html`**    | 6 条清洗规则说明、清洗前后逐步样本数、目标分布对比、2025 逐月样本数                                                                                                                             |
| **`02_split.html`**       | 2025 逐月分布诊断、4 候选切分方案 (A/B/C/D) 评分对比、最终选定方案 B 的理由 + 限制声明                                                                                                          |
| **`03_correlation.html`** | **EDA + 相关性合并**：缺失率/字段分布、电价日均/月均/24h/小时×月热力图、Top 20 Pearson 相关 + day-ahead 可用性标注、数据泄漏识别、峰平谷分段、Top 4 散点拟合、全因子热力图、业务语义解读 |
| **`04_training.html`**    | 残差预测设计、抗过拟合超参表、训练/验证 RMSE 曲线、三集 MAE/RMSE/R²、特征重要性 Top 20                                                                                                         |
| **`05_evaluation.html`**  | 三集汇总、测试集时序对比 + 散点 + 误差分布、按时段/电价四分位/24h 分段诊断、5 方法 baseline 对比、诚实结论与适用建议                                                                            |

---

## 4. 核心设计要点

### 4.1 预测场景：Day-ahead

D-1 日提前预测 D 日全时段电价。这意味着：

- 所有 `_实际` 后缀列（D 日实际负荷/出力）**不可用**
- 所有近期 lag (lag < 24h) **不可用** — 因为 D 日所有点对 D-1 日都是未来
- 日前价 (`日前统一结算点电价`) **可用** — D-1 日 14:00 已出清公布

### 4.2 数据切分（方案 B）

```
训练: 2025-01 ~ 10  (6,898 小时点, 83.5%)
验证: 2025-11       (697 小时点)
测试: 2025-12       (698 小时点)
```

通过 4 方案诊断对比选定（详见 `02_split.html`）：训练占比 ≥ 80%、时序严格 train < val < test、训练-测试漂移最低。**仅使用 2025 数据**，2026 数据完全忽略。

### 4.3 数据泄漏识别

`实时节点电价` 与目标 `实时统一结算点电价` 是同一次出清的两个口径（节点价 vs 加权统一价），**同时刻产生**，相关性 ≈ 0.95。已显式列入 `common.LEAKAGE_COLS` 黑名单，自动从特征中剔除。

### 4.4 残差预测 + Ensemble + α 加权融合

```
ŷ = 日前价 + α · ensemble_mean(XGB_i.predict(residual)),  i = 1..5
其中:
  residual_train = y_train - 日前价_train       (训练目标)
  ensemble:        5 个 XGB, 不同 random_state, 预测取均值
  α:               在验证集上扫描 [0, 1] 学到最优融合权重 (粒度 0.01)
```

**三层设计动机**：

1. **残差预测**：日前价是 D-1 日全市场博弈出清的最优预期，已凝聚海量信号。直接预测 `y` 会让 XGB 重复"学一遍"日前价 + 叠加噪声 — 实测反而比直接复制日前价更差。残差预测让 XGB 只学日前价漏掉的增量信号。
2. **Multi-seed XGB Ensemble** (5 个 seed)：相同超参，仅 `random_state` 不同。subsample/colsample 提供随机源，预测取均值降低单模型方差。在小验证集 (697 h) 上尤其稳定。
3. **α 加权融合**：α=1 表示完全相信 XGB 残差，α=0 退化为日前价 baseline。在验证集上学最优 α (实测 α*≈0.44)，**向日前价方向收缩，抑制 XGB 过激修正**。

### 4.5 day-ahead 合法特征工程

**所有特征的精确计算公式见 `outputs/04_training.html` 第三节**。摘要：

- **周期编码**：`小时_sin/cos = sin/cos(2π·h/24)`，月/星期同理
- **target lag**：`y.shift(24/48/72/120/168)` — 严格 ≥ 24h
- **rolling 统计**：`y.shift(24).rolling(24).mean()` — 窗口 lag_24~lag_47 永不碰当前点
- **差分**：`target_yest_vs_lastweek = y.shift(24) - y.shift(168)`
- **时段 one-hot**：峰 (8-11+18-21)、平、谷 (0-6)
- **业务衍生**：`新能源渗透率_日前 = 新能源_日前 / 省调负荷_日前`
- **显式禁用**：`实时节点电价`(同时刻泄漏)、所有 `_实际` 列、lag < 24h

### 4.6 抗过拟合配置

诊断发现训练集残差 R² ≈ 0.44 vs 测试集 R² ≈ 0.13，明显过拟合。通过容量×正则双管齐下扫描固化：

```python
max_depth=4, min_child_weight=30, reg_lambda=15,
colsample_bytree=0.5, subsample=0.7, learning_rate=0.03,
early_stopping_rounds=80
```

---

## 5. 当前运行结果

| 数据集         |           样本 |         XGB MAE |         B7' MAE |            MAE 改进 |        XGB RMSE | B7' RMSE |           RMSE 改进 | 残差 R² |
| -------------- | -------------: | --------------: | --------------: | ------------------: | --------------: | -------: | ------------------: | -------: |
| 训练           |         6,898h |           40.89 |           44.87 |    **+8.86%** |              — |       — |   **+11.20%** |    0.186 |
| 验证           |           697h |           34.15 |           36.08 |    **+5.35%** |              — |       — |   **+14.69%** |    0.184 |
| **测试** | **698h** | **33.65** | **34.79** | **+3.29%** ✅ | **63.65** |    69.63 | **+8.59%** ✅ |    0.147 |

**测试集 Baseline 横评（按 MAE 升序）**：

| 排名 | 方法                            |             MAE |            RMSE |            MAPE |
| :--: | :------------------------------ | --------------: | --------------: | --------------: |
|  🥇  | **XGB 生产模型 (本项目)** | **33.65** | **63.65** | **17.08** |
|  🥈  | B7' 日前价 (D-1 已知)           |           34.79 |           69.63 |           20.15 |
|  3  | B2' 前 1 天同时刻               |           61.39 |          113.48 |           37.30 |
|  4  | B3' 前 1 周同时刻               |           76.60 |          132.78 |           44.31 |
|  5  | B4 训练集均值                   |          135.55 |          147.00 |           67.45 |

**Ensemble + α 收益（vs 单 XGB α=1）**：

| 指标            | 单 XGB 直接预测 | Ensemble + α=0.44 |       提升       |
| :-------------- | --------------: | -----------------: | :--------------: |
| 测试 MAE        |           38.31 |    **33.65** | **-12.2%** |
| 测试 RMSE       |           65.16 |    **63.65** |      -2.3%      |
| 训练 R²        |           0.359 |    **0.186** |      容量↓      |
| 测试 R²        |           0.106 |    **0.147** |      泛化↑      |
| vs B7' baseline |       输 10.09% | **赢 3.29%** |       质变       |

**关键解读**：Ensemble 取均值降低方差，α=0.44 把 XGB 残差向日前价方向收缩 56%。训练 R² 降而测试 R² 升 — 这是真正的泛化提升信号，而非简单调参的拟合优化。

详细诊断与建议见 `outputs/05_evaluation.html`。

---

## 6. 环境与依赖

- Python 3.10+（推荐 anaconda `pytorch-2.10` 环境）
- 必需库：`pandas` `numpy` `matplotlib` `seaborn` `openpyxl` `scipy` `xgboost>=2.0` `scikit-learn>=1.3` `joblib>=1.3` `tabulate`
- 中文字体：自动探测 `Noto Sans CJK` / `WenQuanYi` / `SimHei` / `Microsoft YaHei`，Linux 服务器无需额外配置

`run.sh` 默认使用 `/data/ztwen2/envs_dir/anaconda3/envs/pytorch-2.10/bin/python`。在其他机器上请修改 `run.sh` 顶部 `PY=` 变量。

---

## 7. 自定义

| 调整目标          | 修改位置                                                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| 目标电价列        | `src/common.py` 中 `TARGET_COL`                                                            |
| 日前价基准列      | `src/common.py` 中 `DA_COL`                                                                |
| 数据泄漏列黑名单  | `src/common.py` 中 `LEAKAGE_COLS`                                                          |
| 切分月份          | `src/common.py` 中 `DEFAULT_*_MONTHS`，或直接编辑 `outputs/split.json` 后跳过 split 模块 |
| 峰平谷划分        | `src/common.py` 中 `_seg_label()` 和 `src/cleaning.py` 中 `_to_datetime` 内的 `_seg` |
| XGB 超参          | `src/training.py` 中 `XGB_CFG`                                                             |
| 原始数据目录      | `src/common.py` 中 `DATA_DIR_DEFAULT`                                                      |
| Python 解释器路径 | `run.sh` 中 `PY` 变量                                                                      |

---

## 8. 数据结构假设

每月一对 Excel 文件（任务设定下仅消费 2025 年文件）：

| 文件                   | Sheet            | 关键字段                                                                           |
| ---------------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `*市场价格趋势.xlsx` | `明细` (96 点) | 日期, 时间, 日前/实时 统一结算点电价, 日前/实时 节点电价, 负荷率                   |
| `*市场供需情况.xlsx` | `日前`         | 省调负荷, 外来送负荷, 新能源, 水电, 光伏, 风电, 发电总出力, 非市场化出力, 竞价空间 |
| `*市场供需情况.xlsx` | `实际`         | 同上（无 外来 / 发电总出力）                                                       |
