# 安徽省电力市场 XGBoost 实时电价预测

针对安徽真实电网交易数据，在 **day-ahead** 场景下预测 `实时统一结算点电价(元/MWh)`。工程按 5 个功能模块独立、单一职责，统一通过 `run.sh` 入口运行；每个模块输出**单一 HTML** 报告（含内嵌图表）。

---

## 1. 工程结构

```
pingwei_data_analyser/
├── run.sh                       # ★ 一键入口 (--all / --module / --clean / --help)
├── src/
│   ├── common.py                共享: 路径 / 字体 / HTML 模板 / 特征工程 / 切分 / 指标
│   ├── cleaning.py              ① 数据清洗 (5 表对齐+中英映射)  → 01_cleaning.html
│   ├── split.py                 ② 数据划分                     → 02_split.html
│   ├── correlation.py           ③ 相关性分析 (含 EDA)          → 03_correlation.html
│   ├── training.py              ④ 模型训练 (残差 + Ensemble + α) → 04_training.html
│   └── evaluation.py            ⑤ 模型评测 + 总览页            → 05_evaluation.html
├── outputs/                     所有产物 (HTML / 模型 / 缓存 / 图表)
│   ├── index.html               ★ 总览导航
│   ├── 01_cleaning.html ~ 05_evaluation.html
│   ├── cleaned_data.pkl         清洗缓存
│   ├── split.json               切分配置 (方案 B)
│   ├── correlation.pkl          相关性矩阵
│   ├── model.joblib             生产模型 (5 seed ensemble + α)
│   ├── metrics.pkl              训练+评估完整指标
│   ├── daily/                   小时粒度逐日对比图 (355 张)
│   ├── daily_15min/             15min 粒度逐日对比图 (355 张)
│   ├── plots/                   HTML 内嵌 PNG
│   └── data_comparison/         新旧数据集对比报告与图表
├── 安徽真实电网交易数据情况/    原始数据 (5 个宽表 + 对照表)
│   ├── 市场交易信息/            96点/24点/1天 3 种粒度
│   ├── 负荷预测/                96点 (日前预测 8 列)
│   ├── 负荷实际/                96点 (实际 6 列, 建模时禁用)
│   ├── 中英文对照表.xlsx        字段中英对照
│   └── 空值情况.txt             已知空值日期说明
├── 外部数据/                    辅助数据（可选）
│   ├── bspi_煤炭指数.csv        BSPI 环渤海动力煤 5500K
│   ├── weather_features.nc      ERA5 天气 F组 (仅 2025 年覆盖)
│   └── read_weather_nc.py       nc 文件读取脚本
├── CLAUDE.md                    项目角色 + 代码规范
├── requirements.md              需求文档
└── README.md                    本文档
```

---

## 2. 快速开始

### 全流程一键运行

```bash
bash run.sh --all
```

依次执行：清洗 → 划分 → 相关性（含 EDA）→ 训练 → 评测。耗时约 150 秒。
完成后打开 `outputs/index.html`。

### 单模块运行（调试用）

```bash
bash run.sh --module cleaning       # ① 数据清洗
bash run.sh --module split          # ② 数据划分
bash run.sh --module correlation    # ③ 相关性分析（含 EDA）
bash run.sh --module training       # ④ 模型训练
bash run.sh --module evaluation     # ⑤ 模型评测
```

多模块运行（自动按依赖顺序）：

```bash
bash run.sh --module training,evaluation
bash run.sh --module "cleaning split training"
```

### 清理与帮助

```bash
bash run.sh --clean    # 清空 outputs/
bash run.sh --help
```

---

## 3. 数据集

### 数据源：安徽真实电网交易数据情况

5 个宽表 + 中英对照表 + 空值说明，混合粒度（15min / 1h / 1day）：

| 文件 | 粒度 | 关键字段 | 用途 |
|---|---|---|---|
| `日前实时出清_96点宽表表头.xlsx` | 15min | 日前/实时出清均价、电量 | 特征（日前）+ 泄漏（实时） |
| `统一结算点电价_24点宽表表头.xlsx` | 1h | 日前/实时统一结算点电价 | **目标列** + 日前价基准 |
| `日前平均申报电价_1day宽表表头.xlsx` | 1day | 日前平均申报电价 | 日频特征 |
| `负荷预测_96点宽表表头.xlsx` | 15min | 系统负荷、发电、新能源等 8 列预测 | day-ahead 合法特征 |
| `负荷实际_96点宽表表头.xlsx` | 15min | 实际发电、新能源出力等 6 列 | 泄漏列（自动过滤） |
| `中英文对照表.xlsx` | — | 英中字段映射 | 字段解释 |

**数据对齐策略**（`src/cleaning.py`）：
- 15min 主键（96 点表）
- 1h 结算价 → 按 `(date, hour_index)` 广播到 15min 4 点
- 1day 申报价 → 按日广播到全天

**时间范围**：2025-01-01 ~ 2026-07-01（含 4 天空值：2024-12-31, 2025-01-12/15, 2026-07-01）
**建模范围**：默认仅使用 **2025 全年**数据（含 2026 数据的对比实验见 `experiments/`）

### 新旧数据集差异

`outputs/data_comparison/` 提供了新旧数据集的对比报告：

- **旧数据**（`2025-2026市场情况`）：逐月 Excel，15min 原生粒度，4 列电价
- **新数据**（当前使用）：5 个宽表，混合粒度，无"节点电价"列

**关键发现**（详见 `data_comparison/comparison_report.html`）：
- 旧数据"实时统一结算点电价"（15min）≈ 新数据"实时出清电价"（15min）（99.6% 一致）
- 新数据"实时统一结算点电价"（1h）是**独立结算量**，非 15min 出清价的算术平均
- 日前统一结算点电价 ≈ 96 点表电量加权（MAE = 0.06）
- 实时统一结算点电价 ≠ 96 点表任何简单聚合（Max 差 45 元）

---

## 4. 五份 HTML 报告

| 报告 | 内容 |
|---|---|
| **`01_cleaning.html`** | 8 条清洗规则、逐步样本数变化、目标分布对比、逐月样本数 |
| **`02_split.html`** | 逐月分布诊断、4 候选切分方案 (A/B/C/D) 评分、选定方案 B 的理由 |
| **`03_correlation.html`** | EDA + 相关性合并：缺失率、日均/月均/24h/小时×月热力图、Top 20 Pearson、泄漏识别、峰平谷分段、Top 4 散点、业务解读 |
| **`04_training.html`** | 残差预测设计、抗过拟合超参、训练曲线、三集 MAE/RMSE/R²、特征重要性 Top 20、Ablation 消融 |
| **`05_evaluation.html`** | 三集汇总、测试集时序+散点+误差分布、分段诊断、5 方法 baseline 横评、逐日图链接 |

---

## 5. 核心设计要点

### 5.1 预测场景：Day-ahead

D-1 日提前预测 D 日全时段电价：
- 所有 `_实际` 列（D 日实际负荷/出力）**不可用** → 自动过滤
- lag < 24h **不可用**（D 日所有点对 D-1 日都是未来）
- 日前价（`日前统一结算点电价`）**可用**（D-1 日 14:00 已出清）
- lag ≥ 24h 的历史实时价**可用**（已公布的历史数据）

### 5.2 数据切分（方案 B）

```
训练: 2025-01 ~ 2025-10   (~7,056h, ~83%)
验证: 2025-11             (~720h)
测试: 2025-12             (~744h)
```

时序严格 train < val < test，无随机打乱。方案对比见 `02_split.html`。

### 5.3 数据泄漏识别

- **`实时出清电价(元/MWh)`**（15min）与目标 `实时统一结算点电价`（1h）源自同一次实时出清，同时刻产生 → 显式加入 `common.LEAKAGE_COLS`
- **所有 `_实际` 后缀列**（负荷实际、发电实际、新能源实际等）→ `select_feature_cols` 中 `if "_实际" in c: continue`

### 5.4 残差预测 + Ensemble + α 加权融合

```
ŷ = 日前价 + α · ensemble_mean(XGB_i.predict(residual)),  i = 1..5

其中:
  residual_train = y_train − 日前价_train
  ensemble:      5 个 XGB，seeds=[42, 7, 137, 2024, 9527]
  α:             验证集扫描 [0, 1] 学最优融合权重
```

三层设计动机：

1. **残差预测**：日前价凝聚了 D-1 日全市场博弈的信号；直接预测 `y` 会让 XGB 重复学一遍日前价 + 叠加噪声，反而更差。残差预测让 XGB 只学日前价漏掉的增量。
2. **5-seed Ensemble**：相同超参，`random_state` 不同；`subsample=0.7 + colsample_bytree=0.5` 提供随机源，均值降低方差。
3. **α 加权融合**：向日前价方向收缩，抑制 XGB 过激修正；α=1 表示完全信 XGB，α=0 退化为 B7'。

### 5.5 day-ahead 合法特征（当前 59 个）

- **周期编码 一阶**：`{小时,月,星期}_sin/cos`
- **周期编码 二阶谐波**：`{小时,月}_sin2/cos2`（捕捉半日/半年）
- **季节 + 工作日**：`季节_{春夏秋冬}` one-hot、`是否工作日`
- **target lag**：`target_lag_{24,48,72,120,168}h`（≥ 24h 合法）
- **rolling 统计**：基于 `shift(24).rolling(24/168)` 的均值/标准差
- **差分**：`target_yest_vs_lastweek = lag_24 − lag_168`
- **时段 one-hot**：峰 (8-11, 18-21) / 平 / 谷 (0-6)
- **业务衍生**：`新能源渗透率_日前`、`竞价空间紧张度_日前`
- **煤价 F组**（4）：`bspi_current/ma7/diff30d/yoy`
- **天气 F组**（8）：`{ghi,wind_speed,t2m,tcc}_{lag1d,diff1d}`（仅 2025 有效）
- **原始市场特征**：日前预测的省调负荷、新能源、风光水电、外送计划、发电总出力等

特征清单与来源见 `04_training.html` 第三节。

### 5.6 抗过拟合超参

```python
max_depth=4, min_child_weight=30, reg_lambda=15,
colsample_bytree=0.5, subsample=0.7, learning_rate=0.03,
early_stopping_rounds=80
```

---

## 6. 当前运行结果

### 主指标（测试集 = 2025-12, 744h）

| 数据集 | 样本 | XGB MAE | B7' MAE | MAE 改进 | RMSE 改进 | R² |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 7,056h | 39.15 | 43.48 | **+9.97%** | +11.78% | 0.195 |
| 验证 | 720h | 32.36 | 35.07 | **+7.73%** | +15.16% | 0.194 |
| **测试** | **744h** | **36.79** | **39.92** | **+7.84%** ✅ | **+8.72%** | **0.122** |

α*=0.44, 5 seed ensemble

### 测试集 Baseline 横评（MAE 升序）

| 排名 | 方法 | MAE | RMSE | MAPE% |
|:---:|:---|---:|---:|---:|
| 🥇 | **XGB 生产模型 (本项目)** | **36.79** | **72.77** | **19.99** |
| 🥈 | B7' 日前价 (D-1 已知) | 39.92 | 79.72 | 21.73 |
| 3 | B2' 前 1 天同时刻 | 58.50 | 104.35 | 40.54 |
| 4 | B3' 前 1 周同时刻 | 70.83 | 120.82 | 52.28 |
| 5 | B4 训练集均值 | 130.91 | 141.67 | 84.80 |

### Ensemble + α 消融（测试集）

| 方法 | MAE | RMSE | vs B7' MAE% |
|:---|---:|---:|---:|
| **Ensemble + α*=0.44 ★** | **36.79** | 72.77 | **+7.84%** |
| Single seed=7 (α=1) | 37.99 | 69.99 | +4.82% |
| Ensemble (α=1) | 38.27 | 69.64 | +4.13% |
| Single seed=2024 (α=1) | 38.31 | 69.31 | +4.03% |
| Single seed=9527 (α=1) | 38.31 | 69.98 | +4.03% |
| Single seed=137 (α=1) | 38.55 | 69.79 | +3.42% |
| Single seed=42 (α=1) | 39.72 | 69.98 | +0.49% |
| B7' 日前价 baseline | 39.92 | 79.72 | 0% |

详细诊断（时段/电价四分位/24h 分段、逐日对比图）见 `outputs/05_evaluation.html`。

---

## 7. 环境与依赖

- Python 3.10+（推荐 anaconda `pytorch-2.10` 环境）
- 必需库：`pandas` `numpy` `matplotlib` `seaborn` `openpyxl` `scipy` `xgboost>=2.0` `scikit-learn>=1.3` `joblib>=1.3` `tabulate`
- 可选：`xarray` `netCDF4`（读取 `weather_features.nc`，不装时天气特征跳过不报错）
- 中文字体：自动探测 `Noto Sans CJK` / `WenQuanYi` / `SimHei` / `Microsoft YaHei`

`run.sh` 默认使用 `/data/ztwen2/envs_dir/anaconda3/envs/pytorch-2.10/bin/python`。其他机器修改 `run.sh` 顶部 `PY=`。

---

## 8. 自定义

| 调整目标 | 修改位置 |
|---|---|
| 目标电价列 | `src/common.py` 中 `TARGET_COL` |
| 日前价基准列 | `src/common.py` 中 `DA_COL` |
| 数据泄漏列黑名单 | `src/common.py` 中 `LEAKAGE_COLS` |
| 切分月份 | `src/common.py` 中 `DEFAULT_*_MONTHS`（或直接编辑 `outputs/split.json` 后跳过 split） |
| 峰平谷划分 | `src/common.py` 中 `_seg_label()` |
| XGB 超参 | `src/training.py` 中 `XGB_CFG` |
| Ensemble seeds | `src/training.py` 中 `ENSEMBLE_SEEDS` |
| 原始数据目录 | `src/common.py` 中 `DATA_DIR_DEFAULT` |
| Python 解释器路径 | `run.sh` 中 `PY` 变量 |

---

## 9. 附加实验

`experiments/` 目录存放独立的对比实验脚本（不修改 src/ 主流程）：

- `no_weather_3way_comparison.py` — 天气特征消融 + 多年数据扩充实验
- 其他实验脚本与结果 pkl

关键结论（详见实验脚本内注释）：
- **天气特征贡献微小**：去掉 8 个天气特征，测试 MAE 从 36.79 → 36.92（+0.13 元）
- **多年数据扩充有效**：训练集 7,056h → 11,400h，测试 MAE 从 36.92 → 36.58（−0.33 元）
- **合并方案**（无天气 + 2025+2026）：测试 MAE = 36.58（vs 当前 −0.21 元）

实验通过独立脚本运行，主流程 `src/` 保持不变，可完整复现。
