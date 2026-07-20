# 项目：安徽省电力市场 XGBoost 实时电价预测

## 角色
我是资深 Python 电力交易预测数据分析师，写工业级、可直接运行的数据分析与建模代码。

## 技术栈
- 在用户 anaconda 环境中的 `pytorch-2.10` 环境下可运行
- 不自行安装库，需要安装请让用户协助
- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scipy
- xgboost>=2.0, scikit-learn>=1.3, joblib>=1.3, openpyxl, tabulate
- 可选：xarray + netCDF4（读取 weather_features.nc，装了才启用天气特征）
- 输出中文不乱码（自动探测 Noto Sans CJK / WenQuanYi / SimHei 等字体）

## 代码规范
- 函数式、模块化、单一职责
- 详细注释，特别是电力业务相关的衍生逻辑
- 异常处理
- 不写多余功能、避免功能冗余（同一类能力只在一处实现）

---

# 工程架构

整个工程分 5 个**功能模块**，单一职责，通过 `run.sh` 统一入口运行；
共享逻辑下沉到 `src/common.py`；所有产物集中输出到 `outputs/`。

```
pingwei_data_analyser/
├── run.sh                    一键入口（--all / --module <name> / --clean / --help）
├── src/
│   ├── common.py             共享工具：路径常量、字体、HTML 模板、特征工程、切分、指标
│   ├── cleaning.py           ① 数据清洗（5 表对齐 + 中英映射）→ outputs/01_cleaning.html
│   ├── split.py              ② 数据划分     → outputs/02_split.html
│   ├── correlation.py        ③ 数据相关性分析（含 EDA） → outputs/03_correlation.html
│   ├── training.py           ④ 模型训练（残差+Ensemble+α） → outputs/04_training.html
│   └── evaluation.py         ⑤ 模型评测     → outputs/05_evaluation.html
├── outputs/                  所有产物（HTML / 模型 / 缓存 / 图表）
├── 安徽真实电网交易数据情况/ 原始数据（5 个宽表 + 中英对照表 + 空值说明）
├── 外部数据/                 辅助数据（bspi_煤炭指数.csv、weather_features.nc）
└── experiments/              独立对比实验（不修改 src/）
```

## 全局固定业务规则

1. **预测目标**：`实时统一结算点电价(元/MWh)`（新数据集中为 1h 粒度，来自 24 点表）
2. **数据源**：安徽真实电网交易数据情况（5 个宽表，混合粒度 15min/1h/1day）
3. **数据范围**：默认仅使用 **2025 全年**数据（含 2026 数据的对比实验放在 `experiments/`）
4. **数据切分**：时序严格按年月，**禁止随机打乱**
   - 训练集：2025-01 ~ 10（≥ 80% 占比）
   - 验证集：2025-11
   - 测试集：2025-12
5. **模型评估指标**：MAE、MSE、RMSE、MAPE
6. **数据泄漏**：`实时出清电价(元/MWh)`（15min 出清价）与目标（1h 统一结算价）源自同一次实时出清，同时刻产生，必须显式列入黑名单
7. **day-ahead 合法性**：
   - 所有 `_实际` 后缀列（负荷实际、发电实际等）在预测时刻未知，禁用
   - 所有 target lag < 24h 在 day-ahead 场景下属未来信息，禁用
   - lag ≥ 24h 的历史实时价合法（已公布的历史数据）
8. **产物路径**：
   - 清洗缓存：`outputs/cleaned_data.pkl`
   - 切分配置：`outputs/split.json`
   - 相关性矩阵：`outputs/correlation.pkl`
   - 生产模型：`outputs/model.joblib`
   - 评估指标：`outputs/metrics.pkl`
   - 所有图表：`outputs/plots/`
   - 逐日对比图：`outputs/daily/`（小时）+ `outputs/daily_15min/`（15min）

## 数据源结构

新数据集 `安徽真实电网交易数据情况/` 包含 5 个宽表：

| 文件 | 粒度 | 关键字段 | 用途 |
|---|---|---|---|
| `市场交易信息/日前实时出清_96点宽表表头.xlsx` | 15min | 日前/实时出清均价、电量 | 特征（日前）+ 泄漏（实时） |
| `市场交易信息/统一结算点电价_24点宽表表头.xlsx` | 1h | 日前/实时统一结算点电价 | **目标列** + 日前价基准 |
| `市场交易信息/日前平均申报电价_1day宽表表头.xlsx` | 1day | 日前平均申报电价 | 日频特征 |
| `负荷预测/负荷预测_96点宽表表头.xlsx` | 15min | 8 列日前预测（负荷、发电、新能源等） | day-ahead 合法特征 |
| `负荷实际/负荷实际_96点宽表表头.xlsx` | 15min | 6 列实际值 | 泄漏列（自动过滤） |

**已知空值**：2024-12-31、2025-01-12、2025-01-15、2026-07-01 目标列全空。

**对齐策略**（`src/cleaning.py::load_data`）：
- 15min 主键（96 点表）
- 1h 结算价 → 按 `(date, hour_index)` 广播到 15min 4 点
- 1day 申报价 → 按日广播到全天
- 中英列名映射（表头是英文，项目沿用中文命名）

## 模块职责与边界

| 模块 | 职责 | 唯一 HTML |
|------|------|----------|
| ① `cleaning.py` | 5 表加载对齐 + 中英映射 + 清洗策略 + 2025 筛选 + 缓存 | `01_cleaning.html` |
| ② `split.py` | 4 切分方案诊断 + 选定方案 B + 持久化 split.json | `02_split.html` |
| ③ `correlation.py` | **EDA + 相关性分析**（合并原 analyzer 全部能力）：缺失率、字段分布、电价日均/月均/24h 模式/小时×月热力图、全因子 Pearson 矩阵、目标 \|r\| 排序、峰平谷分段、Top 主因子散点、业务语义解读 | `03_correlation.html` |
| ④ `training.py` | 特征工程（59 特征）+ 残差预测 + 5-seed Ensemble + α 加权融合 + Ablation | `04_training.html` |
| ⑤ `evaluation.py` | 测试集时序/散点/误差/分段诊断 + Naive baseline 对比 + 逐日对比图 + 总览 index.html | `05_evaluation.html` |

## 代码规范

1. **无功能冗余**：同一类能力只在一处实现。EDA 与相关性分析合并到 `correlation.py`，**不保留独立的 analyzer.py**；不输出 `report.md`，所有报告统一为模块对应的 HTML。
2. **统一报告出口**：每个模块输出**单一 HTML**，HTML 内嵌图表（base64）。不生成 txt/md/混合格式。
3. **共享工具下沉**：路径、字体、HTML 模板、特征工程、切分、指标全部放 `src/common.py`，模块按需 import。
4. **入口归一**：所有运行通过 `run.sh` 触发，模块独立可调（`--module <name>`），全流程一键（`--all`）。
5. **缓存优先**：清洗后必须持久化 `cleaned_data.pkl`；下游模块优先加载缓存，不重复清洗。
6. **业务注释**：所有电力业务相关的衍生特征/规则必须有注释说明物理含义。
7. **异常处理**：清洗、加载、训练全流程必须有 try/except 或前置检查（FileNotFound、列缺失等）。
8. **主流程不改**：探索性实验放到 `experiments/` 独立脚本，不修改 `src/`。
