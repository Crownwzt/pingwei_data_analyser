# 文件清单与含义

本文档逐一说明工程中每个文件和目录的作用、生成方式、依赖关系。

---

## 根目录文档

| 文件 | 大小 | 含义 | 写入者 |
|:---|---:|:---|:---|
| `README.md` | 10K | **项目入口文档**：工程结构、快速开始、HTML 报告说明、设计要点、运行结果、自定义指南 | 人工维护 |
| `CLAUDE.md` | 4.4K | **项目角色 + 代码规范**：技术栈、模块职责与边界、全局业务规则、代码规范 | 人工维护 |
| `requirements.md` | 6.3K | **需求文档**：5 个模块的功能清单、约束条件、输出清单 | 人工维护 |
| `file_desc.md` | 本文件 | **每个文件的含义说明**（本文件） | 人工维护 |
| `.gitignore` | 360B | Git 忽略规则（缓存/产物） | 人工维护 |

---

## 入口脚本

| 文件 | 大小 | 含义 |
|:---|---:|:---|
| `run.sh` | 5.7K | **一键入口**。支持 `--all` 全流程、`--module <names>` 单/多模块（任意分隔: `, ; / +` 空格）、`--clean` 清空 outputs、`--help`。多模块自动按 pipeline 依赖排序+去重 |

调用示例：

```bash
bash run.sh --all                          # 全流程
bash run.sh --module training,evaluation   # 多模块 (逗号)
bash run.sh --module "split training"      # 多模块 (空格)
bash run.sh --module split+training        # 多模块 (加号)
bash run.sh --module training evaluation   # 多模块 (位置参数)
bash run.sh --clean                        # 清空 outputs/
```

---

## 原始数据

| 路径 | 含义 |
|:---|:---|
| `2025-2026市场情况/` | **原始数据目录**。逐月一对 Excel：`YYYY-MM-DD到YYYY-MM-DD市场价格趋势.xlsx`（96 点价格明细）+ `YYYY-MM-DD到YYYY-MM-DD市场供需情况.xlsx`（日前/实际供需）。工程**仅消费 2025 年文件**，2026 数据完全忽略 |

---

## 源代码 `src/`

### 共享工具

| 文件 | 大小 | 含义 |
|:---|---:|:---|
| `src/__init__.py` | 0B | Python 包标识 |
| `src/common.py` | 15K | **共享工具模块**。集中：路径常量 `PATHS`、目标列/日前列/泄漏列常量、中文字体 `setup_cn_font`、HTML 模板/导航 `render_html`、图片 base64 工具 `img_b64/img_tag`、清洗缓存读写 `load_clean`、小时聚合 `aggregate_hourly`、特征工程 `build_features` / `select_feature_cols` / `prepare_features`、时序切分 `split_by_months` / `Splits` 容器、评估指标 `metrics`、目录创建 `ensure_dirs` |

### 5 个功能模块

| 文件 | 大小 | 入口 | 含义 |
|:---|---:|:---|:---|
| `src/cleaning.py` | 13K | `python -m src.cleaning` | **① 数据清洗**。包含 `load_data` + 6 步清洗策略（异常电价过滤、负荷类前向填充、电价缺失剔除、datetime 去重、无效列剔除、仅保留 2025 年）。输出 `cleaned_data.pkl` + `01_cleaning.html` |
| `src/split.py` | 12K | `python -m src.split` | **② 数据划分**。对 4 候选切分方案（A/B/C/D）做 KS 距离+漂移率+时序合法性评分，选定方案 B。输出 `split.json` + `02_split.html` |
| `src/correlation.py` | 21K | `python -m src.correlation` | **③ 数据相关性分析（含 EDA）**。整合原 analyzer.py 全部 EDA 能力 + 相关性分析。包括缺失率/字段分布/电价时序四视图（日均/月均/24h按月/小时×月热力图）/全因子 Pearson 矩阵/Top 20 条形/峰平谷分段/Top 4 散点拟合/业务语义解读。输出 `correlation.pkl` + `03_correlation.html` |
| `src/training.py` | 33K | `python -m src.training` | **④ 模型训练**。特征工程 → 切分 → 5-seed XGB Ensemble 残差预测 → 验证集学 α 加权融合 → Ablation 消融对比（单 seed / Ensemble α=1 / Ensemble + α*）。抗过拟合配置 (depth=4, λ=15, mcw=30, csbt=0.5)。输出 `model.joblib` + `metrics.pkl` + `04_training.html` |
| `src/evaluation.py` | 28K | `python -m src.evaluation` | **⑤ 模型评测**。三集时序对比/散点/误差分布/分段诊断 + Naive baseline 对比 + 三集逐日图按 TrMAE@10% 排序。输出 `05_evaluation.html` + `index.html` + `outputs/daily/{train,val,test}/` |

### 模块依赖关系

```
cleaning  ──→ cleaned_data.pkl
              │
              ├──→ split        ──→ split.json
              │                      │
              ├──→ correlation  ──→ correlation.pkl
              │                      │
              └──→ training (依赖 cleaned_data.pkl + split.json)
                              ──→ model.joblib + metrics.pkl
                                   │
                                   └──→ evaluation (依赖 cleaned_data.pkl + metrics.pkl)
                                                  ──→ 05_evaluation.html + index.html + daily/
```

---

## 产物目录 `outputs/`

### HTML 报告（含 base64 内嵌图）

| 文件 | 大小 | 含义 |
|:---|---:|:---|
| `outputs/index.html` | 3.7K | **总览导航页**。5 份 HTML 入口 + 关键指标摘要 + 重跑指令 |
| `outputs/01_cleaning.html` | 103K | 清洗策略 / 各步骤样本数 / 目标分布对比 / 逐月样本数（2 张内嵌图） |
| `outputs/02_split.html` | 128K | 2025 逐月分布 / 4 方案评分对比 / 选定方案 B 理由 + 限制声明（2 张内嵌图） |
| `outputs/03_correlation.html` | 2.3M | EDA 6 类图 + 相关性 4 类图 + 业务自动解读（9 张内嵌图，文件大主要因高分辨率全因子热力图） |
| `outputs/04_training.html` | 443K | 残差预测设计 / Ensemble + α 融合 / **Ablation 消融对比** / 特征工程公式表 / 超参表 / 训练曲线 / α 扫描曲线 / 三集评估 / 特征重要性（5 张内嵌图） |
| `outputs/05_evaluation.html` | 1.8M | 三集汇总 / 时序对比 / 散点 / 误差分布 / 分段诊断 / Baseline 对比 / **逐日图按 TrMAE@10% 排序**（14 张内嵌图） |

### 序列化产物（机器读取）

| 文件 | 大小 | 含义 | 内容 |
|:---|---:|:---|:---|
| `outputs/cleaned_data.pkl` | 7.7M | 2025 全年清洗缓存 (15min) | DataFrame, 33,792 行 × 33 列 |
| `outputs/split.json` | 191B | 切分配置 | `{year, train_months, val_months, test_months, scheme}` |
| `outputs/correlation.pkl` | 7.4K | 相关性矩阵 | `{target_corr, full_corr, seg_corr}` 三个 DataFrame/Series |
| `outputs/model.joblib` | 282K | **生产模型 bundle** | `{models: List[XGBRegressor]×5, alpha: float, feature_cols, config, seeds}` |
| `outputs/metrics.pkl` | 469K | 训练 + 评估完整指标 | XGB 配置 / seeds / α / α 扫描曲线 / split 配置 / 特征列 / best_iters / 三集 metrics / 特征重要性 / 三集 datetime/y/da/y_pred 完整序列 |

### 图表 `outputs/plots/` （22 张 PNG）

**清洗模块** (2)

| 文件 | 含义 |
|:---|:---|
| `cleaning_dist.png` | 清洗前后目标变量分布对比直方图 |
| `cleaning_monthly_count.png` | 清洗后 2025 逐月样本数 |

**切分模块** (2)

| 文件 | 含义 |
|:---|:---|
| `split_monthly_box.png` | 2025 逐月电价分布箱线图 + 方案 B 切分区段标注 |
| `split_scheme_compare.png` | 4 方案训练占比 / 综合评分 / KS 距离 / 漂移率柱状对比 |

**相关性模块** (10) — 整合自原 analyzer EDA

EDA 部分：
| 文件 | 含义 |
|:---|:---|
| `eda_distributions.png` | 各数值字段分布直方图网格 |
| `eda_price_daily.png` | 全周期 4 类电价日均走势折线 |
| `eda_price_monthly.png` | 月度均价柱状图 |
| `eda_price_24h_by_month.png` | 24 小时电价模式（按月分线） |
| `eda_price_heatmap.png` | 小时 × 月份电价热力图 |

相关性部分：
| 文件 | 含义 |
|:---|:---|
| `corr_top.png` | 与目标 Pearson 相关性 Top 20 条形图 |
| `corr_heatmap.png` | 全因子相关性矩阵热力图 |
| `corr_by_segment.png` | 峰/平/谷分时段相关性差异 |
| `corr_top4_scatter.png` | Top 4 主因子 vs 目标散点图 + 一次拟合 |

（注：还有 1 张是 `eda_missing.png`，缺失率条形图——如本次未缺失则不生成）

**训练模块** (4)

| 文件 | 含义 |
|:---|:---|
| `training_curve.png` | Ensemble 中代表模型的训练/验证 RMSE 曲线 |
| `training_feat_importance.png` | 5-seed Ensemble 特征重要性均值 Top 20 |
| `training_alpha_curve.png` | 验证集 α 扫描曲线，标记 α*（融合权重学习） |
| `training_ablation.png` | Ablation 8 方案对比柱状图（按测试 MAE 升序） |

**评测模块** (5)

| 文件 | 含义 |
|:---|:---|
| `eval_timeseries.png` | 测试集 真实 / 日前价 / XGB 三条线时序对比 |
| `eval_scatter.png` | 测试集 真实 vs 预测 散点图 + y=x 参考 |
| `eval_err_dist.png` | 测试集 XGB vs B7' 误差分布直方图 |
| `eval_segment.png` | 测试集 4 维分段诊断 (峰平谷/电价四分位/24h/24h 改进%) |
| `eval_baseline.png` | 测试集 5 方法 baseline 横评柱状图 |

### 逐日图 `outputs/daily/`

按 **TrMAE@10% 截尾均值** 升序命名：`rank{NN}_TrMAE{score}_{date}.png`

| 子目录 | 张数 | 命名示例 |
|:---|---:|:---|
| `outputs/daily/train/` | 297 | `rank001_TrMAE000.82_2025-05-31.png` ... `rank297_TrMAE134.25_2025-03-08.png` |
| `outputs/daily/val/` | 30 | `rank01_TrMAE008.39_2025-11-06.png` ... `rank30_TrMAE053.87_2025-11-18.png` |
| `outputs/daily/test/` | 31 | `rank01_TrMAE002.90_2025-12-19.png` ... `rank31_TrMAE061.40_2025-12-26.png` |

**每张图内容**：单日 24 小时点的 真实 / 日前价 B7' / XGB 三条线 + XGB 误差填充带。标题包含 4 个指标：TrMAE@10% / MAE / MdAE / P90。

**排序口径 TrMAE@10%**：去掉当日 24 点中误差最高 10% 与最低 10% 后取均值。抗极端值但保留典型水平区分度。

---

## Python 缓存（自动生成，可忽略）

| 路径 | 含义 |
|:---|:---|
| `src/__pycache__/` | Python 字节码缓存，运行后自动生成。`.gitignore` 已忽略 |

---

## 关键产物速查

| 你想要 | 打开 |
|:---|:---|
| 看项目总览 | `outputs/index.html` |
| 看数据清洗细节 | `outputs/01_cleaning.html` |
| 看切分方案为什么选 B | `outputs/02_split.html` |
| 看 EDA + 相关性分析 | `outputs/03_correlation.html` |
| 看模型设计与 ensemble 消融 | `outputs/04_training.html` |
| 看模型测试集表现 + Baseline 对比 | `outputs/05_evaluation.html` |
| 找测试集最差的一天 | `ls outputs/daily/test/ \| tail -1` |
| 找训练集最好的一天 | `ls outputs/daily/train/ \| head -1` |
| 复用生产模型做新数据推理 | `joblib.load("outputs/model.joblib")` → `bundle["models"]` + `bundle["alpha"]` |
| 看完整训练评估数据 | `pickle.load(open("outputs/metrics.pkl","rb"))` |

---

## 重新生成所有产物

```bash
bash run.sh --clean          # 清空 outputs/
bash run.sh --all            # 全流程 (耗时约 400 秒)
```

或者只重跑某些模块（自动按依赖顺序）：

```bash
bash run.sh --module training,evaluation
```
