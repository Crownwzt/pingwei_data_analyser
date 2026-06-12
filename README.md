# 电力市场数据分析工具

针对省级电力市场逐月「市场价格趋势」与「市场供需情况」Excel 数据，自动化完成数据合并、可视化、相关性分析与多格式报告生成。

## 功能

- **数据加载**：扫描目录下逐月 Excel，自动按 `日期+时间` 关联价格 96 点明细与供需（日前/实际）数据
- **整体可视化**：缺失率、各数值字段分布直方图
- **电价时序**：日均走势 / 月均柱状 / 24 小时模式 / 小时×月份热力图
- **相关性分析**（电价预测视角）：
  - 全因子皮尔逊相关性矩阵
  - 目标电价与各因子的 |r| 排序
  - 峰/平/谷分时段相关性差异
  - 主因子散点 + 一次拟合
- **专业解读**：自动给出推升/抑制因子、日内峰谷小时、月度均价波动、新能源挤压效应等结论
- **报告输出**：同时生成 `report.txt`（纯文本）、`report.md`（Markdown）、`report.html`（带样式 + base64 内嵌图表）

## 环境

- Python 3.10+（推荐使用 anaconda `pytorch-2.10` 环境）
- 依赖：`pandas` `numpy` `matplotlib` `seaborn` `openpyxl` `scipy`
- 中文字体：自动探测 `Noto Sans CJK` / `WenQuanYi` / `SimHei` / `Microsoft YaHei` 等，Linux 服务器无需额外配置

## 数据结构假设

每月一对 Excel：

| 文件 | Sheet | 关键字段 |
|------|-------|----------|
| `*市场价格趋势.xlsx` | `明细` (96 点) | 日期, 时间, 日前/实时 统一结算点电价, 日前/实时 节点电价, 负荷率 |
| `*市场供需情况.xlsx` | `日前` | 省调负荷, 外来送负荷, 新能源, 水电, 光伏, 风电, 发电总出力, 非市场化出力, 竞价空间 |
| `*市场供需情况.xlsx` | `实际` | 同上（无 外来 / 发电总出力） |

## 使用

### 命令行

```bash
# 默认调试路径
python analyzer.py

# 指定数据目录
python analyzer.py /path/to/2025-2026市场情况

# 指定输出目录与目标电价
python analyzer.py /path/to/data \
    --output-dir ./out \
    --target "实时统一结算点电价(元/MWh)"
```

### 作为函数调用

```python
from analyzer import main

main(
    data_dir="/data/ztwen2/project_dir/pingwei_data_analyser/2025-2026市场情况",
    target="实时统一结算点电价(元/MWh)",   # 默认就是这个
)
```

主入口 `main(data_dir)` 仅需一个路径参数即可运行。

## 输出

```
./plots/
  ├── 00_missing_rate.png            缺失率
  ├── 01_distributions.png           各字段分布
  ├── 10_price_daily_trend.png       日均电价
  ├── 11_price_monthly_bar.png       月均电价
  ├── 12_price_hourly_by_month.png   24h 模式（按月）
  ├── 13_price_heatmap_hour_month.png 小时×月份热力图
  ├── 20_corr_heatmap.png            全因子相关矩阵
  ├── 21_corr_with_target.png        目标 vs 因子条形图
  ├── 22_corr_with_target_by_seg.png 峰/平/谷 分段
  └── 23_corr_top_factors_scatter.png Top 4 因子散点

./report.txt    纯文本报告
./report.md     Markdown 报告（IDE/GitHub 预览）
./report.html   HTML 报告（浏览器打开，图表内嵌）
```

## 项目结构

```
pingwei_data_analyser/
├── analyzer.py             主分析脚本（约 900 行）
├── CLAUDE.md               项目角色与代码规范
├── requirements.md         需求文档
├── README.md               本文档
├── 2025-2026市场情况/       数据目录（逐月 Excel 对）
├── plots/                  自动生成的图表
└── report.{txt,md,html}    三种格式报告
```

## 自定义

- **目标电价**：修改 `--target` 或 `main(target=...)`。默认 `实时统一结算点电价(元/MWh)`（即实际结算价）。可改为 `日前统一结算点电价(元/MWh)` 或节点电价。
- **峰平谷划分**：`_to_datetime` 函数中 `_seg(h)`，默认 8-11、18-21 为峰，0-6 为谷，其余为平
- **散点图因子数**：`plot_correlation` 中 `top_factors = ... .head(4)`
- **直方图分箱**：`plot_overview` 中 `bins=50`