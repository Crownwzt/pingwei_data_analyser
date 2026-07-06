# 外部数据

从"天气情况/"和"煤价情况/"提炼的核心资产，用于电价预测特征扩展实验。

## 文件清单

| 文件 | 大小 | 内容 | 时间范围 |
|:---|---:|:---|:---|
| `weather_features.nc` | 4.2 MB | 安徽气象 29 特征 (NetCDF, 15min 粒度) | 2025-01-01 ~ 2025-12-31 |
| `weather_features.README.md` | 6.5 KB | 29 个变量的物理含义 + 派生公式 | — |
| `bspi_煤炭指数.csv` | 2.7 KB | 环渤海动力煤 5500K 价格指数 (周频) | 2024-06-05 ~ 2026-06-10 |

## `weather_features.nc` 快速概览

29 个气象/物理派生特征，按物理驱动分组：

| 组 | 特征数 | 示例 |
|:---|---:|:---|
| A. 原始 NWP | 7 | `ghi`, `t2m`, `tcc`, `tp`, `u100/v100`, `sp` |
| C. 光伏派生 | 5 | `ghi_day`, `ghi_squared`, `ghi_clearsky_index`, `ghi_efficiency`, `ghi_per_tcc` |
| D. 风电派生 | 4 | `wind_cube_clipped`, `wind_speed_3hmean/3hstd`, `wind_persistence` |
| E. 负荷派生 | 5 | `t2m_celsius`, `hdd` (制热度日), `cdd` (制冷度日), `t2m_squared`, `t2m_3hmean` |
| F. 滞后差分 | 8 | `{ghi,wind_speed,t2m,tcc}_{lag1d,diff1d}` |

详细定义见 `weather_features.README.md`。

**加载示例**:
```python
import xarray as xr, pandas as pd
ds = xr.open_dataset('外部数据/weather_features.nc')
df = ds.to_dataframe().reset_index().rename(columns={'times': 'datetime'})
```

## `bspi_煤炭指数.csv` 快速概览

| 列 | 类型 | 说明 |
|:---|:---|:---|
| `public_date` | date | 发布日期（每周一次，通常周三） |
| `coal_max_price` | float | 5500 大卡动力煤指数价 (元/吨) |
| `changeprice` | float | 较上期涨跌额 (元/吨) |
| `bl` | float | 较上期涨跌幅 (%) |

**业务意义**: 煤电成本约占我国火电边际成本 70%+。煤价上涨会推高**火电报价下限**，
在竞价空间紧张时段直接抬升出清电价。

**频率对齐**: 煤价周频 → 需要 forward fill 到 15min 粒度（"D 日的煤价" = D 日之前最近一次发布价）。

## 从这里怎么用

已集成到 `src/common.py::build_features()`：

1. **煤价组**（4 特征）：`bspi_current/ma7/diff30d/yoy`  
   - 单 seed α=1 下改善 -6.71%，5-seed+α 下改善 -2.04%  
   - 已落地 ✅

2. **天气 F组**（8 特征）：`{ghi,wind_speed,t2m,tcc}_{lag1d,diff1d}`  
   - 单 seed α=1 下改善 -0.46%，5-seed+α 下改善 -1.04%  
   - 已落地 ✅

3. **天气其他组（A/C/D/E）**：  
   - 分组消融验证无效或负贡献，未落地 ❌

具体增益见 `README.md` 第 5.4 节实验记录。
