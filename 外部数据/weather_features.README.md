# weather_features.nc

## 概述

安徽电网预测任务的扩展天气特征数据集，NetCDF 格式，15 分钟分辨率，2025 全年。
从原始 NWP（`anhui/raw/nwp/*.nc`）区域平均后派生，包含光伏/风电/负荷三类物理驱动的非线性特征。

## 基本信息

- **文件大小**: 4.2 MB
- **时间范围**: 2025-01-01 00:00 ~ 2025-12-31 23:00（北京时间）
- **时间点数**: 35,037（15 分钟分辨率）
- **特征数量**: 29
- **空间分辨率**: 安徽区域整体平均（52×41 网格 → 单值）
- **时区**: Asia/Shanghai (UTC+8)

## 特征清单（按物理驱动分组）

### A. 原始 NWP (7 列)

从原始 nc 文件区域平均得到的 7 个气象通道：

| 特征 | 单位 | 物理意义 |
|------|------|---------|
| `ghi` | W/m² | 短波辐射通量（Global Horizontal Irradiance），驱动光伏出力 |
| `sp` | Pa | 地表气压 |
| `t2m` | K | 2 米气温 |
| `tcc` | 0-1 | 总云量（Total Cloud Cover） |
| `tp` | mm | 总降水 |
| `u100` | m/s | 100 米高度东向风分量 |
| `v100` | m/s | 100 米高度北向风分量 |

### C. 光伏驱动派生 (5 列)

针对光伏出力预测的非线性派生：

| 特征 | 定义 | 物理意义 |
|------|------|---------|
| `ghi_day` | `ghi × 1[6 ≤ hour ≤ 18]` | 只保留白天辐射（夜间 0 是噪声） |
| `ghi_squared` | `ghi²` | 高辐射时光伏效率边际下降 |
| `ghi_clearsky_index` | `ghi / 当日 ghi 最大值` | 晴空指数，反映云遮蔽程度（1=全晴） |
| `ghi_per_tcc` | `ghi / (1 + tcc)` | 云遮蔽前的"理论辐射"代理 |
| `ghi_efficiency` | `ghi × (1 - 0.004×(t°C - 25))` | 光伏组件温升降效（高温降效 0.4%/°C）|

### D. 风电驱动派生 (4 列)

针对风电出力的功率曲线近似 + 稳定性指标：

| 特征 | 定义 | 物理意义 |
|------|------|---------|
| `wind_cube_clipped` | 切入 3 m/s、额定 12 m/s 的限幅 v³ | 风电功率曲线物理近似（3-12 m/s 区间 ∝ v³）|
| `wind_speed_3hmean` | 3 小时滚动均值 | 风电出力短期趋势 |
| `wind_speed_3hstd` | 3 小时滚动标准差 | 风电功率波动率（调度难度指标）|
| `wind_persistence` | 6 小时滚动均值 | 风电出力持续性 |

### E. 负荷驱动派生 (5 列)

针对电网负荷对气温的非线性响应：

| 特征 | 定义 | 物理意义 |
|------|------|---------|
| `t2m_celsius` | `t2m - 273.15` | 摄氏温度 |
| `t2m_squared` | `t2m_celsius²` | 极冷/极热都会升负荷（U 型曲线）|
| `hdd` | `max(0, 18 - t°C)` | 制热度日（Heating Degree Days），>18°C 不需制热 |
| `cdd` | `max(0, t°C - 22)` | 制冷度日（Cooling Degree Days），>22°C 开空调 |
| `t2m_3hmean` | 3 小时滚动均值 | 气温短期趋势 |

### F. 滞后差分 (8 列)

同时段昨日值 + 日际变化量（直接对应残差学习的"今天比昨天的变化"）：

| 特征 | 对象 | 物理意义 |
|------|------|---------|
| `ghi_lag1d` | ghi | 昨天此时刻辐射水平 |
| `ghi_diff1d` | ghi - ghi_lag1d | 今天比昨天辐射多/少多少 |
| `wind_speed_lag1d` | wind_speed | 昨天此时刻风速 |
| `wind_speed_diff1d` | wind_speed - wind_speed_lag1d | 今天比昨天风速增减 |
| `t2m_lag1d` | t2m | 昨天此时刻气温 |
| `t2m_diff1d` | t2m - t2m_lag1d | 今天比昨天气温升降 |
| `tcc_lag1d` | tcc | 昨天此时刻云量 |
| `tcc_diff1d` | tcc - tcc_lag1d | 今天比昨天云量变化 |

注：lag1d 采用同 tod（time of day）滞后，即 shift 24 小时（hourly 基础上）。

## 去除的冗余特征（B 组）

以下 4 个特征在消融实验中证明与 u100/v100 冗余，且会让模型表现变差（MAE 增加 1.15 分），已从最终版本中删除：

- `wind_speed` = √(u100² + v100²)
- `wind_speed_sq` = wind_speed²
- `wind_dir_sin` = sin(arctan2(-u100, -v100))
- `wind_dir_cos` = cos(arctan2(-u100, -v100))

原因：u100/v100 本身就是风的完整表达，LightGBM 能自己组合出 wind_speed，提前派生反而引入分裂冗余。

## 数据质量

### 缺失值

- lag1d / diff1d 特征：前 24 小时（96 个 15 分钟时点）为 NaN（冷启动）
- 其余特征：基本无缺失（原始 nc 文件 2025 年 365 天连续无缺）

### 物理一致性验证

| 验证项 | 相关系数 | 结论 |
|--------|---------|------|
| ghi ↔ 光伏负荷(MW)_实际 | r = +0.932 | 时间对齐正确，物理自洽 |
| ghi ↔ ghi_day | r = +1.000 | 白天保留全部，夜间归零 |
| ghi ↔ ghi_squared | r = +0.952 | 非线性但单调 |
| wind_speed ↔ wind_cube_clipped | r = +0.624 | 弱相关（因限幅 + 切入截断）|

## 使用示例

### Python + xarray

```python
import xarray as xr
import pandas as pd

# 加载
ds = xr.open_dataset('anhui/clean/weather_features.nc')

# 查看结构
print(ds)
print(f"特征数: {len(ds.data_vars)}")
print(f"时间范围: {ds.times.values[0]} ~ {ds.times.values[-1]}")

# 转 DataFrame
df = ds.to_dataframe().reset_index()
df = df.rename(columns={'times': 'datetime'})

# 与其他数据按 datetime 连接
price_df = pd.read_csv('train_clean.csv', parse_dates=['datetime'])
merged = price_df.merge(df, on='datetime', how='left')

ds.close()
```

### 读取单个变量

```python
ghi = ds['ghi'].values         # numpy array, shape (35037,)
times = ds.times.values        # datetime64[ns] array
```

### 查看全局属性

```python
for k, v in ds.attrs.items():
    print(f"{k}: {v}")
```

## 生成方式

本文件由 `anhui/scripts/build_weather_features.py` 生成：

```bash
cd anhui/scripts
python build_weather_features.py --start 2025-01-01 --end 2025-12-31
```

生成流程：
1. 加载 `anhui/raw/nwp/*.nc` 逐日 NWP 文件（365 天）
2. 区域平均（52×41 网格 → 单值）
3. 派生 C/D/E/F 组物理特征
4. 重采样到 15 分钟（线性插值）
5. 写出 NetCDF

## 在建模中的使用

本文件被 `anhui/experiments/exp1-replace/scripts/build_features.py` 自动加载，
与电价历史、日前节点电价等特征拼接成完整训练数据。

最终模型性能（实时节点电价预测）：
- 全局 MAE: 39.41（反超 D-1 基线 31.5%）
- 全局 RMSE: 82.25（反超 32.3%）
- 崩价 MAE: 84.95（反超 40.2%）

天气特征贡献约 +2.0 分 MAE（相对于无天气的纯历史+日前模型）。

## 版本历史

- **当前版本**（29 列，ACDEF 组）：去除冗余 B 组后的精简版
- ~~原版~~（35 列）：含 B 组，已废弃（消融证明 B 组负贡献）

## 许可与引用

本文件基于原始 NWP 数据（`anhui/raw/nwp/`）派生。
使用时请确保符合原始数据的许可协议。
