# 负荷特征使用情况检查报告

## 数据源分类

### ✓ 可用数据（日前预测，D-1已知）
来源：`负荷预测_96点宽表表头.xlsx`

**12个特征列**：
1. `system_load_forecast_mw` - 系统负荷预测（省调负荷_日前）
2. `power_supply_demand_balance_forecast_mw` - 供需平衡预测
3. `external_exchange_plan_mw` - 外送计划
4. `generation_total_forecast_mw` - 总发电预测
5. `non_market_unit_output_forecast_mw` - 非市场机组出力预测
6. `renewable_total_output_forecast_mw` - 新能源总出力预测
7. `wind_output_forecast_mw` - 风电出力预测
8. `solar_output_forecast_mw` - 光伏出力预测
9. `hydro_pumped_storage_output_forecast_mw` - 水电/抽蓄出力预测

**特点**：
- D-1天就已知的预测值
- 符合day-ahead场景
- ✓ 已被模型使用

---

### ✗ 不可用数据（实时实际值，与目标同时产生）
来源：`负荷实际_96点宽表表头.xlsx`

**6个特征列**：
1. `actual_generation_total_mw` - 实际总发电
2. `non_market_unit_output_actual_mw` - 非市场机组实际出力
3. `renewable_total_output_actual_mw` - 新能源实际出力
4. `wind_output_actual_mw` - 风电实际出力
5. `solar_output_actual_mw` - 光伏实际出力
6. `hydro_pumped_storage_output_actual_mw` - 水电/抽蓄实际出力

**特点**：
- 与统一结算点电价同时刻产生
- 如果使用会造成数据泄漏
- ✗ 被 `select_feature_cols` 过滤（含"_实际"关键字）

---

## 代码验证

### src/common.py 第429行
```python
def select_feature_cols(df: pd.DataFrame, target_col: str = TARGET_COL) -> List[str]:
    """day-ahead 合法特征列：剔除目标、_实际、泄漏列、非数值列。"""
    ...
    for c in df.columns:
        ...
        if "_实际" in c:
            continue  # ← 所有实际值列被过滤
        ...
```

### src/cleaning.py 第188-195行
```python
# 合并负荷预测和负荷实际
load_actual_dir = os.path.join(data_dir, "负荷实际")
load_forecast_dir = os.path.join(data_dir, "负荷预测")

fp_load_actual = os.path.join(load_actual_dir, "负荷实际_96点宽表表头.xlsx")
fp_load_forecast = os.path.join(load_forecast_dir, "负荷预测_96点宽表表头.xlsx")
```

清洗阶段会合并两张表，但特征工程阶段会过滤掉"_实际"列。

---

## 结论

### ✓ 正确使用
- **只使用日前预测值**（负荷预测表的12列）
- **不使用实时实际值**（负荷实际表的6列被过滤）
- 符合day-ahead场景的业务约束

### 无数据泄漏风险
- 所有使用的负荷特征都是D-1天已知的预测值
- 实际值通过`if "_实际" in c: continue`被严格过滤
- 满足日前市场预测的合法性要求

---

## 当前模型使用的负荷相关特征

从特征工程（59个特征）中，负荷相关的包括：

**原始预测值**（9个）：
- 省调负荷(MW)_日前
- 新能源负荷(MW)_日前
- 风电出力(MW)_日前
- 光伏出力(MW)_日前
- 水电出力(MW)_日前
- 竞价空间(MW)_日前
- 供需平衡预测(MW)_日前
- 总发电预测(MW)_日前
- 非市场机组出力(MW)_日前

**衍生特征**（2个）：
- 新能源渗透率_日前 = 新能源负荷 / 省调负荷
- 竞价空间紧张度_日前 = 竞价空间 / 省调负荷

**全部都是日前预测值，无泄漏。**

---

## 补充：target lag 特征的合法性

### 用户疑问
> "lag特征不是实际的吗？"

### 回答：lag ≥ 24h 是合法的

**target lag 特征定义**：
```python
TARGET_COL = "实时统一结算点电价(元/MWh)"  # 预测目标

# lag 特征
target_lag_24h = df[TARGET_COL].shift(24)   # D-1天同时刻的实时价
target_lag_48h = df[TARGET_COL].shift(48)   # D-2天同时刻的实时价
target_lag_168h = df[TARGET_COL].shift(168) # D-7天同时刻的实时价
```

**时间可用性分析**：

预测 D 天 12:00 的实时价时：
- ✓ D-1 天 12:00 的实时价：**D-1天已出清并公布**（15min后）
- ✓ D-2 天 12:00 的实时价：**D-2天已公布**
- ✓ D-7 天 12:00 的实时价：**D-7天已公布**

**关键时间线**：
```
D-1 天 00:00  →  日前市场出清（日前价已知）
D-1 天 12:00  →  实时市场出清（实时价已知，15min后公布）
              →  此时可用于 D 天的预测
D   天 12:00  →  【预测目标】实时统一结算点电价
```

### 数据泄漏判断标准

| 特征 | 可用时间 | 是否泄漏 |
|---|---|---|
| `target_lag_24h` | D-1天已公布 | ✓ 合法 |
| `target_lag_168h` | D-7天已公布 | ✓ 合法 |
| `日前价` | D-1天00:00已公布 | ✓ 合法 |
| **负荷实际值（D天同时刻）** | **与目标同时产生** | **✗ 泄漏** |
| **实时价（D天同时刻）** | **与目标同时产生** | **✗ 泄漏** |

### 代码验证

**正确使用**：
```python
# src/common.py 第336行
for lag in [24, 48, 72, 120, 168]:
    df[f"target_lag_{lag}h"] = df[target_col].shift(lag)
```
- 最小 lag = 24h，保证使用的是 D-1 天的历史值 ✓

**正确过滤**：
```python
# src/common.py 第429行
if "_实际" in c:
    continue  # 过滤同时刻实际值
```
- 过滤掉负荷实际值（与目标同时产生）✓

### 结论

**无数据泄漏**：
1. target lag 特征使用的是**历史已公布**的实时价（≥24h前）
2. 负荷实际值被严格过滤（同时刻，会泄漏）
3. 符合日前市场 day-ahead 预测场景
