# -*- coding: utf-8 -*-
"""
① 数据清洗模块
================

职责：
  - 加载原始 Excel (价格趋势 + 供需情况)
  - 仅保留 2025 数据
  - 标准清洗策略 (异常电价、缺失值、去重、无效列)
  - 输出 cleaned_data.pkl
  - 输出 01_cleaning.html (清洗前后统计 + 关键图)

入口：python -m src.cleaning [data_dir]
"""

from __future__ import annotations

import os
import re
import sys
import glob
import argparse
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (
    TARGET_COL, DATA_DIR_DEFAULT, PATHS,
    ensure_dirs, setup_cn_font, render_html, safe_savefig,
    save_pickle, img_tag,
)


# ---------------------------------------------------------------------------
# 数据加载：读取安徽真实电网交易数据的 5 个宽表并对齐到 15min
# ---------------------------------------------------------------------------
# 安徽新数据集：
#   ├── 市场交易信息/日前实时出清_96点宽表表头.xlsx        (15min, 4 价量列 + 日前/实时)
#   ├── 市场交易信息/统一结算点电价_24点宽表表头.xlsx      (1h  , 日前/实时统一结算点电价)
#   ├── 市场交易信息/日前平均申报电价_1day宽表表头.xlsx    (day , 日均申报电价)
#   ├── 负荷预测/负荷预测_96点宽表表头.xlsx                (15min, 日前预测 8 列)
#   └── 负荷实际/负荷实际_96点宽表表头.xlsx                (15min, 实际 6 列)
#
# 对齐策略：以"日前实时出清" 15min 宽表为基准 (52608 行)，其它文件按 datetime 或 date merge。
# 为了兼容既有 XGB 特征工程与 HTML 报告的中文列名，读入后统一映射到项目历史列名。

# 英->中列名映射 (基于对照表 + 项目历史约定)
_EN2CN_MAP = {
    # 供给预测 (day-ahead) → 加 "_日前" 后缀 (与 select_feature_cols 里的 day-ahead 合法性一致)
    "system_load_forecast_mw":                    "省调负荷(MW)_日前",
    "power_supply_demand_balance_forecast_mw":    "供需平衡预测(MW)_日前",
    "external_exchange_plan_mw":                  "外来外送计划(MW)_日前",
    "generation_total_forecast_mw":               "发电总出力预测(MW)_日前",
    "non_market_unit_output_forecast_mw":         "非市场化机组出力(MW)_日前",
    "renewable_total_output_forecast_mw":         "新能源负荷(MW)_日前",
    "wind_output_forecast_mw":                    "风电(MW)_日前",
    "solar_output_forecast_mw":                   "光伏(MW)_日前",
    "hydro_pumped_storage_output_forecast_mw":    "水电抽蓄(MW)_日前",
    # 供给实际 (真实运行) → 加 "_实际" 后缀 (select_feature_cols 里禁用)
    "actual_generation_total_mw":                 "发电总出力(MW)_实际",
    "non_market_unit_output_actual_mw":           "非市场化机组出力(MW)_实际",
    "renewable_total_output_actual_mw":           "新能源负荷(MW)_实际",
    "wind_output_actual_mw":                      "风电(MW)_实际",
    "solar_output_actual_mw":                     "光伏(MW)_实际",
    "hydro_pumped_storage_output_actual_mw":      "水电抽蓄(MW)_实际",
    # 市场交易
    "day_ahead_cleared_energy_mwh":               "日前出清电量(MWh)",
    "real_time_cleared_energy_mwh":               "实时出清电量(MWh)_实际",   # 与目标同期，标记为实际禁用
    "day_ahead_clearing_avg_price_yuan_per_mwh":  "日前节点电价(元/MWh)",
    "real_time_clearing_price_yuan_per_mwh":      "实时出清电价(元/MWh)",     # 15min 出清价, 与目标同时刻产生 → LEAKAGE_COLS
    "day_ahead_unified_settlement_price_yuan_per_mwh":  "日前统一结算点电价(元/MWh)",
    "real_time_unified_settlement_price_yuan_per_mwh":  "实时统一结算点电价(元/MWh)",  # TARGET_COL
    "day_ahead_avg_bid_price_yuan_per_mwh":       "日前平均申报电价(元/MWh)_日前",
}


def _read_excel_normalize(path: str) -> pd.DataFrame:
    """读取 xlsx 并把英文列名翻译为项目中文列名。"""
    df = pd.read_excel(path)
    df = df.rename(columns=_EN2CN_MAP)
    return df


def _parse_datetime_15min(df: pd.DataFrame) -> pd.DataFrame:
    """15min 宽表：解析 date + time_point 到 datetime，并派生 date_key + hour_index。

    电力交易 "区间末点" 命名约定：
      - time_point='00:15' (point_index=1) → [00:00, 00:15) 时段末，归属 D 日第 1 小时
      - time_point='01:00' (point_index=4) → [00:45, 01:00) 时段末，归属 D 日第 1 小时
      - time_point='01:15' (point_index=5) → 归属 D 日第 2 小时
      - time_point='24:00' (point_index=96) → [23:45, 24:00) 时段末，归属 D 日第 24 小时

    hour_index = ceil(point_index / 4) = ((p-1)//4)+1，用于与 24 点表 (date, hour_index) 精确对齐。
    datetime 只作为时间轴显示用：'24:00' 归到当日 23:59 保留独立性。
    """
    df = df.copy()
    date_raw = pd.to_datetime(df["date"])
    df["date_key"] = date_raw.dt.floor("D")
    if "point_index" in df.columns:
        df["hour_index"] = ((df["point_index"].astype(int) - 1) // 4) + 1

    time_str = df["time_point"].astype(str)
    mask_24 = time_str == "24:00"
    date_str = date_raw.dt.strftime("%Y-%m-%d")
    date_str.loc[mask_24] = (date_raw.loc[mask_24] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    time_str = time_str.mask(mask_24, "00:00")

    df["datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    df.loc[mask_24, "datetime"] = df.loc[mask_24, "datetime"] - pd.Timedelta(minutes=1)

    df = df.dropna(subset=["datetime"]).drop(columns=["date", "time_point"], errors="ignore")
    return df


def _parse_datetime_1h(df: pd.DataFrame) -> pd.DataFrame:
    """24 点宽表：保留 date_key + hour_index 作为精确对齐键，同时派生 datetime 便于显示。

    hour_index=k 代表 D 日 [k-1:00, k:00) 区间末，即 hour='k:00' (k=1..23) 或 '24:00' (k=24)。
    """
    df = df.copy()
    date_raw = pd.to_datetime(df["date"])
    df["date_key"] = date_raw.dt.floor("D")
    df["hour_index"] = df["hour_index"].astype(int)

    time_str = df["hour"].astype(str)
    mask_24 = time_str == "24:00"
    date_str = date_raw.dt.strftime("%Y-%m-%d")
    date_str.loc[mask_24] = (date_raw.loc[mask_24] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    time_str = time_str.mask(mask_24, "00:00")

    df["datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    df.loc[mask_24, "datetime"] = df.loc[mask_24, "datetime"] - pd.Timedelta(minutes=1)

    df = df.dropna(subset=["datetime"]).drop(columns=["date", "hour"], errors="ignore")
    return df


def _broadcast_hourly_to_15min_grid(df_h: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """按 (date_key, hour_index) 把 1h 表广播到 15min 网格。

    这是唯一正确的对齐方式：15min 表的 hour_index 由 point_index 派生 (ceil(p/4))，
    与 24 点表的 hour_index 语义完全一致。不使用 datetime.floor('h') 是因为区间末点
    命名约定下 15min '01:00' 属 hour_index=1 而 1h 表 datetime='01:00' 也属 hour_index=1，
    但 floor 会把它们拆到不同小时槽。
    """
    keys = ["date_key", "hour_index"]
    val_cols = [c for c in df_h.columns if c not in keys + ["datetime"]]
    right = df_h[keys + val_cols].drop_duplicates(subset=keys, keep="first")
    merged = grid.merge(right, on=keys, how="left")
    return merged


def _augment_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """给已带 datetime 的宽表补上 年/月/日/小时/星期/是否周末/时段 派生列。"""
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    df["年"] = df["datetime"].dt.year
    df["月"] = df["datetime"].dt.month
    df["日"] = df["datetime"].dt.day
    df["小时"] = df["datetime"].dt.hour
    df["星期"] = df["datetime"].dt.weekday + 1
    df["是否周末"] = (df["星期"] >= 6).astype(int)

    def _seg(h: int) -> str:
        if 8 <= h <= 11 or 18 <= h <= 21: return "峰"
        if 0 <= h <= 6: return "谷"
        return "平"
    df["时段"] = df["小时"].map(_seg)
    return df


def load_data(data_dir: str) -> pd.DataFrame:
    """
    加载安徽真实电网交易数据（5 个宽表），对齐到 15min 粒度并返回宽表。

    对齐策略：
      1. 以 "日前实时出清_96点" (52608 行, 15min) 为主键
      2. "负荷预测_96点" / "负荷实际_96点" 按 datetime 直接 merge
      3. "统一结算点电价_24点" (1h)  → 广播到该小时的 4 个 15min 时段 (同小时同价)
      4. "日前平均申报电价_1day" (day) → 广播到当日全部 96 点
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    market_dir = os.path.join(data_dir, "市场交易信息")
    load_actual_dir = os.path.join(data_dir, "负荷实际")
    load_forecast_dir = os.path.join(data_dir, "负荷预测")

    fp_clear = os.path.join(market_dir, "日前实时出清_96点宽表表头.xlsx")
    fp_settle = os.path.join(market_dir, "统一结算点电价_24点宽表表头.xlsx")
    fp_bid   = os.path.join(market_dir, "日前平均申报电价_1day宽表表头.xlsx")
    fp_load_actual = os.path.join(load_actual_dir, "负荷实际_96点宽表表头.xlsx")
    fp_load_forecast = os.path.join(load_forecast_dir, "负荷预测_96点宽表表头.xlsx")

    for fp in (fp_clear, fp_settle, fp_bid, fp_load_actual, fp_load_forecast):
        if not os.path.exists(fp):
            raise FileNotFoundError(f"缺失原始文件: {fp}")

    # ── 15min 基准表 ──
    df_clear = _parse_datetime_15min(_read_excel_normalize(fp_clear))
    df_clear = df_clear.drop(columns=["point_index"], errors="ignore")

    df_load_fc = _parse_datetime_15min(_read_excel_normalize(fp_load_forecast))
    df_load_fc = df_load_fc.drop(columns=["point_index", "hour_index", "date_key"], errors="ignore")

    df_load_ac = _parse_datetime_15min(_read_excel_normalize(fp_load_actual))
    df_load_ac = df_load_ac.drop(columns=["point_index", "hour_index", "date_key"], errors="ignore")

    # ── 1h 结算点电价 → 按 (date_key, hour_index) 广播到 15min 网格 ──
    df_settle = _parse_datetime_1h(_read_excel_normalize(fp_settle))
    grid = df_clear[["datetime", "date_key", "hour_index"]].copy()
    df_settle_15 = _broadcast_hourly_to_15min_grid(
        df_settle.drop(columns=["datetime"], errors="ignore"),
        grid,
    ).drop(columns=["date_key", "hour_index"], errors="ignore")

    # ── 1day 申报均价 → 按 date_key 广播到 15min ──
    df_bid = _read_excel_normalize(fp_bid)
    df_bid["date_key"] = pd.to_datetime(df_bid["date"]).dt.floor("D")
    df_bid = df_bid.drop(columns=["date"], errors="ignore")
    df_bid_15 = grid[["datetime", "date_key"]].merge(df_bid, on="date_key", how="left") \
                                              .drop(columns=["date_key"])

    # ── 合并 ──
    merged = df_clear.drop(columns=["date_key", "hour_index"], errors="ignore").copy()
    for right in (df_load_fc, df_load_ac, df_settle_15, df_bid_15):
        merged = merged.merge(right, on="datetime", how="left")

    merged["来源文件"] = "安徽真实电网交易数据"
    return _augment_time_columns(merged)


# ---------------------------------------------------------------------------
# 清洗策略
# ---------------------------------------------------------------------------
CLEAN_RULES = [
    ("异常电价过滤", f"剔除 {TARGET_COL} ∉ [0, 2000] 元/MWh 样本",
     "极端报价多为出清异常 / 汇总行污染"),
    ("整日污染剔除", "若某日有效样本数 < 24 (占理论 96 点的 25%) 则整日剔除",
     "上游数据源采集失败的日子 (如 12/19 96 点里 95 个 NaN) 保留 1~几个点参与训练/评估均无意义"),
    ("负荷类缺失填充", "含「负荷/出力/新能源/光伏/风电/水电/竞价空间」关键字列用 forward fill",
     "电力负荷 15min 粒度高度连续，前向填充语义合理"),
    ("电价缺失剔除", "目标列缺失行直接 dropna",
     "目标缺失样本不参与监督学习"),
    ("时序去重", "基于 datetime drop_duplicates(keep='first')",
     "避免多月文件首尾日重叠造成的重复"),
    ("无效列剔除", "缺失率 > 80% 或 std=0 整列剔除",
     "无信号的列只会增加 XGB split 噪声"),
    ("2025 数据筛选", "仅保留 datetime.year == 2025 的样本",
     "任务设定：2026 数据忽略，只用 2025 全年做切分"),
    ("剩余 NaN 兜底", "经过前 6 步仍有任意数值列 NaN 的行整行 dropna",
     "前面已显式处理电价(剔除)/负荷类(ffill)，这一步保证未来若加入新列(气温/燃料价)缺失也不会污染特征矩阵"),
]


def clean(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    price_min: float = 0.0,
    price_max: float = 2000.0,
    missing_thresh: float = 0.8,
) -> Tuple[pd.DataFrame, Dict]:
    """执行清洗并返回 (cleaned_df, stats_dict)。"""
    df_in = df.copy()
    stats = {"n_before": len(df_in), "col_before": df_in.shape[1]}
    # steps: 每步的明细 (步骤名, 前样本数, 本步剔除数, 后样本数, 子原因细分 dict)
    steps: List[Dict] = []

    # 1) 异常电价过滤
    n0 = len(df)
    sub_reasons = {}
    if target_col in df.columns:
        t = df[target_col]
        sub_reasons["目标 NaN"]            = int(t.isna().sum())
        sub_reasons[f"< {price_min:g}"]    = int((t.notna() & (t < price_min)).sum())
        sub_reasons[f"> {price_max:g}"]    = int((t.notna() & (t > price_max)).sum())
        df = df.loc[t.notna() & (t >= price_min) & (t <= price_max)]
    n1 = len(df)
    stats["n_after_price_filter"] = n1
    steps.append({"name": "异常电价过滤", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons})

    # 1.5) 整日污染剔除: 目标有效样本 < 24 (占 96 的 25%) 则整日剔除
    #   前提: 我们已知一日理论 96 点 (15min 粒度); 允许 <96 是因为月末/月初边界
    #   但 <24 就是"上游采集失败", 保留残点无意义 (如 12/19: 96 中 95 个 NaN)
    n0 = len(df)
    sub_reasons = {}
    if target_col in df.columns and "datetime" in df.columns:
        MIN_VALID = 24
        # 用一个临时列判断: 该日目标非 NaN 数量
        day_key = df["datetime"].dt.date
        valid_per_day = df.groupby(day_key)[target_col].apply(lambda s: s.notna().sum())
        bad_days = valid_per_day[valid_per_day < MIN_VALID].index.tolist()
        if bad_days:
            for d in bad_days:
                n_kept = int((day_key == d).sum())
                sub_reasons[str(d)] = f"{n_kept} 点 (原本仅 {int(valid_per_day[d])} 点有效)"
            df = df.loc[~day_key.isin(bad_days)]
    n1 = len(df)
    stats["n_after_bad_day_drop"] = n1
    steps.append({"name": "整日污染剔除", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons,
                  "note": f"阈值: 单日有效样本 < 24 (占 96 的 25%) 则整日剔除"})

    # 2) 负荷类前向填充（不改样本数，统计填充总单元数）
    n0 = len(df)
    load_keys = ["负荷", "出力", "新能源", "光伏", "风电", "水电", "竞价空间", "非市场化"]
    load_cols = [c for c in df.columns if any(k in c for k in load_keys)]
    sub_reasons = {}
    n_filled_total = 0
    for c in load_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            n_nan_before = int(df[c].isna().sum())
            if n_nan_before > 0:
                sub_reasons[c] = n_nan_before
                n_filled_total += n_nan_before
            df[c] = df[c].ffill().fillna(0)
    stats["n_filled_load"] = n_filled_total
    steps.append({"name": "负荷类前向填充", "before": n0, "after": n0,
                  "dropped": 0, "sub": sub_reasons,
                  "note": f"填充 {n_filled_total:,} 个单元 (不剔除样本)"})

    # 3) 电价缺失剔除
    n0 = len(df)
    price_cols = [c for c in df.columns if "电价" in c]
    sub_reasons = {}
    for c in price_cols:
        if c in df.columns:
            sub_reasons[c] = int(df[c].isna().sum())
    df = df.dropna(subset=[c for c in price_cols if c in df.columns])
    n1 = len(df)
    stats["n_after_price_dropna"] = n1
    steps.append({"name": "电价缺失剔除", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons})

    # 4) 去重
    n0 = len(df)
    sub_reasons = {}
    if "datetime" in df.columns:
        n_dup = int(df["datetime"].duplicated().sum())
        sub_reasons["重复 datetime 行"] = n_dup
        df = df.drop_duplicates(subset=["datetime"], keep="first")
    n1 = len(df)
    stats["n_after_dedup"] = n1
    steps.append({"name": "时序去重", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons})

    # 5) 无效列 (剔列不剔行，单列统计原因)
    n_col_before = df.shape[1]
    to_drop_miss = []
    to_drop_std0 = []
    keep_meta = {"datetime", "来源文件", "来源文件_供需", "时段"}
    for c in df.columns:
        if c in keep_meta:
            continue
        if df[c].isna().mean() > missing_thresh:
            to_drop_miss.append(c); continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].std() == 0:
            to_drop_std0.append(c)
    to_drop = to_drop_miss + to_drop_std0
    df = df.drop(columns=to_drop, errors="ignore")
    stats["dropped_cols"] = to_drop
    stats["dropped_cols_miss"] = to_drop_miss
    stats["dropped_cols_std0"] = to_drop_std0
    steps.append({"name": "无效列剔除", "before": n_col_before,
                  "after": df.shape[1], "dropped": len(to_drop),
                  "sub": {f"缺失率>{int(missing_thresh*100)}%": len(to_drop_miss),
                          "std=0":                              len(to_drop_std0)},
                  "note": "剔除整列, 不剔除样本"})

    # 6) 仅 2025
    n0 = len(df)
    if "datetime" in df.columns:
        years = df["datetime"].dt.year
        sub_reasons = {f"{int(y)} 年": int((years == y).sum())
                       for y in sorted(years.unique()) if y != 2025}
        df = df[years == 2025].reset_index(drop=True)
    else:
        sub_reasons = {}
    n1 = len(df)
    stats["n_after_2025"] = n1
    steps.append({"name": "仅 2025 数据筛选", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons})

    # 7) 剩余 NaN 兜底
    n0 = len(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # 按列统计还有几行带 NaN
    sub_reasons = {}
    for c in numeric_cols:
        n_nan = int(df[c].isna().sum())
        if n_nan > 0:
            sub_reasons[c] = n_nan
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    n1 = len(df)
    stats["n_after_nan_safety"] = n1
    stats["n_dropped_by_nan_safety"] = n0 - n1
    steps.append({"name": "剩余 NaN 兜底", "before": n0, "after": n1,
                  "dropped": n0 - n1, "sub": sub_reasons,
                  "note": "前 6 步若有遗漏的 NaN 此处兜底剔除"})

    stats["steps"] = steps
    stats["n_after"] = len(df)
    stats["col_after"] = df.shape[1]
    return df, stats


# ---------------------------------------------------------------------------
# 可视化（清洗前后对比）
# ---------------------------------------------------------------------------
def plot_cleaning_summary(df_before: pd.DataFrame, df_after: pd.DataFrame,
                          target_col: str) -> Dict[str, str]:
    """两张图：清洗前后目标分布对比 + 月度样本数。"""
    setup_cn_font()
    paths = {}

    # 图 1: 清洗前后 target 分布
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df_before[target_col].dropna(), bins=80, color="#C44E52", alpha=0.75)
    axes[0].set_title(f"清洗前  n={len(df_before):,}")
    axes[0].set_xlabel("电价 (元/MWh)"); axes[0].set_ylabel("频次")
    axes[0].grid(alpha=0.3)

    axes[1].hist(df_after[target_col].dropna(), bins=80, color="#4C72B0", alpha=0.85)
    axes[1].set_title(f"清洗后  n={len(df_after):,}")
    axes[1].set_xlabel("电价 (元/MWh)"); axes[1].set_ylabel("频次")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"清洗前后 {target_col} 分布对比", fontsize=13)
    p1 = os.path.join(PATHS["plots"], "cleaning_dist.png")
    safe_savefig(p1); paths["dist"] = p1

    # 图 2: 2025 逐月样本数
    df_mo = df_after.copy()
    df_mo["月"] = df_mo["datetime"].dt.month
    mo = df_mo.groupby("月").size()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(mo.index, mo.values, color="#4C72B0")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("月份"); ax.set_ylabel("样本数 (15min)")
    ax.set_title("2025 清洗后逐月样本数")
    ax.grid(axis="y", alpha=0.3)
    for x, y in zip(mo.index, mo.values):
        ax.text(x, y + 30, f"{y:,}", ha="center", fontsize=9)
    p2 = os.path.join(PATHS["plots"], "cleaning_monthly_count.png")
    safe_savefig(p2); paths["monthly"] = p2
    return paths


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------
def make_html(stats: Dict, plot_paths: Dict[str, str],
              df_after: pd.DataFrame, target_col: str) -> str:
    p: List[str] = []

    p.append('<div class="meta">')
    p.append(f"<p><strong>数据范围</strong>: 仅 2025 年 (2026 数据忽略)</p>")
    p.append(f"<p><strong>清洗前样本</strong>: {stats['n_before']:,} 行 × {stats['col_before']} 列</p>")
    p.append(f"<p><strong>清洗后样本</strong>: {stats['n_after']:,} 行 × {stats['col_after']} 列</p>")
    p.append(f"<p><strong>剔除样本</strong>: {stats['n_before'] - stats['n_after']:,} 行 "
             f"({(1 - stats['n_after']/max(stats['n_before'],1))*100:.2f}%)</p>")
    p.append(f"<p><strong>缓存文件</strong>: <code>{PATHS['cleaned']}</code></p>")
    p.append('</div>')

    p.append("<h2>一、清洗策略</h2>")
    p.append("<table><tr><th>步骤</th><th>规则</th><th>业务理由</th></tr>")
    for name, rule, why in CLEAN_RULES:
        p.append(f"<tr><td><strong>{name}</strong></td><td>{rule}</td>"
                 f"<td class='small'>{why}</td></tr>")
    p.append("</table>")

    p.append("<h2>二、各步骤样本变化（明细）</h2>")
    p.append("<p class='small'>"
             "<strong>本步前</strong>: 进入本步时的样本数。"
             "<strong>本步剔除</strong>: 本规则真实命中的行数。"
             "<strong>占本步前 %</strong>: 本步剔除占本步前样本的比例。"
             "<strong>累计剔除</strong>: 从原始数据到本步累计剔除的样本数。"
             "<strong>剔除原因细分</strong>: 本步内部按子原因拆解。"
             "</p>")
    p.append("<table><tr><th>步骤</th><th>本步前</th><th>本步剔除</th>"
             "<th>占本步前 %</th><th>本步后</th><th>累计剔除</th>"
             "<th>剔除原因细分</th></tr>")

    n_orig = stats["n_before"]
    cum_dropped = 0
    for step in stats.get("steps", []):
        name = step["name"]
        before = step["before"]
        after = step["after"]
        dropped = step["dropped"]
        # 无效列剔除步骤是"剔列不剔行"，不累加到样本累计
        is_col_step = (name == "无效列剔除")
        if not is_col_step:
            cum_dropped += dropped
        pct = (dropped / before * 100) if before > 0 else 0.0

        # 子原因细分渲染
        sub = step.get("sub", {})
        sub_parts = []
        if step.get("note"):
            sub_parts.append(f"<em class='small'>{step['note']}</em>")
        if sub:
            for reason, n in sub.items():
                # n 可能是 int 或 str (整日污染剔除步骤用 str 描述)
                if isinstance(n, str):
                    sub_parts.append(f"<code>{reason}</code>: {n}")
                elif n > 0:
                    sub_parts.append(f"<code>{reason}</code>: {n:,}")
        sub_html = "<br>".join(sub_parts) if sub_parts else "—"

        unit = "列" if is_col_step else "行"
        p.append(f"<tr><td><strong>{name}</strong></td>"
                 f"<td class='num'>{before:,} {unit}</td>"
                 f"<td class='num'>{dropped:,}</td>"
                 f"<td class='num'>{pct:.2f}%</td>"
                 f"<td class='num'>{after:,} {unit}</td>"
                 f"<td class='num'>{cum_dropped:,}</td>"
                 f"<td class='small'>{sub_html}</td></tr>")
    # 末行汇总
    p.append(f"<tr class='winner'><td><strong>总计</strong></td>"
             f"<td class='num'>{n_orig:,} 行</td>"
             f"<td class='num'>{cum_dropped:,}</td>"
             f"<td class='num'>{cum_dropped/max(n_orig,1)*100:.2f}%</td>"
             f"<td class='num'>{stats['n_after']:,} 行</td>"
             f"<td class='num'>{cum_dropped:,}</td>"
             f"<td class='small'>清洗后 {stats['n_after']:,} 行 × {stats['col_after']} 列</td></tr>")
    p.append("</table>")

    if stats["dropped_cols"]:
        p.append(f"<p><strong>剔除列</strong>: {', '.join(stats['dropped_cols'])}</p>")
    else:
        p.append("<p>所有原始列均保留 (无缺失率 &gt;80% 或方差=0 的列)</p>")

    p.append("<h2>三、清洗后目标变量统计</h2>")
    s = df_after[target_col].describe()
    p.append("<table><tr><th>统计量</th><th>值 (元/MWh)</th></tr>")
    for k, v in s.items():
        p.append(f"<tr><td>{k}</td><td class='num'>{v:.2f}</td></tr>")
    p.append("</table>")

    p.append("<h2>四、可视化</h2>")
    p.append("<h3>清洗前后分布对比</h3>")
    p.append(img_tag(plot_paths.get("dist", "")))
    p.append("<h3>2025 清洗后逐月样本数</h3>")
    p.append(img_tag(plot_paths.get("monthly", "")))

    return render_html("① 数据清洗报告", p, PATHS["html_cleaning"])


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(data_dir: str = DATA_DIR_DEFAULT) -> str:
    ensure_dirs()
    setup_cn_font()
    print(f"[INFO] 加载原始数据: {data_dir}")
    df_raw = load_data(data_dir)
    print(f"[INFO] 原始 shape={df_raw.shape}")

    df_clean, stats = clean(df_raw)
    save_pickle(df_clean, PATHS["cleaned"])
    print(f"[INFO] 清洗后 shape={df_clean.shape}, 已保存 {PATHS['cleaned']}")

    plot_paths = plot_cleaning_summary(df_raw, df_clean, TARGET_COL)
    out_html = make_html(stats, plot_paths, df_clean, TARGET_COL)
    print(f"[INFO] 已生成 {out_html}")
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=DATA_DIR_DEFAULT)
    args = ap.parse_args()
    main(args.data_dir)
