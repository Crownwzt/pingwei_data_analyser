# -*- coding: utf-8 -*-
"""
电力市场数据自动化分析工具
============================

输入  : 包含「市场价格趋势」与「市场供需情况」Excel 的目录
输出  : ./plots/ 下的一组图表 + ./report.txt 分析报告

主入口: main(data_dir: str) —— 只需传入数据目录路径即可运行。

数据结构假设（基于实际样本）：
  价格趋势 *.xlsx
    - sheet "明细"  : 日期, 时间, 日前统一结算点电价, 实时统一结算点电价,
                     日前节点电价, 实时节点电价, 负荷率(%)
  供需情况 *.xlsx
    - sheet "日前"  : 日期, 时间, 省调负荷, 外来送负荷, 新能源负荷, 水电负荷,
                     光伏负荷, 风电负荷, 发电总出力, 非市场化出力, 竞价空间
    - sheet "实际"  : 同上（少 外来 / 发电总出力）

作者: 资深 Python 电力交易预测数据分析师
"""

from __future__ import annotations

import os
import re
import sys
import glob
import math
import argparse
import warnings
import traceback
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 服务器环境不依赖显示
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 全局设置：中文字体 / 负号显示 / 风格
# ---------------------------------------------------------------------------
def _setup_matplotlib() -> None:
    """配置 matplotlib，保证中文与负号正常显示。"""
    candidates = [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "SimHei", "Microsoft YaHei", "Source Han Sans CN", "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["figure.autolayout"] = True
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.sans-serif"][0])


# ---------------------------------------------------------------------------
# 1. 数据加载：扫描目录、读取并合并所有月份
# ---------------------------------------------------------------------------
def _list_excels(data_dir: str) -> Tuple[List[str], List[str]]:
    """扫描目录，返回 (价格文件列表, 供需文件列表)，过滤 Excel 临时文件。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    all_files = [f for f in all_files if not os.path.basename(f).startswith("~$")
                 and not os.path.basename(f).startswith(".~")]

    price_files  = [f for f in all_files if "价格趋势" in os.path.basename(f)]
    supply_files = [f for f in all_files if "供需情况" in os.path.basename(f)]
    return price_files, supply_files


def _read_price_one(path: str) -> pd.DataFrame:
    """读取单个价格文件的 96点明细 sheet。"""
    df = pd.read_excel(path, sheet_name="明细")
    df["来源文件"] = os.path.basename(path)
    return df


def _read_supply_one(path: str) -> pd.DataFrame:
    """读取单个供需文件，优先用 日前 sheet（用于预测），并把 实际 sheet 的关键列拼接进来。"""
    xls = pd.ExcelFile(path)
    da = pd.read_excel(xls, sheet_name="日前") if "日前" in xls.sheet_names else pd.DataFrame()
    rt = pd.read_excel(xls, sheet_name="实际") if "实际" in xls.sheet_names else pd.DataFrame()

    if not da.empty:
        da = da.add_suffix("_日前")
        da = da.rename(columns={"日期_日前": "日期", "时间_日前": "时间"})
    if not rt.empty:
        rt = rt.add_suffix("_实际")
        rt = rt.rename(columns={"日期_实际": "日期", "时间_实际": "时间"})

    if da.empty and rt.empty:
        return pd.DataFrame()
    if da.empty:
        merged = rt
    elif rt.empty:
        merged = da
    else:
        merged = pd.merge(da, rt, on=["日期", "时间"], how="outer")
    merged["来源文件_供需"] = os.path.basename(path)
    return merged


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """将 日期+时间 拼成 datetime 索引，并衍生时间特征。"""
    df = df.copy()
    # 统一字符串格式：日期可能已是 datetime64
    if pd.api.types.is_datetime64_any_dtype(df["日期"]):
        date_str = df["日期"].dt.strftime("%Y-%m-%d")
    else:
        date_str = df["日期"].astype(str).str.slice(0, 10)
    time_str = df["时间"].astype(str)
    # 时间列偶现 NaN（汇总行），过滤
    mask = time_str.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").fillna(False)
    df = df.loc[mask].copy()
    date_str = date_str.loc[mask]
    time_str = time_str.loc[mask]

    df["datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df["年"] = df["datetime"].dt.year
    df["月"] = df["datetime"].dt.month
    df["日"] = df["datetime"].dt.day
    df["小时"] = df["datetime"].dt.hour
    df["星期"] = df["datetime"].dt.weekday + 1   # 1~7
    df["是否周末"] = (df["星期"] >= 6).astype(int)
    # 峰平谷划分（南方电力市场常见经验值，可按需调整）
    def _seg(h: int) -> str:
        if 8 <= h <= 11 or 18 <= h <= 21:
            return "峰"
        if 0 <= h <= 6:
            return "谷"
        return "平"
    df["时段"] = df["小时"].map(_seg)
    return df


def load_data(data_dir: str) -> pd.DataFrame:
    """
    加载并合并目录下全部价格 + 供需 Excel。
    返回一个按 datetime 排序的宽表。
    """
    price_files, supply_files = _list_excels(data_dir)
    if not price_files:
        raise FileNotFoundError(f"目录中未发现 *价格趋势*.xlsx: {data_dir}")

    price_df = pd.concat([_read_price_one(f) for f in price_files], ignore_index=True)
    if supply_files:
        supply_df = pd.concat([_read_supply_one(f) for f in supply_files], ignore_index=True)
    else:
        supply_df = pd.DataFrame()

    if supply_df.empty:
        merged = price_df
    else:
        merged = pd.merge(price_df, supply_df, on=["日期", "时间"], how="left")

    merged = _to_datetime(merged)
    return merged


# ---------------------------------------------------------------------------
# 2. 数据可视化
# ---------------------------------------------------------------------------
def _safe_savefig(path: str) -> None:
    """安全保存当前 figure 并关闭，避免内存泄漏。"""
    try:
        plt.savefig(path, bbox_inches="tight")
    finally:
        plt.close()


def _numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0:
            cols.append(c)
    return cols


def plot_overview(df: pd.DataFrame, plots_dir: str) -> Dict[str, str]:
    """整体概览：缺失值热力图、各数值列分布直方图。"""
    out: Dict[str, str] = {}

    # --- 缺失值汇总条形图 ---
    miss = df.isna().mean().sort_values(ascending=True) * 100
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(miss))))
        miss.plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_xlabel("缺失率 (%)")
        ax.set_title("各字段缺失率")
        path = os.path.join(plots_dir, "00_missing_rate.png")
        _safe_savefig(path)
        out["missing"] = path

    # --- 数值列分布直方图（网格） ---
    num_cols = _numeric_columns(
        df, exclude=["年", "月", "日", "小时", "星期", "是否周末"]
    )
    if num_cols:
        n = len(num_cols)
        ncol = 3
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(num_cols):
            ax = axes[i]
            data = df[c].dropna()
            if data.empty:
                ax.set_visible(False); continue
            ax.hist(data, bins=50, color="#4C72B0", alpha=0.85)
            ax.set_title(c, fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("各数值字段分布直方图", fontsize=14, y=1.02)
        path = os.path.join(plots_dir, "01_distributions.png")
        _safe_savefig(path)
        out["distributions"] = path

    return out


def plot_price_timeseries(df: pd.DataFrame, plots_dir: str) -> Dict[str, str]:
    """电价时间序列折线图：全周期日均、月度对比、96点平均。"""
    out: Dict[str, str] = {}
    price_cols = [c for c in [
        "日前统一结算点电价(元/MWh)", "实时统一结算点电价(元/MWh)",
        "日前节点电价(元/MWh)",     "实时节点电价(元/MWh)",
    ] if c in df.columns]
    if not price_cols:
        return out

    # 1) 全周期日均电价折线
    daily = df.set_index("datetime")[price_cols].resample("D").mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    for c in price_cols:
        ax.plot(daily.index, daily[c], label=c, linewidth=1.1)
    ax.set_title("全周期日均电价走势")
    ax.set_xlabel("日期"); ax.set_ylabel("电价 (元/MWh)")
    ax.legend(loc="best", fontsize=9)
    path = os.path.join(plots_dir, "10_price_daily_trend.png")
    _safe_savefig(path); out["daily"] = path

    # 2) 月度均价柱状图
    monthly = (df.assign(年月=df["datetime"].dt.strftime("%Y-%m"))
                 .groupby("年月")[price_cols].mean())
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("月度平均电价")
    ax.set_xlabel("年-月"); ax.set_ylabel("电价 (元/MWh)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    path = os.path.join(plots_dir, "11_price_monthly_bar.png")
    _safe_savefig(path); out["monthly"] = path

    # 3) 24小时电价模式（按月份分线）
    hourly = df.groupby(["月", "小时"])[price_cols[0]].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab20")
    for i, col in enumerate(hourly.columns):
        ax.plot(hourly.index, hourly[col], label=f"{int(col)}月",
                color=cmap(i % 20), linewidth=1.4)
    ax.set_title(f"日内24小时电价模式（按月） — {price_cols[0]}")
    ax.set_xlabel("小时"); ax.set_ylabel("电价 (元/MWh)")
    ax.set_xticks(range(0, 24))
    ax.legend(ncol=4, fontsize=8, loc="best")
    path = os.path.join(plots_dir, "12_price_hourly_by_month.png")
    _safe_savefig(path); out["hourly"] = path

    # 4) 小时×月份 电价热力图
    pivot = df.pivot_table(index="小时", columns="月",
                           values=price_cols[0], aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="RdYlBu_r", annot=False, ax=ax,
                cbar_kws={"label": "电价 (元/MWh)"})
    ax.set_title(f"电价小时×月份热力图 — {price_cols[0]}")
    path = os.path.join(plots_dir, "13_price_heatmap_hour_month.png")
    _safe_savefig(path); out["heatmap"] = path

    return out


def plot_correlation(df: pd.DataFrame, plots_dir: str,
                     target: str) -> Tuple[Dict[str, str], pd.Series, pd.DataFrame]:
    """
    相关性分析：
      - 全因子相关性热力图
      - target 与各因子相关性条形图
      - 分时段（峰平谷）的相关性差异
      - 主因子散点图（前 4 个）
    返回 (图路径dict, target相关性Series, 分时段相关性DataFrame)
    """
    out: Dict[str, str] = {}
    if target not in df.columns:
        return out, pd.Series(dtype=float), pd.DataFrame()

    num_cols = _numeric_columns(
        df, exclude=["年", "日", "是否周末"]  # 保留小时、月、星期参与相关性观察
    )
    corr_df = df[num_cols].corr(method="pearson")

    # 1) 热力图
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(num_cols)),
                                    max(8, 0.4 * len(num_cols))))
    sns.heatmap(corr_df, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("全因子皮尔逊相关性矩阵")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    path = os.path.join(plots_dir, "20_corr_heatmap.png")
    _safe_savefig(path); out["heatmap"] = path

    # 2) target 相关性条形图（按绝对值排序）
    target_corr = corr_df[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(target_corr))))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Pearson 相关系数")
    ax.set_title(f"各因子与「{target}」的相关性（按 |r| 排序）")
    path = os.path.join(plots_dir, "21_corr_with_target.png")
    _safe_savefig(path); out["with_target"] = path

    # 3) 峰/平/谷 分时段的相关性
    seg_corr = {}
    for seg in ["峰", "平", "谷"]:
        sub = df[df["时段"] == seg]
        if len(sub) > 30:
            seg_corr[seg] = sub[num_cols].corr()[target].drop(target)
    seg_corr_df = pd.DataFrame(seg_corr)
    if not seg_corr_df.empty:
        seg_corr_df = seg_corr_df.reindex(target_corr.index)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(seg_corr_df))))
        seg_corr_df.plot(kind="barh", ax=ax, width=0.8,
                         color=["#C44E52", "#55A868", "#4C72B0"])
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Pearson 相关系数")
        ax.set_title(f"各因子与「{target}」相关性 —— 分时段（峰/平/谷）")
        path = os.path.join(plots_dir, "22_corr_with_target_by_seg.png")
        _safe_savefig(path); out["with_target_seg"] = path
    else:
        seg_corr_df = pd.DataFrame()

    # 4) 主因子散点图（与 target |r| 最大的前 4 个）
    top_factors = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()
    if top_factors:
        n = len(top_factors); ncol = 2; nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow))
        axes = np.array(axes).reshape(-1)
        sample = df[[target] + top_factors].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        for i, c in enumerate(top_factors):
            ax = axes[i]
            ax.scatter(sample[c], sample[target], s=6, alpha=0.35, c="#4C72B0")
            # 拟合线
            try:
                z = np.polyfit(sample[c].values, sample[target].values, 1)
                xs = np.linspace(sample[c].min(), sample[c].max(), 100)
                ax.plot(xs, np.polyval(z, xs), color="#C44E52", linewidth=1.5)
            except Exception:
                pass
            r = sample[c].corr(sample[target])
            ax.set_title(f"{c}\nr = {r:.3f}", fontsize=10)
            ax.set_xlabel(c, fontsize=9); ax.set_ylabel(target, fontsize=9)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"主因子 vs 「{target}」散点（含一次拟合）", fontsize=13, y=1.02)
        path = os.path.join(plots_dir, "23_corr_top_factors_scatter.png")
        _safe_savefig(path); out["scatter"] = path

    return out, target_corr.sort_values(key=lambda s: s.abs(), ascending=False), seg_corr_df


# ---------------------------------------------------------------------------
# 3. 报告生成
# ---------------------------------------------------------------------------
def _fmt(v, nd=2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def generate_report_txt(
    df: pd.DataFrame,
    target: str,
    target_corr: pd.Series,
    seg_corr: pd.DataFrame,
    plot_paths: Dict[str, str],
    out_path: str,
    data_dir: str,
) -> None:
    """生成 report.txt 文本报告（保留用于对比）。"""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  电力市场数据分析报告")
    lines.append("=" * 78)
    lines.append(f"数据目录   : {data_dir}")
    lines.append(f"时间范围   : {df['datetime'].min()}  ~  {df['datetime'].max()}")
    lines.append(f"样本规模   : {len(df):,} 行 × {df.shape[1]} 列")
    lines.append(f"分析目标   : {target}")
    lines.append("")

    # —— 1. 数据概况 ——
    lines.append("-" * 78)
    lines.append("一、数据概况")
    lines.append("-" * 78)
    months = df["datetime"].dt.strftime("%Y-%m").unique().tolist()
    lines.append(f"覆盖月份   : {len(months)} 个月  -> {', '.join(months)}")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0].head(10)
    if not miss.empty:
        lines.append("缺失率Top10:")
        for k, v in miss.items():
            lines.append(f"   {k:<35s}  {v:6.2f}%")
    else:
        lines.append("无缺失字段。")
    lines.append("")

    # —— 2. 关键统计 ——
    lines.append("-" * 78)
    lines.append("二、关键字段描述统计")
    lines.append("-" * 78)
    key_cols = [c for c in df.columns
                if any(k in c for k in ["电价", "负荷率", "省调负荷", "新能源", "光伏",
                                        "风电", "水电", "竞价空间", "非市场化", "发电总出力"])
                and pd.api.types.is_numeric_dtype(df[c])]
    desc = df[key_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["count", "mean", "std", "min", "median", "max"]
    lines.append(desc.round(2).to_string())
    lines.append("")

    # —— 3. 相关性分析 ——
    lines.append("-" * 78)
    lines.append("三、电价相关性分析（皮尔逊 r，按 |r| 排序）")
    lines.append("-" * 78)
    if not target_corr.empty:
        lines.append(f"目标变量   : {target}")
        lines.append(f"{'因子':<35s}  {'r':>8s}  {'方向':>6s}  {'强度':>10s}")
        for k, v in target_corr.items():
            direction = "正" if v > 0 else ("负" if v < 0 else "—")
            absv = abs(v)
            if absv >= 0.7:    strength = "强"
            elif absv >= 0.4:  strength = "中"
            elif absv >= 0.2:  strength = "弱"
            else:              strength = "极弱"
            lines.append(f"{k:<35s}  {v:8.3f}  {direction:>6s}  {strength:>10s}")
    lines.append("")

    # —— 4. 分时段对比 ——
    if not seg_corr.empty:
        lines.append("-" * 78)
        lines.append("四、分时段（峰/平/谷）相关性对比")
        lines.append("-" * 78)
        lines.append(seg_corr.round(3).to_string())
        lines.append("")

    # —— 5. 专业解读 ——
    lines.append("-" * 78)
    lines.append("五、电价预测专业解读")
    lines.append("-" * 78)
    insights: List[str] = []
    # 自动从 target_corr 抽取前几个因子做语义化解读
    if not target_corr.empty:
        top_pos = target_corr[target_corr > 0].head(3)
        top_neg = target_corr[target_corr < 0].head(3)
        if not top_pos.empty:
            tops = ", ".join([f"{k}(r={v:.2f})" for k, v in top_pos.items()])
            insights.append(f"1) 推升电价的主要因子：{tops}。这些因子上升通常伴随系统供需偏紧或负荷上行，电价随之走高。")
        if not top_neg.empty:
            tops = ", ".join([f"{k}(r={v:.2f})" for k, v in top_neg.items()])
            insights.append(f"2) 抑制电价的主要因子：{tops}。新能源/水电出力增加挤占火电边际机组，结算点价格趋于回落。")

    # 日内规律
    if target in df.columns:
        peak_h = (df.groupby("小时")[target].mean().idxmax())
        valley_h = (df.groupby("小时")[target].mean().idxmin())
        insights.append(f"3) 日内电价高点出现在 {peak_h} 时附近，低点在 {valley_h} 时附近，呈现典型双峰/单谷型负荷特征。")

    # 峰平谷差异（自动选择 |r| 最大的新能源因子）
    if not seg_corr.empty:
        ne_candidates = [c for c in seg_corr.index
                         if any(k in c for k in ["光伏", "新能源", "风电"])]
        if ne_candidates:
            ne_top = max(ne_candidates,
                         key=lambda c: abs(seg_corr.loc[c].get("峰", 0) or 0))
            v_peak = seg_corr.loc[ne_top].get("峰", np.nan)
            v_valley = seg_corr.loc[ne_top].get("谷", np.nan)
            if pd.notna(v_peak) and pd.notna(v_valley):
                insights.append(
                    f"4) {ne_top} 对电价的相关性在峰段为 r={v_peak:.2f}，谷段为 r={v_valley:.2f}；"
                    f"差异显示新能源对峰时段挤压更明显，谷时段影响有限（谷时段光伏几乎不出力）。"
                )
    # 月度波动
    if target in df.columns:
        m_mean = df.groupby(df["datetime"].dt.strftime("%Y-%m"))[target].mean()
        if not m_mean.empty:
            insights.append(
                f"5) 月度均价波动范围 {m_mean.min():.1f}~{m_mean.max():.1f} 元/MWh，"
                f"最高出现在 {m_mean.idxmax()}，最低出现在 {m_mean.idxmin()}。"
            )

    if insights:
        lines.extend(insights)
    else:
        lines.append("（数据不足以给出语义化解读）")
    lines.append("")

    # —— 6. 图表索引 ——
    lines.append("-" * 78)
    lines.append("六、图表索引（已保存到 ./plots/）")
    lines.append("-" * 78)
    for name, p in sorted(plot_paths.items()):
        lines.append(f"  - {name:<25s} -> {p}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("                              报告结束")
    lines.append("=" * 78)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_report_md(
    df: pd.DataFrame,
    target: str,
    target_corr: pd.Series,
    seg_corr: pd.DataFrame,
    plot_paths: Dict[str, str],
    out_path: str,
    data_dir: str,
) -> None:
    """生成 Markdown 格式报告，嵌入图表。"""
    lines: List[str] = []
    lines.append("# 电力市场数据分析报告\n")
    lines.append(f"**数据目录**: `{data_dir}`  ")
    lines.append(f"**时间范围**: {df['datetime'].min()} ~ {df['datetime'].max()}  ")
    lines.append(f"**样本规模**: {len(df):,} 行 × {df.shape[1]} 列  ")
    lines.append(f"**分析目标**: `{target}`\n")

    # --- 1. 数据概况 ---
    lines.append("---\n## 一、数据概况\n")
    months = df["datetime"].dt.strftime("%Y-%m").unique().tolist()
    lines.append(f"**覆盖月份**: {len(months)} 个月  \n`{', '.join(months)}`\n")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0].head(10)
    if not miss.empty:
        lines.append("### 缺失率 Top 10\n")
        lines.append("| 字段 | 缺失率 (%) |")
        lines.append("|------|----------:|")
        for k, v in miss.items():
            lines.append(f"| {k} | {v:.2f} |")
    else:
        lines.append("✅ 无缺失字段。")
    lines.append("")

    # --- 2. 关键统计 ---
    lines.append("---\n## 二、关键字段描述统计\n")
    key_cols = [c for c in df.columns
                if any(k in c for k in ["电价", "负荷率", "省调负荷", "新能源", "光伏",
                                        "风电", "水电", "竞价空间", "非市场化", "发电总出力"])
                and pd.api.types.is_numeric_dtype(df[c])]
    desc = df[key_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["count", "mean", "std", "min", "median", "max"]
    lines.append(desc.round(2).to_markdown())
    lines.append("")

    # --- 3. 相关性分析 ---
    lines.append("---\n## 三、电价相关性分析（皮尔逊 r）\n")
    if not target_corr.empty:
        lines.append(f"**目标变量**: `{target}`\n")
        lines.append("| 因子 | r | 方向 | 强度 |")
        lines.append("|------|------:|:----:|:----:|")
        for k, v in target_corr.items():
            direction = "➕ 正" if v > 0 else ("➖ 负" if v < 0 else "—")
            absv = abs(v)
            if absv >= 0.7:    strength = "🔴 强"
            elif absv >= 0.4:  strength = "🟠 中"
            elif absv >= 0.2:  strength = "🟡 弱"
            else:              strength = "⚪ 极弱"
            lines.append(f"| {k} | {v:.3f} | {direction} | {strength} |")
    lines.append("")

    # --- 4. 分时段对比 ---
    if not seg_corr.empty:
        lines.append("---\n## 四、分时段（峰/平/谷）相关性对比\n")
        lines.append(seg_corr.round(3).to_markdown())
        lines.append("")

    # --- 5. 专业解读 ---
    lines.append("---\n## 五、电价预测专业解读\n")
    insights: List[str] = []
    if not target_corr.empty:
        top_pos = target_corr[target_corr > 0].head(3)
        top_neg = target_corr[target_corr < 0].head(3)
        if not top_pos.empty:
            tops = ", ".join([f"**{k}** (r={v:.2f})" for k, v in top_pos.items()])
            insights.append(f"1. **推升电价的主要因子**: {tops}  \n   这些因子上升通常伴随系统供需偏紧或负荷上行，电价随之走高。")
        if not top_neg.empty:
            tops = ", ".join([f"**{k}** (r={v:.2f})" for k, v in top_neg.items()])
            insights.append(f"2. **抑制电价的主要因子**: {tops}  \n   新能源/水电出力增加挤占火电边际机组，结算点价格趋于回落。")

    if target in df.columns:
        peak_h = df.groupby("小时")[target].mean().idxmax()
        valley_h = df.groupby("小时")[target].mean().idxmin()
        insights.append(f"3. **日内电价模式**: 高点出现在 **{peak_h} 时**，低点在 **{valley_h} 时**，呈现典型双峰/单谷型负荷特征。")

    if not seg_corr.empty:
        ne_candidates = [c for c in seg_corr.index
                         if any(k in c for k in ["光伏", "新能源", "风电"])]
        if ne_candidates:
            ne_top = max(ne_candidates,
                         key=lambda c: abs(seg_corr.loc[c].get("峰", 0) or 0))
            v_peak = seg_corr.loc[ne_top].get("峰", np.nan)
            v_valley = seg_corr.loc[ne_top].get("谷", np.nan)
            if pd.notna(v_peak) and pd.notna(v_valley):
                insights.append(
                    f"4. **峰谷差异**: {ne_top} 对电价的相关性在峰段为 r={v_peak:.2f}，谷段为 r={v_valley:.2f}；  \n"
                    f"   差异显示新能源对峰时段挤压更明显，谷时段影响有限（谷时段光伏几乎不出力）。"
                )

    if target in df.columns:
        m_mean = df.groupby(df["datetime"].dt.strftime("%Y-%m"))[target].mean()
        if not m_mean.empty:
            insights.append(
                f"5. **月度波动**: 均价范围 **{m_mean.min():.1f} ~ {m_mean.max():.1f}** 元/MWh，  \n"
                f"   最高出现在 **{m_mean.idxmax()}**，最低出现在 **{m_mean.idxmin()}**。"
            )

    if insights:
        for ins in insights:
            lines.append(ins)
            lines.append("")
    else:
        lines.append("（数据不足以给出语义化解读）\n")

    # --- 6. 图表 ---
    lines.append("---\n## 六、可视化图表\n")
    # 按文件名排序，嵌入本地相对路径
    for name, abs_path in sorted(plot_paths.items()):
        rel = os.path.relpath(abs_path, os.path.dirname(out_path))
        lines.append(f"### {name}\n")
        lines.append(f"![{name}]({rel})\n")

    lines.append("---\n*报告结束*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_report_html(
    df: pd.DataFrame,
    target: str,
    target_corr: pd.Series,
    seg_corr: pd.DataFrame,
    plot_paths: Dict[str, str],
    out_path: str,
    data_dir: str,
) -> None:
    """生成 HTML 格式报告，带 CSS 样式，图表 base64 嵌入。"""
    import base64

    def _img_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    html_parts: List[str] = []
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>电力市场数据分析报告</title>
<style>
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; color: #333; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }
h3 { color: #555; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
th { background: #3498db; color: white; font-weight: 600; }
tr:nth-child(even) { background: #f2f2f2; }
tr:hover { background: #e8f4f8; }
.meta { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
.meta p { margin: 5px 0; }
.insight { background: #fff; padding: 15px; margin: 10px 0; border-left: 4px solid #e67e22; border-radius: 3px; }
img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }
hr { border: none; border-top: 2px solid #bdc3c7; margin: 40px 0; }
</style>
</head>
<body>
""")

    html_parts.append("<h1>电力市场数据分析报告</h1>")
    html_parts.append('<div class="meta">')
    html_parts.append(f"<p><strong>数据目录</strong>: <code>{data_dir}</code></p>")
    html_parts.append(f"<p><strong>时间范围</strong>: {df['datetime'].min()} ~ {df['datetime'].max()}</p>")
    html_parts.append(f"<p><strong>样本规模</strong>: {len(df):,} 行 × {df.shape[1]} 列</p>")
    html_parts.append(f"<p><strong>分析目标</strong>: <code>{target}</code></p>")
    html_parts.append("</div><hr>")

    # --- 1. 数据概况 ---
    html_parts.append("<h2>一、数据概况</h2>")
    months = df["datetime"].dt.strftime("%Y-%m").unique().tolist()
    html_parts.append(f"<p><strong>覆盖月份</strong>: {len(months)} 个月<br><code>{', '.join(months)}</code></p>")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0].head(10)
    if not miss.empty:
        html_parts.append("<h3>缺失率 Top 10</h3><table><tr><th>字段</th><th>缺失率 (%)</th></tr>")
        for k, v in miss.items():
            html_parts.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
        html_parts.append("</table>")
    else:
        html_parts.append("<p>✅ 无缺失字段。</p>")

    # --- 2. 关键统计 ---
    html_parts.append("<hr><h2>二、关键字段描述统计</h2>")
    key_cols = [c for c in df.columns
                if any(k in c for k in ["电价", "负荷率", "省调负荷", "新能源", "光伏",
                                        "风电", "水电", "竞价空间", "非市场化", "发电总出力"])
                and pd.api.types.is_numeric_dtype(df[c])]
    desc = df[key_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["count", "mean", "std", "min", "median", "max"]
    html_parts.append(desc.round(2).to_html(border=0))

    # --- 3. 相关性分析 ---
    html_parts.append("<hr><h2>三、电价相关性分析（皮尔逊 r）</h2>")
    if not target_corr.empty:
        html_parts.append(f"<p><strong>目标变量</strong>: <code>{target}</code></p>")
        html_parts.append("<table><tr><th>因子</th><th>r</th><th>方向</th><th>强度</th></tr>")
        for k, v in target_corr.items():
            direction = "➕ 正" if v > 0 else ("➖ 负" if v < 0 else "—")
            absv = abs(v)
            if absv >= 0.7:    strength = "🔴 强"
            elif absv >= 0.4:  strength = "🟠 中"
            elif absv >= 0.2:  strength = "🟡 弱"
            else:              strength = "⚪ 极弱"
            html_parts.append(f"<tr><td>{k}</td><td>{v:.3f}</td><td>{direction}</td><td>{strength}</td></tr>")
        html_parts.append("</table>")

    # --- 4. 分时段对比 ---
    if not seg_corr.empty:
        html_parts.append("<hr><h2>四、分时段（峰/平/谷）相关性对比</h2>")
        html_parts.append(seg_corr.round(3).to_html(border=0))

    # --- 5. 专业解读 ---
    html_parts.append("<hr><h2>五、电价预测专业解读</h2>")
    insights: List[str] = []
    if not target_corr.empty:
        top_pos = target_corr[target_corr > 0].head(3)
        top_neg = target_corr[target_corr < 0].head(3)
        if not top_pos.empty:
            tops = ", ".join([f"<strong>{k}</strong> (r={v:.2f})" for k, v in top_pos.items()])
            insights.append(f"<strong>推升电价的主要因子</strong>: {tops}<br>这些因子上升通常伴随系统供需偏紧或负荷上行，电价随之走高。")
        if not top_neg.empty:
            tops = ", ".join([f"<strong>{k}</strong> (r={v:.2f})" for k, v in top_neg.items()])
            insights.append(f"<strong>抑制电价的主要因子</strong>: {tops}<br>新能源/水电出力增加挤占火电边际机组，结算点价格趋于回落。")

    if target in df.columns:
        peak_h = df.groupby("小时")[target].mean().idxmax()
        valley_h = df.groupby("小时")[target].mean().idxmin()
        insights.append(f"<strong>日内电价模式</strong>: 高点出现在 <strong>{peak_h} 时</strong>，低点在 <strong>{valley_h} 时</strong>，呈现典型双峰/单谷型负荷特征。")

    if not seg_corr.empty:
        ne_candidates = [c for c in seg_corr.index
                         if any(k in c for k in ["光伏", "新能源", "风电"])]
        if ne_candidates:
            ne_top = max(ne_candidates,
                         key=lambda c: abs(seg_corr.loc[c].get("峰", 0) or 0))
            v_peak = seg_corr.loc[ne_top].get("峰", np.nan)
            v_valley = seg_corr.loc[ne_top].get("谷", np.nan)
            if pd.notna(v_peak) and pd.notna(v_valley):
                insights.append(
                    f"<strong>峰谷差异</strong>: {ne_top} 对电价的相关性在峰段为 r={v_peak:.2f}，谷段为 r={v_valley:.2f}；"
                    f"差异显示新能源对峰时段挤压更明显，谷时段影响有限（谷时段光伏几乎不出力）。"
                )

    if target in df.columns:
        m_mean = df.groupby(df["datetime"].dt.strftime("%Y-%m"))[target].mean()
        if not m_mean.empty:
            insights.append(
                f"<strong>月度波动</strong>: 均价范围 <strong>{m_mean.min():.1f} ~ {m_mean.max():.1f}</strong> 元/MWh，"
                f"最高出现在 <strong>{m_mean.idxmax()}</strong>，最低出现在 <strong>{m_mean.idxmin()}</strong>。"
            )

    if insights:
        for ins in insights:
            html_parts.append(f'<div class="insight">{ins}</div>')
    else:
        html_parts.append("<p>（数据不足以给出语义化解读）</p>")

    # --- 6. 图表（base64 嵌入） ---
    html_parts.append("<hr><h2>六、可视化图表</h2>")
    for name, path in sorted(plot_paths.items()):
        b64 = _img_to_base64(path)
        html_parts.append(f"<h3>{name}</h3>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" alt="{name}">')

    html_parts.append("<hr><p style='text-align:center; color:#7f8c8d;'><em>报告结束</em></p>")
    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main(data_dir: str,
         output_dir: Optional[str] = None,
         target: str = "实时统一结算点电价(元/MWh)") -> None:
    """
    主入口：仅需传入数据目录路径。

    参数
    ----
    data_dir   : 数据所在目录
    output_dir : 输出根目录（plots/ 与 report.txt 的父目录），默认当前目录
    target     : 用于相关性分析的目标电价列名，默认 日前统一结算点电价
    """
    _setup_matplotlib()

    output_dir = output_dir or os.getcwd()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[1/4] 加载数据  : {data_dir}")
    df = load_data(data_dir)
    print(f"      合并完成 -> shape = {df.shape}, 时间 {df['datetime'].min()} ~ {df['datetime'].max()}")

    print(f"[2/4] 整体可视化 -> {plots_dir}")
    paths_overview = plot_overview(df, plots_dir)

    print(f"[3/4] 时序与相关性分析（目标: {target}）")
    paths_ts = plot_price_timeseries(df, plots_dir)
    paths_corr, target_corr, seg_corr = plot_correlation(df, plots_dir, target)

    all_paths = {**paths_overview, **paths_ts, **paths_corr}

    print(f"[4/4] 生成报告 -> TXT + Markdown + HTML")
    report_txt = os.path.join(output_dir, "report.txt")
    report_md  = os.path.join(output_dir, "report.md")
    report_html = os.path.join(output_dir, "report.html")

    generate_report_txt(df, target, target_corr, seg_corr, all_paths, report_txt, data_dir)
    generate_report_md(df, target, target_corr, seg_corr, all_paths, report_md, data_dir)
    generate_report_html(df, target, target_corr, seg_corr, all_paths, report_html, data_dir)

    print(f"完成。共生成 {len(all_paths)} 张图")
    print(f"  - TXT  : {report_txt}")
    print(f"  - MD   : {report_md}")
    print(f"  - HTML : {report_html}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="电力市场数据自动化分析工具")
    parser.add_argument(
        "data_dir", nargs="?",
        default="/data/ztwen2/project_dir/pingwei_data_analyser/2025-2026市场情况",
        help="数据目录路径（默认调试路径）",
    )
    parser.add_argument("--output-dir", default=None, help="输出根目录（默认当前目录）")
    parser.add_argument("--target", default="实时统一结算点电价(元/MWh)", help="目标电价列")
    args = parser.parse_args()

    try:
        main(args.data_dir, args.output_dir, args.target)
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # # debug
    # "/data/ztwen2/project_dir/pingwei_data_analyser/2025-2026市场情况"
    _cli()
