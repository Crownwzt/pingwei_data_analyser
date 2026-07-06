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
# 数据加载：扫描目录、读取并合并所有月份的 Excel
# ---------------------------------------------------------------------------
def _list_excels(data_dir: str) -> Tuple[List[str], List[str]]:
    """扫描目录，返回 (价格文件列表, 供需文件列表)，过滤 Excel 临时文件。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    all_files = [f for f in all_files
                 if not os.path.basename(f).startswith("~$")
                 and not os.path.basename(f).startswith(".~")]
    price_files = [f for f in all_files if "价格趋势" in os.path.basename(f)]
    supply_files = [f for f in all_files if "供需情况" in os.path.basename(f)]
    return price_files, supply_files


def _read_price_one(path: str) -> pd.DataFrame:
    """读取单个价格文件的 96 点明细 sheet。"""
    df = pd.read_excel(path, sheet_name="明细")
    df["来源文件"] = os.path.basename(path)
    return df


def _read_supply_one(path: str) -> pd.DataFrame:
    """读取单个供需文件，合并日前+实际 sheet（带列后缀）。"""
    xls = pd.ExcelFile(path)
    da = pd.read_excel(xls, sheet_name="日前") if "日前" in xls.sheet_names else pd.DataFrame()
    rt = pd.read_excel(xls, sheet_name="实际") if "实际" in xls.sheet_names else pd.DataFrame()
    if not da.empty:
        da = da.add_suffix("_日前").rename(
            columns={"日期_日前": "日期", "时间_日前": "时间"})
    if not rt.empty:
        rt = rt.add_suffix("_实际").rename(
            columns={"日期_实际": "日期", "时间_实际": "时间"})
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
    """将 日期+时间 拼成 datetime，并衍生年/月/日/小时/星期/时段。

    电力交易数据约定：
      "日期=D+1, 时间=00:00:00" 实际含义是 "D 日 24:00:00" (当日最后一个时段结束点)。
      直接按字面日期解析会让每月末尾多出 1 个"孤儿"点。
      修复：把这类点回归到 (D+1 日) 00:00:00 - 15min = D 日 23:45:00
    """
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["日期"]):
        date_str = df["日期"].dt.strftime("%Y-%m-%d")
    else:
        date_str = df["日期"].astype(str).str.slice(0, 10)
    time_str = df["时间"].astype(str)
    mask = time_str.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").fillna(False)
    df = df.loc[mask].copy()
    date_str = date_str.loc[mask]
    time_str = time_str.loc[mask]
    df["datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # ★ 修复电力交易时间约定
    # 原始 Excel 里 "日期=D+1, 时间=00:00" 是 "D 日最后一个 15min 时段 [23:45, 24:00) 的结束点"
    # 它与 "D 日 23:45" 是【不同时段】(实测价格不同, 例如 1/30 23:45=280 元 vs 1/31 00:00=250 元)
    # 需保留其数据, 但归到当月最后一天 → 具体做法:
    #   将 "D+1 00:00" 的 datetime 减 1 分钟 → "D 23:59"
    #   这样它既保留独立性、又归属到 D 日, 逐日聚合时归入 D 日的第 24 个小时
    if len(df) > 0:
        is_midnight = ((df["datetime"].dt.hour == 0)
                        & (df["datetime"].dt.minute == 0)
                        & (df["datetime"].dt.second == 0))
        # 只有那些"当日仅此一个点"的 00:00 才是孤儿 (真正的月末闭合时段)
        day_counts = df["datetime"].dt.floor("D").value_counts()
        is_orphan_day = df["datetime"].dt.floor("D").map(day_counts) == 1
        orphan_mask = is_midnight & is_orphan_day
        n_orphans = int(orphan_mask.sum())
        if n_orphans > 0:
            # 减 1 分钟归到前一日 23:59, 归属正确又不与前一日 23:45 冲突
            df.loc[orphan_mask, "datetime"] = (
                df.loc[orphan_mask, "datetime"] - pd.Timedelta(minutes=1)
            )
            df = df.sort_values("datetime").reset_index(drop=True)
            print(f"[电力约定修复] 检测到 {n_orphans} 个月末孤儿点 (次日 00:00 = 前一日 24:00 结束点), "
                  f"已归属到前一日 23:59 (保留独立数据)")

    df["年"] = df["datetime"].dt.year
    df["月"] = df["datetime"].dt.month
    df["日"] = df["datetime"].dt.day
    df["小时"] = df["datetime"].dt.hour
    df["星期"] = df["datetime"].dt.weekday + 1
    df["是否周末"] = (df["星期"] >= 6).astype(int)
    # 峰平谷划分（南方电力市场常见经验值）
    def _seg(h: int) -> str:
        if 8 <= h <= 11 or 18 <= h <= 21: return "峰"
        if 0 <= h <= 6: return "谷"
        return "平"
    df["时段"] = df["小时"].map(_seg)
    return df


def load_data(data_dir: str) -> pd.DataFrame:
    """
    加载并合并目录下全部价格 + 供需 Excel。
    返回按 datetime 排序的宽表。
    """
    price_files, supply_files = _list_excels(data_dir)
    if not price_files:
        raise FileNotFoundError(f"目录中未发现 *价格趋势*.xlsx: {data_dir}")
    price_df = pd.concat([_read_price_one(f) for f in price_files], ignore_index=True)
    supply_df = (pd.concat([_read_supply_one(f) for f in supply_files], ignore_index=True)
                 if supply_files else pd.DataFrame())
    merged = price_df if supply_df.empty else pd.merge(
        price_df, supply_df, on=["日期", "时间"], how="left")
    return _to_datetime(merged)


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
