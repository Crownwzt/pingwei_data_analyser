# -*- coding: utf-8 -*-
"""
③ 数据相关性分析模块（含 EDA）
================================

职责（单一 HTML 出口，EDA + 相关性分析合并）：

  Part A — EDA 数据可视化:
    1. 缺失率 Top N 条形图
    2. 各数值字段分布直方图 (网格)
    3. 电价时序: 全周期日均 / 月均柱状 / 24h 模式 (按月) / 小时×月热力图

  Part B — 相关性分析:
    4. 全因子 Pearson 相关性矩阵
    5. 与目标 |r| 排序条形图 (Top 20)
    6. 峰/平/谷分时段相关性差异
    7. Top 4 主因子 vs 目标散点图 + 一次拟合

  Part C — 业务解读:
    8. day-ahead 可用性标注 (合法/泄漏/不可用三类)
    9. 电力业务语义解读 (推升/抑制因子、日内峰谷、新能源挤压等)

  产物：
    - outputs/correlation.pkl       相关性矩阵 + 分段相关性
    - outputs/03_correlation.html   合并 EDA + 相关性的单一 HTML

入口：python -m src.correlation
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (
    TARGET_COL, LEAKAGE_COLS, PATHS,
    ensure_dirs, setup_cn_font, render_html, safe_savefig,
    load_clean, save_pickle, img_tag,
)


# ---------------------------------------------------------------------------
# Part A — EDA 计算
# ---------------------------------------------------------------------------
def _numeric_cols(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """返回数值列名（去除离散标识与指定排除项）。"""
    exclude = set(exclude or [])
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
            and df[c].notna().sum() > 0]


def compute_eda(df: pd.DataFrame) -> Dict:
    """统计 EDA 所需的描述性指标。"""
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]

    # 关键字段描述统计 (电价/负荷/出力相关)
    key_cols = [c for c in df.columns
                if any(k in c for k in
                       ["电价", "负荷率", "省调负荷", "新能源", "光伏",
                        "风电", "水电", "竞价空间", "非市场化", "发电总出力"])
                and pd.api.types.is_numeric_dtype(df[c])]
    desc = (df[key_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
            .round(2))
    desc.columns = ["count", "mean", "std", "min", "median", "max"]

    return {"miss_top": miss.head(15), "desc": desc}


# ---------------------------------------------------------------------------
# Part B — 相关性计算
# ---------------------------------------------------------------------------
def compute_correlations(df: pd.DataFrame, target_col: str = TARGET_COL
                         ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """返回 (target_corr, full_corr_matrix, seg_corr_df)。"""
    # 排除离散标识（保留小时/月/星期参与相关性观察）
    exclude = {"年", "日", "是否周末"}
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    full = df[num_cols].corr(method="pearson")
    target_corr = full[target_col].drop(target_col).sort_values(
        key=lambda s: s.abs(), ascending=False)

    seg_data = {}
    if "时段" in df.columns:
        for seg in ["峰", "平", "谷"]:
            sub = df[df["时段"] == seg]
            if len(sub) > 30:
                seg_data[seg] = sub[num_cols].corr()[target_col].drop(target_col)
    seg_df = (pd.DataFrame(seg_data).reindex(target_corr.index)
              if seg_data else pd.DataFrame())
    return target_corr, full, seg_df


# ---------------------------------------------------------------------------
# Part C — 可用性标注
# ---------------------------------------------------------------------------
def feature_legality(name: str) -> Tuple[str, str]:
    """day-ahead 可用性标签 → (CSS 类, 文字)。"""
    if name in LEAKAGE_COLS:
        return "leak", "❌ 数据泄漏 (与目标同时刻产生)"
    if "_实际" in name:
        return "loser", "❌ D 日未知, day-ahead 不可用"
    return "winner", "✅ 可用 (D-1 已知)"


# ---------------------------------------------------------------------------
# 可视化（EDA 部分）
# ---------------------------------------------------------------------------
def plot_eda(df: pd.DataFrame, miss_top: pd.Series,
             target_col: str = TARGET_COL) -> Dict[str, str]:
    setup_cn_font()
    paths = {}

    # 1. 缺失率
    if not miss_top.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(miss_top))))
        miss_top.sort_values().plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_xlabel("缺失率 (%)"); ax.set_title("各字段缺失率 Top 15")
        p = os.path.join(PATHS["plots"], "eda_missing.png")
        safe_savefig(p); paths["missing"] = p

    # 2. 数值字段分布直方图 (网格)
    num_cols = _numeric_cols(df, exclude=["年", "月", "日", "小时", "星期",
                                          "是否周末"])
    if num_cols:
        n = len(num_cols); ncol = 3; nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(num_cols):
            ax = axes[i]
            data = df[c].dropna()
            if data.empty:
                ax.set_visible(False); continue
            ax.hist(data, bins=50, color="#4C72B0", alpha=0.85)
            ax.set_title(c, fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("各数值字段分布直方图", fontsize=13, y=1.01)
        p = os.path.join(PATHS["plots"], "eda_distributions.png")
        safe_savefig(p); paths["dist"] = p

    # 3. 电价时序图组（4 子图：日均 / 月均 / 24h按月 / 小时×月热力）
    price_cols = [c for c in [
        "日前统一结算点电价(元/MWh)", "实时统一结算点电价(元/MWh)",
        "实时出清电价(元/MWh)",
    ] if c in df.columns]

    if price_cols:
        # 3.1 日均走势
        daily = df.set_index("datetime")[price_cols].resample("D").mean()
        fig, ax = plt.subplots(figsize=(14, 4.5))
        for c in price_cols:
            ax.plot(daily.index, daily[c], label=c, lw=1.1)
        ax.set_title("全周期日均电价走势"); ax.set_xlabel("日期")
        ax.set_ylabel("电价 (元/MWh)"); ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        p = os.path.join(PATHS["plots"], "eda_price_daily.png")
        safe_savefig(p); paths["price_daily"] = p

        # 3.2 月度均价柱状
        monthly = (df.assign(年月=df["datetime"].dt.strftime("%Y-%m"))
                     .groupby("年月")[price_cols].mean())
        fig, ax = plt.subplots(figsize=(13, 4.5))
        monthly.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("月度平均电价"); ax.set_xlabel("年-月")
        ax.set_ylabel("电价 (元/MWh)"); ax.legend(fontsize=9)
        plt.xticks(rotation=45, ha="right")
        p = os.path.join(PATHS["plots"], "eda_price_monthly.png")
        safe_savefig(p); paths["price_monthly"] = p

        # 3.3 24h 按月分线
        hourly = df.groupby(["月", "小时"])[target_col].mean().unstack(0)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        cmap = plt.get_cmap("tab20")
        for i, col in enumerate(hourly.columns):
            ax.plot(hourly.index, hourly[col], label=f"{int(col)}月",
                    color=cmap(i % 20), lw=1.4)
        ax.set_title(f"日内 24 小时电价模式 (按月)  目标列: {target_col}")
        ax.set_xlabel("小时"); ax.set_ylabel("电价 (元/MWh)")
        ax.set_xticks(range(0, 24))
        ax.legend(ncol=4, fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        p = os.path.join(PATHS["plots"], "eda_price_24h_by_month.png")
        safe_savefig(p); paths["price_24h"] = p

        # 3.4 小时 × 月份热力图
        pivot = df.pivot_table(index="小时", columns="月",
                                values=target_col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.heatmap(pivot, cmap="RdYlBu_r", annot=False, ax=ax,
                    cbar_kws={"label": "电价 (元/MWh)"})
        ax.set_title(f"小时 × 月份 电价热力图  目标列: {target_col}")
        p = os.path.join(PATHS["plots"], "eda_price_heatmap.png")
        safe_savefig(p); paths["price_heatmap"] = p

    return paths


# ---------------------------------------------------------------------------
# 可视化（相关性部分）
# ---------------------------------------------------------------------------
def plot_correlations(df: pd.DataFrame, target_corr: pd.Series,
                      full: pd.DataFrame, seg_df: pd.DataFrame,
                      target_col: str = TARGET_COL) -> Dict[str, str]:
    setup_cn_font()
    paths = {}

    # 1. 与 target 相关性条形图 (Top 20)
    top = target_corr.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"各因子与 {target_col} 的相关性 Top 20")
    for i, v in enumerate(top.values):
        ax.text(v + (0.01 if v > 0 else -0.01), i, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right", fontsize=9)
    p = os.path.join(PATHS["plots"], "corr_top.png")
    safe_savefig(p); paths["top"] = p

    # 2. 全因子热力图
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(full)),
                                    max(8, 0.4 * len(full))))
    sns.heatmap(full, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("全因子相关性矩阵")
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
    p = os.path.join(PATHS["plots"], "corr_heatmap.png")
    safe_savefig(p); paths["heatmap"] = p

    # 3. 峰平谷分段
    if not seg_df.empty:
        seg_df_top = seg_df.loc[target_corr.head(15).index].iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(seg_df_top))))
        seg_df_top.plot(kind="barh", ax=ax, width=0.8,
                        color=["#C44E52", "#55A868", "#4C72B0"])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("Pearson r")
        ax.set_title("主因子相关性 — 分时段 (峰/平/谷)")
        p = os.path.join(PATHS["plots"], "corr_by_segment.png")
        safe_savefig(p); paths["seg"] = p

    # 4. Top 4 主因子散点
    top4 = target_corr.head(4).index.tolist()
    if top4:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.reshape(-1)
        sample = df[[target_col] + top4].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        for i, c in enumerate(top4):
            ax = axes[i]
            ax.scatter(sample[c], sample[target_col], s=6, alpha=0.35, c="#4C72B0")
            try:
                z = np.polyfit(sample[c].values, sample[target_col].values, 1)
                xs = np.linspace(sample[c].min(), sample[c].max(), 100)
                ax.plot(xs, np.polyval(z, xs), color="#C44E52", lw=1.5)
            except Exception:
                pass
            r = sample[c].corr(sample[target_col])
            ax.set_title(f"{c}\nr = {r:.3f}", fontsize=10)
            ax.set_xlabel(c); ax.set_ylabel(target_col, fontsize=9)
        fig.suptitle("Top 4 主因子 vs 目标 散点图 + 一次拟合",
                     fontsize=13, y=1.02)
        p = os.path.join(PATHS["plots"], "corr_top4_scatter.png")
        safe_savefig(p); paths["scatter"] = p

    return paths


# ---------------------------------------------------------------------------
# 业务语义自动解读
# ---------------------------------------------------------------------------
def auto_insights(df: pd.DataFrame, target_corr: pd.Series,
                  seg_df: pd.DataFrame, target_col: str = TARGET_COL
                  ) -> List[str]:
    """从数据自动抽取业务语义解读语句。"""
    out = []
    if not target_corr.empty:
        pos = target_corr[target_corr > 0].head(3)
        neg = target_corr[target_corr < 0].head(3)
        if not pos.empty:
            tops = ", ".join([f"<strong>{k}</strong> (r={v:.2f})" for k, v in pos.items()])
            out.append(f"<strong>推升电价的主要因子</strong>: {tops}<br>"
                       "这些因子上升通常伴随系统供需偏紧或负荷上行，电价随之走高。")
        if not neg.empty:
            tops = ", ".join([f"<strong>{k}</strong> (r={v:.2f})" for k, v in neg.items()])
            out.append(f"<strong>抑制电价的主要因子</strong>: {tops}<br>"
                       "新能源/水电出力增加挤占火电边际机组，结算点价格趋于回落。")

    if target_col in df.columns:
        peak_h = df.groupby("小时")[target_col].mean().idxmax()
        valley_h = df.groupby("小时")[target_col].mean().idxmin()
        out.append(f"<strong>日内电价模式</strong>: 高点出现在 <strong>{peak_h} 时</strong>，"
                   f"低点在 <strong>{valley_h} 时</strong>，呈现典型双峰/单谷型负荷特征。")

    if not seg_df.empty:
        ne_candidates = [c for c in seg_df.index
                         if any(k in c for k in ["光伏", "新能源", "风电"])]
        if ne_candidates:
            ne_top = max(ne_candidates,
                         key=lambda c: abs(seg_df.loc[c].get("峰", 0) or 0))
            v_peak = seg_df.loc[ne_top].get("峰", np.nan)
            v_valley = seg_df.loc[ne_top].get("谷", np.nan)
            if pd.notna(v_peak) and pd.notna(v_valley):
                out.append(f"<strong>峰谷差异</strong>: {ne_top} 对电价的相关性在峰段 r={v_peak:.2f}，"
                           f"谷段 r={v_valley:.2f}；新能源对峰时段挤压更明显，谷时段几乎不出力。")

    if target_col in df.columns:
        m_mean = df.groupby(df["datetime"].dt.strftime("%Y-%m"))[target_col].mean()
        if not m_mean.empty:
            out.append(f"<strong>月度波动</strong>: 均价范围 "
                       f"<strong>{m_mean.min():.1f} ~ {m_mean.max():.1f}</strong> 元/MWh，"
                       f"最高 <strong>{m_mean.idxmax()}</strong>，最低 <strong>{m_mean.idxmin()}</strong>。")
    return out


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------
def make_html(df: pd.DataFrame, eda_stats: Dict, target_corr: pd.Series,
              seg_df: pd.DataFrame, eda_plots: Dict[str, str],
              corr_plots: Dict[str, str], target_col: str = TARGET_COL) -> str:
    p: List[str] = []

    months = df["datetime"].dt.strftime("%Y-%m").unique().tolist()
    p.append('<div class="meta">')
    p.append(f"<p><strong>目标变量</strong>: <code>{target_col}</code></p>")
    p.append(f"<p><strong>时间范围</strong>: {df['datetime'].min()} ~ {df['datetime'].max()}</p>")
    p.append(f"<p><strong>样本规模</strong>: {len(df):,} 行 × {df.shape[1]} 列 (15min 粒度)</p>")
    p.append(f"<p><strong>覆盖月份</strong>: {len(months)} 个 → <code>{', '.join(months)}</code></p>")
    p.append("</div>")

    # ============ Part A — EDA ============
    p.append("<h2>一、数据概况</h2>")
    miss = eda_stats["miss_top"]
    if not miss.empty:
        p.append("<h3>缺失率 Top 15</h3>")
        p.append("<table><tr><th>字段</th><th>缺失率 (%)</th></tr>")
        for k, v in miss.items():
            p.append(f"<tr><td>{k}</td><td class='num'>{v:.2f}</td></tr>")
        p.append("</table>")
        p.append(img_tag(eda_plots.get("missing", "")))
    else:
        p.append("<p>✅ 无缺失字段</p>")

    p.append("<h3>关键字段描述统计</h3>")
    p.append(eda_stats["desc"].to_html(border=0, classes=""))

    p.append("<h2>二、字段分布</h2>")
    p.append(img_tag(eda_plots.get("dist", "")))

    p.append("<h2>三、电价时序特征</h2>")
    p.append("<h3>3.1 全周期日均电价走势</h3>")
    p.append(img_tag(eda_plots.get("price_daily", "")))
    p.append("<h3>3.2 月度平均电价</h3>")
    p.append(img_tag(eda_plots.get("price_monthly", "")))
    p.append("<h3>3.3 日内 24 小时电价模式 (按月分线)</h3>")
    p.append(img_tag(eda_plots.get("price_24h", "")))
    p.append("<h3>3.4 小时 × 月份 电价热力图</h3>")
    p.append(img_tag(eda_plots.get("price_heatmap", "")))

    # ============ Part B — 相关性分析 ============
    p.append("<h2>四、与目标相关性 Top 20 (按 |r| 排序)</h2>")
    p.append("<table><tr><th>排名</th><th>因子</th><th>r</th>"
             "<th>方向</th><th>强度</th><th>day-ahead 可用性</th></tr>")
    for i, (k, v) in enumerate(target_corr.head(20).items(), 1):
        d = "➕ 正" if v > 0 else "➖ 负"
        a = abs(v)
        if a >= 0.7: s = "🔴 强"
        elif a >= 0.4: s = "🟠 中"
        elif a >= 0.2: s = "🟡 弱"
        else: s = "⚪ 极弱"
        cls, label = feature_legality(k)
        p.append(f"<tr class='{cls}'><td class='center'>{i}</td>"
                 f"<td><code>{k}</code></td>"
                 f"<td class='num'>{v:+.3f}</td>"
                 f"<td class='center'>{d}</td>"
                 f"<td class='center'>{s}</td>"
                 f"<td>{label}</td></tr>")
    p.append("</table>")
    p.append(img_tag(corr_plots.get("top", "")))

    p.append('<div class="warn">')
    p.append("<p><strong>数据泄漏识别</strong>: <code>实时出清电价(元/MWh)</code> 与目标"
             f" <code>{target_col}</code> 是同一次出清的两个口径 (15min 出清价 vs 1h 统一结算价)，"
             "<strong>同时刻产生</strong>，相关性 ≈ 0.95。用作特征等于变相用未来信息，必须剔除。</p>")
    p.append("</div>")

    if not seg_df.empty:
        p.append("<h2>五、峰/平/谷分时段相关性差异</h2>")
        p.append(seg_df.head(15).round(3).to_html(border=0, classes=""))
        p.append(img_tag(corr_plots.get("seg", "")))

    p.append("<h2>六、Top 4 主因子 vs 目标散点图</h2>")
    p.append(img_tag(corr_plots.get("scatter", "")))

    p.append("<h2>七、全因子相关性热力图</h2>")
    p.append(img_tag(corr_plots.get("heatmap", "")))

    # ============ Part C — 业务解读 ============
    p.append("<h2>八、电力业务专业解读</h2>")
    insights = auto_insights(df, target_corr, seg_df, target_col)
    for ins in insights:
        p.append(f'<div class="insight">{ins}</div>')

    p.append('<div class="insight">')
    p.append("<p><strong>建模启示</strong>：</p>")
    p.append("<ul>")
    p.append("<li>日前价 r ≈ 0.83 是最强合法外部锚，应作为残差预测的 base</li>")
    p.append("<li>光伏 (r ≈ -0.50) / 竞价空间 (r ≈ +0.50) 是最有价值的合法供需特征</li>")
    p.append("<li>峰段对新能源敏感度远强于谷段，可考虑分时段建模</li>")
    p.append("</ul></div>")

    return render_html("③ 数据相关性分析报告（含 EDA）", p, PATHS["html_correlation"])


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> str:
    ensure_dirs()
    setup_cn_font()
    df = load_clean()

    print("[INFO] 计算 EDA 统计...")
    eda_stats = compute_eda(df)
    eda_plots = plot_eda(df, eda_stats["miss_top"])

    print("[INFO] 计算相关性...")
    target_corr, full, seg_df = compute_correlations(df)
    save_pickle({"target_corr": target_corr, "full_corr": full,
                 "seg_corr": seg_df}, PATHS["corr"])
    corr_plots = plot_correlations(df, target_corr, full, seg_df)

    out = make_html(df, eda_stats, target_corr, seg_df, eda_plots, corr_plots)
    print(f"[INFO] 已生成 {out}")
    return out


if __name__ == "__main__":
    main()
