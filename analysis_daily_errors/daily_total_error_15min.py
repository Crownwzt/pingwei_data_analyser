"""
按"日总绝对误差"（一天 24h 累计误差 = sum|真实-预测|）重排测试集逐日图，
并在每张图下方拼接 15min 分辨率子图。

数据源（只读）：
  - outputs/metrics.pkl     小时级 y_test / da_test / y_pred / dt_test
  - 2025-2026市场情况/2025-12-01到2025-12-30市场价格趋势.xlsx  15min 明细

产物（全部写入本目录）：
  analysis_daily_errors/
    ├── daily_total_error_15min.py          本脚本
    ├── daily_by_total_error/               按日总误差排序的逐日图（rank01_最大累计误差 → rank29_最小）
    │   └── rank01_TotalErr{XXXX.X}_2025-MM-DD.png
    ├── summary_total_error.csv             每日各指标汇总
    └── ranking_comparison_total.png        排序对比

图结构（每张 PNG）：
  上子图  ┌── 小时级：真实值 / 日前 / XGB 预测 (24 点)
          └── 标注每小时 |真实-预测| 的绝对误差与该点 MAPE
  中子图  └── 15min 统一结算点电价：实时 vs 日前 (96 点，同日)
  下子图  └── 15min 节点电价：实时 vs 日前 (96 点，同日)
  注：中/下子图共享 x 轴与 y 轴范围，可上下对齐、直接目视对比幅度
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
METRICS_PKL = os.path.join(PROJECT, "outputs", "metrics.pkl")
XLSX_15MIN = os.path.join(
    PROJECT, "2025-2026市场情况", "2025-12-01到2025-12-30市场价格趋势.xlsx",
)
OUT_DIR = os.path.join(ROOT, "daily_by_total_error")
os.makedirs(OUT_DIR, exist_ok=True)


def setup_font():
    # 中文字体候选，按优先级排列：
    #   前段 = Linux/Windows 常见 CJK 字体（与 src/common.py:setup_cn_font 一致，
    #          保证在原 Linux 服务器上选中的字体不变）
    #   后段 = macOS 自带 CJK 字体（Linux 字体在 macOS 上全部缺失，
    #          若不补充会回退到 DejaVu Sans —— 该字体无中文字形，图例会显示成方框）
    cand = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei",
            "PingFang SC", "PingFang HK", "Hiragino Sans GB",
            "Arial Unicode MS", "STHeiti", "Songti SC"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in cand if c in avail), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    else:
        print("[WARN] 未找到任何中文字体，图中中文将显示为方框")
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ---------- 加载 ----------
def load_test_hourly() -> pd.DataFrame:
    with open(METRICS_PKL, "rb") as f:
        m = pickle.load(f)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(m["dt_test"]),
        "y_true": m["y_test"],
        "da": m["da_test"],
        "y_pred": m["metrics"]["test"]["y_pred"],
    })
    df["date"] = df["datetime"].dt.date
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    # MAPE 分母保护：真实值太小时用一个下限，避免爆炸
    denom = df["y_true"].abs().clip(lower=10.0)
    df["mape"] = df["abs_err"] / denom * 100.0
    return df


def load_15min() -> pd.DataFrame:
    xl = pd.ExcelFile(XLSX_15MIN)
    d = xl.parse("明细")
    d["datetime"] = pd.to_datetime(d["日期"].astype(str) + " " + d["时间"].astype(str))
    d = d.rename(columns={
        "日前统一结算点电价(元/MWh)": "da",
        "实时统一结算点电价(元/MWh)": "y_true",
        # 节点电价：与统一结算点价的差额 = 阻塞 + 网损分量，
        # 用于对比分析同一时刻"节点 vs 统一结算点"的价差来源
        "日前节点电价(元/MWh)": "da_node",
        "实时节点电价(元/MWh)": "rt_node",
    })
    d["date"] = d["datetime"].dt.date
    cols = ["datetime", "date", "y_true", "da", "da_node", "rt_node"]
    return d[cols].sort_values("datetime").reset_index(drop=True)


# ---------- 每日统计 ----------
def _trmae(errs: np.ndarray, trim: float = 0.10) -> float:
    n = len(errs)
    if n == 0:
        return float("nan")
    k = int(n * trim)
    s = np.sort(errs)
    core = s[k: n - k] if n > 2 * k else s
    return float(core.mean())


def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in hourly.groupby("date"):
        errs = g["abs_err"].to_numpy()
        mapes = g["mape"].to_numpy()
        rows.append({
            "date": d,
            "n": len(g),
            "mae": float(errs.mean()),
            "trmae10": _trmae(errs, 0.10),
            "total_error": float(errs.sum()),  # 日总绝对误差 = sum|真实-预测|
            "max_abs_err": float(errs.max()),
            "max_mape": float(mapes.max()),
            "mean_mape": float(mapes.mean()),
            "peak_hour": int(g.loc[g["abs_err"].idxmax(), "datetime"].hour),
            "peak_y_true": float(g.loc[g["abs_err"].idxmax(), "y_true"]),
            "peak_y_pred": float(g.loc[g["abs_err"].idxmax(), "y_pred"]),
        })
    return pd.DataFrame(rows).sort_values("total_error", ascending=False).reset_index(drop=True)


# ---------- 绘图 ----------
def plot_one_day(hourly_day: pd.DataFrame, fifteen_day: pd.DataFrame,
                 meta: Dict, out_path: str) -> None:
    # 三子图：小时级预测 / 15min 统一结算点价 / 15min 节点价
    # 后两者拆开画，避免 4 条线挤在同一坐标系里互相遮挡；
    # 两个 15min 子图共享 x 轴（同一时间基准），便于上下对齐读同一时刻
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 11),
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
        sharex=False,
    )
    ax3.sharex(ax2)

    # 上：小时级对比（沿用 outputs/daily/test 图例配色）
    hd = hourly_day.sort_values("datetime")
    ax1.plot(hd["datetime"], hd["y_true"], label="真实",
             color="#4C72B0", lw=1.8, marker="o", ms=4)
    ax1.plot(hd["datetime"], hd["da"], label="日前价(B7')",
             color="#888", lw=1.2, ls=":", marker="s", ms=3)
    ax1.plot(hd["datetime"], hd["y_pred"], label="XGB 预测",
             color="#C44E52", lw=1.6, marker="x", ms=5)
    ax1.fill_between(hd["datetime"], hd["y_true"], hd["y_pred"],
                     color="#C44E52", alpha=0.12, label="XGB 误差区间")

    ax1.set_ylabel("电价 (元/MWh)")
    ax1.set_title(
        f"[test] {meta['date']}  日内 24 点预测  "
        f"日总误差={meta['total_error']:.1f}元  "
        f"MAE={meta['mae']:.2f}  TrMAE={meta['trmae10']:.2f}",
        fontsize=11,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # 中：15min 统一结算点电价（实时 vs 日前）
    # 下：15min 节点电价（实时 vs 日前）—— 与上方统一结算点价的差额即阻塞 + 网损分量
    if fifteen_day is not None and len(fifteen_day) > 0:
        fd = fifteen_day.sort_values("datetime")

        # 两个子图统一配色语义：实时=蓝色实线，日前=灰色虚线
        for ax, rt_col, da_col, tag in (
            (ax2, "y_true", "da", "统一结算点"),
            (ax3, "rt_node", "da_node", "节点"),
        ):
            ax.plot(fd["datetime"], fd[rt_col], label=f"实时{tag}电价 (15min)",
                    color="#4C72B0", lw=1.3)
            ax.plot(fd["datetime"], fd[da_col], label=f"日前{tag}电价 (15min)",
                    color="#888", lw=1.2, ls=":")
            ax.fill_between(fd["datetime"], fd[rt_col], fd[da_col],
                            where=fd[rt_col] > fd[da_col],
                            color="#C44E52", alpha=0.12, label="实时>日前")
            ax.fill_between(fd["datetime"], fd[rt_col], fd[da_col],
                            where=fd[rt_col] < fd[da_col],
                            color="#4C72B0", alpha=0.10, label="实时<日前")

        # 两个 15min 子图共用同一 y 轴范围，保证幅度可直接目视对比
        lo = float(min(fd[["y_true", "da", "rt_node", "da_node"]].min()))
        hi = float(max(fd[["y_true", "da", "rt_node", "da_node"]].max()))
        pad = max((hi - lo) * 0.08, 5.0)
        ax2.set_ylim(lo - pad, hi + pad)
        ax3.set_ylim(lo - pad, hi + pad)

    ax2.set_ylabel("统一结算点价 (元/MWh)")
    ax2.set_title("15min 统一结算点电价：实时 vs 日前", fontsize=10)
    ax3.set_xlabel("时间")
    ax3.set_ylabel("节点价 (元/MWh)")
    ax3.set_title("15min 节点电价：实时 vs 日前", fontsize=10)
    for ax in (ax2, ax3):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_ranking_comparison(daily: pd.DataFrame, out_path: str) -> None:
    d = daily.copy()
    d["rank_totalerr"] = d["total_error"].rank(ascending=False).astype(int)
    d["rank_trmae"] = d["trmae10"].rank(ascending=False).astype(int)
    d = d.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    w = 0.4
    ax.bar(x - w / 2, d["rank_trmae"], w, label="rank by TrMAE", color="#337ab7")
    ax.bar(x + w / 2, d["rank_totalerr"], w, label="rank by 日总误差", color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels([str(dd)[-5:] for dd in d["date"]], rotation=45, fontsize=8)
    ax.set_ylabel("排名 (数字越小 = 误差越大)")
    ax.set_title("两种排序对比：TrMAE vs 日总绝对误差")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ---------- 主流程 ----------
def main():
    font = setup_font()
    print(f"[0/4] 中文字体: {font or '未找到（中文将显示为方框）'}")
    print(f"[1/4] 加载小时级测试集数据: {METRICS_PKL}")
    hourly = load_test_hourly()
    print(f"      测试集 n = {len(hourly)},  {hourly['date'].nunique()} 天")

    print(f"[2/4] 加载 15min 明细: {XLSX_15MIN}")
    f15 = load_15min()
    print(f"      15min n = {len(f15)},  {f15['date'].nunique()} 天")

    print("[3/4] 计算每日汇总并按 TotalError 排序")
    daily = daily_summary(hourly)
    daily.to_csv(os.path.join(ROOT, "summary_total_error.csv"),
                 index=False, encoding="utf-8-sig")
    print(daily[["date", "n", "mae", "trmae10", "total_error", "max_abs_err", "peak_hour"]].to_string(index=False))

    print(f"[4/4] 生成逐日图 → {OUT_DIR}")
    rank_pad = max(2, len(str(len(daily))))
    for i, row in daily.iterrows():
        rank = i + 1
        date = row["date"]
        hourly_day = hourly[hourly["date"] == date]
        f15_day = f15[f15["date"] == date]

        fname = f"rank{rank:0{rank_pad}d}_TotalErr{row['total_error']:07.1f}_{date}.png"
        out = os.path.join(OUT_DIR, fname)
        plot_one_day(hourly_day, f15_day, row.to_dict(), out)

    plot_ranking_comparison(daily, os.path.join(ROOT, "ranking_comparison_total.png"))
    print("\n完成。")


if __name__ == "__main__":
    main()
