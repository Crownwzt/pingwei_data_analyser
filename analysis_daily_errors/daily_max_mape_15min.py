"""
按"日内单点最大 MAPE"重排测试集逐日图，并在每张图下方拼接 15min 分辨率子图。

数据源（只读）：
  - outputs/metrics.pkl     小时级 y_test / da_test / y_pred / dt_test
  - 2025-2026市场情况/2025-12-01到2025-12-30市场价格趋势.xlsx  15min 明细

产物（全部写入本目录）：
  analysis_daily_errors/
    ├── daily_max_mape_15min.py             本脚本
    ├── daily_by_max_mape/                  按 max MAPE 排序的逐日图（rank01_最坏 → rank29_最好）
    │   └── rank01_MaxMAPE{XX}_2025-MM-DD.png
    ├── summary_max_mape.csv                每日各指标汇总
    └── ranking_comparison.png              两种排序（TrMAE vs MaxMAPE）对比

图结构（每张 PNG）：
  上子图  ┌── 小时级：真实值 / 日前 / XGB 预测 (24 点)
          └── 标注每小时 |真实-预测| 的绝对误差与该点 MAPE
  下子图  └── 15min 真实值 vs 15min 日前 (96 点，同日)
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
OUT_DIR = os.path.join(ROOT, "daily_by_max_mape")
os.makedirs(OUT_DIR, exist_ok=True)


def setup_font():
    # 与 src/common.py:setup_cn_font 完全一致：Noto Sans CJK JP 在 ttc 中
    # 包含全部中文汉字，是 matplotlib 能识别到的可靠中文字体
    cand = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in cand if c in avail), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


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
    })
    d["date"] = d["datetime"].dt.date
    return d[["datetime", "date", "y_true", "da"]].sort_values("datetime").reset_index(drop=True)


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
            "max_abs_err": float(errs.max()),
            "max_mape": float(mapes.max()),
            "mean_mape": float(mapes.mean()),
            "peak_hour": int(g.loc[g["abs_err"].idxmax(), "datetime"].hour),
            "peak_y_true": float(g.loc[g["abs_err"].idxmax(), "y_true"]),
            "peak_y_pred": float(g.loc[g["abs_err"].idxmax(), "y_pred"]),
        })
    return pd.DataFrame(rows).sort_values("max_mape", ascending=False).reset_index(drop=True)


# ---------- 绘图 ----------
def plot_one_day(hourly_day: pd.DataFrame, fifteen_day: pd.DataFrame,
                 meta: Dict, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1.2, 1.0]},
        sharex=False,
    )

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

    idx_max = hd["abs_err"].idxmax()
    peak = hd.loc[idx_max]
    ax1.axvline(peak["datetime"], color="red", ls=":", alpha=0.5)
    ax1.annotate(
        f"最大误差 {peak['abs_err']:.1f} 元\nMAPE={peak['mape']:.1f}%\n"
        f"真={peak['y_true']:.0f}  预={peak['y_pred']:.0f}",
        xy=(peak["datetime"], max(peak["y_true"], peak["y_pred"])),
        xytext=(10, 25), textcoords="offset points",
        fontsize=9, color="red",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.85),
    )

    ax1.set_ylabel("电价 (元/MWh)")
    ax1.set_title(
        f"[test] {meta['date']}  日内 24 点预测  "
        f"MaxMAPE={meta['max_mape']:.1f}%  "
        f"MaxAbsErr={meta['max_abs_err']:.1f}  "
        f"MAE={meta['mae']:.2f}  TrMAE={meta['trmae10']:.2f}",
        fontsize=11,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # 下：15min 真实值 vs 日前（同风格）
    if fifteen_day is not None and len(fifteen_day) > 0:
        fd = fifteen_day.sort_values("datetime")
        ax2.plot(fd["datetime"], fd["y_true"], label="实时价 (15min)",
                 color="#4C72B0", lw=1.2)
        ax2.plot(fd["datetime"], fd["da"], label="日前价 (15min)",
                 color="#888", lw=1.2, ls=":")
        ax2.fill_between(fd["datetime"], fd["y_true"], fd["da"],
                         where=fd["y_true"] > fd["da"],
                         color="#C44E52", alpha=0.12, label="实时>日前")
        ax2.fill_between(fd["datetime"], fd["y_true"], fd["da"],
                         where=fd["y_true"] < fd["da"],
                         color="#4C72B0", alpha=0.10, label="实时<日前")
    ax2.set_xlabel("时间"); ax2.set_ylabel("15min 电价 (元/MWh)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_ranking_comparison(daily: pd.DataFrame, out_path: str) -> None:
    d = daily.copy()
    d["rank_maxmape"] = d["max_mape"].rank(ascending=False).astype(int)
    d["rank_trmae"] = d["trmae10"].rank(ascending=False).astype(int)
    d = d.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    w = 0.4
    ax.bar(x - w / 2, d["rank_trmae"], w, label="rank by TrMAE (原)", color="#337ab7")
    ax.bar(x + w / 2, d["rank_maxmape"], w, label="rank by MaxMAPE (新)", color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels([str(dd)[-5:] for dd in d["date"]], rotation=45, fontsize=8)
    ax.set_ylabel("排名 (数字越小 = 误差越大)")
    ax.set_title("两种排序对比：TrMAE vs 日内单点最大 MAPE")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ---------- 主流程 ----------
def main():
    setup_font()
    print(f"[1/4] 加载小时级测试集数据: {METRICS_PKL}")
    hourly = load_test_hourly()
    print(f"      测试集 n = {len(hourly)},  {hourly['date'].nunique()} 天")

    print(f"[2/4] 加载 15min 明细: {XLSX_15MIN}")
    f15 = load_15min()
    print(f"      15min n = {len(f15)},  {f15['date'].nunique()} 天")

    print("[3/4] 计算每日汇总并按 MaxMAPE 排序")
    daily = daily_summary(hourly)
    daily.to_csv(os.path.join(ROOT, "summary_max_mape.csv"),
                 index=False, encoding="utf-8-sig")
    print(daily[["date", "n", "mae", "trmae10", "max_abs_err", "max_mape", "peak_hour"]].to_string(index=False))

    print(f"[4/4] 生成逐日图 → {OUT_DIR}")
    rank_pad = max(2, len(str(len(daily))))
    for i, row in daily.iterrows():
        rank = i + 1
        date = row["date"]
        hourly_day = hourly[hourly["date"] == date]
        f15_day = f15[f15["date"] == date]

        fname = f"rank{rank:0{rank_pad}d}_MaxMAPE{row['max_mape']:06.2f}_{date}.png"
        out = os.path.join(OUT_DIR, fname)
        plot_one_day(hourly_day, f15_day, row.to_dict(), out)

    plot_ranking_comparison(daily, os.path.join(ROOT, "ranking_comparison.png"))
    print("\n完成。")


if __name__ == "__main__":
    main()
