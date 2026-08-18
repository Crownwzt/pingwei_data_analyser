#!/usr/bin/env python3
"""
按小时统计训练集/测试集的收益分布，识别不该交易的时段
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_font():
    for name in ["PingFang HK", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]:
        try:
            f = font_manager.findfont(name, fallback_to_default=False)
            if "ttf" in f.lower() or "otf" in f.lower():
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
        except:
            continue
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def load_metrics(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 重构为 DataFrame
    result = {}
    for split in ["train", "val", "test"]:
        dt_key = f"dt_{split}"
        y_key = f"y_{split}"
        da_key = f"da_{split}"

        df = pd.DataFrame({
            "datetime": pd.to_datetime(data[dt_key]),
            "y_true": data[y_key],
            "y_pred": data["metrics"][split]["y_pred"],
            "da": data[da_key],
        })
        result[split] = df

    return result


def compute_hourly_benefit(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个小时的收益 = |日前-真实| - |XGB-真实|"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["xgb_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["da_err"] = (df["da"] - df["y_true"]).abs()
    df["benefit"] = df["da_err"] - df["xgb_err"]  # 正=XGB更好

    hourly = df.groupby("hour").agg({
        "benefit": ["mean", "std", "count"],
        "xgb_err": "mean",
        "da_err": "mean",
    }).reset_index()
    hourly.columns = ["hour", "benefit_mean", "benefit_std", "n", "xgb_err_mean", "da_err_mean"]
    return hourly


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(ROOT)
    METRICS_PKL = os.path.join(PROJECT_ROOT, "outputs", "metrics.pkl")
    OUT_DIR = os.path.join(ROOT, "hourly_benefit")
    os.makedirs(OUT_DIR, exist_ok=True)

    font = setup_font()
    print(f"[1/5] 中文字体: {font or '未找到'}")

    print(f"[2/5] 加载 metrics: {METRICS_PKL}")
    data = load_metrics(METRICS_PKL)

    val_df = data["val"]
    test_df = data["test"]

    print(f"      val n={len(val_df)}, test n={len(test_df)}")

    print("[3/5] 计算各集合的小时收益分布")
    val_hourly = compute_hourly_benefit(val_df)
    test_hourly = compute_hourly_benefit(test_df)

    # 合并
    summary = val_hourly[["hour", "benefit_mean"]].rename(columns={"benefit_mean": "val_benefit"})
    summary = summary.merge(
        test_hourly[["hour", "benefit_mean"]].rename(columns={"benefit_mean": "test_benefit"}),
        on="hour"
    )
    summary["both_negative"] = (summary["val_benefit"] < 0) & (summary["test_benefit"] < 0)

    print("\n小时收益统计 (元):")
    print(summary.to_string(index=False))

    # 保存
    summary.to_csv(os.path.join(OUT_DIR, "hourly_benefit_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n[4/5] 绘制对比图")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = summary["hour"]
    ax1.plot(x, summary["val_benefit"], marker="o", label="验证集", color="#337ab7", lw=2)
    ax1.plot(x, summary["test_benefit"], marker="s", label="测试集", color="#d9534f", lw=2)
    ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.fill_between(x, 0, summary["val_benefit"], where=(summary["val_benefit"] < 0),
                      alpha=0.2, color="#337ab7", label="验证集负收益区")
    ax1.fill_between(x, 0, summary["test_benefit"], where=(summary["test_benefit"] < 0),
                      alpha=0.2, color="#d9534f", label="测试集负收益区")

    # 标注双负时段
    double_neg = summary[summary["both_negative"]]
    for _, row in double_neg.iterrows():
        ax1.axvspan(row["hour"] - 0.3, row["hour"] + 0.3, color="red", alpha=0.15, zorder=0)
        ax1.text(row["hour"], min(row["val_benefit"], row["test_benefit"]) - 2,
                 "双负", ha="center", fontsize=8, color="red", weight="bold")

    ax1.set_ylabel("平均收益 (元)")
    ax1.set_title("小时维度收益分布 (XGB vs 日前价)\n红色区域 = 验证&测试都是负收益", fontsize=12, weight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    # 下图：误差对比
    ax2.bar(x - 0.2, val_hourly["xgb_err_mean"], width=0.4, label="XGB误差(val)", color="#5cb85c", alpha=0.7)
    ax2.bar(x + 0.2, val_hourly["da_err_mean"], width=0.4, label="日前误差(val)", color="#f0ad4e", alpha=0.7)
    ax2.set_xlabel("小时 (0-23)")
    ax2.set_ylabel("平均绝对误差 (元)")
    ax2.set_title("验证集小时误差对比", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hourly_benefit_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      已保存: hourly_benefit_comparison.png")

    print("\n[5/5] 计算优化后收益")
    skip_hours = set(summary[summary["both_negative"]]["hour"])
    print(f"      双负时段: {sorted(skip_hours)}")

    # 原策略：全天交易
    test_original = test_df.copy()
    test_original["benefit"] = (test_original["da"] - test_original["y_true"]).abs() - \
                                (test_original["y_pred"] - test_original["y_true"]).abs()
    total_original = test_original["benefit"].sum()

    # 优化策略：跳过双负时段
    test_optimized = test_original[~test_original["datetime"].dt.hour.isin(skip_hours)]
    total_optimized = test_optimized["benefit"].sum()

    skipped_hours_count = len(test_original) - len(test_optimized)
    improvement = total_optimized - total_original

    print(f"\n=== 测试集收益对比 ===")
    print(f"原策略 (全天交易):      {total_original:>8.1f} 元  (n={len(test_original)} 小时)")
    print(f"优化策略 (跳过双负):    {total_optimized:>8.1f} 元  (n={len(test_optimized)} 小时)")
    print(f"跳过时段数:             {skipped_hours_count} 小时")
    print(f"收益提升:               {improvement:>8.1f} 元  ({improvement/total_original*100:+.1f}%)")

    # 保存结果
    result = {
        "skip_hours": sorted(skip_hours),
        "original_benefit": float(total_original),
        "optimized_benefit": float(total_optimized),
        "improvement": float(improvement),
        "skipped_count": int(skipped_hours_count),
    }
    with open(os.path.join(OUT_DIR, "optimization_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"双负时段: {sorted(skip_hours)}\n")
        f.write(f"原策略收益: {total_original:.1f} 元\n")
        f.write(f"优化收益: {total_optimized:.1f} 元\n")
        f.write(f"提升: {improvement:.1f} 元 ({improvement/total_original*100:+.1f}%)\n")

    print(f"\n完成。输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
