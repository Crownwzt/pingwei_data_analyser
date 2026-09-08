#!/usr/bin/env python3
"""
用验证集收益给 24 个小时排序，然后看"只交易 Top-K 个验证集优势小时"时
测试集的收益表现，帮助选出整体收益率最高的时段子集。

指标（均在测试集上评估）：
  - 每小时平均收益 = 选中时段的总收益 / 选中的小时样本数（衡量效率）
  - 累计总收益     = 选中时段的总收益（衡量规模）

收益定义：benefit = |日前-真实| - |XGB-真实|，正=XGB更优
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_font():
    # macOS 的 PingFang.ttc 首个字面缺少部分简体字形（如"杀""远"），优先级下调，
    # 改由 Hiragino Sans GB / Arial Unicode MS 兜底，避免图中出现方框
    for name in ["Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", "SimHei", "STHeiti", "PingFang HK"]:
        try:
            f = font_manager.findfont(name, fallback_to_default=False)
            if f and (f.lower().endswith(".ttf") or f.lower().endswith(".otf") or ".ttc" in f.lower()):
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def load_split(pkl_path: str, split: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data[f"dt_{split}"]),
        "y_true": data[f"y_{split}"],
        "y_pred": data["metrics"][split]["y_pred"],
        "da": data[f"da_{split}"],
    })
    df["hour"] = df["datetime"].dt.hour
    df["da_err"] = (df["da"] - df["y_true"]).abs()
    df["xgb_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["benefit"] = df["da_err"] - df["xgb_err"]
    return df


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT = os.path.dirname(ROOT)
    METRICS_PKL = os.path.join(PROJECT, "outputs", "metrics.pkl")
    OUT_DIR = os.path.join(ROOT, "topk_hour_selection")
    os.makedirs(OUT_DIR, exist_ok=True)

    font = setup_font()
    print(f"[1/4] 中文字体: {font or '未找到'}")

    print(f"[2/4] 加载 metrics: {METRICS_PKL}")
    val = load_split(METRICS_PKL, "val")
    test = load_split(METRICS_PKL, "test")
    print(f"      val n={len(val)}, test n={len(test)}")

    # 用全验证集排序，找出最优 K（斩杀线 = 验证集全时段平均收益）
    val_mean_all = val["benefit"].mean()
    test_mean_all = test["benefit"].mean()
    print(f"[3/4] 全时段基准: val={val_mean_all:.3f}元/h, test={test_mean_all:.3f}元/h")

    # 验证集按小时平均收益排序（从高到低）
    val_rank = val.groupby("hour")["benefit"].mean().sort_values(ascending=False)
    hours_sorted = val_rank.index.tolist()

    # 找最优 K：从高到低遍历，第一个 < val_mean_all 的位置
    optimal_k = 24  # 默认全选
    for i, h in enumerate(hours_sorted):
        if val_rank[h] < val_mean_all:
            optimal_k = i  # 在这个位置停止（不包含当前这个）
            break

    print(f"      验证集斩杀线: {val_mean_all:.3f}元/h")
    print(f"      最优 K = {optimal_k} (小时: {hours_sorted[:optimal_k]})")

    # 逐步加入 Top-K 小时，在 val/test 上分别评估
    rows = []
    for k in range(1, 25):
        topk_hours = set(hours_sorted[:k])
        sub_val = val[val["hour"].isin(topk_hours)]
        sub_test = test[test["hour"].isin(topk_hours)]
        rows.append({
            "k": k,
            "hours": sorted(topk_hours),
            "val_hour_benefit": val_rank.iloc[k - 1],  # 第 k 个小时的收益
            "val_mean_benefit": sub_val["benefit"].mean(),
            "test_mean_benefit": sub_test["benefit"].mean(),
            "test_total_benefit": sub_test["benefit"].sum(),
            "n_samples": len(sub_test),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "topk_selection.csv"), index=False, encoding="utf-8-sig")

    # 用验证集选出的 K 在测试集上验证
    test_result = res[res["k"] == optimal_k].iloc[0]
    print(f"      测试集验证 (K={optimal_k}): 每小时={test_result['test_mean_benefit']:.3f}元, 累计={test_result['test_total_benefit']:.1f}元")
    print(f"      对比全时段 test 基准 {test_mean_all:.3f}元, 提升 {(test_result['test_mean_benefit']/test_mean_all-1)*100:.1f}%")

    # 测试集按小时平均收益（用于左下对比图）
    test_rank = test.groupby("hour")["benefit"].mean()

    # 测试集每小时统计：收益成因分析（日前误差、价格波动、实时-日前价差）
    test_hourly_stats = test.groupby("hour").agg(
        y_mean=("y_true", "mean"),
        y_std=("y_true", "std"),
        da_err=("da_err", "mean"),
        xgb_err=("xgb_err", "mean"),
        benefit=("benefit", "mean"),
    ).reset_index()

    print("[4/4] 绘图")
    plot_topk_curves(res, optimal_k, val_mean_all, test_mean_all,
                     val_rank, test_rank, hours_sorted, test_hourly_stats, OUT_DIR)
    print(f"完成。输出目录: {OUT_DIR}")
    return res


def plot_topk_curves(res, optimal_k, val_mean_all, test_mean_all,
                     val_rank, test_rank, hours_sorted, test_hourly_stats, out_dir):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])  # 底部跨两列

    k = res["k"]

    # 左上：验证集 vs 测试集 Top-K 曲线（斩杀线策略）
    ax1.plot(k, res["val_mean_benefit"], marker="o", lw=2, color="#5DA9E9",
             label="验证集(11月)")
    ax1.plot(k, res["test_mean_benefit"], marker="s", lw=2, color="#2E86AB",
             label="测试集(12月)")
    ax1.axhline(val_mean_all, color="#5DA9E9", ls="--", lw=1.5,
                label=f"验证集全时段基准 ({val_mean_all:.3f}元)")
    ax1.axhline(test_mean_all, color="gray", ls="--", lw=1.5,
                label=f"测试集全时段基准 ({test_mean_all:.3f}元)")

    # 标注验证集选出的最优 K（斩杀线法）
    ax1.axvline(optimal_k, color="red", ls=":", lw=2, alpha=0.6,
                label=f"斩杀线 K={optimal_k}")
    if optimal_k > 0:
        opt_val = res[res["k"] == optimal_k]["val_mean_benefit"].values[0]
        opt_test = res[res["k"] == optimal_k]["test_mean_benefit"].values[0]
        ax1.scatter([optimal_k], [opt_val], s=200, color="red", zorder=5, marker="*")
        ax1.scatter([optimal_k], [opt_test], s=200, color="red", zorder=5, marker="*")

    ax1.set_xlabel("Top-K 小时数 (按验证集收益排序)", fontsize=11)
    ax1.set_ylabel("每小时平均收益 (元)", fontsize=11)
    ax1.set_title("每小时平均收益 vs Top-K (斩杀线法选K)", fontsize=12, weight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # 右上：测试集累计总收益
    ax2.plot(k, res["test_total_benefit"], marker="s", lw=2, color="#A23B72", label="测试集累计总收益")
    ax2.axhline(test_mean_all * len(res), color="gray", ls="--", lw=1.5, alpha=0)  # 占位，不显示
    best_k_total = res.loc[res["test_total_benefit"].idxmax()]
    ax2.scatter([best_k_total["k"]], [best_k_total["test_total_benefit"]], s=150, color="red", zorder=5, label=f"最优 Top-{int(best_k_total['k'])}")
    ax2.set_xlabel("Top-K 小时数", fontsize=11)
    ax2.set_ylabel("测试集累计总收益 (元)", fontsize=11)
    ax2.set_title("测试集：累计总收益 vs Top-K", fontsize=12, weight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # 左下：验证集排序下，验证集 vs 测试集每小时收益对比
    val_sorted = [val_rank[h] for h in hours_sorted]
    test_sorted = [test_rank[h] for h in hours_sorted]
    x_labels = [f"{h}h" for h in hours_sorted]
    x_pos = np.arange(24)

    ax3.bar(x_pos - 0.2, val_sorted, width=0.4, label="验证集(11月)",
            color="#5DA9E9", alpha=0.85)
    ax3.bar(x_pos + 0.2, test_sorted, width=0.4, label="测试集(12月)",
            color="#2E86AB", alpha=0.85)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axhline(val_mean_all, color="#5DA9E9", ls="--", lw=1.5, alpha=0.5,
                label=f"验证集基准 {val_mean_all:.2f}")

    # 标注斩杀线位置
    if optimal_k < 24:
        ax3.axvspan(optimal_k - 0.5, 24, color="red", alpha=0.1,
                    label=f"K>{optimal_k} 被斩杀")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax3.set_xlabel("小时 (按验证集收益从高到低排列)", fontsize=11)
    ax3.set_ylabel("小时平均收益 (元)", fontsize=11)
    ax3.set_title("24小时收益：验证集排序下 验证 vs 测试对比", fontsize=12, weight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, axis="y")

    # 右下：参与样本数 vs Top-K
    ax4.plot(k, res["n_samples"], marker="^", lw=2, color="#F18F01", label="测试集参与样本数")
    ax4.set_xlabel("Top-K 小时数", fontsize=11)
    ax4.set_ylabel("测试集参与小时数 (n)", fontsize=11)
    ax4.set_title("测试集：参与样本数 vs Top-K", fontsize=12, weight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 底部跨列：收益成因分析（自然小时顺序 0-23）
    #   左轴 = 日前误差(柱) + XGB误差(柱)  右轴 = 实时价波动std(线) + benefit(线)
    #   一眼看出因果链：波动大 → 日前失准 → 模型纠错空间大 → 收益高
    s = test_hourly_stats.sort_values("hour")
    hx = s["hour"].to_numpy()
    ax5.bar(hx - 0.2, s["da_err"], width=0.4, label="日前误差 |日前-实时|",
            color="#F0AD4E", alpha=0.85)
    ax5.bar(hx + 0.2, s["xgb_err"], width=0.4, label="XGB误差 |预测-实时|",
            color="#5CB85C", alpha=0.85)
    ax5.set_xlabel("小时 (0-23)", fontsize=11)
    ax5.set_ylabel("平均绝对误差 (元)", fontsize=11)
    ax5.set_xticks(range(24))
    ax5.grid(alpha=0.3, axis="y")

    ax5b = ax5.twinx()
    ax5b.plot(hx, s["y_std"], marker="o", lw=2, color="#C0392B",
              label="实时价波动 std")
    ax5b.plot(hx, s["benefit"], marker="s", lw=2, color="#2E86AB",
              label="收益 benefit")
    ax5b.set_ylabel("实时价波动 std / 收益 (元)", fontsize=11)

    # 合并双轴图例
    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = ax5b.get_legend_handles_labels()
    ax5.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax5.set_title(
        "收益成因：白天光伏大发→实时价暴跌+波动放大→日前价失准→模型纠错空间大→收益集中在 9-15 点",
        fontsize=12, weight="bold")

    plt.savefig(os.path.join(out_dir, "topk_hour_selection.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      已保存: topk_hour_selection.png")


if __name__ == "__main__":
    main()
