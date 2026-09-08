#!/usr/bin/env python3
"""
时段选择策略的稳健性检验 —— 回答 topk_hour_selection.py 没有回答的三个问题：

  Q1 收益稳不稳？   K=7 的"每小时 7.43 元"是长期优势，还是被少数极端日拉高的假象
  Q2 排序能迁移吗？ 验证集(11月)的小时排序，在测试集(12月)上是否仍然成立
  Q3 复杂方法值吗？ 斩杀线法 vs 朴素固定窗(9-15h) vs 全时段，谁的性价比更高

收益定义与 topk_hour_selection.py 完全一致：
    benefit = |日前价 - 实时价| - |XGB预测 - 实时价|     正值 = XGB 优于日前价
单位口径：元/MWh（按每小时交易 1 MWh 计）；日度指标 = 该日选中时段的聚合值
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 复用 topk_hour_selection 的字体探测与数据加载，同类能力不重复实现
from topk_hour_selection import setup_font, load_split

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
METRICS_PKL = os.path.join(PROJECT, "outputs", "metrics.pkl")
CLEANED_PKL = os.path.join(PROJECT, "outputs", "cleaned_data.pkl")
OUT_DIR = os.path.join(ROOT, "topk_robustness")

# 待对比的候选时段方案：斩杀线法产出 / 业务经验固定窗 / 不筛选
STRATEGIES = {
    "斩杀线Top7": [8, 9, 10, 11, 13, 14, 15],
    "固定窗9-15h": list(range(9, 16)),
    "固定窗8-15h": list(range(8, 16)),
    "全时段24h": list(range(24)),
}
FOCUS = "斩杀线Top7"  # 报告推荐方案，作为风险剖面的主角


def add_date(df: pd.DataFrame) -> pd.DataFrame:
    """补日历日字段，日度风险指标需要按天聚合"""
    df = df.copy()
    df["date"] = df["datetime"].dt.normalize()
    return df


def eval_strategy(test: pd.DataFrame, hours: list) -> dict:
    """在测试集上评估一个时段方案：效率 / 规模 / 风险 三类指标一起给出"""
    sub = test[test["hour"].isin(set(hours))]
    daily_mean = sub.groupby("date")["benefit"].mean()
    daily_sum = sub.groupby("date")["benefit"].sum()
    return {
        "K": len(hours),
        "每小时均值": sub["benefit"].mean(),
        "每小时中位数": sub["benefit"].median(),
        "累计收益": sub["benefit"].sum(),
        "收益保留率%": sub["benefit"].sum() / test["benefit"].sum() * 100,
        "参与时长%": len(sub) / len(test) * 100,
        "逐时胜率%": (sub["benefit"] > 0).mean() * 100,
        "亏损日占比%": (daily_mean < 0).mean() * 100,
        "日均值std": daily_mean.std(),
        "最差日": daily_mean.min(),
        "Top3日贡献%": daily_sum.nlargest(3).sum() / daily_sum.sum() * 100,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple:
    """自助法 95% 置信区间。逐时收益分布极度右偏（少数极端日主导），
    均值的标准误不能用正态近似，必须重采样"""
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_transfer(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Q2：验证集小时排序能否迁移到测试集。
    Spearman 只看名次不看数值，是"排序是否复现"的正确度量"""
    vr = val.groupby("hour")["benefit"].mean()
    tr = test.groupby("hour")["benefit"].mean()
    rho, p_rho = stats.spearmanr(vr.values, tr.values)
    r, p_r = stats.pearsonr(vr.values, tr.values)
    val_top7 = set(vr.nlargest(7).index)
    test_top7 = set(tr.nlargest(7).index)
    return {
        "spearman": rho, "spearman_p": p_rho,
        "pearson": r, "pearson_p": p_r,
        "val_top7": sorted(val_top7),
        "test_top7": sorted(test_top7),
        "overlap": sorted(val_top7 & test_top7),
        "val_only": sorted(val_top7 - test_top7),
        "test_only": sorted(test_top7 - val_top7),
        "val_rank": vr, "test_rank": tr,
    }


def extreme_dependence(test: pd.DataFrame, hours: list) -> pd.DataFrame:
    """Q1：逐步剔除收益最高的 N 天，看均值塌陷多快。
    塌陷越快 → 优势越依赖运气，越不能当作稳定收益预期"""
    sub = test[test["hour"].isin(set(hours))]
    daily = sub.groupby("date")["benefit"].sum().sort_values(ascending=False)
    rows = []
    for n in range(0, 6):
        keep = sub[~sub["date"].isin(daily.index[:n])]
        rows.append({
            "剔除最佳日数": n,
            "剩余天数": keep["date"].nunique(),
            "每小时均值": keep["benefit"].mean(),
            "累计收益": keep["benefit"].sum(),
        })
    return pd.DataFrame(rows)


def pv_causal_check(test: pd.DataFrame) -> pd.DataFrame:
    """核实报告的因果链：光伏出力 → 净负荷下压 → 电价崩塌+波动放大 → 日前失准。
    光伏/负荷取 _实际 列：此处是事后归因分析，不参与建模，无泄漏问题"""
    df = pd.read_pickle(CLEANED_PKL)
    dt = pd.to_datetime(df["datetime"])
    price_col = next(c for c in df.columns if "实时统一结算点电价" in c)
    raw = pd.DataFrame({
        "hour": dt.dt.hour,
        "month": dt.dt.month,
        "光伏": df["光伏负荷(MW)_实际"].values,
        "省调负荷": df["省调负荷(MW)_实际"].values,
        "电价": df[price_col].values,
    })
    raw = raw[raw["month"] == 12]  # 与测试集对齐
    g = raw.groupby("hour").agg(
        光伏出力MW=("光伏", "mean"),
        省调负荷MW=("省调负荷", "mean"),
        实时价=("电价", "mean"),
        价格std=("电价", "std"),
    )
    g["净负荷MW"] = g["省调负荷MW"] - g["光伏出力MW"]  # 净负荷 = 总负荷 - 光伏，决定火电边际报价
    err = test.groupby("hour").agg(日前误差=("da_err", "mean"), 收益=("benefit", "mean"))
    return g.join(err)


def plot_robustness(test, cmp_df, transfer, decay, ci_map, out_dir):
    """4 张子图：日度收益分布 / 极端日依赖 / 排序迁移 / 方案对比"""
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # 左上：K=7 逐日收益柱状图 —— 直观展示"少数日撑起全部收益"
    sub = test[test["hour"].isin(set(STRATEGIES[FOCUS]))]
    daily = sub.groupby("date")["benefit"].mean().sort_index()
    colors = ["#5CB85C" if v > 0 else "#D9534F" for v in daily]
    ax1.bar(range(len(daily)), daily.values, color=colors, alpha=0.85)
    ax1.axhline(daily.mean(), color="#2E86AB", ls="--", lw=2,
                label=f"均值 {daily.mean():.2f}元")
    ax1.axhline(daily.median(), color="#F0AD4E", ls="--", lw=2,
                label=f"中位数 {daily.median():.2f}元")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(range(0, len(daily), 4))
    ax1.set_xticklabels([d.strftime("%m-%d") for d in daily.index[::4]], rotation=45, fontsize=8)
    ax1.set_xlabel("测试集日期 (2025-12)", fontsize=11)
    ax1.set_ylabel("当日选中时段平均收益 (元/MWh)", fontsize=11)
    ax1.set_title(f"日度收益分布：{(daily < 0).mean() * 100:.0f}% 的交易日亏损，"
                  f"均值远高于中位数", fontsize=12, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    # 右上：剔除最佳日后的均值塌陷曲线
    ax2.plot(decay["剔除最佳日数"], decay["每小时均值"], marker="o", lw=2.5, color="#A23B72")
    for _, r in decay.iterrows():
        ax2.annotate(f"{r['每小时均值']:.2f}", (r["剔除最佳日数"], r["每小时均值"]),
                     textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax2.axhline(test["benefit"].mean(), color="gray", ls="--", lw=1.5,
                label=f"全时段基准 {test['benefit'].mean():.2f}元")
    ax2.set_xlabel("剔除收益最高的前 N 个交易日", fontsize=11)
    ax2.set_ylabel("每小时平均收益 (元/MWh)", fontsize=11)
    ax2.set_title("极端日依赖度：剔除 3 天即跌破全时段基准", fontsize=12, weight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # 左下：验证集 vs 测试集小时收益散点（排序迁移）
    vr, tr = transfer["val_rank"], transfer["test_rank"]
    ax3.scatter(vr.values, tr.values, s=90, color="#2E86AB", alpha=0.75, zorder=3)
    for h in vr.index:
        ax3.annotate(f"{h}h", (vr[h], tr[h]), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.axvline(vr.mean(), color="#5DA9E9", ls=":", lw=1.5,
                label=f"验证集斩杀线 {vr.mean():.2f}元")
    ax3.set_xlabel("验证集(11月) 小时平均收益 (元/MWh)", fontsize=11)
    ax3.set_ylabel("测试集(12月) 小时平均收益 (元/MWh)", fontsize=11)
    ax3.set_title(f"排序迁移性：Spearman ρ={transfer['spearman']:.2f} "
                  f"(中等，非完全复现)", fontsize=12, weight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # 右下：方案对比 —— 效率(柱) vs 亏损日占比(线)
    names = list(cmp_df.index)
    x = np.arange(len(names))
    ax4.bar(x, cmp_df["每小时均值"], width=0.5, color="#5DA9E9", alpha=0.9,
            label="每小时均值 (元/MWh)")
    for i, v in enumerate(cmp_df["每小时均值"]):
        lo, hi = ci_map[names[i]]
        ax4.plot([i, i], [lo, hi], color="#C0392B", lw=2, zorder=4)
        ax4.annotate(f"{v:.2f}", (i, hi), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9, weight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylabel("每小时平均收益 (元/MWh)", fontsize=11)
    ax4.set_title("方案对比：斩杀线Top7 与固定窗9-15h 收益几乎相同\n"
                  "(红线=Bootstrap 95%置信区间，彼此大幅重叠)", fontsize=12, weight="bold")
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(alpha=0.3, axis="y")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "topk_robustness.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    font = setup_font()
    print(f"[1/6] 中文字体: {font or '未找到'}")

    if not os.path.exists(METRICS_PKL):
        raise FileNotFoundError(f"缺少 {METRICS_PKL}，请先运行 bash run.sh --module training")
    print(f"[2/6] 加载 {METRICS_PKL}")
    val = add_date(load_split(METRICS_PKL, "val"))
    test = add_date(load_split(METRICS_PKL, "test"))
    print(f"      val n={len(val)} ({val['date'].nunique()}天), "
          f"test n={len(test)} ({test['date'].nunique()}天)")

    print("[3/6] Q1 收益稳健性 + 方案对比")
    cmp_df = pd.DataFrame({k: eval_strategy(test, v) for k, v in STRATEGIES.items()}).T
    ci_map = {}
    for name, hours in STRATEGIES.items():
        vals = test[test["hour"].isin(set(hours))]["benefit"].values
        ci_map[name] = bootstrap_ci(vals)
    cmp_df["CI下界"] = [ci_map[n][0] for n in cmp_df.index]
    cmp_df["CI上界"] = [ci_map[n][1] for n in cmp_df.index]
    print(cmp_df.round(2).to_string())

    decay = extreme_dependence(test, STRATEGIES[FOCUS])
    print("\n      极端日依赖:")
    print(decay.round(2).to_string(index=False))

    print("\n[4/6] Q2 排序迁移性")
    transfer = rank_transfer(val, test)
    print(f"      Spearman ρ={transfer['spearman']:.3f} (p={transfer['spearman_p']:.4f})  "
          f"Pearson r={transfer['pearson']:.3f}")
    print(f"      验证集Top7={transfer['val_top7']}  测试集Top7={transfer['test_top7']}")
    print(f"      重叠={transfer['overlap']}  仅验证集={transfer['val_only']}  "
          f"仅测试集={transfer['test_only']}")

    print("\n[5/6] 因果链核实（光伏 → 净负荷 → 电价 → 日前误差）")
    try:
        pv = pv_causal_check(test)
        print(pv.round(1).to_string())
        print(f"      corr(光伏出力, 实时价) = {pv['光伏出力MW'].corr(pv['实时价']):.3f}")
        print(f"      corr(光伏出力, 价格std) = {pv['光伏出力MW'].corr(pv['价格std']):.3f}")
        print(f"      corr(日前误差, 收益) = {pv['日前误差'].corr(pv['收益']):.3f}")
        pv.round(3).to_csv(os.path.join(OUT_DIR, "pv_causal_check.csv"), encoding="utf-8-sig")
    except (FileNotFoundError, KeyError, StopIteration) as e:
        print(f"      跳过（清洗缓存不可用或列缺失）: {e}")

    print("\n[6/6] 绘图")
    cmp_df.round(3).to_csv(os.path.join(OUT_DIR, "strategy_comparison.csv"), encoding="utf-8-sig")
    decay.round(3).to_csv(os.path.join(OUT_DIR, "extreme_dependence.csv"),
                          index=False, encoding="utf-8-sig")
    path = plot_robustness(test, cmp_df, transfer, decay, ci_map, OUT_DIR)
    print(f"      已保存: {os.path.basename(path)}")
    print(f"完成。输出目录: {OUT_DIR}")
    return cmp_df, transfer, decay


if __name__ == "__main__":
    main()
