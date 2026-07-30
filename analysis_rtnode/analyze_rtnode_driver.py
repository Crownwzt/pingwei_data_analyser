"""
实时节点电价驱动因素分析（只读原始数据，产物写入 analysis_rtnode/）

目标：把"实时节点电价"当作实时价格波动的桥梁变量，反推是哪些客观物理量
      驱动了实时价格与日前价格的分歧，从而定位可获取的增量特征。

分析方法：
  1) 基线相关性：实时节点电价 vs 所有其他字段（Pearson + Spearman）
  2) 偏差建模：定义 Δrt = 实时节点电价 - 日前节点电价（这才是"实时相对日前的偏差"）
     然后看 Δrt 与所有"实际 - 日前"偏差变量（新能源偏差、负荷偏差、竞价空间偏差…）的相关性
     —— 这一步是核心：能告诉我们"哪些物理量偏离预期，会让实时价格偏离日前价格"
  3) 分位数条件相关：只在实时价异常波动样本（|Δrt| > q90）上重新算相关性
  4) 分时段相关：按峰/平/谷、月份、小时切片，看驱动因素是否季节/时段依赖
  5) XGBoost 特征重要性交叉验证：用所有"实际"字段预测 Δrt，看谁贡献最大

产物：
  analysis_rtnode/
    ├── analyze_rtnode_driver.py       本脚本
    ├── report.html                    分析报告（含图表 base64）
    ├── correlations.csv               相关性明细表
    ├── residual_feature_importance.csv XGB 特征重要性
    └── plots/                          图表 PNG
"""
from __future__ import annotations

import base64
import io
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PKL = os.path.join(os.path.dirname(ROOT), "outputs", "cleaned_data.pkl")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- 字体（沿用项目风格，中文不乱码） ----------
def setup_font():
    for name in ["Noto Sans CJK SC", "Noto Sans CJK JP",
                 "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------- 数据准备 ----------
RT_NODE = "实时节点电价(元/MWh)"
DA_NODE = "日前节点电价(元/MWh)"
RT_UNIFIED = "实时统一结算点电价(元/MWh)"

PAIRED_ACTUAL = [
    "省调负荷(MW)", "新能源负荷(MW)", "水电负荷(MW)",
    "光伏负荷(MW)", "风电负荷(MW)", "非市场化出力(MW)",
    "竞价空间(MW)",
]


def load_data() -> pd.DataFrame:
    df = pd.read_pickle(DATA_PKL)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造 (实际 - 日前) 偏差列。这些偏差量在预测时刻并不可提前知晓，
    但它们是"实时价偏离日前价"的物理成因——我们要找的正是它们。
    """
    out = df.copy()
    for name in PAIRED_ACTUAL:
        col_da = f"{name}_日前"
        col_rt = f"{name}_实际"
        if col_da in out.columns and col_rt in out.columns:
            out[f"Δ{name}"] = out[col_rt] - out[col_da]
    # 关键：实时节点电价 - 日前节点电价 = 实时价的"意外分量"
    out["Δ实时节点电价"] = out[RT_NODE] - out[DA_NODE]
    # 相对偏差（更直观）
    out["Δ新能源占比"] = (
        (out["新能源负荷(MW)_实际"] - out["新能源负荷(MW)_日前"])
        / out["省调负荷(MW)_实际"].replace(0, np.nan)
    )
    out["Δ负荷率"] = (
        (out["省调负荷(MW)_实际"] - out["省调负荷(MW)_日前"])
        / out["省调负荷(MW)_日前"].replace(0, np.nan)
    )
    return out


# ---------- 相关性分析 ----------
def numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in exclude]


def correlation_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    exclude = ["年", "月", "日", "小时", "星期", "是否周末"]
    # (1) 实时节点电价的裸相关性
    base_cols = numeric_features(df, exclude + [RT_NODE])
    corr_rt = pd.DataFrame({
        "pearson": [df[RT_NODE].corr(df[c], method="pearson") for c in base_cols],
        "spearman": [df[RT_NODE].corr(df[c], method="spearman") for c in base_cols],
    }, index=base_cols)
    corr_rt["abs_p"] = corr_rt["pearson"].abs()
    corr_rt = corr_rt.sort_values("abs_p", ascending=False).drop(columns="abs_p")

    # (2) 实时相对日前的偏差 Δrt 与各 Δ 变量的相关性
    delta_cols = [c for c in df.columns if c.startswith("Δ") and c != "Δ实时节点电价"]
    corr_delta = pd.DataFrame({
        "pearson": [df["Δ实时节点电价"].corr(df[c], method="pearson") for c in delta_cols],
        "spearman": [df["Δ实时节点电价"].corr(df[c], method="spearman") for c in delta_cols],
    }, index=delta_cols)
    corr_delta["abs_p"] = corr_delta["pearson"].abs()
    corr_delta = corr_delta.sort_values("abs_p", ascending=False).drop(columns="abs_p")

    # (3) 异常波动样本（|Δrt| ≥ q90）上的相关性
    thr = df["Δ实时节点电价"].abs().quantile(0.90)
    extreme = df[df["Δ实时节点电价"].abs() >= thr]
    corr_extreme = pd.DataFrame({
        "pearson_extreme": [extreme["Δ实时节点电价"].corr(extreme[c]) for c in delta_cols],
        "n_samples": len(extreme),
    }, index=delta_cols)
    corr_extreme["abs_p"] = corr_extreme["pearson_extreme"].abs()
    corr_extreme = corr_extreme.sort_values("abs_p", ascending=False).drop(columns="abs_p")

    # (4) 按时段（峰/平/谷）
    seg_corr = {}
    for seg in df["时段"].dropna().unique():
        sub = df[df["时段"] == seg]
        seg_corr[seg] = pd.Series(
            [sub["Δ实时节点电价"].corr(sub[c]) for c in delta_cols],
            index=delta_cols, name=seg,
        )
    seg_df = pd.DataFrame(seg_corr)

    return {
        "corr_rt": corr_rt,
        "corr_delta": corr_delta,
        "corr_extreme": corr_extreme,
        "corr_by_segment": seg_df,
        "extreme_threshold": thr,
        "n_extreme": len(extreme),
    }


# ---------- XGBoost 特征重要性（残差建模） ----------
def xgb_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    用所有"实际"字段（包括 Δ 偏差）预测 Δ实时节点电价。
    这不是要做预测——是要看"哪些实际物理量最能解释实时价偏离日前价"。
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
    except ImportError:
        return pd.DataFrame()

    delta_cols = [c for c in df.columns if c.startswith("Δ") and c != "Δ实时节点电价"]
    actual_cols = [c for c in df.columns if c.endswith("_实际")]
    time_cols = ["小时", "星期", "是否周末", "月"]
    feats = delta_cols + actual_cols + time_cols
    feats = [c for c in feats if c in df.columns]

    data = df[feats + ["Δ实时节点电价"]].dropna()
    X, y = data[feats], data["Δ实时节点电价"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    imp = pd.DataFrame({
        "feature": feats,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    r2_tr = model.score(X_tr, y_tr)
    r2_te = model.score(X_te, y_te)
    imp.attrs["r2_train"] = r2_tr
    imp.attrs["r2_test"] = r2_te
    return imp


# ---------- 绘图 ----------
def _save_fig(name: str) -> str:
    p = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    return p


def plot_corr_rt(corr_rt: pd.DataFrame) -> str:
    top = corr_rt.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#d9534f" if v > 0 else "#5bc0de" for v in top["pearson"]]
    ax.barh(range(len(top)), top["pearson"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title("实时节点电价 与 各字段 Pearson 相关性 (Top 20)")
    return _save_fig("corr_rtnode_raw.png")


def plot_corr_delta(corr_delta: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    d = corr_delta.iloc[::-1]
    colors = ["#d9534f" if v > 0 else "#5bc0de" for v in d["pearson"]]
    ax.barh(range(len(d)), d["pearson"], color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title("Δ实时节点电价 与 各 Δ 变量（实际 - 日前）相关性")
    return _save_fig("corr_delta.png")


def plot_seg_corr(seg_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    seg_df.plot(kind="barh", ax=ax, width=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r（分时段）")
    ax.set_title("Δ实时节点电价 分时段相关性（峰/平/谷）")
    ax.legend(title="时段", fontsize=9)
    return _save_fig("corr_by_segment.png")


def plot_feat_imp(imp: pd.DataFrame) -> str:
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top)), top["gain"], color="#337ab7")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("XGB gain importance")
    ax.set_title(
        f"XGB 特征重要性：预测 Δ实时节点电价"
        f"（R²_train={imp.attrs.get('r2_train', 0):.3f}, "
        f"R²_test={imp.attrs.get('r2_test', 0):.3f}）"
    )
    return _save_fig("xgb_feat_importance.png")


def plot_extreme_scatter(df: pd.DataFrame, top_features: List[str]) -> str:
    fig, axes = plt.subplots(1, min(3, len(top_features)),
                             figsize=(5 * min(3, len(top_features)), 4))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_features[:3]):
        sub = df[[feat, "Δ实时节点电价"]].dropna().sample(
            min(5000, len(df)), random_state=42)
        ax.scatter(sub[feat], sub["Δ实时节点电价"], s=4, alpha=0.3, color="#5cb85c")
        r = sub[feat].corr(sub["Δ实时节点电价"])
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Δ实时节点电价")
        ax.set_title(f"{feat}\nr={r:.3f}", fontsize=10)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
    return _save_fig("delta_scatter.png")


# ---------- HTML 报告 ----------
def img_b64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_html(results: Dict, plots: Dict, imp: pd.DataFrame) -> str:
    corr_rt = results["corr_rt"]
    corr_delta = results["corr_delta"]
    corr_extreme = results["corr_extreme"]
    seg_df = results["corr_by_segment"]

    def tbl(df: pd.DataFrame, top_n: int = 15) -> str:
        return df.head(top_n).to_html(
            float_format=lambda x: f"{x:.3f}", classes="tbl")

    def img(name: str) -> str:
        b64 = img_b64(plots.get(name, ""))
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'

    top_delta_features = corr_delta.head(3).index.tolist()

    html_head = """<!doctype html>
<html><head><meta charset="utf-8">
<title>实时节点电价驱动因素分析</title>
<style>
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
        max-width: 1100px; margin: 24px auto; padding: 0 20px; color: #222; }
h1 { border-bottom: 3px solid #337ab7; padding-bottom: 8px; }
h2 { color: #337ab7; margin-top: 32px; }
.insight { background: #f6f8fa; border-left: 4px solid #337ab7;
            padding: 10px 16px; margin: 12px 0; }
.warn { background: #fff4e5; border-left: 4px solid #f0ad4e;
         padding: 10px 16px; margin: 12px 0; }
table.tbl { border-collapse: collapse; margin: 8px 0; }
table.tbl th, table.tbl td {
    border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }
table.tbl th { background: #f0f0f0; }
code { background: #f4f4f4; padding: 1px 6px; border-radius: 3px; }
</style></head><body>
"""
    body_parts = [html_head]
    body_parts.append("<h1>实时节点电价驱动因素分析</h1>")
    body_parts.append(
        "<div class='insight'>"
        "<p><strong>研究问题</strong>：<code>实时节点电价</code>与目标"
        "<code>实时统一结算点电价</code>高度相关但同时刻产生，属信息泄漏被排除。"
        "但它并非客观物理量，其波动必然由其他<em>实际观测到的物理量</em>驱动。"
        "本分析将<code>实时节点电价</code>作为桥梁，反向定位客观物理特征。</p>"
        "</div>"
    )

    body_parts.append("<h2>1. 实时节点电价 与全部字段的相关性</h2>")
    body_parts.append("<p>这是<strong>裸相关</strong>，会同时反映共同趋势和真正因果链。</p>")
    body_parts.append(img("corr_rtnode_raw"))
    body_parts.append(f"<details><summary>相关性 Top 15 表格</summary>{tbl(corr_rt)}</details>")

    body_parts.append("<h2>2. 核心：Δ实时节点电价 与 Δ物理量的相关性</h2>")
    body_parts.append(
        "<p>定义 <code>Δrt = 实时节点电价 - 日前节点电价</code>，即实时相对日前的意外分量。"
        "再看它与各 <code>Δx = x_实际 - x_日前</code> 的相关性。"
        "这一步剔除了共同趋势的干扰，得到<strong>真正驱动实时价偏离日前价的物理因素</strong>。</p>"
    )
    body_parts.append(img("corr_delta"))
    body_parts.append(
        f"<div class='warn'><p><strong>Top 3 驱动因子</strong>："
        f"{', '.join(top_delta_features)}</p></div>"
    )
    body_parts.append(tbl(corr_delta, top_n=20))

    body_parts.append("<h2>3. 异常波动样本下的驱动因素</h2>")
    body_parts.append(
        f"<p>筛选 |Δrt| ≥ q90 (阈值={results['extreme_threshold']:.1f} 元/MWh，"
        f"共 {results['n_extreme']} 样本) 的极端波动时段。</p>"
    )
    body_parts.append(tbl(corr_extreme, top_n=15))

    body_parts.append("<h2>4. 分时段相关性（峰/平/谷）</h2>")
    body_parts.append(img("corr_by_segment"))

    body_parts.append("<h2>5. XGB 特征重要性交叉验证</h2>")
    body_parts.append(img("xgb_feat_importance"))
    if not imp.empty:
        body_parts.append(tbl(imp, top_n=20))
    else:
        body_parts.append("<p>xgboost 未安装，跳过</p>")

    body_parts.append("<h2>6. Top 驱动因子散点</h2>")
    body_parts.append(img("delta_scatter"))

    body_parts.append("<h2>7. 结论与外部数据获取建议</h2>")
    conclusion = "<div class='insight'><p>实时价格异常波动的主要驱动因素（按 |r| 排序）：</p><ol>"
    for f in top_delta_features:
        r = corr_delta.loc[f, "pearson"]
        conclusion += f"<li><code>{f}</code>（r={r:.3f}）</li>"
    conclusion += (
        "</ol>"
        "<p>这些偏差量在预测时刻不可提前知晓，但可通过以下方式利用：</p>"
        "<ul>"
        "<li><strong>历史波动性代理</strong>：过去 N 日同时段偏差的滚动均值/标准差</li>"
        "<li><strong>预报质量特征</strong>：多源预报的分歧度作为不确定性代理</li>"
        "<li><strong>短期滚动预测</strong>：15min~4h 提前量场景可直接输入最新实际偏差</li>"
        "</ul></div>"
    )
    body_parts.append(conclusion)
    body_parts.append(
        "<div class='warn'><p><strong>本分析范围声明</strong>：仅使用项目内 "
        "<code>cleaned_data.pkl</code>。如需定位到机组/联络线级别，需接入调度侧数据。</p></div>"
    )
    body_parts.append("</body></html>")
    return "\n".join(body_parts)


# ---------- 主流程 ----------
def main():
    setup_font()
    print(f"[1/6] 加载数据: {DATA_PKL}")
    df = load_data()
    print(f"      shape={df.shape}")

    print("[2/6] 构造 Δ 偏差列")
    df = build_deviation_features(df)

    print("[3/6] 相关性分析")
    results = correlation_analysis(df)

    print("[4/6] XGB 特征重要性")
    imp = xgb_feature_importance(df)
    if not imp.empty:
        imp.to_csv(os.path.join(ROOT, "residual_feature_importance.csv"),
                   index=False, encoding="utf-8-sig")

    print("[5/6] 绘图")
    plots = {
        "corr_rtnode_raw": plot_corr_rt(results["corr_rt"]),
        "corr_delta": plot_corr_delta(results["corr_delta"]),
        "corr_by_segment": plot_seg_corr(results["corr_by_segment"]),
    }
    if not imp.empty:
        plots["xgb_feat_importance"] = plot_feat_imp(imp)
    top3 = results["corr_delta"].head(3).index.tolist()
    plots["delta_scatter"] = plot_extreme_scatter(df, top3)

    print("[6/6] 生成 HTML 报告")
    html = render_html(results, plots, imp)
    report_path = os.path.join(ROOT, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    results["corr_delta"].to_csv(
        os.path.join(ROOT, "correlations.csv"), encoding="utf-8-sig")

    print(f"\n完成。报告：{report_path}")
    print(f"极端阈值: |Δrt| ≥ {results['extreme_threshold']:.2f} 元/MWh")
    print(f"极端样本数: {results['n_extreme']}")
    print("\n=== Δ实时节点电价 相关性 Top 10 ===")
    print(results["corr_delta"].head(10).to_string())
    if not imp.empty:
        print(f"\n=== XGB R²_test={imp.attrs['r2_test']:.3f} ===")
        print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
