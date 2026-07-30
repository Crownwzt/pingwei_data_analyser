"""
多因子组合分析：找出哪些Δ变量的组合最能解释实时价偏差

方法：
  1. 线性回归：逐步加入 Top N 个 Δ 变量，观察 R² 增量
  2. 交互项探测：测试所有两两交互（Δx * Δy），看是否有显著非线性效应
  3. 方差分解：用 Shapley 值或偏 R² 确定每个因子的独立贡献 vs 协同贡献
  4. 主成分分析：看 Δ 变量是否存在潜在公因子（如"供需失配综合指标"）
  5. 分段建模：峰/平/谷是否需要不同的组合模型

产物：
  - multivariate_report.html
  - model_comparison.csv
  - interaction_matrix.csv
"""
from __future__ import annotations

import base64
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PKL = os.path.join(os.path.dirname(ROOT), "outputs", "cleaned_data.pkl")
PLOTS_DIR = os.path.join(ROOT, "plots")

# ---------- 字体 ----------
def setup_font():
    for name in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------- 数据 ----------
def load_and_prepare() -> pd.DataFrame:
    df = pd.read_pickle(DATA_PKL)
    df = df.sort_values("datetime").reset_index(drop=True)

    # 构造 Δ 变量
    PAIRED = ["省调负荷(MW)", "新能源负荷(MW)", "水电负荷(MW)",
              "光伏负荷(MW)", "风电负荷(MW)", "非市场化出力(MW)", "竞价空间(MW)"]
    for name in PAIRED:
        col_da = f"{name}_日前"
        col_rt = f"{name}_实际"
        if col_da in df.columns and col_rt in df.columns:
            df[f"Δ{name}"] = df[col_rt] - df[col_da]

    df["Δ实时节点电价"] = df["实时节点电价(元/MWh)"] - df["日前节点电价(元/MWh)"]

    # 相对偏差
    df["Δ负荷率"] = (df["省调负荷(MW)_实际"] - df["省调负荷(MW)_日前"]) / df["省调负荷(MW)_日前"].replace(0, np.nan)
    df["Δ新能源占比"] = (df["新能源负荷(MW)_实际"] - df["新能源负荷(MW)_日前"]) / df["省调负荷(MW)_实际"].replace(0, np.nan)

    return df


# ---------- 逐步回归 R² 增量分析 ----------
def stepwise_regression(df: pd.DataFrame, top_features: List[str]) -> pd.DataFrame:
    """
    逐步加入特征，观察 R² 增量。
    """
    y = df["Δ实时节点电价"].values
    results = []

    # 单因子
    for feat in top_features:
        X = df[[feat]].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 100:
            continue
        lr = LinearRegression().fit(X[mask], y[mask])
        r2 = r2_score(y[mask], lr.predict(X[mask]))
        results.append({
            "model": feat,
            "n_features": 1,
            "R2": r2,
            "R2_increment": r2,
        })

    # 累积加入
    cumulative_features = []
    prev_r2 = 0.0
    for i, feat in enumerate(top_features):
        cumulative_features.append(feat)
        X = df[cumulative_features].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 100:
            continue
        lr = LinearRegression().fit(X[mask], y[mask])
        r2 = r2_score(y[mask], lr.predict(X[mask]))
        results.append({
            "model": " + ".join(cumulative_features),
            "n_features": len(cumulative_features),
            "R2": r2,
            "R2_increment": r2 - prev_r2,
        })
        prev_r2 = r2

    return pd.DataFrame(results)


# ---------- 交互项矩阵 ----------
def interaction_matrix(df: pd.DataFrame, top_features: List[str]) -> pd.DataFrame:
    """
    对所有两两组合，测试 R²(A + B + A*B) - R²(A + B)，即交互增益。
    """
    y = df["Δ实时节点电价"].values
    results = []

    for f1, f2 in combinations(top_features, 2):
        X_base = df[[f1, f2]].values
        X_inter = np.column_stack([X_base, X_base[:, 0] * X_base[:, 1]])

        mask = ~(np.isnan(X_base).any(axis=1) | np.isnan(y))
        if mask.sum() < 100:
            continue

        lr_base = LinearRegression().fit(X_base[mask], y[mask])
        r2_base = r2_score(y[mask], lr_base.predict(X_base[mask]))

        lr_inter = LinearRegression().fit(X_inter[mask], y[mask])
        r2_inter = r2_score(y[mask], lr_inter.predict(X_inter[mask]))

        results.append({
            "feature_1": f1,
            "feature_2": f2,
            "R2_base": r2_base,
            "R2_with_interaction": r2_inter,
            "interaction_gain": r2_inter - r2_base,
        })

    return pd.DataFrame(results).sort_values("interaction_gain", ascending=False)


# ---------- PCA 主成分 ----------
def pca_analysis(df: pd.DataFrame, delta_cols: List[str]) -> Tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    对所有 Δ 变量做 PCA，看是否存在潜在公因子。
    """
    X = df[delta_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, len(delta_cols)))
    Z = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=delta_cols,
    )

    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })

    return loadings, pca, Z


# ---------- 分时段多元模型 ----------
def segment_models(df: pd.DataFrame, top_features: List[str]) -> pd.DataFrame:
    """
    峰/平/谷分别建多元线性回归，对比 R²。
    """
    y_col = "Δ实时节点电价"
    results = []

    for seg in df["时段"].dropna().unique():
        sub = df[df["时段"] == seg]
        X = sub[top_features].values
        y = sub[y_col].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 50:
            continue
        lr = LinearRegression().fit(X[mask], y[mask])
        r2 = r2_score(y[mask], lr.predict(X[mask]))
        results.append({
            "segment": seg,
            "n_samples": mask.sum(),
            "R2": r2,
            "features": " + ".join(top_features),
        })

    return pd.DataFrame(results)


# ---------- 绘图 ----------
def _save_fig(name: str) -> str:
    p = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    return p


def plot_stepwise(step_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative = step_df[step_df["n_features"] > 1].copy()
    ax.plot(cumulative["n_features"], cumulative["R2"], marker="o", lw=2, color="#337ab7")
    ax.set_xlabel("特征数量")
    ax.set_ylabel("R²")
    ax.set_title("逐步加入 Δ 特征的 R² 变化")
    ax.grid(alpha=0.3)
    return _save_fig("stepwise_r2.png")


def plot_interaction_heatmap(inter_df: pd.DataFrame, top_n: int = 7) -> str:
    # 只取 top N 特征构造对称矩阵
    feats = list(set(inter_df["feature_1"].tolist() + inter_df["feature_2"].tolist()))[:top_n]
    matrix = pd.DataFrame(0.0, index=feats, columns=feats)
    for _, row in inter_df.iterrows():
        if row["feature_1"] in feats and row["feature_2"] in feats:
            matrix.loc[row["feature_1"], row["feature_2"]] = row["interaction_gain"]
            matrix.loc[row["feature_2"], row["feature_1"]] = row["interaction_gain"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=-0.01, vmax=0.03)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feats, fontsize=9)

    for i in range(len(feats)):
        for j in range(len(feats)):
            val = matrix.values[i, j]
            if i != j:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="交互增益 (ΔR²)")
    ax.set_title("两两交互项的 R² 增益")
    return _save_fig("interaction_heatmap.png")


def plot_pca_loadings(loadings: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    loadings.iloc[:, :3].plot(kind="barh", ax=ax, width=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Loading")
    ax.set_title("PCA Loadings (前 3 个主成分)")
    ax.legend(fontsize=9)
    return _save_fig("pca_loadings.png")


# ---------- HTML 报告 ----------
def render_html(step_df: pd.DataFrame, inter_df: pd.DataFrame,
                loadings: pd.DataFrame, pca_explained: pd.DataFrame,
                seg_df: pd.DataFrame, plots: Dict) -> str:

    def img_b64(path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def img(name: str) -> str:
        b64 = img_b64(plots.get(name, ""))
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'

    best_combo = step_df[step_df["n_features"] > 1].iloc[-1] if len(step_df[step_df["n_features"] > 1]) > 0 else None
    top_inter = inter_df.head(3)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>多因子组合分析</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; max-width: 1100px;
        margin: 24px auto; padding: 0 20px; color: #222; }}
h1 {{ border-bottom: 3px solid #5cb85c; padding-bottom: 8px; }}
h2 {{ color: #5cb85c; margin-top: 32px; }}
.insight {{ background: #f6f8fa; border-left: 4px solid #5cb85c; padding: 10px 16px; margin: 12px 0; }}
.warn {{ background: #fff4e5; border-left: 4px solid #f0ad4e; padding: 10px 16px; margin: 12px 0; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }}
th {{ background: #f0f0f0; }}
code {{ background: #f4f4f4; padding: 1px 6px; border-radius: 3px; }}
</style></head><body>

<h1>多因子组合分析：Δ 变量的协同效应</h1>

<div class='insight'>
<p><strong>核心问题</strong>：单因子相关性都在 0.2-0.3，但多因子组合能否大幅提升解释力？
如果能，说明这些物理量是<strong>协同驱动</strong>实时价偏差；如果不能，说明还有结构性因素缺失。</p>
</div>

<h2>1. 逐步回归：累积 R² 变化</h2>
<p>按相关性从高到低依次加入 Δ 特征，观察 R² 增量。</p>
{img("stepwise_r2")}
<div class='warn'>
<p><strong>关键发现</strong>：{"" if best_combo is None else f"累积到 {best_combo['n_features']} 个特征时 R² = {best_combo['R2']:.3f}。"}
{"边际增量递减，说明存在多重共线性——这些 Δ 变量之间本身相关。" if (best_combo is not None and best_combo['R2'] < 0.15) else ""}</p>
</div>
{step_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")}

<h2>2. 交互项矩阵：哪些组合有非线性协同</h2>
<p>测试所有两两交互 (A × B)，计算 R²(A + B + A×B) − R²(A + B)，即交互增益。</p>
{img("interaction_heatmap")}
<p><strong>Top 3 交互对</strong>：</p>
{top_inter.to_html(index=False, float_format=lambda x: f"{x:.4f}")}

<h2>3. PCA 主成分分解</h2>
<p>看这些 Δ 变量是否存在潜在公因子（如"供需失配综合指数"）。</p>
<p><strong>方差解释</strong>：</p>
{pca_explained.to_html(index=False, float_format=lambda x: f"{x:.3f}")}
{img("pca_loadings")}
<div class='insight'>
<p><strong>解读</strong>：PC1 累积方差 {pca_explained.iloc[0]['cumulative']:.1%}，
如果这个值 > 50%，说明存在一个"主导模式"（如供需整体失配）；
如果前 3 个 PC 才达到 70%，说明驱动因素相对独立。</p>
</div>

<h2>4. 分时段建模对比</h2>
<p>峰/平/谷的物理驱动机制可能不同，分段建模看差异。</p>
{seg_df.to_html(index=False, float_format=lambda x: f"{x:.3f}")}

<h2>5. 结论与数据获取建议</h2>
<div class='warn'>
<p><strong>如果多元 R² 显著高于单因子（如 > 0.15）</strong>：
说明这些 Δ 变量的<strong>组合</strong>确实能更好解释实时价，
建议<strong>不需要单独追某一个数据源</strong>，而是：</p>
<ul>
<li>构造"综合供需失配指数"（如 PC1 或加权组合）</li>
<li>用历史滚动统计（如过去 7 日同时段的 Δ 标准差）作为波动性代理</li>
<li>重点获取<strong>交互增益最大的那对</strong>变量</li>
</ul>
</div>

<div class='insight'>
<p><strong>如果多元 R² 仍 < 0.15</strong>：
说明当前可观测的物理量组合不足以解释实时价偏差，真正的驱动因素在数据外，
需要获取<strong>调度侧结构性数据</strong>（机组检修、阻塞、备用指令、联络线实时调度）。</p>
</div>

</body></html>"""
    return html


# ---------- 主流程 ----------
def main():
    setup_font()
    print("[1/6] 加载数据并构造 Δ 变量")
    df = load_and_prepare()

    delta_cols = [c for c in df.columns if c.startswith("Δ") and c != "Δ实时节点电价"]
    # 按单因子相关性排序，取 Top 7
    corrs = {c: df[c].corr(df["Δ实时节点电价"]) for c in delta_cols}
    top_features = sorted(corrs, key=lambda x: abs(corrs[x]), reverse=True)[:7]
    print(f"      Top 7 特征: {top_features}")

    print("[2/6] 逐步回归")
    step_df = stepwise_regression(df, top_features)

    print("[3/6] 交互项矩阵")
    inter_df = interaction_matrix(df, top_features)

    print("[4/6] PCA 主成分")
    loadings, pca, Z = pca_analysis(df, delta_cols)
    pca_explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })

    print("[5/6] 分时段建模")
    seg_df = segment_models(df, top_features[:5])

    print("[6/6] 绘图与报告")
    plots = {
        "stepwise_r2": plot_stepwise(step_df),
        "interaction_heatmap": plot_interaction_heatmap(inter_df),
        "pca_loadings": plot_pca_loadings(loadings),
    }

    html = render_html(step_df, inter_df, loadings, pca_explained, seg_df, plots)
    report_path = os.path.join(ROOT, "multivariate_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    step_df.to_csv(os.path.join(ROOT, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    inter_df.to_csv(os.path.join(ROOT, "interaction_matrix.csv"), index=False, encoding="utf-8-sig")

    print(f"\n完成。报告：{report_path}")
    print("\n=== 逐步回归 R² ===")
    print(step_df[step_df["n_features"] > 1].to_string(index=False))
    print("\n=== Top 5 交互对 ===")
    print(inter_df.head(5).to_string(index=False))
    print("\n=== PCA 方差解释 ===")
    print(pca_explained.to_string(index=False))


if __name__ == "__main__":
    main()
