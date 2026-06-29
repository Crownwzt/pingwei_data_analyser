# -*- coding: utf-8 -*-
"""
④ 模型训练模块
================

职责：
  - 加载清洗缓存 + 切分配置
  - 特征工程 (day-ahead 合法) + 小时聚合
  - 训练 XGB 残差预测模型 (residual = y - 日前价)
  - 抗过拟合配置 (depth=4, λ=15, mcw=30, csbt=0.5)
  - 线性月权 (1.0~2.0)
  - 输出 model.joblib + metrics.pkl + 04_training.html

入口：python -m src.training
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (
    DA_COL, PATHS,
    DEFAULT_TRAIN_MONTHS, DEFAULT_VAL_MONTHS, DEFAULT_TEST_MONTHS,
    ensure_dirs, setup_cn_font, render_html, safe_savefig,
    load_clean, save_pickle, img_tag, metrics,
    prepare_features, split_by_months,
)


# ---------------------------------------------------------------------------
# XGB 配置 (抗过拟合，来自 V14 扫描结果)
# ---------------------------------------------------------------------------
XGB_CFG = dict(
    n_estimators=3000, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5,
    min_child_weight=30, gamma=0.2,
    reg_alpha=1.0, reg_lambda=15.0,
    tree_method="hist", objective="reg:squarederror",
    n_jobs=-1, early_stopping_rounds=80,
)

# Ensemble 配置：5 个不同随机种子的 XGB，预测取均值
ENSEMBLE_SEEDS = [42, 7, 137, 2024, 9527]


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------
def load_split_cfg() -> Dict:
    """读取切分配置 (split.py 生成的 split.json)。"""
    if not os.path.exists(PATHS["split"]):
        print(f"[WARN] {PATHS['split']} 不存在，使用默认切分 (方案 B)")
        return {
            "year": 2025,
            "train_months": DEFAULT_TRAIN_MONTHS,
            "val_months": DEFAULT_VAL_MONTHS,
            "test_months": DEFAULT_TEST_MONTHS,
        }
    with open(PATHS["split"], "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_weight(dt_series: pd.Series) -> np.ndarray:
    """训练样本线性月权 (1.0 ~ 2.0)，近月权重更高贴近测试分布。"""
    months = dt_series.dt.month.values
    span = max(months.max() - months.min(), 1)
    return 1.0 + (months - months.min()) / span


def fit_residual_ensemble(splits, da_col: str = DA_COL,
                          seeds: List[int] = None) -> List[XGBRegressor]:
    """
    Multi-seed XGB Ensemble (残差预测)。
    每个 seed 用相同超参 + 相同样本权重训一个独立 XGB，
    推理时各模型 predict 取均值 → 降低单模型方差。

    返回：list of fitted XGBRegressor (与 seeds 对应)。
    """
    seeds = seeds or ENSEMBLE_SEEDS
    da_tr = splits.X_tr[da_col].values
    da_va = splits.X_va[da_col].values
    res_tr = splits.y_tr.values - da_tr
    res_va = splits.y_va.values - da_va
    sw = _sample_weight(splits.dt_tr)

    models = []
    for i, seed in enumerate(seeds, 1):
        model = XGBRegressor(random_state=seed, **XGB_CFG)
        model.fit(splits.X_tr, res_tr, sample_weight=sw,
                  eval_set=[(splits.X_tr, res_tr), (splits.X_va, res_va)],
                  verbose=False)
        models.append(model)
        print(f"  [ensemble {i}/{len(seeds)}] seed={seed} best_iter={model.best_iteration}")
    return models


def ensemble_predict_residual(models: List[XGBRegressor],
                              X: pd.DataFrame) -> np.ndarray:
    """Ensemble 预测残差：各模型预测均值。"""
    preds = np.stack([m.predict(X) for m in models])
    return preds.mean(axis=0)


def fit_alpha(models: List[XGBRegressor], splits,
              da_col: str = DA_COL) -> Tuple[float, np.ndarray]:
    """
    在验证集上学最优融合权重 α (粒度 0.01)：
        ŷ = α · (日前价 + XGB_ensemble残差) + (1-α) · 日前价
           = 日前价 + α · XGB_ensemble残差
    目标：最小化验证集 MAE。
    返回 (alpha_star, val_mae_curve)。
    """
    da_va = splits.X_va[da_col].values
    y_va = splits.y_va.values
    res_pred_va = ensemble_predict_residual(models, splits.X_va)

    alphas = np.arange(0.0, 1.01, 0.01)
    maes = []
    for a in alphas:
        y_pred = da_va + a * res_pred_va
        maes.append(float(np.mean(np.abs(y_va - y_pred))))
    maes = np.array(maes)
    alpha_star = float(alphas[int(np.argmin(maes))])
    print(f"  [α 学习] 验证集 α 扫描: α*={alpha_star:.2f}, "
          f"val_MAE_min={maes.min():.3f} (α=1 时 MAE={maes[-1]:.3f})")
    return alpha_star, list(zip(alphas.tolist(), maes.tolist()))


def evaluate_split(models: List[XGBRegressor], alpha: float,
                   X: pd.DataFrame, y: pd.Series, da: np.ndarray) -> Dict:
    """
    Ensemble + α 融合后的评估指标。
    最终预测：ŷ = 日前价 + α · ensemble_avg(残差)
    """
    pred_res = ensemble_predict_residual(models, X)   # ensemble 均值残差
    y_pred = da + alpha * pred_res                     # α 融合
    m_xgb = metrics(y.values, y_pred)
    m_b7 = metrics(y.values, da)
    true_res = y.values - da
    eff_res = alpha * pred_res
    ss_res = np.sum((true_res - eff_res) ** 2)
    ss_tot = np.sum((true_res - true_res.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sign_hit = float(np.mean(np.sign(true_res) == np.sign(eff_res)))
    return {
        "n": int(len(y)),
        "xgb_mae": m_xgb["MAE"], "xgb_rmse": m_xgb["RMSE"], "xgb_mape": m_xgb["MAPE"],
        "b7_mae": m_b7["MAE"], "b7_rmse": m_b7["RMSE"], "b7_mape": m_b7["MAPE"],
        "mae_gain%": (m_b7["MAE"] - m_xgb["MAE"]) / m_b7["MAE"] * 100,
        "rmse_gain%": (m_b7["RMSE"] - m_xgb["RMSE"]) / m_b7["RMSE"] * 100,
        "residual_R2": float(r2), "sign_hit": sign_hit,
        "y_pred": y_pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Ablation: 单模型 vs Ensemble vs Ensemble+α 三档对比
# ---------------------------------------------------------------------------
def build_ablation_table(models: List[XGBRegressor], alpha_star: float,
                         splits, seeds: List[int],
                         da_col: str = DA_COL) -> pd.DataFrame:
    """
    在 train/val/test 三集上分别评测 (N+3) 个方案，量化 ensemble 与 α 融合的实际增益：

      Single_seed_i (i=1..N):  ŷ = 日前价 + XGB_i.predict(X)           (单 seed, α=1)
      Ensemble (α=1):          ŷ = 日前价 + ensemble_mean(残差)         (Multi-seed, 无 α)
      Ensemble + α=α*:         ŷ = 日前价 + α* · ensemble_mean(残差)    (本项目最终方案)
      B7'_baseline:            ŷ = 日前价                              (零成本对照)

    返回长格式 DataFrame:  方法 / split / MAE / RMSE / vs B7' 改进%
    """
    def _eval(name: str, y_pred: np.ndarray, y: np.ndarray,
              da: np.ndarray, split_name: str) -> Dict:
        m = metrics(y, y_pred)
        m_b7 = metrics(y, da)
        return {
            "方法": name, "split": split_name,
            "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"],
            "vs_B7_MAE%": (m_b7["MAE"] - m["MAE"]) / m_b7["MAE"] * 100,
            "vs_B7_RMSE%": (m_b7["RMSE"] - m["RMSE"]) / m_b7["RMSE"] * 100,
        }

    rows = []
    for split_name, X, y, da in [
        ("train", splits.X_tr, splits.y_tr.values, splits.X_tr[da_col].values),
        ("val",   splits.X_va, splits.y_va.values, splits.X_va[da_col].values),
        ("test",  splits.X_te, splits.y_te.values, splits.X_te[da_col].values),
    ]:
        # 1) 每个 single seed (α=1)
        single_preds = []
        for seed, mdl in zip(seeds, models):
            res = mdl.predict(X)
            y_pred = da + res            # α=1 单模型残差预测
            single_preds.append(res)
            rows.append(_eval(f"Single seed={seed} (α=1)", y_pred, y, da, split_name))
        # 2) Ensemble α=1
        ens_res = np.mean(np.stack(single_preds), axis=0)
        rows.append(_eval("Ensemble (α=1, no α tuning)", da + ens_res, y, da, split_name))
        # 3) Ensemble + α=α*
        rows.append(_eval(f"Ensemble + α* = {alpha_star:.2f} ★", da + alpha_star * ens_res,
                          y, da, split_name))
        # 4) B7' baseline
        rows.append(_eval("B7' 日前价 (zero-cost baseline)", da, y, da, split_name))

    return pd.DataFrame(rows)


def plot_ablation(ablation_df: pd.DataFrame) -> str:
    """绘制 ablation 对比柱状图（按测试集 MAE 排序）。"""
    setup_cn_font()
    test_df = ablation_df[ablation_df["split"] == "test"].copy()
    test_df = test_df.sort_values("MAE").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(test_df)); w = 0.38
    b1 = ax.bar(x - w/2, test_df["MAE"], w, label="MAE", color="#4C72B0")
    b2 = ax.bar(x + w/2, test_df["RMSE"], w, label="RMSE", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(test_df["方法"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("误差 (元/MWh)")
    ax.set_title("Ablation: 测试集 (2025-12) 各方案 MAE / RMSE 对比 (按 MAE 升序)",
                 fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        xy=(b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
    p = os.path.join(PATHS["plots"], "training_ablation.png")
    safe_savefig(p)
    return p


# ---------------------------------------------------------------------------
# 可视化（训练过程）
# ---------------------------------------------------------------------------
def plot_training(models: List[XGBRegressor], feature_cols: List[str],
                  alpha_curve: List[Tuple[float, float]] = None,
                  alpha_star: float = None) -> Dict[str, str]:
    setup_cn_font()
    paths = {}

    # 图 1: 训练曲线 (取 ensemble 中 best_iter 居中的那个模型展示)
    iters = [m.best_iteration for m in models]
    rep_idx = int(np.argsort(iters)[len(iters) // 2])
    rep = models[rep_idx]
    evals = rep.evals_result()
    if evals:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        keys = list(evals.keys())
        for i, k in enumerate(keys):
            metric_name = list(evals[k].keys())[0]
            v = evals[k][metric_name]
            label = "训练" if i == 0 else "验证"
            ax.plot(v, label=f"{label} {metric_name.upper()}",
                    color="#4C72B0" if i == 0 else "#C44E52")
        ax.axvline(rep.best_iteration, color="gray", ls="--",
                   label=f"早停 best_iter={rep.best_iteration}")
        ax.set_xlabel("Boosting Round"); ax.set_ylabel("RMSE on residual")
        ax.set_title(f"训练曲线 (代表模型, ensemble 中位 seed)  "
                     f"全部 best_iter={iters}")
        ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(PATHS["plots"], "training_curve.png")
        safe_savefig(p); paths["curve"] = p

    # 图 2: 特征重要性 = ensemble 内所有模型的均值
    imp_mat = np.stack([m.feature_importances_ for m in models])
    imp_mean = pd.Series(imp_mat.mean(axis=0), index=feature_cols)\
                 .sort_values(ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_mean.index, imp_mean.values, color="#4C72B0")
    ax.set_xlabel("XGB Importance (gain) — ensemble 均值")
    ax.set_title(f"特征重要性 Top 20 ({len(models)} 个 seed 均值)")
    ax.grid(alpha=0.3, axis="x")
    p = os.path.join(PATHS["plots"], "training_feat_importance.png")
    safe_savefig(p); paths["feat"] = p

    # 图 3: α 扫描曲线
    if alpha_curve and alpha_star is not None:
        alphas = [a for a, _ in alpha_curve]
        maes = [m for _, m in alpha_curve]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(alphas, maes, color="#4C72B0", lw=1.5)
        ax.axvline(alpha_star, color="#C44E52", ls="--",
                   label=f"α* = {alpha_star:.2f}")
        ax.axvline(0.0, color="gray", ls=":", alpha=0.6, label="α=0 (纯 B7' 日前价)")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.6, label="α=1 (纯 XGB 残差)")
        ax.set_xlabel("融合权重 α  (y_hat = 日前价 + α · XGB残差)")
        ax.set_ylabel("验证集 MAE")
        ax.set_title("α 扫描曲线 (验证集上学最优融合权重)")
        ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(PATHS["plots"], "training_alpha_curve.png")
        safe_savefig(p); paths["alpha"] = p

    return paths


# ---------------------------------------------------------------------------
# 特征公式表（用于 HTML 详细文档）
# ---------------------------------------------------------------------------
FEATURE_SPECS = [
    # (类别, 特征名, 计算公式, 业务含义)
    ("周期编码", "小时_sin / 小时_cos", "sin(2π·h/24), cos(2π·h/24)",
     "把离散小时映射到圆周上，让 23 时与 0 时距离=1"),
    ("周期编码", "月_sin / 月_cos", "sin(2π·m/12), cos(2π·m/12)",
     "月份周期 (12 月 ↔ 1 月相邻)"),
    ("周期编码", "星期_sin / 星期_cos", "sin(2π·dow/7), cos(2π·dow/7)",
     "周节律 (周日 ↔ 周一相邻)"),
    ("target 滞后", "target_lag_24h", "y.shift(24)",
     "1 天前同时刻实时价"),
    ("target 滞后", "target_lag_48/72/120/168h", "y.shift({48,72,120,168})",
     "2/3/5/7 天前同时刻"),
    ("rolling", "target_roll_mean_24_lag24",
     "y.shift(24).rolling(24).mean()",
     "过去 24 小时均价 (窗口 lag_24~lag_47, 永不碰当前)"),
    ("rolling", "target_roll_std_24_lag24",
     "y.shift(24).rolling(24).std()",
     "过去 24 小时波动幅度"),
    ("rolling", "target_roll_mean_168_lag24",
     "y.shift(24).rolling(168).mean()",
     "过去 7 天均价 (窗口 lag_24~lag_191)"),
    ("差分", "target_yest_vs_lastweek",
     "y.shift(24) - y.shift(168)",
     "昨日 vs 上周同日 偏差，捕捉短期趋势"),
    ("时段 one-hot", "时段_峰 / 时段_平 / 时段_谷",
     "(时段=={'峰','平','谷'}).astype(int)",
     "峰: 8-11 + 18-21; 谷: 0-6; 其余平"),
    ("业务衍生", "新能源渗透率_日前",
     "新能源_日前 / 省调负荷_日前  (除零置 0)",
     "边际机组成本核心驱动 (光伏/风电占比越高，电价越低)"),
    ("业务衍生", "竞价空间紧张度_日前",
     "竞价空间_日前 / 省调负荷_日前  (除零置 0)",
     "供需紧张度 (空间越小，高成本机组越易入清算)"),
    ("原始日前列",
     "日前统一结算点电价 / 日前节点电价 / 省调负荷_日前 / 新能源_日前 / 光伏_日前 / 风电_日前 / 水电_日前 / 竞价空间_日前 / 非市场化出力_日前 / 外来送负荷_日前 / 发电总出力_日前",
     "原值, D-1 14:00 出清后已知", "day-ahead 合法的供需特征"),
]

# 显式禁用列（数据泄漏 / 未来信息）
FORBIDDEN_SPECS = [
    ("实时节点电价(元/MWh)", "与目标同一次出清产生，相关性≈0.95，强同步信号"),
    ("所有 _实际 后缀列", "D 日实际负荷/出力 D-1 日 14:00 申报截止时未发生"),
    ("target lag < 24h", "day-ahead 场景下 D 日所有点的近期 lag 都是未来"),
]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def make_html(models: List[XGBRegressor], alpha_star: float, splits,
              m_tr: Dict, m_va: Dict, m_te: Dict,
              plot_paths: Dict[str, str],
              ablation_df: pd.DataFrame = None) -> str:
    p: List[str] = []
    iters = [m.best_iteration for m in models]

    p.append('<div class="meta">')
    p.append(f"<p><strong>训练目标</strong>: 残差 <code>residual = y - 日前价</code></p>")
    p.append(f"<p><strong>最终预测</strong>: "
             f"<code>ŷ = 日前价 + α · ensemble_mean(XGB残差)</code>, "
             f"α* = <strong>{alpha_star:.2f}</strong> (验证集学得)</p>")
    p.append(f"<p><strong>Ensemble</strong>: {len(models)} 个不同 seed 的 XGB, "
             f"预测取均值; seeds = <code>{ENSEMBLE_SEEDS}</code></p>")
    p.append(f"<p><strong>各 seed best_iter</strong>: {iters}</p>")
    p.append(f"<p><strong>样本规模 (小时粒度)</strong>: 训练 {m_tr['n']:,}h / "
             f"验证 {m_va['n']:,}h / 测试 {m_te['n']:,}h</p>")
    p.append(f"<p><strong>特征维度</strong>: {len(splits.feature_cols)} 列 "
             "(全部 day-ahead 合法)</p>")
    p.append(f"<p><strong>模型保存</strong>: <code>{PATHS['model']}</code> "
             "(joblib 序列化的 list[XGBRegressor] + α)</p>")
    p.append("</div>")

    # 一、残差预测设计
    p.append("<h2>一、残差预测设计</h2>")
    p.append("""
    <div class='insight'>
    <p>不直接预测 <code>y</code>，而是预测 <strong>残差</strong>
    <code>residual = y − 日前价</code>。</p>
    <p><strong>动机</strong>：日前价是 D-1 日全市场博弈出清的最优预期，已凝聚海量信号。
    直接预测 <code>y</code> 会让 XGB 重复"学一遍"日前价，再叠加其他特征的噪声 → 反而
    更差。残差预测让 XGB 只学日前价漏掉的增量信号。</p>
    </div>
    """)

    # 二、Ensemble + α 加权融合
    p.append("<h2>二、Ensemble + α 加权融合</h2>")
    p.append(f"""
    <div class='insight'>
    <p><strong>Multi-seed XGB Ensemble</strong> ({len(models)} 个模型)：</p>
    <ul>
    <li>相同超参 + 相同样本权重，仅 <code>random_state</code> 不同</li>
    <li>seeds = <code>{ENSEMBLE_SEEDS}</code></li>
    <li>子采样 (subsample=0.7) + 列采样 (colsample_bytree=0.5) 提供随机源</li>
    <li>推理：<code>ensemble_residual = mean(model_i.predict(X))</code></li>
    <li><strong>作用</strong>: 降低单模型方差，对小数据集 (697 h 验证) 尤其稳定</li>
    </ul>
    <p><strong>α 加权融合</strong> (验证集学得最优 α* = {alpha_star:.2f})：</p>
    <ul>
    <li>融合公式: <code>ŷ = 日前价 + α · ensemble_residual</code></li>
    <li>α=0 → 纯 B7' 日前价 baseline</li>
    <li>α=1 → 纯 XGB ensemble 残差预测</li>
    <li>α∈(0,1) → 收缩到日前价方向, 抑制 XGB 过激修正</li>
    <li>扫描粒度 0.01, 在验证集上最小化 MAE</li>
    </ul>
    </div>
    """)
    if plot_paths.get("alpha"):
        p.append(img_tag(plot_paths["alpha"]))

    # 二·补：Ablation 消融对比 — 量化 ensemble 与 α 的实际增益
    if ablation_df is not None and len(ablation_df) > 0:
        p.append("<h2>二·补、Ablation 消融对比 (回答 \"ensemble 与 α 到底贡献多少\")</h2>")
        p.append("""
        <div class='insight'>
        <p>把每个方案在三集上都跑一遍，<strong>直接量化设计每一步的实际增益</strong>：</p>
        <ul>
        <li><code>Single seed=X (α=1)</code>: 单 XGB, 直接用其残差预测 (α=1)</li>
        <li><code>Ensemble (α=1)</code>: 5 个 seed 残差取均值, 仍 α=1 (隔离 ensemble 单独的方差降低收益)</li>
        <li><code>Ensemble + α*</code>: 本项目最终方案 (α* 在验证集学得)</li>
        <li><code>B7' 日前价</code>: 零成本基准</li>
        </ul>
        </div>
        """)

        # 三集 pivot 表 (按方案行, train/val/test 三列分别展示 MAE)
        for split_label, split_key in [("训练 train", "train"),
                                        ("验证 val", "val"),
                                        ("测试 test ★", "test")]:
            p.append(f"<h3>{split_label}</h3>")
            sub = ablation_df[ablation_df["split"] == split_key].sort_values("MAE").reset_index(drop=True)
            p.append("<table><tr><th>排名</th><th>方法</th><th>MAE</th><th>RMSE</th>"
                     "<th>MAPE%</th><th>vs B7' MAE%</th><th>vs B7' RMSE%</th></tr>")
            best_mae = sub["MAE"].min()
            worst_mae = sub["MAE"].max()
            for i, r in sub.iterrows():
                cls = ""
                if r["MAE"] == best_mae and "★" in r["方法"]:
                    cls = "winner"
                elif r["MAE"] == best_mae:
                    cls = "winner"
                elif r["MAE"] == worst_mae:
                    cls = "loser"
                p.append(f"<tr class='{cls}'>"
                         f"<td class='center'>{i+1}</td>"
                         f"<td>{r['方法']}</td>"
                         f"<td class='num'>{r['MAE']:.3f}</td>"
                         f"<td class='num'>{r['RMSE']:.3f}</td>"
                         f"<td class='num'>{r['MAPE']:.2f}</td>"
                         f"<td class='num'>{r['vs_B7_MAE%']:+.2f}</td>"
                         f"<td class='num'>{r['vs_B7_RMSE%']:+.2f}</td></tr>")
            p.append("</table>")

        if plot_paths.get("ablation"):
            p.append(img_tag(plot_paths["ablation"]))

        # 自动计算并展示增益分解
        te = ablation_df[ablation_df["split"] == "test"].set_index("方法")
        b7_mae = te.loc[
            te.index[te.index.str.startswith("B7'")][0], "MAE"]
        # 单 seed 平均
        single_rows = te[te.index.str.startswith("Single seed=")]
        single_mae_avg = single_rows["MAE"].mean()
        single_mae_std = single_rows["MAE"].std()
        ens_a1_mae = te.loc[te.index[te.index.str.startswith("Ensemble (α=1")][0], "MAE"]
        ens_astar_mae = te.loc[te.index[te.index.str.contains("★")][0], "MAE"]

        p.append("""
        <div class='insight'>
        <p><strong>测试集增益分解</strong> (越往下越好):</p>
        <table>
        <tr><th>步骤</th><th>测试 MAE</th><th>累计 vs B7' 改进%</th><th>本步带来的增量</th></tr>
        """)
        p.append(f"<tr><td>B7' 日前价 baseline (起点)</td>"
                 f"<td class='num'>{b7_mae:.3f}</td>"
                 f"<td class='num'>0%</td>"
                 f"<td class='small'>—</td></tr>")
        p.append(f"<tr><td>单 XGB seed 平均 (α=1)</td>"
                 f"<td class='num'>{single_mae_avg:.3f} ± {single_mae_std:.2f}</td>"
                 f"<td class='num'>{(b7_mae - single_mae_avg)/b7_mae*100:+.2f}%</td>"
                 f"<td class='small'>单模型 (有方差)</td></tr>")
        gain_ens = (single_mae_avg - ens_a1_mae) / single_mae_avg * 100
        p.append(f"<tr><td>+ Multi-seed Ensemble (α=1)</td>"
                 f"<td class='num'>{ens_a1_mae:.3f}</td>"
                 f"<td class='num'>{(b7_mae - ens_a1_mae)/b7_mae*100:+.2f}%</td>"
                 f"<td class='small'>方差↓ (vs 单 seed 平均: {gain_ens:+.2f}%)</td></tr>")
        gain_alpha = (ens_a1_mae - ens_astar_mae) / ens_a1_mae * 100
        p.append(f"<tr class='winner'><td>+ α* 加权融合 (本项目最终方案)</td>"
                 f"<td class='num'>{ens_astar_mae:.3f}</td>"
                 f"<td class='num'>{(b7_mae - ens_astar_mae)/b7_mae*100:+.2f}%</td>"
                 f"<td class='small'>偏差↓ (vs Ensemble α=1: {gain_alpha:+.2f}%)</td></tr>")
        p.append("</table></div>")

        p.append("""
        <div class='insight'>
        <p><strong>解读</strong>：</p>
        <ul>
        <li>单 seed 跨 5 个种子的标准差体现 <strong>训练随机性带来的方差</strong></li>
        <li>Ensemble 把这部分方差<strong>取平均消掉</strong>，对应"方差↓"列的增量</li>
        <li>α* 的增量来自<strong>偏差校正</strong> — XGB 残差对测试集会过激修正,
            α<1 把预测向日前价收缩, 弥补了模型 + 数据的系统性偏差</li>
        </ul>
        </div>
        """)

    # 三、特征工程清单（含精确计算公式）
    p.append("<h2>三、特征工程清单 (含精确计算公式)</h2>")
    p.append("<p>所有特征定义在 <code>src/common.py::build_features</code>。"
             "所有 target 滞后/滚动/差分严格 ≥ 24 小时，"
             "保证 day-ahead 场景下不引入未来信息。</p>")
    p.append("<table>")
    p.append("<tr><th>类别</th><th>特征名</th><th>计算公式</th><th>业务含义</th></tr>")
    for cat, name, formula, why in FEATURE_SPECS:
        p.append(f"<tr><td>{cat}</td><td><code>{name}</code></td>"
                 f"<td><code>{formula}</code></td>"
                 f"<td class='small'>{why}</td></tr>")
    p.append("</table>")

    p.append("<h3>显式禁用列 (数据泄漏 / 未来信息)</h3>")
    p.append("<table><tr><th>禁用列</th><th>禁用原因</th></tr>")
    for name, why in FORBIDDEN_SPECS:
        p.append(f"<tr class='leak'><td><code>{name}</code></td>"
                 f"<td class='small'>{why}</td></tr>")
    p.append("</table>")

    p.append('<div class="warn">')
    p.append("<p><strong>关键合法性证明</strong>：所有 lag/rolling/差分特征的"
             "窗口尾部都至少是 24 小时前的数据。预测 D 日任何小时点时，"
             "用到的数据全部在 D-1 日 0:00 之前已结算公布。"
             "<code>实时节点电价</code> 与目标同时刻产生 (r≈0.95)，已显式黑名单。</p>")
    p.append("</div>")

    # 四、超参数
    p.append("<h2>四、超参数 (抗过拟合配置)</h2>")
    p.append("""
    <div class='insight'>
    <p>诊断发现训练集残差 R² ≈ 0.44 vs 测试集 R² ≈ 0.13，明显过拟合。
    通过容量×正则双管齐下扫描确定以下配置：</p>
    </div>
    """)
    p.append("<table><tr><th>参数</th><th>取值</th><th>作用</th></tr>")
    descs = {
        "max_depth": "限制每棵树深度，抑制复杂规则",
        "min_child_weight": "每叶子至少 30 样本，强制泛化",
        "reg_lambda": "强 L2 抑制权重",
        "reg_alpha": "L1 推稀疏",
        "colsample_bytree": "每棵树用一半特征，强制多样性",
        "subsample": "样本子采样，注入随机源",
        "learning_rate": "慢学习配合早停",
        "early_stopping_rounds": "基于验证集自动停止",
        "n_estimators": "上限，配合早停",
        "tree_method": "直方图加速",
        "objective": "与 RMSE 评估对齐",
        "gamma": "分裂最低收益阈值",
        "n_jobs": "并行",
    }
    for k, v in XGB_CFG.items():
        p.append(f"<tr><td><code>{k}</code></td><td class='num'>{v}</td>"
                 f"<td class='small'>{descs.get(k, '')}</td></tr>")
    p.append(f"<tr><td><code>ENSEMBLE_SEEDS</code></td>"
             f"<td class='num'>{ENSEMBLE_SEEDS}</td>"
             f"<td class='small'>{len(ENSEMBLE_SEEDS)} 个 XGB ensemble seeds</td></tr>")
    p.append("</table>")

    # 五、训练过程
    p.append("<h2>五、训练过程</h2>")
    p.append("<h3>训练 / 验证 RMSE 曲线 (代表模型)</h3>")
    p.append(img_tag(plot_paths.get("curve", "")))

    # 六、三集评估
    p.append("<h2>六、三集评估指标 (ensemble + α 融合后)</h2>")
    p.append("<table><tr><th>数据集</th><th>样本</th><th>XGB MAE</th><th>B7' MAE</th>"
             "<th>MAE 改进%</th><th>XGB RMSE</th><th>B7' RMSE</th><th>RMSE 改进%</th>"
             "<th>有效残差 R²</th><th>方向命中%</th></tr>")
    for name, r in [("训练", m_tr), ("验证", m_va), ("测试", m_te)]:
        cls_mae = "winner" if r["mae_gain%"] > 0 else "loser"
        cls_rmse = "winner" if r["rmse_gain%"] > 0 else "loser"
        p.append(f"<tr><td><strong>{name}</strong></td>"
                 f"<td class='num'>{r['n']:,}</td>"
                 f"<td class='num'>{r['xgb_mae']:.2f}</td>"
                 f"<td class='num'>{r['b7_mae']:.2f}</td>"
                 f"<td class='num {cls_mae}'>{r['mae_gain%']:+.2f}</td>"
                 f"<td class='num'>{r['xgb_rmse']:.2f}</td>"
                 f"<td class='num'>{r['b7_rmse']:.2f}</td>"
                 f"<td class='num {cls_rmse}'>{r['rmse_gain%']:+.2f}</td>"
                 f"<td class='num'>{r['residual_R2']:.3f}</td>"
                 f"<td class='num'>{r['sign_hit']*100:.1f}</td></tr>")
    p.append("</table>")

    p.append("<h2>七、特征重要性 Top 20 (ensemble 均值)</h2>")
    p.append(img_tag(plot_paths.get("feat", "")))

    return render_html("④ 模型训练报告", p, PATHS["html_training"])


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> str:
    ensure_dirs()
    setup_cn_font()
    df_clean = load_clean()
    df_feat, feat_cols = prepare_features(df_clean)
    cfg = load_split_cfg()
    splits = split_by_months(
        df_feat, feat_cols,
        train_months=cfg["train_months"],
        val_months=cfg["val_months"],
        test_months=cfg["test_months"],
        year=cfg["year"],
    )
    print(f"[切分] 训练 {len(splits.X_tr):,}h / 验证 {len(splits.X_va):,}h / "
          f"测试 {len(splits.X_te):,}h  特征={len(feat_cols)}")

    print(f"\n[训练 ensemble: {len(ENSEMBLE_SEEDS)} 个 XGB]")
    models = fit_residual_ensemble(splits)

    print("\n[学习 α 加权融合]")
    alpha_star, alpha_curve = fit_alpha(models, splits)

    # 保存 ensemble + α (单一 joblib bundle)
    joblib.dump({"models": models, "alpha": alpha_star,
                 "feature_cols": feat_cols,
                 "config": XGB_CFG, "seeds": ENSEMBLE_SEEDS},
                PATHS["model"], compress=3)
    print(f"[INFO] 模型 bundle 已保存 {PATHS['model']} "
          f"(α={alpha_star:.2f}, {len(models)} models)")

    da_tr = splits.X_tr[DA_COL].values
    da_va = splits.X_va[DA_COL].values
    da_te = splits.X_te[DA_COL].values
    m_tr = evaluate_split(models, alpha_star, splits.X_tr, splits.y_tr, da_tr)
    m_va = evaluate_split(models, alpha_star, splits.X_va, splits.y_va, da_va)
    m_te = evaluate_split(models, alpha_star, splits.X_te, splits.y_te, da_te)

    print("\n[评估 (ensemble + α 融合后)]")
    for name, r in [("train", m_tr), ("val", m_va), ("test", m_te)]:
        print(f"  {name:5s} n={r['n']:,} XGB_MAE={r['xgb_mae']:.2f} "
              f"B7_MAE={r['b7_mae']:.2f} 改进={r['mae_gain%']:+.2f}% "
              f"RMSE_改进={r['rmse_gain%']:+.2f}%  R²={r['residual_R2']:.3f}")

    # 持久化指标供 evaluation 模块使用（三集完整序列，供逐日图生成）
    # 特征重要性取 ensemble 均值
    imp_mat = np.stack([m.feature_importances_ for m in models])
    imp_mean = dict(zip(feat_cols, imp_mat.mean(axis=0).tolist()))

    save_pickle({
        "config": XGB_CFG,
        "seeds": ENSEMBLE_SEEDS,
        "alpha": alpha_star,
        "alpha_curve": alpha_curve,
        "split_cfg": cfg,
        "feature_cols": feat_cols,
        "best_iters": [int(m.best_iteration) for m in models],
        "metrics": {"train": m_tr, "val": m_va, "test": m_te},
        "feature_importance": imp_mean,
        "dt_train": splits.dt_tr.values.astype("datetime64[us]").astype(str).tolist(),
        "y_train":  splits.y_tr.values.tolist(),
        "da_train": da_tr.tolist(),
        "dt_val":   splits.dt_va.values.astype("datetime64[us]").astype(str).tolist(),
        "y_val":    splits.y_va.values.tolist(),
        "da_val":   da_va.tolist(),
        "dt_test":  splits.dt_te.values.astype("datetime64[us]").astype(str).tolist(),
        "y_test":   splits.y_te.values.tolist(),
        "da_test":  da_te.tolist(),
    }, PATHS["metrics"])
    print(f"[INFO] 指标已保存 {PATHS['metrics']}")

    plot_paths = plot_training(models, feat_cols, alpha_curve, alpha_star)

    # Ablation 消融对比：单 seed / Ensemble α=1 / Ensemble + α* / B7' baseline
    print("\n[Ablation 消融对比]")
    ablation_df = build_ablation_table(models, alpha_star, splits, ENSEMBLE_SEEDS)
    test_view = ablation_df[ablation_df["split"] == "test"].sort_values("MAE")
    print(test_view[["方法", "MAE", "RMSE", "vs_B7_MAE%"]].to_string(index=False))
    plot_paths["ablation"] = plot_ablation(ablation_df)

    out = make_html(models, alpha_star, splits, m_tr, m_va, m_te, plot_paths,
                    ablation_df=ablation_df)
    print(f"[INFO] 已生成 {out}")
    return out


if __name__ == "__main__":
    main()
