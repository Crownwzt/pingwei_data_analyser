# -*- coding: utf-8 -*-
"""
共享工具模块
=============

集中以下能力，避免各模块重复：
  - 路径常量      PATHS
  - 中文字体配置  setup_cn_font()
  - HTML 渲染    html_*(), render_html()
  - 数据加载/切分 load_clean, split_by_strategy, aggregate_hourly
  - 特征工程     build_features, select_feature_cols
  - 评估指标     metrics()
  - 图片转 b64   img_b64(), img_tag()

业务约定：
  - 目标列        TARGET_COL    实时统一结算点电价(元/MWh)
  - 日前价基准    DA_COL        日前统一结算点电价(元/MWh) (D-1 已知)
  - 数据泄漏列    LEAKAGE_COLS  实时节点电价（与目标同时刻产生）
  - 时序粒度      15min → 聚合为小时 (24 点/天)
  - 默认切分      train=2025-01~10 / val=2025-11 / test=2025-12
"""

from __future__ import annotations

import os
import base64
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. 全局常量
# ---------------------------------------------------------------------------
TARGET_COL = "实时统一结算点电价(元/MWh)"
DA_COL = "日前统一结算点电价(元/MWh)"

# 与目标同时刻产生的强同步信号（实时节点价、实时统一结算价是同一次出清的两个口径）
LEAKAGE_COLS = {"实时节点电价(元/MWh)"}

# 默认切分（dataset_partitioning 选定的方案 B）
DEFAULT_TRAIN_MONTHS = list(range(1, 11))   # 2025-01 ~ 10
DEFAULT_VAL_MONTHS = [11]
DEFAULT_TEST_MONTHS = [12]
DEFAULT_DATA_YEAR = 2025

# 工程路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR_DEFAULT = os.path.join(ROOT, "2025-2026市场情况")
OUT_DIR = os.path.join(ROOT, "outputs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
DAILY_DIR = os.path.join(OUT_DIR, "daily")

PATHS = {
    "cleaned": os.path.join(OUT_DIR, "cleaned_data.pkl"),
    "split": os.path.join(OUT_DIR, "split.json"),
    "model": os.path.join(OUT_DIR, "model.joblib"),
    "metrics": os.path.join(OUT_DIR, "metrics.pkl"),
    "corr": os.path.join(OUT_DIR, "correlation.pkl"),
    "html_cleaning": os.path.join(OUT_DIR, "01_cleaning.html"),
    "html_split": os.path.join(OUT_DIR, "02_split.html"),
    "html_correlation": os.path.join(OUT_DIR, "03_correlation.html"),
    "html_training": os.path.join(OUT_DIR, "04_training.html"),
    "html_evaluation": os.path.join(OUT_DIR, "05_evaluation.html"),
    "html_index": os.path.join(OUT_DIR, "index.html"),
    "plots": PLOTS_DIR,
    "daily": DAILY_DIR,
    "daily_train": os.path.join(DAILY_DIR, "train"),
    "daily_val":   os.path.join(DAILY_DIR, "val"),
    "daily_test":  os.path.join(DAILY_DIR, "test"),
}


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for k in ("daily_train", "daily_val", "daily_test"):
        os.makedirs(PATHS[k], exist_ok=True)


# ---------------------------------------------------------------------------
# 2. 中文字体（matplotlib 全局）
# ---------------------------------------------------------------------------
def setup_cn_font() -> None:
    import matplotlib.font_manager as fm
    cand = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
    avail = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in cand if c in avail), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 130


# ---------------------------------------------------------------------------
# 3. HTML 工具
# ---------------------------------------------------------------------------
HTML_CSS = """
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 1200px;
       margin: 30px auto; padding: 0 24px; background: #f7f8fa; color: #2c3e50; }
h1 { color: #1f3a5f; border-bottom: 4px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 32px; border-left: 5px solid #3498db; padding-left: 14px; }
h3 { color: #34495e; margin-top: 22px; }
.meta { background:#ecf0f1; padding:14px 18px; border-radius:6px; margin:14px 0; }
.insight { background:white; padding:14px 18px; border-left:4px solid #16a085;
           border-radius:4px; margin:12px 0; }
.warn { background:#fff4e3; padding:14px 18px; border-left:5px solid #e67e22;
        border-radius:6px; margin:14px 0; }
table { border-collapse:collapse; width:100%; background:white; margin:14px 0;
        box-shadow:0 2px 4px rgba(0,0,0,0.08); }
th, td { border:1px solid #dde2e7; padding:8px 12px; font-size:13px; }
th { background:#3498db; color:white; text-align:center; }
td.num { font-family:Consolas,monospace; text-align:right; }
td.center { text-align:center; }
.winner { background:#d5f5e3 !important; font-weight:600; }
.loser { background:#fadbd8 !important; }
.leak { background:#fcf3cf !important; }
code { background:#ecf0f1; padding:2px 6px; border-radius:3px; }
img { max-width:100%; border:1px solid #ddd; border-radius:6px;
      box-shadow:0 2px 8px rgba(0,0,0,0.12); margin:10px 0; }
.small { color:#7f8c8d; font-size:12px; }
.nav { background:#34495e; padding:10px; border-radius:6px; margin:0 0 20px 0; }
.nav a { color:white; margin-right:18px; text-decoration:none; }
.nav a:hover { text-decoration:underline; }
"""

NAV_HTML = """
<div class="nav">
  <a href="index.html">🏠 总览</a>
  <a href="01_cleaning.html">① 数据清洗</a>
  <a href="02_split.html">② 数据划分</a>
  <a href="03_correlation.html">③ 相关性分析</a>
  <a href="04_training.html">④ 模型训练</a>
  <a href="05_evaluation.html">⑤ 模型评测</a>
</div>
"""


def img_b64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path: str, alt: str = "") -> str:
    b64 = img_b64(path)
    if not b64:
        return f"<p class='small' style='color:#c0392b;'>⚠️ 图片缺失: {path}</p>"
    return f"<img src='data:image/png;base64,{b64}' alt='{alt}'>"


def render_html(title: str, body_parts: List[str], out_path: str) -> str:
    """渲染统一 HTML 文档（带导航条）。"""
    html = (f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
            f"<title>{title}</title><style>{HTML_CSS}</style></head><body>"
            f"{NAV_HTML}<h1>{title}</h1>"
            + "\n".join(body_parts) +
            "</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def safe_savefig(path: str) -> None:
    try:
        plt.savefig(path, bbox_inches="tight")
    finally:
        plt.close()


# ---------------------------------------------------------------------------
# 4. 数据加载 / 切分 / 聚合
# ---------------------------------------------------------------------------
def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_clean() -> pd.DataFrame:
    """加载清洗后的 2025 全年 15min 缓存（由 cleaning.py 生成）。"""
    if not os.path.exists(PATHS["cleaned"]):
        raise FileNotFoundError(f"清洗缓存不存在: {PATHS['cleaned']}, 请先运行 src/cleaning.py")
    return load_pickle(PATHS["cleaned"])


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """15min → 小时（4 点取均值），保留时间衍生列。"""
    df = df.copy()
    df["datetime_hour"] = df["datetime"].dt.floor("h")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    agg_dict = {c: "mean" for c in numeric_cols}
    if "时段" in df.columns:
        agg_dict["时段"] = "first"
    df_hour = df.groupby("datetime_hour", as_index=False).agg(agg_dict)
    df_hour = df_hour.rename(columns={"datetime_hour": "datetime"})
    df_hour["年"] = df_hour["datetime"].dt.year
    df_hour["月"] = df_hour["datetime"].dt.month
    df_hour["日"] = df_hour["datetime"].dt.day
    df_hour["小时"] = df_hour["datetime"].dt.hour
    df_hour["星期"] = df_hour["datetime"].dt.weekday + 1
    df_hour["是否周末"] = (df_hour["星期"] >= 6).astype(int)
    if "时段" not in df_hour.columns:
        df_hour["时段"] = df_hour["小时"].map(_seg_label)
    return df_hour.sort_values("datetime").reset_index(drop=True)


def _seg_label(h: int) -> str:
    if 8 <= h <= 11 or 18 <= h <= 21:
        return "峰"
    if 0 <= h <= 6:
        return "谷"
    return "平"


@dataclass
class Splits:
    """train/val/test 三集容器。"""
    X_tr: pd.DataFrame; y_tr: pd.Series; dt_tr: pd.Series
    X_va: pd.DataFrame; y_va: pd.Series; dt_va: pd.Series
    X_te: pd.DataFrame; y_te: pd.Series; dt_te: pd.Series
    feature_cols: List[str]


def split_by_months(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_months: List[int] = None,
    val_months: List[int] = None,
    test_months: List[int] = None,
    year: int = DEFAULT_DATA_YEAR,
    target_col: str = TARGET_COL,
) -> Splits:
    """按 (year, 月份列表) 切分，返回 Splits。"""
    train_months = train_months or DEFAULT_TRAIN_MONTHS
    val_months = val_months or DEFAULT_VAL_MONTHS
    test_months = test_months or DEFAULT_TEST_MONTHS

    df = df.copy()
    df["_y"] = df["datetime"].dt.year
    df["_m"] = df["datetime"].dt.month
    masks = {
        "train": (df["_y"] == year) & (df["_m"].isin(train_months)),
        "val":   (df["_y"] == year) & (df["_m"].isin(val_months)),
        "test":  (df["_y"] == year) & (df["_m"].isin(test_months)),
    }
    parts = {}
    for k, m in masks.items():
        sub = df.loc[m]
        parts[k] = {
            "X": sub[feature_cols].copy(),
            "y": sub[target_col].copy(),
            "dt": sub["datetime"].copy(),
        }
    return Splits(
        X_tr=parts["train"]["X"], y_tr=parts["train"]["y"], dt_tr=parts["train"]["dt"],
        X_va=parts["val"]["X"],   y_va=parts["val"]["y"],   dt_va=parts["val"]["dt"],
        X_te=parts["test"]["X"],  y_te=parts["test"]["y"],  dt_te=parts["test"]["dt"],
        feature_cols=feature_cols,
    )


# ---------------------------------------------------------------------------
# 5. day-ahead 合法特征工程
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    构造 day-ahead 场景合法特征。所有 target lag 至少 24 小时；所有 "_实际" 列禁用。

    新增列：
      - 周期编码    小时_sin/cos, 月_sin/cos, 星期_sin/cos
      - target lag  lag_{24,48,72,120,168}h
      - rolling     基于 shift(24).rolling，窗口 24h/168h
      - 差分        target_yest_vs_lastweek = lag_24 - lag_168
      - 时段 one-hot 时段_峰/平/谷
      - 业务衍生    新能源渗透率_日前, 竞价空间紧张度_日前
    """
    df = df.sort_values("datetime").reset_index(drop=True).copy()

    # 周期编码
    df["小时_sin"] = np.sin(2 * np.pi * df["小时"] / 24)
    df["小时_cos"] = np.cos(2 * np.pi * df["小时"] / 24)
    df["月_sin"] = np.sin(2 * np.pi * df["月"] / 12)
    df["月_cos"] = np.cos(2 * np.pi * df["月"] / 12)
    df["星期_sin"] = np.sin(2 * np.pi * df["星期"] / 7)
    df["星期_cos"] = np.cos(2 * np.pi * df["星期"] / 7)

    # target lag / rolling（≥ 24h 才合法）
    if target_col in df.columns:
        for lag in [24, 48, 72, 120, 168]:
            df[f"target_lag_{lag}h"] = df[target_col].shift(lag)
        base = df[target_col].shift(24)
        df["target_roll_mean_24_lag24"] = base.rolling(24, min_periods=6).mean()
        df["target_roll_std_24_lag24"] = base.rolling(24, min_periods=6).std()
        df["target_roll_mean_168_lag24"] = base.rolling(168, min_periods=24).mean()
        df["target_yest_vs_lastweek"] = df[target_col].shift(24) - df[target_col].shift(168)

    # 时段 one-hot
    if "时段" in df.columns:
        for s in ["峰", "平", "谷"]:
            df[f"时段_{s}"] = (df["时段"] == s).astype(int)

    # 业务衍生
    if "省调负荷(MW)_日前" in df.columns and "新能源负荷(MW)_日前" in df.columns:
        df["新能源渗透率_日前"] = (df["新能源负荷(MW)_日前"]
                                   / df["省调负荷(MW)_日前"].replace(0, np.nan)).fillna(0)
    if "竞价空间(MW)_日前" in df.columns and "省调负荷(MW)_日前" in df.columns:
        df["竞价空间紧张度_日前"] = (df["竞价空间(MW)_日前"]
                                     / df["省调负荷(MW)_日前"].replace(0, np.nan)).fillna(0)
    return df


def select_feature_cols(df: pd.DataFrame, target_col: str = TARGET_COL) -> List[str]:
    """day-ahead 合法特征列：剔除目标、_实际、泄漏列、非数值列。"""
    blacklist = {target_col, "datetime", "日期", "时间", "时段",
                 "来源文件", "来源文件_供需", "_y", "_m"} | LEAKAGE_COLS
    cols = []
    for c in df.columns:
        if c in blacklist:
            continue
        if "_实际" in c:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def prepare_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """完整特征工程流水线：15min → 小时聚合 → 特征 → dropna(lag NaN)。"""
    df_hour = aggregate_hourly(df_clean)
    df_feat = build_features(df_hour)
    feat_cols = select_feature_cols(df_feat)
    # Bug 修复：差分特征 target_yest_vs_lastweek 依赖 shift(168) 也会产生 NaN，
    # 显式纳入 dropna 集合，避免依赖 lag_168h "顺带" 兜底。
    nan_dep_cols = [c for c in df_feat.columns
                    if c.startswith("target_lag_")
                    or c.startswith("target_roll_")
                    or c.startswith("target_yest_")
                    or c.startswith("target_diff_")]
    df_feat = df_feat.dropna(subset=nan_dep_cols).reset_index(drop=True)
    return df_feat, feat_cols


# ---------------------------------------------------------------------------
# 6. 评估指标
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE / MSE / RMSE / MAPE (y>10 过滤防除 0 爆炸)。"""
    err = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mask = np.asarray(y_true) > 10
    mape = (float(np.mean(np.abs(err[mask] / np.asarray(y_true)[mask]))) * 100
            if mask.sum() > 0 else float("nan"))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
