#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN 天气特征对比 v3 - 深度诊断
==================================
基于 v2 的发现（GCN L2 norm 严重不平稳，min=0.048 max=72.8）：
1. 对 GCN embedding 做标准化 (StandardScaler)
2. 试 GCN + F 组共存（互补测试）
3. 看 XGB 特征重要性中 GCN 32 维的地位

对比方案：
- A: 无天气
- B: F 组 (原始 8 维) ★ 当前生产
- C: GCN raw (32 维)
- D: GCN 标准化 (32 维)
- E: F 组 + GCN 标准化 (8 + 32 = 40 维)
- F: F 组 + GCN 标准化 lag+diff (8 + 64 = 72 维)
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from common import load_clean, prepare_features, TARGET_COL, DA_COL, Splits
from training import fit_residual_ensemble, fit_alpha, evaluate_split

OUT_DIR = Path("experiments")

print("="*70)
print("GCN v3 - 标准化 + 组合方案")
print("="*70)

# 主流程
df_clean = load_clean()
df_feat, all_feats = prepare_features(df_clean)
weather_kw = ['ghi', 'wind_speed', 't2m', 'tcc']
F_feats = [f for f in all_feats if any(kw in f for kw in weather_kw)]
non_weather = [f for f in all_feats if f not in F_feats]

# GCN
df_gcn = pd.read_csv('外部数据/feature-weather_timeseries_features_202607211656.csv')
df_gcn['datetime'] = pd.to_datetime(df_gcn['feature_time']).dt.tz_localize(None)
def parse_emb(s):
    return [float(x) for x in s.strip('{}').split(',')]
df_gcn['emb'] = df_gcn['embedding'].apply(parse_emb)

# 小时聚合
gcn_raw_cols = [f"gcn_raw_{i}" for i in range(32)]
emb_mat = np.array(df_gcn['emb'].tolist())
df_gcn_expanded = pd.DataFrame(emb_mat, columns=gcn_raw_cols)
df_gcn_expanded['datetime'] = df_gcn['datetime'].dt.floor('h').values
df_gcn_h = df_gcn_expanded.groupby('datetime')[gcn_raw_cols].mean().reset_index()
df_gcn_h = df_gcn_h.sort_values('datetime').reset_index(drop=True)

# 标准化
scaler = StandardScaler()
gcn_std_cols = [f"gcn_std_{i}" for i in range(32)]
df_gcn_h[gcn_std_cols] = scaler.fit_transform(df_gcn_h[gcn_raw_cols].values)

# lag24h + diff24h（标准化后）
gcn_lag_cols = [f"gcn_std_lag_{i}" for i in range(32)]
gcn_diff_cols = [f"gcn_std_diff_{i}" for i in range(32)]
for i in range(32):
    df_gcn_h[gcn_lag_cols[i]] = df_gcn_h[gcn_std_cols[i]].shift(24)
    df_gcn_h[gcn_diff_cols[i]] = df_gcn_h[gcn_std_cols[i]] - df_gcn_h[gcn_std_cols[i]].shift(24)

# 合并
df_all = df_feat.merge(df_gcn_h, on='datetime', how='left')
# 前后向填充
all_gcn = gcn_raw_cols + gcn_std_cols + gcn_lag_cols + gcn_diff_cols
for c in all_gcn:
    df_all[c] = df_all[c].ffill().bfill()

df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month
mask_train = (df_all['year']==2025) & df_all['month'].isin(range(1,11))
mask_val = (df_all['year']==2025) & (df_all['month']==11)
mask_test = (df_all['year']==2025) & (df_all['month']==12)
print(f"训练/验证/测试: {mask_train.sum()}/{mask_val.sum()}/{mask_test.sum()}")

def make_splits(feats):
    return Splits(
        X_tr=df_all.loc[mask_train, feats], y_tr=df_all.loc[mask_train, TARGET_COL], dt_tr=df_all.loc[mask_train, 'datetime'],
        X_va=df_all.loc[mask_val, feats],   y_va=df_all.loc[mask_val, TARGET_COL],   dt_va=df_all.loc[mask_val, 'datetime'],
        X_te=df_all.loc[mask_test, feats],  y_te=df_all.loc[mask_test, TARGET_COL],  dt_te=df_all.loc[mask_test, 'datetime'],
        feature_cols=feats,
    )

seeds = [42, 7, 137, 2024, 9527]

def run(name, feats):
    print(f"\n{'='*70}\n{name} ({len(feats)} 特征)\n{'='*70}")
    splits = make_splits(feats)
    models = fit_residual_ensemble(splits, seeds=seeds)
    alpha, _ = fit_alpha(models, splits)
    m = evaluate_split(models, alpha, splits.X_te, splits.y_te, splits.X_te[DA_COL].values)
    print(f"→ test MAE={m['xgb_mae']:.2f}, gain={m['mae_gain%']:+.2f}%, α*={alpha:.2f}")
    return {'name': name, 'metrics_test': m, 'alpha': alpha,
            'n_features': len(feats), 'models': models, 'feat_cols': feats}

results = {}
results['A'] = run('A: 无天气', non_weather)
results['B'] = run('B: F 组原版 (8)', all_feats)
results['C'] = run('C: GCN raw (32)', non_weather + gcn_raw_cols)
results['D'] = run('D: GCN 标准化 (32)', non_weather + gcn_std_cols)
results['E'] = run('E: F 组 + GCN std (40)', all_feats + gcn_std_cols)
results['F'] = run('F: F 组 + GCN std lag+diff (72)', all_feats + gcn_lag_cols + gcn_diff_cols)

joblib.dump(results, OUT_DIR / "gcn_weather_v3.pkl")

print("\n" + "="*70)
print("对比结果汇总")
print("="*70)
print(f"{'方案':<40s} {'#feat':<7s} {'测试MAE':<10s} {'vs A':<10s}")
print("-"*80)
mae_a = results['A']['metrics_test']['xgb_mae']
for k in ['A','B','C','D','E','F']:
    r = results[k]
    m = r['metrics_test']['xgb_mae']
    delta = m - mae_a
    print(f"{r['name']:<40s} {r['n_features']:<7d} {m:<10.2f} {delta:+.2f}")

# ========== 特征重要性分析 ==========
print("\n" + "="*70)
print("特征重要性对比: 方案 E (F 组 + GCN std) 中 GCN 排名")
print("="*70)
model_e = results['E']['models'][0]  # seed=42
imp = pd.Series(model_e.feature_importances_, index=results['E']['feat_cols'])
imp_sorted = imp.sort_values(ascending=False)
print("\nTop 20 特征:")
for i, (feat, val) in enumerate(imp_sorted.head(20).items(), 1):
    is_gcn = 'gcn' in feat.lower()
    is_F = any(kw in feat for kw in weather_kw)
    tag = '[GCN]' if is_gcn else ('[F组]' if is_F else '')
    print(f"  {i:2d}. {feat:<40s}: {val:.4f}  {tag}")

gcn_ranks = [i for i, feat in enumerate(imp_sorted.index, 1) if 'gcn' in feat.lower()]
F_ranks = [i for i, feat in enumerate(imp_sorted.index, 1) if any(kw in feat for kw in weather_kw)]
print(f"\nF 组特征排名: 平均 {np.mean(F_ranks):.1f}，最高 {min(F_ranks)}")
print(f"GCN 特征排名: 平均 {np.mean(gcn_ranks):.1f}，最高 {min(gcn_ranks)}")
print(f"→ 排名越靠前，重要性越高")

print("\n实验完成。")
