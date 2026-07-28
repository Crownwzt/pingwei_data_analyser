#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN 天气特征 vs 原始天气 F组对比实验 v2
============================================
核心修正：
之前 GCN 直接用当前时刻的 32 维 embedding，而原始 F 组是
{ghi,wind_speed,t2m,tcc}_{lag1d,diff1d}（滞后差分特征）。
本实验对 GCN embedding 也做相同的滞后差分处理，公平对比。

对比方案：
- A: 无天气（51 特征）
- B: 原始 F 组（8 维 = 4 变量 × {lag1d, diff1d}）→ 59 特征
- C1: GCN 当前值（32 维原始）→ 83 特征
- C2: GCN 滞后（32 维 lag24h）→ 83 特征
- C3: GCN 滞后差分（64 维 = 32 lag + 32 diff）→ 115 特征
- C4: GCN 三形态（96 维 = 32 current + 32 lag + 32 diff）→ 147 特征
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from common import load_clean, prepare_features, TARGET_COL, DA_COL, Splits
from training import fit_residual_ensemble, fit_alpha, evaluate_split

OUT_DIR = Path("experiments")

print("="*70)
print("GCN 天气对比实验 v2（含滞后差分特征工程）")
print("="*70)

# ========== 1. 主流程数据 ==========
print("\n[1/6] 加载主流程数据...")
df_clean = load_clean()
df_feat, all_feats = prepare_features(df_clean)

weather_kw = ['ghi', 'wind_speed', 't2m', 'tcc']
weather_feats_F = [f for f in all_feats if any(kw in f for kw in weather_kw)]
non_weather_feats = [f for f in all_feats if f not in weather_feats_F]
print(f"  主流程: {len(all_feats)} 特征")
print(f"  天气 F 组: {len(weather_feats_F)}，非天气: {len(non_weather_feats)}")

# ========== 2. 加载 GCN + 做 4 种特征工程 ==========
print("\n[2/6] 加载 GCN + 特征工程...")
df_gcn = pd.read_csv('外部数据/feature-weather_timeseries_features_202607211656.csv')
df_gcn['datetime'] = pd.to_datetime(df_gcn['feature_time']).dt.tz_localize(None)

def parse_emb(s):
    return [float(x) for x in s.strip('{}').split(',')]

df_gcn['embedding_list'] = df_gcn['embedding'].apply(parse_emb)
emb_current_cols = [f"gcn_current_{i}" for i in range(32)]
df_gcn[emb_current_cols] = pd.DataFrame(df_gcn['embedding_list'].tolist(), index=df_gcn.index)
df_gcn = df_gcn[['datetime'] + emb_current_cols].sort_values('datetime').reset_index(drop=True)

# 小时聚合（4 个 15min 取均值）
df_gcn['datetime_h'] = df_gcn['datetime'].dt.floor('h')
df_gcn_h = df_gcn.groupby('datetime_h')[emb_current_cols].mean().reset_index()
df_gcn_h.rename(columns={'datetime_h': 'datetime'}, inplace=True)
df_gcn_h = df_gcn_h.sort_values('datetime').reset_index(drop=True)
print(f"  GCN 小时聚合: {len(df_gcn_h)} 行")

# 特征工程 1：lag24h（昨日同时刻）
emb_lag_cols = [f"gcn_lag1d_{i}" for i in range(32)]
for i, col in enumerate(emb_current_cols):
    df_gcn_h[emb_lag_cols[i]] = df_gcn_h[col].shift(24)

# 特征工程 2：diff1d（今日 - 昨日）
emb_diff_cols = [f"gcn_diff1d_{i}" for i in range(32)]
for i, col in enumerate(emb_current_cols):
    df_gcn_h[emb_diff_cols[i]] = df_gcn_h[col] - df_gcn_h[col].shift(24)

print(f"  GCN 特征扩展: current(32) + lag1d(32) + diff1d(32) = 96 维")

# ========== 3. 合并到主流程特征表 ==========
print("\n[3/6] 合并到主流程特征表...")
df_all = df_feat.merge(df_gcn_h, on='datetime', how='left')
print(f"  合并后: {len(df_all)} 行")

# 处理缺失（GCN 8月缺失几天 + lag24h 头 24h 也缺）
all_gcn_cols = emb_current_cols + emb_lag_cols + emb_diff_cols
missing_before = df_all[emb_current_cols[0]].isna().sum()
print(f"  GCN current 缺失: {missing_before} 行")

# 对 current 做前向后向填充（补 8 月缺失）
for c in emb_current_cols:
    df_all[c] = df_all[c].ffill().bfill()
# 对 lag/diff 也一样
for c in emb_lag_cols + emb_diff_cols:
    df_all[c] = df_all[c].ffill().bfill()

# ========== 4. 数据切分 ==========
print("\n[4/6] 数据切分...")
df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month

mask_train = (df_all['year']==2025) & df_all['month'].isin(range(1,11))
mask_val   = (df_all['year']==2025) & (df_all['month']==11)
mask_test  = (df_all['year']==2025) & (df_all['month']==12)

print(f"  训练/验证/测试: {mask_train.sum()}/{mask_val.sum()}/{mask_test.sum()}")

# ========== 5. 训练 6 个方案 ==========
def make_splits(feats):
    return Splits(
        X_tr=df_all.loc[mask_train, feats], y_tr=df_all.loc[mask_train, TARGET_COL], dt_tr=df_all.loc[mask_train, 'datetime'],
        X_va=df_all.loc[mask_val, feats],   y_va=df_all.loc[mask_val, TARGET_COL],   dt_va=df_all.loc[mask_val, 'datetime'],
        X_te=df_all.loc[mask_test, feats],  y_te=df_all.loc[mask_test, TARGET_COL],  dt_te=df_all.loc[mask_test, 'datetime'],
        feature_cols=feats,
    )

seeds = [42, 7, 137, 2024, 9527]

def train_and_eval(name, feats):
    print(f"\n{'='*70}\n[5/6] 训练方案 {name} ({len(feats)} 特征)\n{'='*70}")
    splits = make_splits(feats)
    models = fit_residual_ensemble(splits, seeds=seeds)
    alpha, _ = fit_alpha(models, splits)
    m = evaluate_split(models, alpha, splits.X_te, splits.y_te, splits.X_te[DA_COL].values)
    print(f"  test: MAE={m['xgb_mae']:.2f}, gain={m['mae_gain%']:+.2f}%, α*={alpha:.2f}")
    return {'name': name, 'metrics_test': m, 'alpha': alpha,
            'n_features': len(feats), 'models': models}

results = {}
results['A'] = train_and_eval('A: 无天气', non_weather_feats)
results['B'] = train_and_eval('B: 原始 F组 (lag+diff, 8维)', all_feats)
results['C1'] = train_and_eval('C1: GCN 当前 (32维)', non_weather_feats + emb_current_cols)
results['C2'] = train_and_eval('C2: GCN lag1d (32维)', non_weather_feats + emb_lag_cols)
results['C3'] = train_and_eval('C3: GCN lag+diff (64维)', non_weather_feats + emb_lag_cols + emb_diff_cols)
results['C4'] = train_and_eval('C4: GCN 三形态 (96维)', non_weather_feats + emb_current_cols + emb_lag_cols + emb_diff_cols)

# ========== 6. 保存与对比 ==========
print("\n[6/6] 保存结果...")
joblib.dump(results, OUT_DIR / "gcn_weather_comparison_v2.pkl")

print("\n" + "="*70)
print("对比结果")
print("="*70)
print(f"{'方案':<32s} {'特征数':<8s} {'测试MAE':<10s} {'vs B7%':<10s} {'vs A':<10s} {'α*':<6s}")
print("-"*80)

mae_a = results['A']['metrics_test']['xgb_mae']
for k, r in results.items():
    m = r['metrics_test']
    delta = m['xgb_mae'] - mae_a
    print(f"{r['name']:<32s} {r['n_features']:<8d} {m['xgb_mae']:<10.2f} +{m['mae_gain%']:<9.2f} {delta:+.2f}      {r['alpha']:<6.2f}")

print("\n" + "="*70)
print("增益分解")
print("="*70)
maes = {k: results[k]['metrics_test']['xgb_mae'] for k in results}
print(f"A → B（原始 F组 lag+diff）: {maes['B']-maes['A']:+.2f} 元")
print(f"A → C1（GCN 当前）:          {maes['C1']-maes['A']:+.2f} 元")
print(f"A → C2（GCN lag1d）:         {maes['C2']-maes['A']:+.2f} 元")
print(f"A → C3(GCN lag+diff):       {maes['C3']-maes['A']:+.2f} 元")
print(f"A → C4(GCN 三形态):         {maes['C4']-maes['A']:+.2f} 元")

best = min(results.items(), key=lambda x: x[1]['metrics_test']['xgb_mae'])
print(f"\n最优方案: {best[1]['name']}, MAE = {best[1]['metrics_test']['xgb_mae']:.2f}")
print("\n实验完成。")
