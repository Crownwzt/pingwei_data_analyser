#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无天气特征的训练集扩充实验（完整版）
======================================

目标：
对比当前方案（含天气 + 7056h）vs 无天气多年扩充方案

考虑因素：
1. 天气特征的实际可用性（2026年无真实天气数据）
2. lag特征在年边界的损失（2025年前7天丢失）
3. 真实可用的训练集规模

实验配置：
- A: 当前方案（含天气59特征，2025-01-08~10-31 实际约6778h）
- B: 无天气扩充（51特征，2025-01-08~10-31 + 2026-01-01~06-30）

可还原：
- 不修改src/目录下的代码
- 所有结果保存到 experiments/
- 修改已备份（.backup文件）
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from common import load_clean, prepare_features, DA_COL, TARGET_COL, LEAKAGE_COLS
from training import fit_residual_ensemble, fit_alpha, evaluate_split

OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("无天气特征的训练集扩充实验")
print("="*70)

# ========== 1. 加载数据 ==========
print("\n[1/5] 加载数据...")
df_clean = load_clean()
print(f"  清洗后: {len(df_clean)} 行")
by_year = df_clean.groupby(df_clean['datetime'].dt.year).size()
for year, cnt in by_year.items():
    print(f"    {year}年: {cnt:,} 行")

# ========== 2. 特征工程 ==========
print("\n[2/5] 特征工程...")
df_feat, all_feats = prepare_features(df_clean)
print(f"  特征工程后: {len(df_feat)} 行（损失{len(df_clean)-len(df_feat)}行，{(len(df_clean)-len(df_feat))/len(df_clean)*100:.1f}%）")

# 检查实际数据范围
df_2025 = df_feat[df_feat['datetime'].dt.year == 2025]
df_2026 = df_feat[df_feat['datetime'].dt.year == 2026]
print(f"\n  2025年: {len(df_2025)} 行")
print(f"    实际范围: {df_2025['datetime'].min()} ~ {df_2025['datetime'].max()}")
print(f"  2026年: {len(df_2026)} 行")
print(f"    实际范围: {df_2026['datetime'].min()} ~ {df_2026['datetime'].max()}")

# 识别天气特征
weather_kw = ['ghi', 'wind_speed', 't2m', 'tcc']
weather_feats = [f for f in all_feats if any(kw in f for kw in weather_kw)]
non_weather_feats = [f for f in all_feats if f not in weather_feats and f not in LEAKAGE_COLS]

print(f"\n  特征分类:")
print(f"    总特征: {len(all_feats)}")
print(f"    天气特征: {len(weather_feats)} 个")
print(f"    非天气特征: {len(non_weather_feats)} 个")

# ========== 3. 数据切分 ==========
print("\n[3/5] 数据切分...")
df_feat['year'] = df_feat['datetime'].dt.year
df_feat['month'] = df_feat['datetime'].dt.month

# 方案A: 含天气 + 2025单年
mask_train_a = (df_feat['year']==2025) & df_feat['month'].isin(range(1,11))
mask_val = (df_feat['year']==2025) & (df_feat['month']==11)
mask_test = (df_feat['year']==2025) & (df_feat['month']==12)

# 方案B: 无天气 + 多年
mask_train_b = ((df_feat['year']==2025) & df_feat['month'].isin(range(1,11))) | \
               ((df_feat['year']==2026) & df_feat['month'].isin(range(1,7)))

print(f"  方案A（含天气+2025）: train={mask_train_a.sum():,}h, val={mask_val.sum()}, test={mask_test.sum()}, feats={len(all_feats)}")
print(f"  方案B（无天气+多年）: train={mask_train_b.sum():,}h, val={mask_val.sum()}, test={mask_test.sum()}, feats={len(non_weather_feats)}")
print(f"  训练集增量: +{mask_train_b.sum()-mask_train_a.sum():,}h (+{(mask_train_b.sum()-mask_train_a.sum())/mask_train_a.sum()*100:.1f}%)")

# ========== 4. 训练模型 ==========
from collections import namedtuple
Splits = namedtuple('Splits', ['X_tr','y_tr','dt_tr','X_va','y_va','dt_va','X_te','y_te','dt_te'])

def make_splits(df, feats, tr_mask, va_mask, te_mask):
    return Splits(
        df.loc[tr_mask, feats], df.loc[tr_mask, TARGET_COL], df.loc[tr_mask, 'datetime'],
        df.loc[va_mask, feats], df.loc[va_mask, TARGET_COL], df.loc[va_mask, 'datetime'],
        df.loc[te_mask, feats], df.loc[te_mask, TARGET_COL], df.loc[te_mask, 'datetime'])

seeds = [42, 7, 137, 2024, 9527]

# 方案A
print("\n[4/5] 训练方案A（含天气+2025）...")
splits_a = make_splits(df_feat, all_feats, mask_train_a, mask_val, mask_test)
models_a = fit_residual_ensemble(splits_a, seeds=seeds)
alpha_a, alpha_curve_a = fit_alpha(models_a, splits_a)
m_a_train = evaluate_split(models_a, alpha_a, splits_a.X_tr, splits_a.y_tr, splits_a.X_tr[DA_COL].values)
m_a_val = evaluate_split(models_a, alpha_a, splits_a.X_va, splits_a.y_va, splits_a.X_va[DA_COL].values)
m_a_test = evaluate_split(models_a, alpha_a, splits_a.X_te, splits_a.y_te, splits_a.X_te[DA_COL].values)
print(f"  训练: MAE={m_a_train['xgb_mae']:.2f}, gain={m_a_train['mae_gain%']:+.2f}%")
print(f"  验证: MAE={m_a_val['xgb_mae']:.2f}, gain={m_a_val['mae_gain%']:+.2f}%")
print(f"  测试: MAE={m_a_test['xgb_mae']:.2f}, gain={m_a_test['mae_gain%']:+.2f}%, α*={alpha_a:.2f}")

# 方案B
print("\n[4/5] 训练方案B（无天气+多年）...")
splits_b = make_splits(df_feat, non_weather_feats, mask_train_b, mask_val, mask_test)
models_b = fit_residual_ensemble(splits_b, seeds=seeds)
alpha_b, alpha_curve_b = fit_alpha(models_b, splits_b)
m_b_train = evaluate_split(models_b, alpha_b, splits_b.X_tr, splits_b.y_tr, splits_b.X_tr[DA_COL].values)
m_b_val = evaluate_split(models_b, alpha_b, splits_b.X_va, splits_b.y_va, splits_b.X_va[DA_COL].values)
m_b_test = evaluate_split(models_b, alpha_b, splits_b.X_te, splits_b.y_te, splits_b.X_te[DA_COL].values)
print(f"  训练: MAE={m_b_train['xgb_mae']:.2f}, gain={m_b_train['mae_gain%']:+.2f}%")
print(f"  验证: MAE={m_b_val['xgb_mae']:.2f}, gain={m_b_val['mae_gain%']:+.2f}%")
print(f"  测试: MAE={m_b_test['xgb_mae']:.2f}, gain={m_b_test['mae_gain%']:+.2f}%, α*={alpha_b:.2f}")

# ========== 5. 保存结果 ==========
print("\n[5/5] 保存结果...")
results = {
    'A_with_weather_2025': {
        'train_size': len(splits_a.X_tr),
        'n_features': len(all_feats),
        'metrics_train': m_a_train,
        'metrics_val': m_a_val,
        'metrics_test': m_a_test,
        'alpha': alpha_a,
        'models': models_a,
    },
    'B_no_weather_multiyear': {
        'train_size': len(splits_b.X_tr),
        'n_features': len(non_weather_feats),
        'metrics_train': m_b_train,
        'metrics_val': m_b_val,
        'metrics_test': m_b_test,
        'alpha': alpha_b,
        'models': models_b,
    },
}

joblib.dump(results, OUT_DIR / "no_weather_expand_results.pkl")
print(f"  已保存: {OUT_DIR}/no_weather_expand_results.pkl")

# ========== 6. 输出对比 ==========
print("\n" + "="*70)
print("实验结果对比")
print("="*70)
print(f"{'方案':<30s} {'训练集':<12s} {'特征':<8s} {'测试MAE':<12s} {'vs B7%':<10s} {'α*':<8s}")
print("-"*70)
print(f"{'A 含天气+2025':<30s} {len(splits_a.X_tr):<12,d} {len(all_feats):<8d} "
      f"{m_a_test['xgb_mae']:<12.2f} +{m_a_test['mae_gain%']:<9.2f} {alpha_a:<8.2f}")
print(f"{'B 无天气+多年':<30s} {len(splits_b.X_tr):<12,d} {len(non_weather_feats):<8d} "
      f"{m_b_test['xgb_mae']:<12.2f} +{m_b_test['mae_gain%']:<9.2f} {alpha_b:<8.2f}")

delta_mae = m_b_test['xgb_mae'] - m_a_test['xgb_mae']
delta_n = len(splits_b.X_tr) - len(splits_a.X_tr)
print("="*70)
print(f"训练集增量: +{delta_n:,}h (+{delta_n/len(splits_a.X_tr)*100:.1f}%)")
print(f"测试MAE变化: {delta_mae:+.2f} 元")
print(f"验证MAE变化: {m_b_val['xgb_mae'] - m_a_val['xgb_mae']:+.2f} 元")
print()

if delta_mae < -0.2:
    print(f"✓ 方案B更优，降低 {-delta_mae:.2f} 元")
elif delta_mae > 0.2:
    print(f"✗ 方案B更差，升高 {delta_mae:.2f} 元")
else:
    print(f"→ 两方案接近（差异 {delta_mae:+.2f} 元）")

print("\n实验完成。")
