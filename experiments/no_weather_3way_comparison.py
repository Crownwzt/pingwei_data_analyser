#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无天气特征的训练集扩充实验（完整三方对比）
============================================

对比方案：
- A: 含天气 + 2025单年（7,056h, 59特征）
- B: 无天气 + 2025单年（7,056h, 51特征）
- C: 无天气 + 多年（11,400h, 51特征）

分解增益：
- A vs B：天气特征的贡献
- B vs C：数据扩充的贡献
- A vs C：综合改进

数据损失说明：
- 15min → 1h：粒度转换，正常操作
- lag特征损失：年边界和空值导致的真实损失
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

print("="*70)
print("无天气特征训练集扩充实验（三方对比）")
print("="*70)

# ========== 1. 加载数据 ==========
print("\n[1/5] 加载数据...")
df_clean = load_clean()
print(f"  清洗后（15min粒度）: {len(df_clean)} 行")
by_year = df_clean.groupby(df_clean['datetime'].dt.year).size()
for year, cnt in by_year.items():
    print(f"    {year}年: {cnt:,} 行")

expected_1h = len(df_clean) / 4
print(f"\n  预期1h粒度: {expected_1h:,.0f} 行（15min÷4）")

# ========== 2. 特征工程 ==========
print("\n[2/5] 特征工程...")
df_feat, all_feats = prepare_features(df_clean)
print(f"  特征工程后（1h粒度）: {len(df_feat)} 行")

# 真实损失分析
loss_total = len(df_clean) - len(df_feat)
loss_granularity = len(df_clean) - expected_1h
loss_real = expected_1h - len(df_feat)

print(f"\n  数据变化分析:")
print(f"    15min原始: {len(df_clean):,} 行")
print(f"    粒度转换: -{loss_granularity:,.0f} 行（15min→1h，{loss_granularity/len(df_clean)*100:.1f}%）")
print(f"    预期1h: {expected_1h:,.0f} 行")
print(f"    lag损失: -{loss_real:,.0f} 行（年边界+空值，{loss_real/expected_1h*100:.1f}%）")
print(f"    实际1h: {len(df_feat):,} 行")

# 检查实际数据范围
df_2025 = df_feat[df_feat['datetime'].dt.year == 2025]
df_2026 = df_feat[df_feat['datetime'].dt.year == 2026]
print(f"\n  实际数据范围:")
print(f"    2025: {len(df_2025):,}h, {df_2025['datetime'].min()} ~ {df_2025['datetime'].max()}")
print(f"    2026: {len(df_2026):,}h, {df_2026['datetime'].min()} ~ {df_2026['datetime'].max()}")

# 特征分类
weather_kw = ['ghi', 'wind_speed', 't2m', 'tcc']
weather_feats = [f for f in all_feats if any(kw in f for kw in weather_kw)]
non_weather_feats = [f for f in all_feats if f not in weather_feats and f not in LEAKAGE_COLS]

print(f"\n  特征分类:")
print(f"    总特征: {len(all_feats)}")
print(f"    天气特征: {len(weather_feats)}")
print(f"    非天气特征: {len(non_weather_feats)}")

# ========== 3. 数据切分 ==========
print("\n[3/5] 数据切分...")
df_feat['year'] = df_feat['datetime'].dt.year
df_feat['month'] = df_feat['datetime'].dt.month

mask_train_2025 = (df_feat['year']==2025) & df_feat['month'].isin(range(1,11))
mask_train_multi = ((df_feat['year']==2025) & df_feat['month'].isin(range(1,11))) | \
                   ((df_feat['year']==2026) & df_feat['month'].isin(range(1,7)))
mask_val = (df_feat['year']==2025) & (df_feat['month']==11)
mask_test = (df_feat['year']==2025) & (df_feat['month']==12)

print(f"  A（含天气+2025）: train={mask_train_2025.sum():,}h, feats={len(all_feats)}")
print(f"  B（无天气+2025）: train={mask_train_2025.sum():,}h, feats={len(non_weather_feats)}")
print(f"  C（无天气+多年）: train={mask_train_multi.sum():,}h, feats={len(non_weather_feats)}")
print(f"  验证/测试: val={mask_val.sum()}h, test={mask_test.sum()}h（统一）")

# ========== 4. 训练模型 ==========
from collections import namedtuple
Splits = namedtuple('Splits', ['X_tr','y_tr','dt_tr','X_va','y_va','dt_va','X_te','y_te','dt_te'])

def make_splits(df, feats, tr_mask, va_mask, te_mask):
    return Splits(
        df.loc[tr_mask, feats], df.loc[tr_mask, TARGET_COL], df.loc[tr_mask, 'datetime'],
        df.loc[va_mask, feats], df.loc[va_mask, TARGET_COL], df.loc[va_mask, 'datetime'],
        df.loc[te_mask, feats], df.loc[te_mask, TARGET_COL], df.loc[te_mask, 'datetime'])

seeds = [42, 7, 137, 2024, 9527]

# 方案A: 含天气+2025
print("\n[4/5] 训练方案A（含天气+2025）...")
splits_a = make_splits(df_feat, all_feats, mask_train_2025, mask_val, mask_test)
models_a = fit_residual_ensemble(splits_a, seeds=seeds)
alpha_a, _ = fit_alpha(models_a, splits_a)
m_a = evaluate_split(models_a, alpha_a, splits_a.X_te, splits_a.y_te, splits_a.X_te[DA_COL].values)
print(f"  test: MAE={m_a['xgb_mae']:.2f}, gain={m_a['mae_gain%']:+.2f}%, α*={alpha_a:.2f}")

# 方案B: 无天气+2025
print("\n[4/5] 训练方案B（无天气+2025）...")
splits_b = make_splits(df_feat, non_weather_feats, mask_train_2025, mask_val, mask_test)
models_b = fit_residual_ensemble(splits_b, seeds=seeds)
alpha_b, _ = fit_alpha(models_b, splits_b)
m_b = evaluate_split(models_b, alpha_b, splits_b.X_te, splits_b.y_te, splits_b.X_te[DA_COL].values)
print(f"  test: MAE={m_b['xgb_mae']:.2f}, gain={m_b['mae_gain%']:+.2f}%, α*={alpha_b:.2f}")

# 方案C: 无天气+多年
print("\n[4/5] 训练方案C（无天气+多年）...")
splits_c = make_splits(df_feat, non_weather_feats, mask_train_multi, mask_val, mask_test)
models_c = fit_residual_ensemble(splits_c, seeds=seeds)
alpha_c, _ = fit_alpha(models_c, splits_c)
m_c = evaluate_split(models_c, alpha_c, splits_c.X_te, splits_c.y_te, splits_c.X_te[DA_COL].values)
print(f"  test: MAE={m_c['xgb_mae']:.2f}, gain={m_c['mae_gain%']:+.2f}%, α*={alpha_c:.2f}")

# ========== 5. 保存结果 ==========
print("\n[5/5] 保存结果...")
results = {
    'A_with_weather_2025': {
        'train_size': len(splits_a.X_tr),
        'n_features': len(all_feats),
        'metrics_test': m_a,
        'alpha': alpha_a,
        'models': models_a,
    },
    'B_no_weather_2025': {
        'train_size': len(splits_b.X_tr),
        'n_features': len(non_weather_feats),
        'metrics_test': m_b,
        'alpha': alpha_b,
        'models': models_b,
    },
    'C_no_weather_multiyear': {
        'train_size': len(splits_c.X_tr),
        'n_features': len(non_weather_feats),
        'metrics_test': m_c,
        'alpha': alpha_c,
        'models': models_c,
    },
}

joblib.dump(results, OUT_DIR / "no_weather_expand_3way.pkl")
print(f"  已保存: {OUT_DIR}/no_weather_expand_3way.pkl")

# ========== 6. 三方对比 ==========
print("\n" + "="*70)
print("三方对比结果")
print("="*70)
print(f"{'方案':<25s} {'训练集':<10s} {'特征':<6s} {'测试MAE':<10s} {'vs B7%':<10s} {'α*':<6s}")
print("-"*70)
print(f"{'A 含天气+2025':<25s} {len(splits_a.X_tr):<10,d} {len(all_feats):<6d} "
      f"{m_a['xgb_mae']:<10.2f} +{m_a['mae_gain%']:<9.2f} {alpha_a:<6.2f}")
print(f"{'B 无天气+2025':<25s} {len(splits_b.X_tr):<10,d} {len(non_weather_feats):<6d} "
      f"{m_b['xgb_mae']:<10.2f} +{m_b['mae_gain%']:<9.2f} {alpha_b:<6.2f}")
print(f"{'C 无天气+多年':<25s} {len(splits_c.X_tr):<10,d} {len(non_weather_feats):<6d} "
      f"{m_c['xgb_mae']:<10.2f} +{m_c['mae_gain%']:<9.2f} {alpha_c:<6.2f}")

print("\n" + "="*70)
print("增益分解")
print("="*70)
delta_ab = m_b['xgb_mae'] - m_a['xgb_mae']
delta_bc = m_c['xgb_mae'] - m_b['xgb_mae']
delta_ac = m_c['xgb_mae'] - m_a['xgb_mae']

print(f"天气特征贡献（A→B）: {delta_ab:+.2f} 元")
print(f"  去掉8个天气特征，MAE变化 {delta_ab:+.2f} 元")
if abs(delta_ab) < 0.2:
    print(f"  → 天气特征贡献微小（<0.2元）")
else:
    print(f"  → 天气特征{'有价值' if delta_ab < 0 else '贡献小'}")

print(f"\n数据扩充收益（B→C）: {delta_bc:+.2f} 元")
print(f"  增加 {len(splits_c.X_tr)-len(splits_b.X_tr):,}h 训练数据（+{(len(splits_c.X_tr)-len(splits_b.X_tr))/len(splits_b.X_tr)*100:.1f}%）")
if delta_bc < -0.2:
    print(f"  ✓ 扩充有效，降低 {-delta_bc:.2f} 元")
else:
    print(f"  → 扩充效果有限")

print(f"\n综合效果（A→C）: {delta_ac:+.2f} 元")
if delta_ac < -0.2:
    print(f"  ✓ 方案C最优，降低 {-delta_ac:.2f} 元")
elif delta_ac > 0.2:
    print(f"  ✗ 方案A更优")
else:
    print(f"  → A和C接近")

print("\n实验完成。")
