#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN 天气特征 vs 原始天气 F组对比实验
==================================

对比方案（全部与主流程 src/ 一致，只改天气特征）：
- A: 无天气特征（51 特征 = 59 - 8 天气）
- B: 原始天气 F组（59 特征）★ 当前生产
- C: GCN 天气 32 维（51 非天气 + 32 GCN = 83 特征）

固定：
- 训练集 2025-01~10, 验证 11, 测试 12
- 5 seed ensemble + α 融合
- 完整特征工程（含煤价、周期编码、lag、rolling 等）
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from common import load_clean, prepare_features, TARGET_COL, DA_COL, LEAKAGE_COLS, Splits
from training import fit_residual_ensemble, fit_alpha, evaluate_split

OUT_DIR = Path("experiments")

print("="*70)
print("GCN 天气特征 vs 原始天气 F组对比实验")
print("="*70)

# ========== 1. 主流程数据（含所有 59 特征）==========
print("\n[1/6] 加载 + 特征工程（含煤价+原始天气 F组）...")
df_clean = load_clean()
df_feat, all_feats = prepare_features(df_clean)
print(f"  主流程特征: {len(all_feats)} 个（含煤价 4 + 天气 F组 8）")

# 分离天气特征
weather_kw = ['ghi', 'wind_speed', 't2m', 'tcc']
weather_feats_F = [f for f in all_feats if any(kw in f for kw in weather_kw)]
non_weather_feats = [f for f in all_feats if f not in weather_feats_F]
print(f"  天气 F组特征（8 维）: {len(weather_feats_F)} 个")
print(f"  非天气特征（含煤价）: {len(non_weather_feats)} 个")

# ========== 2. 加载 GCN 天气特征 ==========
print("\n[2/6] 加载 GCN 天气特征...")
df_gcn = pd.read_csv('外部数据/feature-weather_timeseries_features_202607211656.csv')
df_gcn['datetime'] = pd.to_datetime(df_gcn['feature_time']).dt.tz_localize(None)

def parse_emb(s):
    s = s.strip('{}')
    return [float(x) for x in s.split(',')]

df_gcn['embedding_list'] = df_gcn['embedding'].apply(parse_emb)
emb_cols = [f"weather_gcn_emb_{i}" for i in range(32)]
df_gcn[emb_cols] = pd.DataFrame(df_gcn['embedding_list'].tolist(), index=df_gcn.index)
df_gcn = df_gcn[['datetime'] + emb_cols]
print(f"  GCN 特征: {len(df_gcn)} 行, 32 维")

# ========== 3. 小时聚合 GCN（与主流程一致）==========
print("\n[3/6] 合并 GCN 到主流程特征表...")
# 主流程 df_feat 是小时粒度（prepare_features 内部已 floor('h')）
df_gcn['datetime_h'] = df_gcn['datetime'].dt.floor('h')
df_gcn_hourly = df_gcn.groupby('datetime_h')[emb_cols].mean().reset_index()
df_gcn_hourly.rename(columns={'datetime_h': 'datetime'}, inplace=True)

# 用 datetime 作为 key merge
df_feat_c = df_feat.merge(df_gcn_hourly, on='datetime', how='left')

# 检查缺失
gcn_missing = df_feat_c[emb_cols[0]].isna().sum()
print(f"  GCN 缺失: {gcn_missing} 行（超出GCN时间范围）")

# 用 forward fill 处理少量缺失，避免损失数据
for c in emb_cols:
    df_feat_c[c] = df_feat_c[c].ffill().bfill()

# 验证：确保数据长度一致
assert len(df_feat) == len(df_feat_c), "合并后行数变化"
print(f"  合并后: {len(df_feat_c)} 行（与主流程一致）")

# ========== 4. 数据切分 ==========
print("\n[4/6] 数据切分...")
df_feat['year'] = df_feat['datetime'].dt.year
df_feat['month'] = df_feat['datetime'].dt.month
df_feat_c['year'] = df_feat_c['datetime'].dt.year
df_feat_c['month'] = df_feat_c['datetime'].dt.month

mask_train = (df_feat['year']==2025) & df_feat['month'].isin(range(1,11))
mask_val = (df_feat['year']==2025) & (df_feat['month']==11)
mask_test = (df_feat['year']==2025) & (df_feat['month']==12)

mask_train_c = (df_feat_c['year']==2025) & df_feat_c['month'].isin(range(1,11))
mask_val_c = (df_feat_c['year']==2025) & (df_feat_c['month']==11)
mask_test_c = (df_feat_c['year']==2025) & (df_feat_c['month']==12)

print(f"  训练/验证/测试: {mask_train.sum()}h / {mask_val.sum()}h / {mask_test.sum()}h")

# ========== 5. 训练三个方案 ==========
feats_A = non_weather_feats  # 无天气
feats_B = all_feats  # 原始 F组（主流程当前）
feats_C = non_weather_feats + emb_cols  # GCN

print(f"\n方案特征数:")
print(f"  A（无天气）: {len(feats_A)}")
print(f"  B（原始 F组）: {len(feats_B)}")
print(f"  C（GCN 32维）: {len(feats_C)}")

def make_splits(df, feats, tr, va, te):
    return Splits(
        X_tr=df.loc[tr, feats], y_tr=df.loc[tr, TARGET_COL], dt_tr=df.loc[tr, 'datetime'],
        X_va=df.loc[va, feats], y_va=df.loc[va, TARGET_COL], dt_va=df.loc[va, 'datetime'],
        X_te=df.loc[te, feats], y_te=df.loc[te, TARGET_COL], dt_te=df.loc[te, 'datetime'],
        feature_cols=feats,
    )

seeds = [42, 7, 137, 2024, 9527]

def train_one(name, df, feats, tr, va, te):
    print(f"\n[5/6] 训练方案 {name}...")
    splits = make_splits(df, feats, tr, va, te)
    models = fit_residual_ensemble(splits, seeds=seeds)
    alpha, _ = fit_alpha(models, splits)
    m = evaluate_split(models, alpha, splits.X_te, splits.y_te, splits.X_te[DA_COL].values)
    print(f"  test: MAE={m['xgb_mae']:.2f}, gain={m['mae_gain%']:+.2f}%, α*={alpha:.2f}")
    return {'metrics_test': m, 'alpha': alpha, 'models': models,
            'n_features': len(feats), 'n_train': len(splits.X_tr)}

results = {}
results['A_no_weather'] = train_one('A（无天气）', df_feat, feats_A, mask_train, mask_val, mask_test)
results['B_original_F'] = train_one('B（原始天气 F组）', df_feat, feats_B, mask_train, mask_val, mask_test)
results['C_gcn'] = train_one('C（GCN 32维）', df_feat_c, feats_C, mask_train_c, mask_val_c, mask_test_c)

# ========== 6. 保存与对比 ==========
print("\n[6/6] 保存结果...")
joblib.dump(results, OUT_DIR / "gcn_weather_comparison.pkl")

m_a = results['A_no_weather']['metrics_test']
m_b = results['B_original_F']['metrics_test']
m_c = results['C_gcn']['metrics_test']

print("\n" + "="*70)
print("对比结果")
print("="*70)
print(f"{'方案':<25s} {'特征数':<8s} {'训练集':<10s} {'测试MAE':<10s} {'vs B7%':<10s} {'α*':<6s}")
print("-"*70)
for name, r in [
    ('A 无天气', results['A_no_weather']),
    ('B 原始天气 F组 (8维)', results['B_original_F']),
    ('C GCN 天气 (32维)', results['C_gcn']),
]:
    m = r['metrics_test']
    print(f"{name:<25s} {r['n_features']:<8d} {r['n_train']:<10,d} "
          f"{m['xgb_mae']:<10.2f} +{m['mae_gain%']:<9.2f} {r['alpha']:<6.2f}")

print("\n" + "="*70)
print("增益分解")
print("="*70)

delta_ab = m_b['xgb_mae'] - m_a['xgb_mae']
delta_ac = m_c['xgb_mae'] - m_a['xgb_mae']
delta_bc = m_c['xgb_mae'] - m_b['xgb_mae']

print(f"原始天气 F组贡献（A→B）: {delta_ab:+.2f} 元")
print(f"GCN 天气贡献（A→C）:     {delta_ac:+.2f} 元")
print(f"GCN vs 原始 F组（B→C）:  {delta_bc:+.2f} 元")

print(f"\n对照主流程当前生产:")
print(f"  含天气 F组（主流程）: MAE = 36.79（bash run.sh --all 的结果）")
print(f"  本实验 B（对齐）:     MAE = {m_b['xgb_mae']:.2f}")

# 结论
print("\n" + "="*70)
print("结论")
print("="*70)
best_name, best_mae = min([('A', m_a['xgb_mae']), ('B', m_b['xgb_mae']), ('C', m_c['xgb_mae'])], key=lambda x: x[1])
print(f"  最优方案: {best_name}, MAE = {best_mae:.2f}")

if delta_bc < -0.1:
    print(f"  ✓ GCN 优于原始 F组，降低 {-delta_bc:.2f} 元")
elif delta_bc > 0.1:
    print(f"  ✗ 原始 F组 优于 GCN，GCN 反而增加 {delta_bc:.2f} 元")
else:
    print(f"  → GCN 和原始 F组 接近（差异 <0.1元）")

print("\n实验完成。")
