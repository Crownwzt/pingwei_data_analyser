# -*- coding: utf-8 -*-
"""
独立读取 weather_features.nc 的脚本（快速版，懒加载）
======================================================

用法:
    python read_weather_nc.py                    # 查看基本信息（秒开）
    python read_weather_nc.py --preview          # 查看前 10 行数据
    python read_weather_nc.py --stats            # 数值列统计
    python read_weather_nc.py --to-csv out.csv   # 导出到 CSV
    python read_weather_nc.py --col ghi          # 只读某一列

依赖: xarray, netCDF4
"""

import argparse
import sys
from pathlib import Path

DEFAULT_NC_PATH = Path(__file__).parent / "weather_features.nc"


def open_dataset(nc_path):
    """懒加载打开 .nc 文件（不读数据到内存）"""
    try:
        import xarray as xr
    except ImportError:
        print("[ERROR] 需要安装 xarray: pip install xarray netCDF4")
        sys.exit(1)

    if not Path(nc_path).exists():
        print(f"[ERROR] 文件不存在: {nc_path}")
        sys.exit(1)

    return xr.open_dataset(nc_path)


def show_info(ds):
    """显示基本信息（只读元数据，不加载数据）"""
    print("\n" + "=" * 70)
    print("基本信息")
    print("=" * 70)

    # 维度
    dims = dict(ds.sizes)
    print(f"维度:       {dims}")

    # 时间范围（懒读）
    time_var = None
    for cand in ['times', 'time', 'datetime']:
        if cand in ds.coords or cand in ds.data_vars:
            time_var = cand
            break

    if time_var:
        t0 = ds[time_var].values[0]
        t1 = ds[time_var].values[-1]
        n = len(ds[time_var])
        print(f"时间范围:   {t0}  ~  {t1}")
        print(f"时间点数:   {n}")

    # 列信息
    var_list = list(ds.data_vars)
    print(f"\n变量列表 ({len(var_list)} 个):")
    for i, col in enumerate(var_list, 1):
        var = ds[col]
        print(f"  {i:2d}. {col:<30s}  dtype={str(var.dtype):<10s}  shape={var.shape}")


def show_preview(ds, n=10):
    """显示前 n 行数据（只读前 n 个时间点）"""
    print("\n" + "=" * 70)
    print(f"前 {n} 行数据")
    print("=" * 70)
    # isel 只读前 n 个
    sub = ds.isel({list(ds.sizes)[0]: slice(0, n)})
    df = sub.to_dataframe().reset_index()
    print(df.to_string())


def show_stats(ds):
    """显示数值列统计"""
    print("\n" + "=" * 70)
    print("数值列统计")
    print("=" * 70)
    import pandas as pd
    stats = {}
    for col in ds.data_vars:
        arr = ds[col].values
        if arr.dtype.kind in ('f', 'i'):
            import numpy as np
            stats[col] = {
                'count': len(arr) - np.isnan(arr).sum(),
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
            }
    print(pd.DataFrame(stats).T.to_string())


def show_col(ds, col_name, n=20):
    """显示某一列前 n 个值"""
    print(f"\n=== {col_name} 前 {n} 个值 ===")
    if col_name not in ds.data_vars:
        print(f"[ERROR] 列不存在: {col_name}")
        print(f"可用列: {list(ds.data_vars)}")
        return
    vals = ds[col_name].values[:n]
    print(vals)


def to_csv(ds, out_path):
    """导出为 CSV（此步会转 DataFrame，会慢一些）"""
    print(f"\n[转换] to_dataframe ...")
    df = ds.to_dataframe().reset_index()
    print(f"[导出] {out_path}  ({len(df)} 行) ...")
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"[完成] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="读取 weather_features.nc（快速版）")
    parser.add_argument("--nc-path", default=str(DEFAULT_NC_PATH),
                         help=f"nc 文件路径（默认: {DEFAULT_NC_PATH}）")
    parser.add_argument("--preview", action="store_true", help="显示前 10 行")
    parser.add_argument("--stats", action="store_true", help="数值列统计")
    parser.add_argument("--col", metavar="COL_NAME", help="只显示某一列")
    parser.add_argument("--to-csv", metavar="OUT_PATH", help="导出为 CSV")
    args = parser.parse_args()

    ds = open_dataset(args.nc_path)
    print(f"[加载] {args.nc_path}")
    show_info(ds)

    if args.preview:
        show_preview(ds)
    if args.stats:
        show_stats(ds)
    if args.col:
        show_col(ds, args.col)
    if args.to_csv:
        to_csv(ds, args.to_csv)

    ds.close()
    print("\n[完成]")


if __name__ == "__main__":
    main()
