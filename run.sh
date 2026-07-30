#!/usr/bin/env bash
# =============================================================================
# 平圩电力市场 XGBoost 实时电价预测 — 一键入口
# =============================================================================
# 用法：
#   bash run.sh --all                            # 全流程
#   bash run.sh --module cleaning                # 单模块
#   bash run.sh --module cleaning,split          # 多模块 (逗号分隔)
#   bash run.sh --module "training evaluation"   # 多模块 (空格分隔)
#   bash run.sh --module split+training          # 多模块 (加号分隔)
#   bash run.sh --module training evaluation     # 多模块 (位置参数)
#   bash run.sh --clean
#   bash run.sh --help
#
# 多模块运行时:
#   - 自动按 pipeline 依赖顺序排序 (cleaning → split → correlation → training → evaluation)
#   - 自动去重
#   - 任一模块失败立即停止
#
# 依赖：在 anaconda 环境 pytorch-2.10 下运行
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 优先使用环境变量 PY 指定的解释器；否则回退到当前 PATH 上的 python
PY="${PY:-python}"

# 模块顺序（pipeline 依赖顺序）—— MODULES_ORDER[i] = 第 i 个模块名
MODULES_ORDER=("cleaning" "split" "correlation" "training" "evaluation")

usage() {
    cat <<EOF
用法:
  bash run.sh --all                            # 全流程一键
  bash run.sh --module <name>[,<name>...]      # 单/多模块运行
  bash run.sh --module <a> <b> ...             # 多模块 (位置参数)
  bash run.sh --clean                          # 清空 outputs/
  bash run.sh --help

模块名 (任意分隔: , ; / + 空格 都接受):
  cleaning      ① 数据清洗            → outputs/01_cleaning.html
  split         ② 数据划分            → outputs/02_split.html
  correlation   ③ 数据相关性分析(含EDA) → outputs/03_correlation.html
  training      ④ 模型训练            → outputs/04_training.html
  evaluation    ⑤ 模型评测            → outputs/05_evaluation.html

多模块示例:
  bash run.sh --module training,evaluation
  bash run.sh --module "split training evaluation"
  bash run.sh --module training+evaluation
  bash run.sh --module split correlation training

多模块运行时:
  - 自动按 pipeline 依赖顺序排序
  - 自动去重 (重复模块名只跑一次)
  - 任一模块失败立即停止 (后续模块不会执行)

完整流水线产物:
  outputs/cleaned_data.pkl, split.json, correlation.pkl, model.joblib, metrics.pkl
  outputs/01~05_*.html + outputs/index.html (总览导航)
  outputs/plots/*.png
EOF
}

# 单模块执行 (内部使用)
run_module() {
    local mod="$1"
    echo ""
    echo "===================================================================="
    echo "  ▶ 运行模块: src.$mod"
    echo "===================================================================="
    "$PY" -m "src.$mod"
}

# 检查模块名是否合法
is_valid_module() {
    local mod="$1"
    for m in "${MODULES_ORDER[@]}"; do
        [[ "$m" == "$mod" ]] && return 0
    done
    return 1
}

# 解析所有模块参数 → 去重 + 按依赖顺序排序，结果存到全局数组 SORTED_MODULES
resolve_modules() {
    # 把所有位置参数拼起来，把 [ , ; / + ] 和空白都当分隔符
    local raw="$*"
    # shellcheck disable=SC2206
    local tokens=( $(echo "$raw" | tr ',;/+' ' ') )

    if [[ ${#tokens[@]} -eq 0 ]]; then
        echo "[ERROR] --module 需要至少一个模块名"
        usage; exit 1
    fi

    # 验证 + 收集到 space-separated string (兼容 bash 3.2)
    local seen=""
    local invalid=()
    for t in "${tokens[@]}"; do
        [[ -z "$t" ]] && continue
        if is_valid_module "$t"; then
            # 用空格分隔避免关联数组
            if [[ ! " $seen " =~ " $t " ]]; then
                seen="$seen $t "
            fi
        else
            invalid+=("$t")
        fi
    done

    if [[ ${#invalid[@]} -gt 0 ]]; then
        echo "[ERROR] 未知模块: ${invalid[*]}"
        echo "可选: ${MODULES_ORDER[*]}"
        exit 1
    fi

    # 按 MODULES_ORDER 顺序输出 (天然去重 + 拓扑序)
    SORTED_MODULES=()
    for m in "${MODULES_ORDER[@]}"; do
        if [[ " $seen " =~ " $m " ]]; then
            SORTED_MODULES+=("$m")
        fi
    done
}

run_modules() {
    resolve_modules "$@"
    echo "[INFO] 准备运行 ${#SORTED_MODULES[@]} 个模块 (按依赖顺序): ${SORTED_MODULES[*]}"
    local t0; t0=$(date +%s)
    for mod in "${SORTED_MODULES[@]}"; do
        run_module "$mod"
    done
    local t1; t1=$(date +%s)
    echo ""
    echo "===================================================================="
    echo "  ✓ ${#SORTED_MODULES[@]} 个模块运行完成 (耗时 $((t1 - t0)) 秒)"
    echo "  模块: ${SORTED_MODULES[*]}"
    [[ " ${SORTED_MODULES[*]} " == *" evaluation "* ]] && \
        echo "  打开 outputs/index.html 查看总览"
    echo "===================================================================="
}

run_all() {
    run_modules "${MODULES_ORDER[@]}"
}

clean_outputs() {
    echo "[WARN] 即将清空 outputs/ 目录"
    rm -rf "$PROJECT_ROOT/outputs"
    mkdir -p "$PROJECT_ROOT/outputs/plots"
    echo "[INFO] 已清空"
}

# ---- 入口分发 ----
if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

case "$1" in
    --all|all)
        run_all
        ;;
    --module|-m)
        shift
        if [[ $# -eq 0 ]]; then
            echo "[ERROR] --module 需要至少一个模块名"; usage; exit 1
        fi
        run_modules "$@"
        ;;
    --clean)
        clean_outputs
        ;;
    --help|-h|help)
        usage
        ;;
    *)
        echo "[ERROR] 未知参数: $1"
        usage
        exit 1
        ;;
esac
