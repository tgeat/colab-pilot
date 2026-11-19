#!/usr/bin/env bash
set -euo pipefail

# ================= 配置区域 =================
SESSION="${SESSION:-rclone-relay}"                  # Colab CLI 会话名称
GPU_TYPE="${GPU_TYPE:-T4}"                          # 可选: T4 / L4 / G4 / A100 / H100
GDRIVE_DIR="${GDRIVE_DIR:-gdrive:colab_relay}"      # Google Drive 目标文件夹
LOCAL_RCLONE_CONF="${LOCAL_RCLONE_CONF:-$(rclone config file | tail -n 1 | tr -d '\r')}"
# ============================================

if [[ -z "${LOCAL_RCLONE_CONF}" || ! -f "${LOCAL_RCLONE_CONF}" ]]; then
    echo "❌ 未找到有效的本机 rclone.conf 授权文件: ${LOCAL_RCLONE_CONF:-<empty>}" >&2
    exit 1
fi

if [[ ! -f "train.py" ]]; then
    echo "❌ 当前目录下未找到 train.py 训练脚本" >&2
    exit 1
fi

strip_ansi() {
    sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g'
}

cleanup_on_signal() {
    echo "⚠️ 捕获中断信号，正在销毁云端实例 (${SESSION}) 防止后台持续计费..."
    colab stop -s "$SESSION" 2>/dev/null || true
    exit 130
}
trap cleanup_on_signal INT TERM

echo "🔑 使用本机授权文件: ${LOCAL_RCLONE_CONF}"

# 0. 确保网盘目录存在，并按需清理旧的完成标记文件
rclone mkdir -- "${GDRIVE_DIR}"
if [[ "${RESET_DONE_FLAG:-0}" == "1" ]]; then
    rclone deletefile -- "${GDRIVE_DIR}/training_done.flag" 2>/dev/null || true
fi

ROUND=1
FAST_HANDOVER_STREAK=0
while true; do
    if [[ "$(rclone lsf -- "${GDRIVE_DIR}/training_done.flag" 2>/dev/null || true)" == "training_done.flag" ]]; then
        echo "🎉 检测到网盘中的完成标记 training_done.flag，全部接力训练已圆满结束！"
        break
    fi

    echo "===================================================="
    echo "🚀 启动第 ${ROUND} 棒接力实例 | GPU: ${GPU_TYPE} | 时间: $(date)"
    echo "===================================================="

    # 1. 清理可能残留的同名旧实例，启动全新的高内存 GPU 实例
    colab stop -s "$SESSION" 2>/dev/null || true
    colab new -s "$SESSION" --gpu "$GPU_TYPE" --high-mem

    # 2. 上传本机 rclone.conf 与 train.py 到 Colab
    echo "📤 正在上传 rclone.conf 与 train.py 到云端..."
    if ! colab upload -s "$SESSION" "$LOCAL_RCLONE_CONF" content/rclone.conf || ! colab upload -s "$SESSION" train.py content/train.py; then
        echo "❌ 上传文件失败，立即销毁实例停止计费..." >&2
        colab stop -s "$SESSION" 2>/dev/null || true
        exit 1
    fi

    # 3. 在云端安装 rclone 工具及 Python 依赖包（校验输出标记，防止 colab CLI 在远程异常时仍返回 0）
    echo "🔧 正在云端安装 rclone 环境..."
    set +e
    RCLONE_OUT=$(echo "import shutil, subprocess; shutil.which('rclone') or subprocess.run(['bash', '-c', 'apt-get update -qq && apt-get install -y -qq rclone'], check=True); print('RCLONE_READY')" | colab exec -s "$SESSION" --timeout 600 2>&1)
    RCLONE_RC=$?
    set -e
    printf '%s\n' "$RCLONE_OUT"
    if [[ "$RCLONE_RC" -ne 0 || "$RCLONE_OUT" != *"RCLONE_READY"* ]]; then
        echo "❌ 云端安装 rclone 失败 (exit=${RCLONE_RC})，立即销毁实例停止计费..." >&2
        colab stop -s "$SESSION" 2>/dev/null || true
        exit 1
    fi

    if [[ -f "requirements.txt" ]] && grep -q '[^[:space:]]' "requirements.txt"; then
        echo "📦 正在安装 requirements.txt 依赖..."
        set +e
        INSTALL_OUT=$(colab install -s "$SESSION" -r requirements.txt 2>&1)
        INSTALL_RC=$?
        set -e
        printf '%s\n' "$INSTALL_OUT"
        INSTALL_CLEAN=$(printf '%s' "$INSTALL_OUT" | strip_ansi)
        if [[ "$INSTALL_RC" -ne 0 || "$INSTALL_CLEAN" != *"Installation Complete"* || "$INSTALL_CLEAN" == *"CalledProcessError"* || "$INSTALL_CLEAN" == *"Traceback"* ]]; then
            echo "❌ 云端依赖安装失败 (exit=${INSTALL_RC})，立即销毁实例停止计费..." >&2
            colab stop -s "$SESSION" 2>/dev/null || true
            exit 1
        fi
    fi

    # 4. 执行训练脚本并捕获输出与退出码（使用 -f train.py，无论成败均先释放实例停止计费）
    echo "⏳ 开始云端训练（第 ${ROUND} 棒，单棒限时 22.5 小时，中途自动同步网盘）..."
    ROUND_LOG=$(mktemp)
    ROUND_START_TS=$(date +%s)
    ROUND_RESET_FLAG="0"
    if [[ "$ROUND" -eq 1 && "${RESET_DONE_FLAG:-0}" == "1" ]]; then
        ROUND_RESET_FLAG="1"
    fi
    set +e
    colab exec -s "$SESSION" --timeout 86400 \
        --env "GDRIVE_DIR=${GDRIVE_DIR}" \
        --env "RELAY_ROUND=${ROUND}" \
        --env "RESET_DONE_FLAG=${ROUND_RESET_FLAG}" \
        -f train.py 2>&1 | tee "$ROUND_LOG"
    EXEC_RC=${PIPESTATUS[0]}
    set -e
    ROUND_ELAPSED=$(( $(date +%s) - ROUND_START_TS ))

    # 5. 销毁当前实例，停止扣除计算单元
    echo "🧹 本棒运行结束 (exit=${EXEC_RC}, elapsed=${ROUND_ELAPSED}s)，正在销毁云端实例释放计费..."
    colab stop -s "$SESSION" 2>/dev/null || true

    CLEAN_LOG=$(mktemp)
    strip_ansi < "$ROUND_LOG" > "$CLEAN_LOG"
    rm -f "$ROUND_LOG"

    HAS_COMPLETED=0
    HAS_HANDOVER=0
    HAS_FATAL_ERROR=0
    grep -q "TRAINING_COMPLETED" "$CLEAN_LOG" && HAS_COMPLETED=1 || true
    grep -q "HANDOVER_REQUESTED" "$CLEAN_LOG" && HAS_HANDOVER=1 || true
    if grep -Eq "Traceback \(most recent call last\)|SyntaxError:|IndentationError:|CLOUD_SMOKE_FAILED" "$CLEAN_LOG"; then
        HAS_FATAL_ERROR=1
    fi
    rm -f "$CLEAN_LOG"

    if [[ "$EXEC_RC" -ne 0 || "$HAS_FATAL_ERROR" -eq 1 ]]; then
        echo "❌ [熔断止损] 第 ${ROUND} 棒 train.py 发生异常或崩溃 (exit=${EXEC_RC})，已销毁实例并终止接力循环！" >&2
        exit "${EXEC_RC:-1}"
    fi

    if [[ "$HAS_COMPLETED" -eq 1 ]]; then
        echo "🎉 本棒输出 TRAINING_COMPLETED，全部训练任务已完成！"
        break
    fi

    if [[ "$HAS_HANDOVER" -ne 1 ]]; then
        echo "❌ [熔断止损] 第 ${ROUND} 棒未输出 HANDOVER_REQUESTED 或 TRAINING_COMPLETED 即退出，拒绝无限开新机死循环！" >&2
        exit 1
    fi

    if [[ "$ROUND_ELAPSED" -lt 60 ]]; then
        FAST_HANDOVER_STREAK=$((FAST_HANDOVER_STREAK + 1))
        if [[ "$FAST_HANDOVER_STREAK" -ge 3 ]]; then
            echo "❌ [熔断止损] 连续 ${FAST_HANDOVER_STREAK} 棒在 60s 内极速触发换棒，疑似配置异常，已中止接力循环！" >&2
            exit 1
        fi
    else
        FAST_HANDOVER_STREAK=0
    fi

    ROUND=$((ROUND + 1))
    echo "💤 休息 15 秒后开启第 ${ROUND} 棒..."
    sleep 15
done

# 训练全部结束后，将最终权重从网盘拉回本机备份
echo "📥 正在将最终训练好的 checkpoint.pt 下载回本机当前目录..."
rclone copy -- "${GDRIVE_DIR}/checkpoint.pt" ./ --multi-thread-streams=16 -P
echo "✅ 全部流程执行完毕！"
