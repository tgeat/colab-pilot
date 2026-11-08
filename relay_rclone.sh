#!/bin/bash
set -e

# ================= 配置区域 =================
SESSION="${SESSION:-rclone-relay}"                  # Colab CLI 会话名称
GPU_TYPE="${GPU_TYPE:-T4}"                          # 可选: T4 / L4 / G4 / A100 / H100
GDRIVE_DIR="${GDRIVE_DIR:-gdrive:colab_relay}"      # Google Drive 目标文件夹
LOCAL_RCLONE_CONF=$(rclone config file | tail -n 1) # 自动获取本机 rclone.conf 路径
# ============================================

echo "🔑 使用本机授权文件: ${LOCAL_RCLONE_CONF}"

# 0. 确保网盘目录存在，并清理旧的完成标记文件
rclone mkdir "${GDRIVE_DIR}"
rclone deletefile "${GDRIVE_DIR}/training_done.flag" 2>/dev/null || true

ROUND=1
while true; do
    # 检查 Google Drive 中是否已生成完成标记 training_done.flag
    if [ "$(rclone lsf "${GDRIVE_DIR}/training_done.flag" 2>/dev/null)" == "training_done.flag" ]; then
        echo "🎉 检测到网盘中的完成标记 training_done.flag，全部接力训练已圆满结束！"
        break
    fi

    echo "===================================================="
    echo "🚀 启动第 ${ROUND} 棒接力实例 | GPU: ${GPU_TYPE} | 时间: $(date)"
    echo "===================================================="

    # 1. 清理可能残留的同名旧实例，启动全新的高内存 GPU 实例
    colab stop -s "$SESSION" 2>/dev/null || true
    colab new -s "$SESSION" --gpu "$GPU_TYPE" --high-mem

    # 2. 将本机的 rclone.conf 免密钥匙上传到 Colab
    echo "📤 正在上传 rclone.conf 授权文件到云端..."
    colab upload -s "$SESSION" "$LOCAL_RCLONE_CONF" content/rclone.conf

    # 3. 在云端安装 rclone 工具及 Python 依赖包
    echo "🔧 正在云端安装 rclone 环境..."
    echo "import subprocess; subprocess.run(['bash', '-c', 'apt-get update -qq && apt-get install -y -qq rclone'], check=True)" | colab exec -s "$SESSION"

    if [ -f "requirements.txt" ]; then
        echo "📦 正在安装 requirements.txt 依赖..."
        colab install -s "$SESSION" -r requirements.txt
    fi

    # 4. 执行训练脚本（务必保留 --timeout 86400 以支持长耗时运行）
    echo "⏳ 开始云端训练（单棒限时 22.5 小时，中途每 30 分钟自动同步网盘）..."
    colab exec -s "$SESSION" --timeout 86400 -f train.py || true

    # 5. 销毁当前实例，停止扣除计算单元并重置 24 小时周期
    echo "🧹 本棒结束，正在销毁云端实例释放计费..."
    colab stop -s "$SESSION" 2>/dev/null || true

    ROUND=$((ROUND + 1))
    echo "💤 休息 15 秒后检查状态并准备下一棒..."
    sleep 15
done

# 训练全部结束后，将最终权重从网盘拉回本机备份
echo "📥 正在将最终训练好的 checkpoint.pt 下载回本机当前目录..."
rclone copy "${GDRIVE_DIR}/checkpoint.pt" ./ --multi-thread-streams=16 -P
echo "✅ 全部流程执行完毕！"
