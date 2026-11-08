# train.py
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ================= 配置区域（支持环境变量动态覆盖与本机 Smoke Test） =================
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1" or "--smoke-test" in sys.argv
SMOKE_PASS = int(os.environ.get("SMOKE_PASS", "0"))
if "--smoke-pass" in sys.argv:
    idx = sys.argv.index("--smoke-pass")
    if idx + 1 < len(sys.argv):
        SMOKE_PASS = int(sys.argv[idx + 1])

_DEFAULT_ROOT = "/content" if os.path.isdir("/content") and not SMOKE_TEST else os.getcwd()
GDRIVE_DIR = os.environ.get("GDRIVE_DIR", "gdrive:colab_relay")          # Google Drive 中的工作目录名
GDRIVE_DATASET_DIR = os.environ.get("GDRIVE_DATASET_DIR", "")            # Google Drive 数据集路径（可选）
RCLONE_CONF = os.environ.get("RCLONE_CONF", os.path.join(_DEFAULT_ROOT, "rclone.conf"))      # 传到 Colab 上的授权文件路径
LOCAL_CKPT = os.environ.get("CHECKPOINT_PATH", os.environ.get("LOCAL_CKPT", os.path.join(_DEFAULT_ROOT, "checkpoint.pt")))      # 本地高速磁盘上的断点路径
DONE_FLAG = os.environ.get("DONE_FLAG_PATH", os.environ.get("DONE_FLAG", os.path.join(_DEFAULT_ROOT, "training_done.flag")))    # 训练全部完成的标记文件
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(_DEFAULT_ROOT, "results"))          # 训练结果输出目录
TRAINING_STATE_FILE = os.environ.get(
    "TRAINING_STATE_FILE",
    os.path.join(_DEFAULT_ROOT, ".colab_pilot_training_state.json"),
)  # 可视化面板实时训练进度状态文件

MAX_SESSION_SECONDS = float(os.environ.get("MAX_SESSION_SECONDS", 22.5 * 3600))    # 单次实例满 22.5 小时主动退出交接
BACKUP_INTERVAL_SECONDS = float(os.environ.get("BACKUP_INTERVAL_SECONDS", 1800))   # 每隔 30 分钟自动往网盘备份一次权重
TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", 500))                            # 总训练轮数
RELAY_ROUND = int(os.environ.get("RELAY_ROUND", 1))
MODEL_NAME = os.environ.get("MODEL_NAME", "mlp-128x128-relay")
# ====================================================================


def write_training_state(**kwargs):
    """写入实时训练进度，供本机可视化面板通过 Contents API 无阻塞读取"""
    try:
        data = {
            "mode": "dl_relay",
            "framework": "pytorch",
            "model_name": MODEL_NAME,
            "round": RELAY_ROUND,
            "gdrive_dir": GDRIVE_DIR,
            "dataset_dir": GDRIVE_DATASET_DIR or "/content/dataset",
            "results_dir": RESULTS_DIR,
        }
        if os.path.exists(TRAINING_STATE_FILE):
            with open(TRAINING_STATE_FILE, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        data.update(kwargs)
        data["updated_at"] = time.time()
        tmp = TRAINING_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, TRAINING_STATE_FILE)
    except Exception:
        pass


def rclone_pull(remote_file, local_dir="/content"):
    """从 Google Drive 16 线程并行拉取文件到 Colab 本地高速盘"""
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return
    cmd = [
        "rclone", "copy",
        "--config", RCLONE_CONF,
        "--multi-thread-streams", "16",
        "--transfers", "16",
        "--retries", "5",
        "--low-level-retries", "20",
        f"{GDRIVE_DIR}/{remote_file}",
        local_dir
    ]
    subprocess.run(cmd, check=False)


def rclone_push(local_file, remote_sub=""):
    """将 Colab 本地文件或目录 16 线程静默上传回 Google Drive"""
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return
    target = f"{GDRIVE_DIR}/{remote_sub}".rstrip("/")
    cmd = [
        "rclone", "copy",
        "--config", RCLONE_CONF,
        "--multi-thread-streams", "16",
        "--drive-chunk-size", "128M",
        "--retries", "5",
        "--low-level-retries", "20",
        local_file,
        target
    ]
    subprocess.run(cmd, check=True)


def save_checkpoint_safely(state, filepath):
    """原子化保存：先写临时文件再替换，防止保存中途断电损坏权重文件"""
    tmp_path = filepath + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, filepath)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": MODEL_NAME,
                    "epoch": state.get("epoch", 0) + 1,
                    "total_epochs": TOTAL_EPOCHS,
                    "loss": round(float(state.get("loss", 0.0)), 6),
                    "updated_at": time.time(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


def main():
    start_time = time.time()
    last_backup_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    write_training_state(
        status="pulling_checkpoint",
        pid=os.getpid(),
        gdrive_dir=GDRIVE_DIR,
        dataset_status="smoke_local" if SMOKE_TEST else "ready",
        sync_status="pulling",
        total_epochs=TOTAL_EPOCHS,
        max_session_seconds=MAX_SESSION_SECONDS,
        backup_interval_seconds=BACKUP_INTERVAL_SECONDS,
        session_start_time=start_time,
    )

    # 1. 开机自动从 Google Drive 拉取上一棒的 Checkpoint
    print("🔄 正在检查 Google Drive 中是否有历史断点文件...", flush=True)
    rclone_pull("checkpoint.pt", os.path.dirname(LOCAL_CKPT) or "/content")

    # （可选）若配置了 Google Drive 数据集目录，自动拉取到 /content/dataset
    if GDRIVE_DATASET_DIR and not SMOKE_TEST:
        write_training_state(dataset_status="pulling_from_gdrive")
        rclone_pull(GDRIVE_DATASET_DIR, "/content/dataset")
        write_training_state(dataset_status="synced_from_gdrive")

    # 2. 初始化模型、求解器与优化器（请替换为你的实际模型代码）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(128, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0

    # 3. 恢复历史断点状态
    if os.path.exists(LOCAL_CKPT):
        print(f"📦 发现历史断点 {LOCAL_CKPT}，正在恢复状态...", flush=True)
        try:
            ckpt = torch.load(LOCAL_CKPT, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(LOCAL_CKPT, map_location=device)
        saved_epoch = ckpt.get("epoch", -1) + 1
        if saved_epoch >= TOTAL_EPOCHS and os.environ.get("RESET_DONE_FLAG") == "1":
            print(
                f"🔄 历史断点已完成 {saved_epoch}/{TOTAL_EPOCHS} 轮且已开启重新训练，重置为从第 0 轮开始。",
                flush=True,
            )
            start_epoch = 0
        else:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = saved_epoch
            print(f"✅ 成功从第 {start_epoch} 轮 (Epoch) 继续训练！", flush=True)
    else:
        print("🆕 未找到历史断点，从第 0 轮开始全新训练。", flush=True)

    # 4. 主训练 / 数值模拟循环
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # --- 在此处替换为你的单轮训练或数值模拟计算逻辑 ---
        x = torch.randn(64, 128, device=device)
        target = torch.randn(64, 128, device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        time.sleep(0.05 if SMOKE_TEST else 2.0)
        # ------------------------------------------------

        # 先高速保存到本地 checkpoint.pt 与 results/training_metrics.json
        save_checkpoint_safely({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_val,
        }, LOCAL_CKPT)

        now = time.time()
        elapsed_session = now - start_time
        avg_epoch_sec = elapsed_session / max(1, (epoch - start_epoch + 1))
        eta_sec = avg_epoch_sec * max(0, TOTAL_EPOCHS - (epoch + 1))

        write_training_state(
            status="training",
            sync_status="local_saved",
            current_epoch=epoch + 1,
            start_epoch=start_epoch,
            total_epochs=TOTAL_EPOCHS,
            loss=round(loss_val, 6),
            lr=1e-3,
            elapsed_seconds=round(elapsed_session, 1),
            eta_seconds=round(eta_sec, 1),
            next_backup_in_seconds=max(0, round(BACKUP_INTERVAL_SECONDS - (now - last_backup_time), 1)),
            handover_in_seconds=max(0, round(MAX_SESSION_SECONDS - elapsed_session, 1)),
        )

        # 本机 Smoke Test Pass #1：验证第 1 轮断点保存与状态上报后立即模拟换棒退出
        if SMOKE_TEST and SMOKE_PASS == 1 and (epoch + 1) >= 1:
            write_training_state(status="handover_exit", current_epoch=epoch + 1)
            print("HANDOVER_REQUESTED", flush=True)
            return

        # 本机 Smoke Test Pass #2：验证从第 1 轮断点续训到第 2 轮后完成验证
        if SMOKE_TEST and SMOKE_PASS == 2 and (epoch + 1) >= 2:
            break

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            print(
                f"[Epoch {epoch + 1}/{TOTAL_EPOCHS}] loss={loss_val:.4f} | "
                f"本棒已运行 {elapsed_session / 60:.1f}m | 距下次网盘热备 {max(0, BACKUP_INTERVAL_SECONDS - (now - last_backup_time)):.0f}s",
                flush=True,
            )

        # 每隔 30 分钟（或本棒第 1 轮真机冒烟、或最后一轮）将本地权重与 results/ 同步至 Google Drive
        synced_this_epoch = False
        if (epoch == start_epoch and not SMOKE_TEST) or (now - last_backup_time >= BACKUP_INTERVAL_SECONDS) or (epoch == TOTAL_EPOCHS - 1):
            print(f"[Epoch {epoch + 1}] ☁️ 正在通过 rclone (16线程) 将最新权重与结果备份至 Google Drive ({GDRIVE_DIR})...", flush=True)
            write_training_state(status="syncing_gdrive", sync_status="backing_up", current_epoch=epoch + 1)
            rclone_push(LOCAL_CKPT)
            if os.path.isdir(RESULTS_DIR) and os.listdir(RESULTS_DIR):
                rclone_push(RESULTS_DIR, "results")
            last_backup_time = time.time()
            synced_this_epoch = True
            if epoch == start_epoch and not SMOKE_TEST:
                print(
                    f"CLOUD_SMOKE_PASSED [Epoch {epoch + 1}] ✅ Colab 云端真机首步冒烟通过（GPU 显存、数据流与网盘断点写入均正常），无缝进入正式训练！",
                    flush=True,
                )
            write_training_state(
                status="training",
                sync_status="synced",
                last_backup_epoch=epoch + 1,
                last_backup_time=last_backup_time,
            )

        # 检查是否达到单次实例 22.5 小时安全上限
        elapsed_session = time.time() - start_time
        if elapsed_session >= MAX_SESSION_SECONDS and epoch < TOTAL_EPOCHS - 1:
            print(f"⏰ 当前实例已运行 {elapsed_session/3600:.2f} 小时，即将触及 24h 上限。", flush=True)
            if not synced_this_epoch:
                print("☁️ 正在执行退出前最终网盘同步...", flush=True)
                write_training_state(status="handover_sync", sync_status="backing_up", current_epoch=epoch + 1)
                rclone_push(LOCAL_CKPT)
                if os.path.isdir(RESULTS_DIR) and os.listdir(RESULTS_DIR):
                    rclone_push(RESULTS_DIR, "results")
            write_training_state(
                status="handover_exit",
                sync_status="synced",
                current_epoch=epoch + 1,
                last_backup_epoch=epoch + 1,
                last_backup_time=time.time(),
            )
            print("HANDOVER_REQUESTED", flush=True)
            print("👋 安全退出当前实例，交由本机脚本启动下一棒新实例！", flush=True)
            return

    # 5. 全部轮次完成，生成完成标记并上传至网盘
    with open(DONE_FLAG, "w", encoding="utf-8") as f:
        f.write("COMPLETED")
    rclone_push(DONE_FLAG)
    if os.path.isdir(RESULTS_DIR) and os.listdir(RESULTS_DIR):
        rclone_push(RESULTS_DIR, "results")
    write_training_state(
        status="completed",
        sync_status="synced",
        current_epoch=TOTAL_EPOCHS if not SMOKE_TEST else max(2, start_epoch + 1),
        total_epochs=TOTAL_EPOCHS,
        completed_at=time.time(),
    )
    print("TRAINING_COMPLETED", flush=True)
    print("🎉 所有训练轮次已 100% 完成！完成标记已上传至 Google Drive。", flush=True)


if __name__ == "__main__":
    main()
