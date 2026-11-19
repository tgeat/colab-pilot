# train.py
import json
import math
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
_INITIAL_GDRIVE_DIR = (
    os.environ.get("RELAY_GDRIVE_DIR")
    or os.environ.get("GDRIVE_DIR")
    or "gdrive:colab_relay"
).strip().rstrip("/")
GDRIVE_DIR = _INITIAL_GDRIVE_DIR
GDRIVE_DATASET_DIR = os.environ.get("GDRIVE_DATASET_DIR", "").strip()
RCLONE_CONF = os.environ.get(
    "RCLONE_CONFIG",
    os.environ.get("RCLONE_CONF", os.path.join(_DEFAULT_ROOT, "rclone.conf")),
)
LOCAL_CKPT = os.environ.get(
    "CHECKPOINT_PATH",
    os.environ.get("LOCAL_CKPT", os.path.join(_DEFAULT_ROOT, "checkpoint.pt")),
)
DONE_FLAG = os.environ.get(
    "DONE_FLAG_PATH",
    os.environ.get("DONE_FLAG", os.path.join(_DEFAULT_ROOT, "training_done.flag")),
)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(_DEFAULT_ROOT, "results"))
TRAINING_STATE_FILE = os.environ.get(
    "TRAINING_STATE_FILE",
    os.path.join(_DEFAULT_ROOT, ".colab_pilot_training_state.json"),
)

MAX_SESSION_SECONDS = float(os.environ.get("MAX_SESSION_SECONDS", 22.5 * 3600))
BACKUP_INTERVAL_SECONDS = float(os.environ.get("BACKUP_INTERVAL_SECONDS", 1800))
TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", 500))
RELAY_ROUND = int(os.environ.get("RELAY_ROUND", 1))
MODEL_NAME = os.environ.get("MODEL_NAME", "mlp-128x128-relay")
# ====================================================================


def _sanitize_json(obj):
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _effective_gdrive_dir():
    if GDRIVE_DIR != _INITIAL_GDRIVE_DIR:
        return GDRIVE_DIR.strip().rstrip("/")
    env_gd = os.environ.get("RELAY_GDRIVE_DIR") or os.environ.get("GDRIVE_DIR")
    if env_gd and env_gd.strip():
        return env_gd.strip().rstrip("/")
    return GDRIVE_DIR.strip().rstrip("/")


def _session_isolation_prefix():
    raw_sess = (
        os.environ.get("RELAY_SESSION")
        or os.environ.get("SESSION_PREFIX")
        or os.environ.get("COLAB_SESSION")
        or os.environ.get("SESSION")
        or ""
    ).strip()
    clean_sess = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in raw_sess).strip("-.")
    if clean_sess:
        return f"{clean_sess}.{os.getpid()}"
    return f"pid{os.getpid()}"


def _atomic_write_json(filepath, payload):
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{filepath}.{_session_isolation_prefix()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, filepath)


def write_training_state(**kwargs):
    """写入实时训练进度，供本机可视化面板通过 Contents API 无阻塞读取"""
    try:
        gd = _effective_gdrive_dir()
        data = {
            "mode": "dl_relay",
            "framework": "pytorch",
            "model_name": MODEL_NAME,
            "round": int(os.environ.get("RELAY_ROUND", RELAY_ROUND)),
            "gdrive_dir": gd,
            "dataset_dir": GDRIVE_DATASET_DIR or "/content/dataset",
            "results_dir": RESULTS_DIR,
        }
        if os.path.exists(TRAINING_STATE_FILE):
            try:
                with open(TRAINING_STATE_FILE, "r", encoding="utf-8") as f:
                    data.update(json.load(f))
            except Exception:
                pass
        data.update(kwargs)
        data["updated_at"] = time.time()
        _atomic_write_json(TRAINING_STATE_FILE, data)
    except Exception:
        pass


def _rclone_base_cmd():
    cmd = ["rclone"]
    conf_path = os.environ.get("RCLONE_CONFIG") or os.environ.get("RCLONE_CONF") or RCLONE_CONF
    if conf_path and os.path.exists(conf_path):
        cmd.extend(["--config", conf_path])
    cmd.extend(["--retries", "5", "--low-level-retries", "20"])
    return cmd


def _resolve_remote_path(remote_spec):
    spec = remote_spec.strip()
    if ":" in spec:
        return spec.rstrip("/")
    return f"{_effective_gdrive_dir()}/{spec.lstrip('/')}"


def _list_remote_entries(remote_dir):
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return set()
    res = subprocess.run(
        _rclone_base_cmd() + ["lsf", "--", remote_dir],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode == 0:
        return {ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()}
    err_txt = ((res.stderr or "") + "\n" + (res.stdout or "")).lower()
    if "directory not found" in err_txt or "object not found" in err_txt or "file not found" in err_txt:
        return set()
    raise RuntimeError(f"rclone 查询远端目录失败 ({remote_dir}): {(res.stderr or '').strip()[:240]}")


def _remote_file_exists(remote_dir, filename):
    entries = _list_remote_entries(remote_dir)
    return filename in entries or f"{filename}/" in entries


def rclone_pull_checkpoint(local_ckpt):
    """
    安全拉取历史断点：
    - 若云盘存在有效断点：拉取并校验本地非空，拉取失败立即抛错保护；
    - 若云盘不存在主断点，但存在历史备份或训练结果产物（处于续训状态）且 RELAY_ROUND>1 未重置：报错拒绝归零覆写；
    - 若云盘经查询确认尚无任何断点及训练产物（全新非续训状态）：即便 RELAY_ROUND>1（如前序分配失败重试导致），
      输出清晰诊断提示后允许按实际无断点状态从头启动。
    """
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return False
    gd = _effective_gdrive_dir()
    ckpt_name = os.path.basename(local_ckpt)
    stem, ext = os.path.splitext(ckpt_name)
    prev_name = f"{stem}_prev{ext or '.pt'}"
    effective_round = int(os.environ.get("RELAY_ROUND", RELAY_ROUND))
    reset_flag = os.environ.get("RESET_DONE_FLAG", "0") == "1"

    entries = _list_remote_entries(gd)
    exists = ckpt_name in entries
    has_prior_state = (
        prev_name in entries
        or "results/" in entries
        or "results" in entries
        or "training_done.flag" in entries
        or "TRAINING_COMPLETED.flag" in entries
    )

    if not exists:
        if os.path.exists(local_ckpt) and os.path.getsize(local_ckpt) > 0:
            return True
        if effective_round > 1 and not reset_flag:
            if has_prior_state:
                raise RuntimeError(
                    f"第 {effective_round} 棒接力异常：Google Drive ({gd}) 存在历史训练产物但未找到 {ckpt_name}，拒绝从第 0 轮重跑覆写！"
                )
            print(
                f"⚠️ [接力诊断] 当前 RELAY_ROUND={effective_round}，但已确认云盘目录 ({gd}) 尚无任何历史断点或训练产物"
                f"（非续训状态，可能由前序实例分配失败重试导致），将按实际无断点状态从第 0 轮启动。",
                flush=True,
            )
        return False

    parent = os.path.dirname(os.path.abspath(local_ckpt))
    if parent:
        os.makedirs(parent, exist_ok=True)
    cmd = _rclone_base_cmd() + [
        "copyto",
        "--multi-thread-streams",
        "16",
        "--",
        f"{gd}/{ckpt_name}",
        local_ckpt,
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(local_ckpt) or os.path.getsize(local_ckpt) == 0:
        raise RuntimeError(f"从 Google Drive ({gd}) 拉取 {ckpt_name} 后本地文件为空或缺失")
    return True


def rclone_pull(remote_spec, local_dir="/content", required=True):
    """从 Google Drive 16 线程并行拉取目录或文件到 Colab 本地高速盘（支持完整 remote:path 或相对子路径）"""
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return
    os.makedirs(local_dir, exist_ok=True)
    src = _resolve_remote_path(remote_spec)
    cmd = _rclone_base_cmd() + [
        "copy",
        "--multi-thread-streams",
        "16",
        "--transfers",
        "16",
        "--",
        src,
        local_dir,
    ]
    subprocess.run(cmd, check=required)


def rclone_push(local_path, remote_sub=""):
    """将 Colab 本地文件或目录 16 线程上传回 Google Drive（对断点文件执行进程/会话隔离临时上传+旧版轮转+原子替换）"""
    if SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
        return
    gd = _effective_gdrive_dir()
    iso_prefix = _session_isolation_prefix()
    local_ckpt_target = os.environ.get("CHECKPOINT_PATH") or os.environ.get("LOCAL_CKPT") or LOCAL_CKPT
    if os.path.isfile(local_path) and not remote_sub and os.path.abspath(local_path) == os.path.abspath(local_ckpt_target):
        ckpt_name = os.path.basename(local_path)
        stem, ext = os.path.splitext(ckpt_name)
        remote_ckpt = f"{gd}/{ckpt_name}"
        remote_tmp = f"{gd}/.{iso_prefix}.{ckpt_name}.uploading"
        remote_prev = f"{gd}/{stem}_prev{ext or '.pt'}"
        subprocess.run(
            _rclone_base_cmd()
            + ["copyto", "--multi-thread-streams", "16", "--drive-chunk-size", "128M", "--", local_path, remote_tmp],
            check=True,
        )
        if _remote_file_exists(gd, ckpt_name):
            subprocess.run(_rclone_base_cmd() + ["copyto", "--", remote_ckpt, remote_prev], check=False)
        subprocess.run(_rclone_base_cmd() + ["moveto", "--", remote_tmp, remote_ckpt], check=True)
        return

    target = f"{gd}/{remote_sub}".rstrip("/")
    cmd = _rclone_base_cmd() + [
        "copy",
        "--multi-thread-streams",
        "16",
        "--drive-chunk-size",
        "128M",
        "--",
        local_path,
        target,
    ]
    subprocess.run(cmd, check=True)


def save_checkpoint_safely(state, filepath):
    """原子化保存：先写带进程/会话隔离前缀的临时文件再替换，防止保存中途断电或并发写入损坏权重文件"""
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{filepath}.{_session_isolation_prefix()}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, filepath)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    try:
        raw_loss = float(state.get("loss", 0.0))
        _atomic_write_json(
            metrics_path,
            {
                "model_name": MODEL_NAME,
                "epoch": state.get("epoch", 0) + 1,
                "total_epochs": TOTAL_EPOCHS,
                "loss": round(raw_loss, 6) if math.isfinite(raw_loss) else None,
                "updated_at": time.time(),
            },
        )
    except Exception:
        pass


def load_checkpoint_safely(filepath, map_location="cpu"):
    """安全反序列化 PyTorch 断点：默认启用 weights_only=True 防御不可信网盘 Pickle 执行（H-11）"""
    allow_unsafe = os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "0") == "1"
    try:
        ckpt = torch.load(filepath, map_location=map_location, weights_only=True)
    except TypeError:
        if not allow_unsafe:
            raise RuntimeError(
                "当前 PyTorch 版本不支持安全反序列化 (weights_only=True)；如确认断点可信，请设置 ALLOW_UNSAFE_TORCH_LOAD=1"
            )
        ckpt = torch.load(filepath, map_location=map_location)
    except Exception as safe_err:
        if allow_unsafe:
            ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
        else:
            raise RuntimeError(
                f"安全反序列化 (weights_only=True) 拒绝加载 {filepath}"
                f"（如断点包含自定义类且确认 Google Drive 来源可信，可设置 ALLOW_UNSAFE_TORCH_LOAD=1）: {safe_err}"
            ) from safe_err
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"断点根对象类型非法: {type(ckpt).__name__}")
    return ckpt


def main():
    start_time = time.time()
    last_backup_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gd = _effective_gdrive_dir()
    if os.environ.get("RESET_DONE_FLAG") == "1":
        if os.path.exists(DONE_FLAG):
            try:
                os.remove(DONE_FLAG)
            except OSError:
                pass
        if not (SMOKE_TEST and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1"):
            subprocess.run(
                _rclone_base_cmd() + ["deletefile", "--", f"{gd}/{os.path.basename(DONE_FLAG)}"],
                capture_output=True,
                check=False,
            )

    write_training_state(
        status="pulling_checkpoint",
        pid=os.getpid(),
        gdrive_dir=gd,
        dataset_status="smoke_local" if SMOKE_TEST else "ready",
        sync_status="pulling",
        total_epochs=TOTAL_EPOCHS,
        max_session_seconds=MAX_SESSION_SECONDS,
        backup_interval_seconds=BACKUP_INTERVAL_SECONDS,
        session_start_time=start_time,
    )

    # 1. 开机自动从 Google Drive 拉取上一棒的 Checkpoint
    print("🔄 正在检查 Google Drive 中是否有历史断点文件...", flush=True)
    rclone_pull_checkpoint(LOCAL_CKPT)

    # （可选）若配置了 Google Drive 数据集目录，自动拉取到 /content/dataset
    if GDRIVE_DATASET_DIR and not SMOKE_TEST:
        write_training_state(dataset_status="pulling_from_gdrive")
        rclone_pull(GDRIVE_DATASET_DIR, "/content/dataset", required=True)
        write_training_state(dataset_status="synced_from_gdrive")

    # 2. 初始化模型、求解器与优化器（请替换为你的实际模型代码）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(128, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0

    # 3. 恢复历史断点状态（以实际是否存在有效断点文件为准）
    if os.path.exists(LOCAL_CKPT):
        if os.path.getsize(LOCAL_CKPT) == 0:
            raise RuntimeError(f"本地断点文件为空或已损坏: {LOCAL_CKPT}")
        print(f"📦 发现历史断点 {LOCAL_CKPT}，正在恢复状态...", flush=True)
        ckpt = load_checkpoint_safely(LOCAL_CKPT, map_location=device)
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
        if not math.isfinite(loss_val):
            write_training_state(status="diverged", current_epoch=epoch + 1, loss=None)
            raise RuntimeError(f"Epoch {epoch + 1} 检测到非有限 Loss ({loss_val})，已中止训练防止污染网盘断点")
        time.sleep(0.05 if SMOKE_TEST else 2.0)
        # ------------------------------------------------

        # 先高速保存到本地 checkpoint.pt 与 results/training_metrics.json
        save_checkpoint_safely(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_val,
            },
            LOCAL_CKPT,
        )

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
        if (
            (epoch == start_epoch and not SMOKE_TEST)
            or (now - last_backup_time >= BACKUP_INTERVAL_SECONDS)
            or (epoch == TOTAL_EPOCHS - 1)
        ):
            print(
                f"[Epoch {epoch + 1}] ☁️ 正在通过 rclone (16线程) 将最新权重与结果备份至 Google Drive ({gd})...",
                flush=True,
            )
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
            print(f"⏰ 当前实例已运行 {elapsed_session / 3600:.2f} 小时，即将触及 24h 上限。", flush=True)
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
    tmp_done = f"{DONE_FLAG}.{_session_isolation_prefix()}.tmp"
    with open(tmp_done, "w", encoding="utf-8") as f:
        f.write("COMPLETED")
    os.replace(tmp_done, DONE_FLAG)
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
