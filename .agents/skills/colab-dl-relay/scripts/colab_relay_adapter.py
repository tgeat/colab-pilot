#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 通用深度学习跨实例串行接力适配器 (colab_relay_adapter.py)
==========================================================================
支持三种预配置接入模式（在放入 Colab 之前于本机完成改造与 Smoke Test）：
1. 模式 A (White-box Hook)：适合自写 PyTorch / JAX / TensorFlow 训练循环
2. 模式 B (Framework Callback)：适合 HuggingFace Trainer / PyTorch Lightning / Ultralytics YOLO
3. 模式 C (Black-box CLI Wrapper)：零代码侵入包装任意第三方命令行训练工程

核心职责：
- 开机通过 rclone 从 Google Drive 拉取数据集与最新 checkpoint.pt
- 训练期间实时原子更新 `/content/.colab_pilot_training_state.json`（供 Colab Pilot 仪表盘 4s 轮询）
- 周期性将 checkpoint.pt 与 results/ 增量备份到 Google Drive
- 满 22.5 小时（或本机 Smoke Test 换棒验证）安全保存快照并输出 HANDOVER_REQUESTED
- 全部 Epoch 完成后上传最终权重、results/ 与 training_done.flag，输出 TRAINING_COMPLETED
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _default_state_file() -> str:
    if os.path.isdir("/content"):
        return "/content/.colab_pilot_training_state.json"
    return os.path.abspath(".colab_pilot_training_state.json")


def _default_rclone_conf() -> str:
    if os.path.exists("/content/rclone.conf"):
        return "/content/rclone.conf"
    appdata = os.environ.get("APPDATA")
    if appdata:
        cand = os.path.join(appdata, "rclone", "rclone.conf")
        if os.path.exists(cand):
            return cand
    return os.path.expanduser("~/.config/rclone/rclone.conf")


class ColabRelayContext:
    """通用深度学习串行接力上下文（支持本机 Smoke Test 与云端 Colab 正式接力）"""

    def __init__(
        self,
        model_name: str = "deep-learning-model",
        framework: str = "pytorch",
        total_epochs: int = int(os.environ.get("TOTAL_EPOCHS", "100")),
        gdrive_dir: str = os.environ.get("GDRIVE_DIR", "gdrive:colab_relay"),
        gdrive_dataset_dir: Optional[str] = os.environ.get("GDRIVE_DATASET_DIR"),
        local_dataset_dir: str = os.environ.get("LOCAL_DATASET_DIR", "./dataset"),
        results_dir: str = os.environ.get("RESULTS_DIR", "./results"),
        checkpoint_path: str = os.environ.get("CHECKPOINT_PATH", "checkpoint.pt"),
        done_flag_path: str = os.environ.get("DONE_FLAG_PATH", "training_done.flag"),
        max_session_seconds: float = float(os.environ.get("MAX_SESSION_SECONDS", str(int(22.5 * 3600)))),
        backup_interval_seconds: float = float(os.environ.get("BACKUP_INTERVAL_SECONDS", "1800")),
        smoke_test: bool = os.environ.get("SMOKE_TEST", "0") == "1",
        smoke_pass: int = int(os.environ.get("SMOKE_PASS", "0")),
        round_num: int = int(os.environ.get("RELAY_ROUND", "1")),
    ):
        self.model_name = model_name
        self.framework = framework
        self.total_epochs = total_epochs
        self.gdrive_dir = gdrive_dir.rstrip("/")
        self.gdrive_dataset_dir = gdrive_dataset_dir.rstrip("/") if gdrive_dataset_dir else None
        self.local_dataset_dir = local_dataset_dir
        self.results_dir = results_dir
        self.checkpoint_path = checkpoint_path
        self.done_flag_path = done_flag_path
        self.max_session_seconds = max_session_seconds
        self.backup_interval_seconds = backup_interval_seconds
        self.smoke_test = smoke_test
        self.smoke_pass = smoke_pass
        self.round_num = round_num

        self.state_file = os.environ.get("TRAINING_STATE_FILE", _default_state_file())
        self.rclone_conf = os.environ.get("RCLONE_CONFIG", _default_rclone_conf())
        self.start_time = time.time()
        self.last_backup_time = time.time()
        self.last_backup_epoch: Optional[int] = None
        self.dataset_status = "local_ready" if self.smoke_test else "checking"
        self.sync_status = "idle"
        self.best_metric: Optional[float] = None
        self.cloud_smoke_verified: bool = False
        self.epoch_durations: List[float] = []
        self._last_epoch_ts = time.time()

        os.makedirs(self.results_dir, exist_ok=True)

    # ─────────────────────────── rclone 封装 ───────────────────────────

    def _rclone_base(self) -> List[str]:
        rclone_bin = shutil.which("rclone") or os.path.expanduser("~/.local/bin/rclone")
        if os.name == "nt" and not rclone_bin.endswith(".exe") and os.path.exists(rclone_bin + ".exe"):
            rclone_bin += ".exe"
        cmd = [rclone_bin]
        if self.rclone_conf and os.path.exists(self.rclone_conf):
            cmd.extend(["--config", self.rclone_conf])
        cmd.extend(["--retries=5", "--low-level-retries=20", "--stats=0", "-q"])
        return cmd

    def _run_rclone(self, args: List[str], timeout: int = 180) -> bool:
        if self.smoke_test and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
            return True
        try:
            cmd = self._rclone_base() + args
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return res.returncode == 0
        except Exception as e:
            print(f"[Relay] rclone 告警: {e}", flush=True)
            return False

    def prepare_cloud_workspace(self, reset_done_flag: bool = False) -> None:
        """开机第一阶段：从 Google Drive 拉取数据集与最新断点"""
        if reset_done_flag or os.environ.get("RESET_DONE_FLAG", "0") == "1":
            if os.path.exists(self.done_flag_path):
                os.remove(self.done_flag_path)
            self._run_rclone(["deletefile", f"{self.gdrive_dir}/{os.path.basename(self.done_flag_path)}"])

        if self.smoke_test:
            self.dataset_status = "smoke_local"
            self.report_state(status="smoke_init", current_epoch=0)
            return

        # 1. 拉取 Google Drive 数据集（若配置了 gdrive_dataset_dir）
        if self.gdrive_dataset_dir:
            self.dataset_status = "pulling_from_gdrive"
            self.report_state(status="pulling_dataset", current_epoch=0)
            os.makedirs(self.local_dataset_dir, exist_ok=True)
            print(f"[Relay] 正在从 {self.gdrive_dataset_dir} 拉取数据集到 {self.local_dataset_dir} ...", flush=True)
            ok = self._run_rclone(["copy", self.gdrive_dataset_dir, self.local_dataset_dir, "--transfers=8"], timeout=900)
            self.dataset_status = "synced_from_gdrive" if ok else "dataset_pull_warn"
        else:
            self.dataset_status = "ready"

        # 2. 拉取最新断点与历史 results
        self.sync_status = "pulling_checkpoint"
        self.report_state(status="pulling_checkpoint", current_epoch=0)
        ckpt_name = os.path.basename(self.checkpoint_path)
        self._run_rclone(["copyto", f"{self.gdrive_dir}/{ckpt_name}", self.checkpoint_path], timeout=300)
        self._run_rclone(["copy", f"{self.gdrive_dir}/results", self.results_dir], timeout=300)
        self.sync_status = "ready"

    def backup_to_gdrive(self, epoch: Optional[int] = None, include_results: bool = True) -> bool:
        """将当前 checkpoint.pt 与 results/ 增量备份到 Google Drive"""
        self.sync_status = "backing_up"
        ok = True
        if os.path.exists(self.checkpoint_path):
            ckpt_name = os.path.basename(self.checkpoint_path)
            ok = self._run_rclone(["copyto", self.checkpoint_path, f"{self.gdrive_dir}/{ckpt_name}"], timeout=300) and ok
        if include_results and os.path.isdir(self.results_dir) and os.listdir(self.results_dir):
            ok = self._run_rclone(["copy", self.results_dir, f"{self.gdrive_dir}/results"], timeout=300) and ok
        self.last_backup_time = time.time()
        if epoch is not None:
            self.last_backup_epoch = epoch
        self.sync_status = "synced" if ok else "backup_warn"
        return ok

    # ─────────────────────────── 状态文件上报 ───────────────────────────

    def report_state(
        self,
        status: str = "training",
        current_epoch: int = 0,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        now = time.time()
        elapsed = round(now - self.start_time, 1)
        next_bk = max(0.0, round(self.backup_interval_seconds - (now - self.last_backup_time), 1))
        handover_in = max(0.0, round(self.max_session_seconds - elapsed, 1))

        if metrics:
            for k in ("best_acc", "best_metric", "val_acc", "acc", "f1", "mAP"):
                if k in metrics and isinstance(metrics[k], (int, float)):
                    val = float(metrics[k])
                    if self.best_metric is None or val > self.best_metric:
                        self.best_metric = round(val, 5)

        eta_sec: Optional[float] = None
        if self.epoch_durations and current_epoch < self.total_epochs:
            avg_ep = sum(self.epoch_durations[-10:]) / len(self.epoch_durations[-10:])
            eta_sec = round(avg_ep * max(0, self.total_epochs - current_epoch), 1)

        payload = {
            "mode": "dl_relay",
            "framework": self.framework,
            "model_name": self.model_name,
            "status": status,
            "round": self.round_num,
            "current_epoch": current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": current_step,
            "total_steps": total_steps,
            "loss": round(float(loss), 6) if loss is not None else None,
            "best_metric": self.best_metric,
            "metrics": metrics or {},
            "lr": lr,
            "gdrive_dir": self.gdrive_dir,
            "dataset_dir": self.gdrive_dataset_dir or self.local_dataset_dir,
            "results_dir": self.results_dir,
            "dataset_status": self.dataset_status,
            "sync_status": self.sync_status,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_sec,
            "next_backup_in_seconds": next_bk,
            "handover_in_seconds": handover_in,
            "last_backup_epoch": self.last_backup_epoch,
            "last_backup_time": round(self.last_backup_time, 1),
            "timestamp": now,
        }
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_file)
        except Exception:
            pass

    # ─────────────────────────── PyTorch / 通用断点管理 ───────────────────────────

    def load_torch_checkpoint(
        self,
        models: Any = None,
        optimizers: Any = None,
        schedulers: Any = None,
        scaler: Any = None,
        map_location: str = "cpu",
    ) -> Tuple[int, Dict[str, Any]]:
        """
        加载 PyTorch 断点（支持单模型/多模型、单优化器/多优化器、LR Scheduler、AMP GradScaler）。
        返回 (start_epoch, extra_dict)，若无断点则返回 (1, {})。
        """
        if not os.path.exists(self.checkpoint_path):
            print(f"[Relay] 未发现历史断点 ({self.checkpoint_path})，将从 Epoch 1 开始全新的训练。", flush=True)
            return 1, {}

        try:
            import torch

            ckpt = torch.load(self.checkpoint_path, map_location=map_location, weights_only=False)
            last_epoch = int(ckpt.get("epoch", 0))

            if models is not None and "models" in ckpt:
                model_list = models if isinstance(models, (list, tuple)) else [models]
                for idx, m in enumerate(model_list):
                    if idx < len(ckpt["models"]) and hasattr(m, "load_state_dict"):
                        m.load_state_dict(ckpt["models"][idx])
            elif models is not None and "model_state_dict" in ckpt:
                m = models[0] if isinstance(models, (list, tuple)) else models
                if hasattr(m, "load_state_dict"):
                    m.load_state_dict(ckpt["model_state_dict"])

            if optimizers is not None and "optimizers" in ckpt:
                opt_list = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
                for idx, opt in enumerate(opt_list):
                    if idx < len(ckpt["optimizers"]) and hasattr(opt, "load_state_dict"):
                        opt.load_state_dict(ckpt["optimizers"][idx])
            elif optimizers is not None and "optimizer_state_dict" in ckpt:
                opt = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
                if hasattr(opt, "load_state_dict"):
                    opt.load_state_dict(ckpt["optimizer_state_dict"])

            if schedulers is not None and "schedulers" in ckpt:
                sch_list = schedulers if isinstance(schedulers, (list, tuple)) else [schedulers]
                for idx, sch in enumerate(sch_list):
                    if idx < len(ckpt["schedulers"]) and hasattr(sch, "load_state_dict"):
                        sch.load_state_dict(ckpt["schedulers"][idx])

            if scaler is not None and ckpt.get("scaler") and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(ckpt["scaler"])

            self.best_metric = ckpt.get("best_metric", self.best_metric)
            start_epoch = last_epoch + 1
            print(
                f"[Relay] 成功从断点恢复：上次完成 Epoch {last_epoch}，本次从 Epoch {start_epoch}/{self.total_epochs} 续训！",
                flush=True,
            )
            return start_epoch, ckpt.get("extra", {})
        except Exception as e:
            print(f"[Relay] 加载断点异常 ({e})，将从 Epoch 1 开始。", flush=True)
            return 1, {}

    def save_torch_checkpoint(
        self,
        epoch: int,
        loss: float,
        models: Any = None,
        optimizers: Any = None,
        schedulers: Any = None,
        scaler: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """原子保存 PyTorch 断点文件，防止断电产生半截文件"""
        import torch

        model_list = (
            list(models)
            if isinstance(models, (list, tuple))
            else ([models] if models is not None else [])
        )
        opt_list = (
            list(optimizers)
            if isinstance(optimizers, (list, tuple))
            else ([optimizers] if optimizers is not None else [])
        )
        sch_list = (
            list(schedulers)
            if isinstance(schedulers, (list, tuple))
            else ([schedulers] if schedulers is not None else [])
        )

        state: Dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "loss": float(loss),
            "best_metric": self.best_metric,
            "metrics": metrics or {},
            "extra": extra or {},
            "timestamp": time.time(),
            "models": [m.state_dict() for m in model_list if hasattr(m, "state_dict")],
            "optimizers": [o.state_dict() for o in opt_list if hasattr(o, "state_dict")],
            "schedulers": [s.state_dict() for s in sch_list if hasattr(s, "state_dict")],
            "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        }
        # 兼容旧版单模型字段
        if state["models"]:
            state["model_state_dict"] = state["models"][0]
        if state["optimizers"]:
            state["optimizer_state_dict"] = state["optimizers"][0]

        tmp = self.checkpoint_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.checkpoint_path)

        # 同时在 results/ 下写入训练指标摘要
        summary_file = os.path.join(self.results_dir, "training_metrics.json")
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": self.model_name,
                        "last_epoch": epoch,
                        "total_epochs": self.total_epochs,
                        "loss": round(float(loss), 6),
                        "best_metric": self.best_metric,
                        "metrics": metrics or {},
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def step_end(
        self,
        epoch: int,
        loss: float,
        models: Any = None,
        optimizers: Any = None,
        schedulers: Any = None,
        scaler: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        在每个 Epoch 结束时调用：
        1. 原子保存本地断点与 results/training_metrics.json
        2. 更新 `/content/.colab_pilot_training_state.json`
        3. 判断是否需要定期备份到 Google Drive
        4. 判断是否达到 22.5 小时安全换棒阈值（或 Smoke Pass #1 断点保存验证）
        返回 True 表示需要立即中断当前实例交棒给下一台实例。
        """
        now = time.time()
        self.epoch_durations.append(max(0.01, now - self._last_epoch_ts))
        self._last_epoch_ts = now

        if models is not None:
            self.save_torch_checkpoint(
                epoch=epoch,
                loss=loss,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                metrics=metrics,
                extra=extra,
            )

        self.report_state(
            status="training",
            current_epoch=epoch,
            loss=loss,
            metrics=metrics,
            lr=lr,
        )

        # Smoke Test Pass 1：跑完第 1 个 Epoch 立即触发模拟换棒，验证断点落盘
        if self.smoke_test and self.smoke_pass == 1 and epoch >= 1:
            self.backup_to_gdrive(epoch=epoch)
            self.report_state(status="handover", current_epoch=epoch, loss=loss, metrics=metrics, lr=lr)
            print(f"[SmokeTest] Pass #1 已在 Epoch {epoch} 保存断点并写入遥测状态。", flush=True)
            print("HANDOVER_REQUESTED", flush=True)
            return True

        # Smoke Test Pass 2：验证从 Epoch 2 续训成功后即可提前通过
        if self.smoke_test and self.smoke_pass == 2 and epoch >= 2:
            self.finish_training(final_epoch=epoch, final_loss=loss, metrics=metrics)
            return True

        # Colab 云端真机首步冒烟（Cloud First-Step Smoke）：
        # 在云端跑完第 1 轮后立即强制试写一次 Google Drive，验证真实 GPU 显存、数据流与网盘写入完全畅通
        if not self.smoke_test and not self.cloud_smoke_verified:
            print(f"[CloudSmoke] 正在执行 Colab 云端真机首步网盘写入核验 (Epoch {epoch}) ...", flush=True)
            if not self.backup_to_gdrive(epoch=epoch):
                self.report_state(status="cloud_smoke_failed", current_epoch=epoch, loss=loss)
                raise RuntimeError("Colab 云端真机首步冒烟失败：无法将 checkpoint.pt 同步至 Google Drive")
            self.cloud_smoke_verified = True
            self.report_state(status="training", current_epoch=epoch, loss=loss, metrics=metrics, lr=lr)
            print(
                f"CLOUD_SMOKE_PASSED [Epoch {epoch}] ✅ Colab 云端真机首步冒烟通过（GPU 显存、数据流与网盘断点同步均正常），无缝进入正式训练！",
                flush=True,
            )

        # 周期性 Google Drive 热备份
        if (now - self.last_backup_time) >= self.backup_interval_seconds:
            print(f"[Relay] 触发定时网盘热备份 (Epoch {epoch}) ...", flush=True)
            self.backup_to_gdrive(epoch=epoch)
            self.report_state(status="training", current_epoch=epoch, loss=loss, metrics=metrics, lr=lr)

        # 达到单实例安全时长阈值（默认 22.5 小时），触发先存后拆换棒
        if (now - self.start_time) >= self.max_session_seconds and epoch < self.total_epochs:
            print(
                f"[Relay] 当前实例已达到安全换棒时长 ({self.max_session_seconds}s)，正在上传断点至 {self.gdrive_dir} ...",
                flush=True,
            )
            self.backup_to_gdrive(epoch=epoch)
            self.report_state(status="handover", current_epoch=epoch, loss=loss, metrics=metrics, lr=lr)
            print("HANDOVER_REQUESTED", flush=True)
            return True

        return False

    def finish_training(
        self,
        final_epoch: int,
        final_loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """训练全部完成：上传最终模型、results/ 与 training_done.flag"""
        with open(self.done_flag_path, "w", encoding="utf-8") as f:
            f.write(f"completed_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"epochs={final_epoch}\n")

        self.backup_to_gdrive(epoch=final_epoch, include_results=True)
        done_name = os.path.basename(self.done_flag_path)
        self._run_rclone(["copyto", self.done_flag_path, f"{self.gdrive_dir}/{done_name}"], timeout=120)

        self.report_state(
            status="completed",
            current_epoch=final_epoch,
            loss=final_loss,
            metrics=metrics,
        )
        print("TRAINING_COMPLETED", flush=True)


# ─────────────────────── 模式 B：HuggingFace / Lightning 框架回调工厂 ───────────────────────


def make_hf_trainer_callback(relay_ctx: ColabRelayContext):
    """生成可直接传入 HuggingFace `Trainer(callbacks=[...])` 的接力回调"""
    from transformers import TrainerCallback

    class _HFRelayCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(round(state.epoch or 0))
            loss = None
            if state.log_history:
                for item in reversed(state.log_history):
                    if "loss" in item:
                        loss = float(item["loss"])
                        break
            control.should_save = True
            relay_ctx.backup_to_gdrive(epoch=epoch, include_results=True)
            relay_ctx.report_state(
                status="training",
                current_epoch=epoch,
                loss=loss,
                current_step=state.global_step,
                total_steps=state.max_steps,
            )
            if (time.time() - relay_ctx.start_time) >= relay_ctx.max_session_seconds:
                relay_ctx.report_state(status="handover", current_epoch=epoch, loss=loss)
                print("HANDOVER_REQUESTED", flush=True)
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if (time.time() - relay_ctx.start_time) < relay_ctx.max_session_seconds:
                relay_ctx.finish_training(final_epoch=int(round(state.epoch or relay_ctx.total_epochs)))

    return _HFRelayCallback()


# ─────────────────────── 模式 C：黑盒命令行包装器 (--wrap) ───────────────────────


def run_blackbox_wrapper() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 黑盒深度学习训练接力包装器")
    parser.add_argument("--wrap", required=True, help="原始训练命令，例如 'python main.py --epochs 100'")
    parser.add_argument("--model-name", default="custom-dl-job", help="模型或实验名称")
    parser.add_argument("--framework", default="pytorch-cli", help="训练框架标识")
    parser.add_argument("--total-epochs", type=int, default=int(os.environ.get("TOTAL_EPOCHS", "100")))
    parser.add_argument("--gdrive-dir", default=os.environ.get("GDRIVE_DIR", "gdrive:colab_relay"))
    parser.add_argument("--gdrive-dataset-dir", default=os.environ.get("GDRIVE_DATASET_DIR"))
    parser.add_argument("--local-dataset-dir", default=os.environ.get("LOCAL_DATASET_DIR", "./dataset"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("CHECKPOINT_PATH", "checkpoint.pt"))
    args = parser.parse_args()

    ctx = ColabRelayContext(
        model_name=args.model_name,
        framework=args.framework,
        total_epochs=args.total_epochs,
        gdrive_dir=args.gdrive_dir,
        gdrive_dataset_dir=args.gdrive_dataset_dir,
        local_dataset_dir=args.local_dataset_dir,
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint_path,
    )
    ctx.prepare_cloud_workspace()

    epoch_re = re.compile(r"(?:epoch|ep)[\s:=]+(\d+)(?:/(\d+))?", re.IGNORECASE)
    loss_re = re.compile(r"(?:loss)[\s:=]+([0-9]+\.[0-9]+)", re.IGNORECASE)

    cur_epoch = 0
    cur_loss: Optional[float] = None
    handover_triggered = False

    proc = subprocess.Popen(
        args.wrap,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m_ep = epoch_re.search(line)
        if m_ep:
            cur_epoch = int(m_ep.group(1))
            if m_ep.group(2):
                ctx.total_epochs = int(m_ep.group(2))
        m_loss = loss_re.search(line)
        if m_loss:
            try:
                cur_loss = float(m_loss.group(1))
            except ValueError:
                pass
        if m_ep or m_loss:
            ctx.report_state(status="training", current_epoch=cur_epoch, loss=cur_loss)

        if (time.time() - ctx.start_time) >= ctx.max_session_seconds and not handover_triggered:
            handover_triggered = True
            print("[RelayWrapper] 已达单实例最大安全时长，发送 SIGINT 保存退出...", flush=True)
            proc.send_signal(signal.SIGINT)
            break

    rc = proc.wait()
    if handover_triggered:
        ctx.backup_to_gdrive(epoch=cur_epoch, include_results=True)
        ctx.report_state(status="handover", current_epoch=cur_epoch, loss=cur_loss)
        print("HANDOVER_REQUESTED", flush=True)
        return 0
    if rc == 0:
        ctx.finish_training(final_epoch=cur_epoch or ctx.total_epochs, final_loss=cur_loss)
    return rc


if __name__ == "__main__":
    sys.exit(run_blackbox_wrapper())
