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
- 开机通过 rclone 从 Google Drive 拉取数据集与最新 checkpoint.pt（严格校验返回码，杜绝静默归零覆写）
- 训练期间实时原子更新 `/content/.colab_pilot_training_state.json`（供 Colab Pilot 仪表盘轮询）
- 周期性将 checkpoint.pt（带远端上一版轮转保护与临时文件原子改名）与 results/ 增量备份到 Google Drive
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
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _safe_float(v: Any) -> Optional[float]:
    """安全转换为有限浮点数（支持 Tensor / numpy 标量）；无法转换或 NaN/Inf 时返回 None。"""
    if v is None:
        return None
    try:
        f = float(v.item() if hasattr(v, "item") else v)
    except Exception:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _finite(obj: Any) -> Any:
    """递归清洗：非有限浮点数 (NaN / ±Infinity) 转为 None，保证输出严格合法的 JSON。"""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return _finite(obj.item())
        except Exception:
            return None
    return obj


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


def _atomic_write_text(path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


class ColabRelayContext:
    """通用深度学习串行接力上下文（支持本机 Smoke Test 与云端 Colab 正式接力）"""

    def __init__(
        self,
        model_name: str = "deep-learning-model",
        framework: str = "pytorch",
        total_epochs: Optional[int] = None,
        gdrive_dir: Optional[str] = None,
        gdrive_dataset_dir: Optional[str] = None,
        local_dataset_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        done_flag_path: Optional[str] = None,
        max_session_seconds: Optional[float] = None,
        backup_interval_seconds: Optional[float] = None,
        smoke_test: Optional[bool] = None,
        smoke_pass: Optional[int] = None,
        round_num: Optional[int] = None,
    ):
        self.model_name = model_name
        self.framework = framework
        self.total_epochs = (
            int(total_epochs)
            if total_epochs is not None
            else int(os.environ.get("TOTAL_EPOCHS", "100"))
        )
        raw_gd = gdrive_dir if gdrive_dir is not None else os.environ.get("GDRIVE_DIR", "gdrive:colab_relay")
        self.gdrive_dir = raw_gd.rstrip("/")
        raw_ds = gdrive_dataset_dir if gdrive_dataset_dir is not None else os.environ.get("GDRIVE_DATASET_DIR")
        self.gdrive_dataset_dir = raw_ds.rstrip("/") if raw_ds else None
        self.local_dataset_dir = (
            local_dataset_dir
            if local_dataset_dir is not None
            else os.environ.get("LOCAL_DATASET_DIR", "./dataset")
        )
        self.results_dir = (
            results_dir if results_dir is not None else os.environ.get("RESULTS_DIR", "./results")
        )
        self.checkpoint_path = (
            checkpoint_path
            if checkpoint_path is not None
            else os.environ.get("CHECKPOINT_PATH", "checkpoint.pt")
        )
        self.done_flag_path = (
            done_flag_path
            if done_flag_path is not None
            else os.environ.get("DONE_FLAG_PATH", "training_done.flag")
        )
        self.max_session_seconds = (
            float(max_session_seconds)
            if max_session_seconds is not None
            else float(os.environ.get("MAX_SESSION_SECONDS", str(int(22.5 * 3600))))
        )
        self.backup_interval_seconds = (
            float(backup_interval_seconds)
            if backup_interval_seconds is not None
            else float(os.environ.get("BACKUP_INTERVAL_SECONDS", "1800"))
        )
        self.smoke_test = (
            bool(smoke_test)
            if smoke_test is not None
            else (os.environ.get("SMOKE_TEST", "0") == "1")
        )
        self.smoke_pass = (
            int(smoke_pass) if smoke_pass is not None else int(os.environ.get("SMOKE_PASS", "0"))
        )
        self.round_num = (
            int(round_num) if round_num is not None else int(os.environ.get("RELAY_ROUND", "1"))
        )

        self.state_file = os.environ.get("TRAINING_STATE_FILE", _default_state_file())
        self.rclone_conf = os.environ.get("RCLONE_CONFIG", os.environ.get("RCLONE_CONF", _default_rclone_conf()))
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
        ckpt_parent = os.path.dirname(os.path.abspath(self.checkpoint_path))
        if ckpt_parent:
            os.makedirs(ckpt_parent, exist_ok=True)

    # ─────────────────────────── rclone 封装 ───────────────────────────

    def _rclone_base(self) -> List[str]:
        rclone_bin = shutil.which("rclone") or os.path.expanduser("~/.local/bin/rclone")
        if os.name == "nt" and not rclone_bin.endswith(".exe") and os.path.exists(rclone_bin + ".exe"):
            rclone_bin += ".exe"
        cmd = [rclone_bin]
        if self.rclone_conf and os.path.exists(self.rclone_conf):
            cmd.extend(["--config", self.rclone_conf])
        cmd.extend(["--retries=5", "--low-level-retries=20", "--stats=0"])
        return cmd

    def _run_rclone_proc(self, args: List[str], timeout: int = 180) -> Optional[subprocess.CompletedProcess]:
        if self.smoke_test and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        try:
            cmd = self._rclone_base() + ["-q"] + args
            return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except Exception as e:
            print(f"[Relay] rclone 告警: {e}", flush=True)
            return None

    def _run_rclone(self, args: List[str], timeout: int = 180) -> bool:
        res = self._run_rclone_proc(args, timeout=timeout)
        return res is not None and res.returncode == 0

    def _rclone_remote_file_exists(self, remote_dir: str, filename: str, timeout: int = 90) -> Optional[bool]:
        """
        检查远端目录是否存在指定文件：
        - 返回 True：远端明确存在该文件
        - 返回 False：远端目录不存在或文件不存在（首次训练正常状态）
        - 返回 None：rclone 发生网络/鉴权/限流错误（严禁当作“无断点”从头覆写！）
        """
        if self.smoke_test and os.environ.get("SMOKE_SKIP_RCLONE", "1") == "1":
            return False
        try:
            cmd = self._rclone_base() + ["lsf", "--files-only", "--", remote_dir]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except Exception as e:
            print(f"[Relay] 查询远端断点状态异常: {e}", flush=True)
            return None

        if res.returncode == 0:
            names = {ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()}
            return filename in names

        err_lower = ((res.stderr or "") + "\n" + (res.stdout or "")).lower()
        if "directory not found" in err_lower or "object not found" in err_lower or "file not found" in err_lower:
            return False
        print(f"[Relay] rclone lsf 失败 (code={res.returncode}): {(res.stderr or '').strip()[:240]}", flush=True)
        return None

    def prepare_cloud_workspace(self, reset_done_flag: bool = False) -> None:
        """开机第一阶段：从 Google Drive 拉取数据集与最新断点（C-3 强校验防覆写）"""
        reset_all = reset_done_flag or os.environ.get("RESET_DONE_FLAG", "0") == "1"
        if reset_all:
            if os.path.exists(self.done_flag_path):
                os.remove(self.done_flag_path)
            self._run_rclone(["deletefile", "--", f"{self.gdrive_dir}/{os.path.basename(self.done_flag_path)}"])

        if self.smoke_test:
            self.dataset_status = "smoke_local"
            self.report_state(status="smoke_init", current_epoch=0)
            return

        # 1. 拉取 Google Drive 数据集（若配置了 gdrive_dataset_dir，失败立即抛错止损）
        if self.gdrive_dataset_dir:
            self.dataset_status = "pulling_from_gdrive"
            self.report_state(status="pulling_dataset", current_epoch=0)
            os.makedirs(self.local_dataset_dir, exist_ok=True)
            remote_ds = (
                self.gdrive_dataset_dir
                if ":" in self.gdrive_dataset_dir
                else f"{self.gdrive_dir}/{self.gdrive_dataset_dir.lstrip('/')}"
            )
            print(f"[Relay] 正在从 {remote_ds} 拉取数据集到 {self.local_dataset_dir} ...", flush=True)
            ok = self._run_rclone(["copy", "--transfers=8", "--", remote_ds, self.local_dataset_dir], timeout=900)
            if not ok:
                self.dataset_status = "dataset_pull_failed"
                self.report_state(status="dataset_pull_failed", current_epoch=0)
                raise RuntimeError(f"从 Google Drive 拉取数据集失败 ({remote_ds})，已中止启动防止空数据集训练")
            self.dataset_status = "synced_from_gdrive"
        else:
            self.dataset_status = "ready"

        # 2. 拉取最新断点与历史 results（严格区分“首次无断点”与“网络/限流失败”）
        self.sync_status = "pulling_checkpoint"
        self.report_state(status="pulling_checkpoint", current_epoch=0)
        ckpt_name = os.path.basename(self.checkpoint_path)
        remote_exists = self._rclone_remote_file_exists(self.gdrive_dir, ckpt_name)

        if remote_exists is None:
            self.sync_status = "pull_failed"
            self.report_state(status="checkpoint_pull_failed", current_epoch=0)
            raise RuntimeError(
                f"无法查询 Google Drive ({self.gdrive_dir}) 中的断点状态，拒绝盲目从 Epoch 1 重跑覆盖网盘权重"
            )

        if self.round_num > 1 and not reset_all and not remote_exists:
            self.sync_status = "pull_failed"
            self.report_state(status="checkpoint_missing", current_epoch=0)
            raise RuntimeError(
                f"第 {self.round_num} 棒接力异常：Google Drive ({self.gdrive_dir}) 中不存在 {ckpt_name}，已中止启动防止从头重训"
            )

        if remote_exists:
            pulled = self._run_rclone(
                ["copyto", "--", f"{self.gdrive_dir}/{ckpt_name}", self.checkpoint_path],
                timeout=300,
            )
            if not pulled or not os.path.exists(self.checkpoint_path) or os.path.getsize(self.checkpoint_path) == 0:
                self.sync_status = "pull_failed"
                self.report_state(status="checkpoint_pull_failed", current_epoch=0)
                raise RuntimeError(
                    f"Google Drive 存在历史断点 {self.gdrive_dir}/{ckpt_name} 但拉取失败，已中止启动防止覆盖历史进度"
                )

        self._run_rclone(["copy", "--", f"{self.gdrive_dir}/results", self.results_dir], timeout=300)
        self.sync_status = "ready"

    def backup_to_gdrive(self, epoch: Optional[int] = None, include_results: bool = True) -> bool:
        """
        将当前 checkpoint.pt 与 results/ 备份到 Google Drive（C-3 事务级保护）：
        1. 先上传到远端 `.tmp` 临时对象，网络中断绝不破坏线上正式 `checkpoint.pt`；
        2. 将线上已有 `checkpoint.pt` 服务端轮转为 `checkpoint_prev.pt`；
        3. 将 `.tmp` 原子改名为正式 `checkpoint.pt`。
        """
        self.sync_status = "backing_up"
        ok = True
        if os.path.exists(self.checkpoint_path):
            ckpt_name = os.path.basename(self.checkpoint_path)
            stem, ext = os.path.splitext(ckpt_name)
            prev_name = f"{stem}_prev{ext or '.pt'}"
            remote_ckpt = f"{self.gdrive_dir}/{ckpt_name}"
            remote_tmp = f"{self.gdrive_dir}/.{ckpt_name}.uploading"
            remote_prev = f"{self.gdrive_dir}/{prev_name}"

            up_ok = self._run_rclone(["copyto", "--", self.checkpoint_path, remote_tmp], timeout=300)
            if up_ok:
                if self._rclone_remote_file_exists(self.gdrive_dir, ckpt_name) is True:
                    self._run_rclone(["copyto", "--", remote_ckpt, remote_prev], timeout=180)
                mv_ok = self._run_rclone(["moveto", "--", remote_tmp, remote_ckpt], timeout=180)
                ok = mv_ok and ok
            else:
                ok = False

        if include_results and os.path.isdir(self.results_dir) and os.listdir(self.results_dir):
            ok = self._run_rclone(["copy", "--", self.results_dir, f"{self.gdrive_dir}/results"], timeout=300) and ok
        if ok:
            self.last_backup_time = time.time()
            if epoch is not None:
                self.last_backup_epoch = epoch
        self.sync_status = "synced" if ok else "backup_warn"
        return ok

    # ─────────────────────────── 状态文件上报 ───────────────────────────

    def _update_best_metric(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not metrics:
            return
        for k in ("best_acc", "best_metric", "val_acc", "acc", "f1", "mAP"):
            if k in metrics:
                val = _safe_float(metrics[k])
                if val is None:
                    continue
                if self.best_metric is None or val > self.best_metric:
                    self.best_metric = round(val, 5)

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

        self._update_best_metric(metrics)

        eta_sec: Optional[float] = None
        if self.epoch_durations and current_epoch < self.total_epochs:
            avg_ep = sum(self.epoch_durations[-10:]) / len(self.epoch_durations[-10:])
            eta_sec = round(avg_ep * max(0, self.total_epochs - current_epoch), 1)

        loss_f = _safe_float(loss)
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
            "loss": round(loss_f, 6) if loss_f is not None else None,
            "best_metric": self.best_metric,
            "metrics": metrics or {},
            "lr": _safe_float(lr),
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
            raw = json.dumps(_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
            _atomic_write_text(self.state_file, raw)
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
        返回 (start_epoch, extra_dict)，若本地无断点文件则返回 (1, {})。
        注意：若本地存在断点文件但损坏或结构不匹配，立即抛出 RuntimeError，严禁吞错归零重训（C-3）！
        """
        if not os.path.exists(self.checkpoint_path):
            if self.round_num > 1 and os.environ.get("RESET_DONE_FLAG", "0") != "1":
                raise RuntimeError(
                    f"第 {self.round_num} 棒接力启动失败：本地未找到断点文件 {self.checkpoint_path}"
                )
            print(f"[Relay] 未发现历史断点 ({self.checkpoint_path})，将从 Epoch 1 开始全新的训练。", flush=True)
            return 1, {}

        try:
            import torch

            allow_unsafe = os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "0") == "1"
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=map_location, weights_only=True)
            except TypeError:
                if not allow_unsafe:
                    raise RuntimeError(
                        "当前 PyTorch 版本不支持安全反序列化 (weights_only=True)；如确认断点来源可信，请设置环境变量 ALLOW_UNSAFE_TORCH_LOAD=1"
                    )
                ckpt = torch.load(self.checkpoint_path, map_location=map_location)
            except Exception as safe_err:
                if allow_unsafe:
                    ckpt = torch.load(self.checkpoint_path, map_location=map_location, weights_only=False)
                else:
                    raise RuntimeError(
                        f"安全反序列化 (weights_only=True) 拒绝加载 {self.checkpoint_path}"
                        f"（如断点包含自定义类且确认 Google Drive 来源可信，可设置 ALLOW_UNSAFE_TORCH_LOAD=1）: {safe_err}"
                    ) from safe_err

            if not isinstance(ckpt, dict):
                raise ValueError(f"断点根对象类型非法: {type(ckpt).__name__}")

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

            loaded_best = _safe_float(ckpt.get("best_metric"))
            if loaded_best is not None:
                self.best_metric = loaded_best
            start_epoch = last_epoch + 1
            print(
                f"[Relay] 成功从断点恢复：上次完成 Epoch {last_epoch}，本次从 Epoch {start_epoch}/{self.total_epochs} 续训！",
                flush=True,
            )
            return start_epoch, ckpt.get("extra", {})
        except Exception as e:
            raise RuntimeError(
                f"加载已有断点文件 {self.checkpoint_path} 失败（已中止启动，防止从 Epoch 1 重跑覆盖 Google Drive 权重）: {e}"
            ) from e

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

        self._update_best_metric(metrics)

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
            "loss": _safe_float(loss),
            "best_metric": self.best_metric,
            "metrics": _finite(metrics or {}),
            "extra": extra or {},
            "timestamp": time.time(),
            "models": [m.state_dict() for m in model_list if hasattr(m, "state_dict")],
            "optimizers": [o.state_dict() for o in opt_list if hasattr(o, "state_dict")],
            "schedulers": [s.state_dict() for s in sch_list if hasattr(s, "state_dict")],
            "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        }
        if state["models"]:
            state["model_state_dict"] = state["models"][0]
        if state["optimizers"]:
            state["optimizer_state_dict"] = state["optimizers"][0]

        ckpt_parent = os.path.dirname(os.path.abspath(self.checkpoint_path))
        if ckpt_parent:
            os.makedirs(ckpt_parent, exist_ok=True)
        tmp = f"{self.checkpoint_path}.{os.getpid()}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.checkpoint_path)

        summary_file = os.path.join(self.results_dir, "training_metrics.json")
        try:
            loss_f = _safe_float(loss)
            raw_summary = json.dumps(
                _finite(
                    {
                        "model_name": self.model_name,
                        "last_epoch": epoch,
                        "total_epochs": self.total_epochs,
                        "loss": round(loss_f, 6) if loss_f is not None else None,
                        "best_metric": self.best_metric,
                        "metrics": metrics or {},
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            _atomic_write_text(summary_file, raw_summary)
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
        now = time.time()
        self.epoch_durations.append(max(0.01, now - self._last_epoch_ts))
        self._last_epoch_ts = now
        self._update_best_metric(metrics)

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

        # Colab 云端真机首步冒烟（Cloud First-Step Smoke）
        if not self.smoke_test and not self.cloud_smoke_verified:
            print(f"[CloudSmoke] 正在执行 Colab 云端真机首步网盘写入核验 (Epoch {epoch}) ...", flush=True)
            if not self.backup_to_gdrive(epoch=epoch):
                self.report_state(status="cloud_smoke_failed", current_epoch=epoch, loss=loss)
                print("CLOUD_SMOKE_FAILED", flush=True)
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
            if not self.backup_to_gdrive(epoch=epoch):
                raise RuntimeError("换棒前上传断点至 Google Drive 失败，已中止换棒标记输出以保护进度")
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
        flag_text = (
            f"completed_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"epochs={final_epoch}\n"
        )
        _atomic_write_text(self.done_flag_path, flag_text)

        if not self.backup_to_gdrive(epoch=final_epoch, include_results=True):
            raise RuntimeError("最终模型权重上传 Google Drive 失败")
        done_name = os.path.basename(self.done_flag_path)
        if not self._run_rclone(["copyto", "--", self.done_flag_path, f"{self.gdrive_dir}/{done_name}"], timeout=120):
            raise RuntimeError("training_done.flag 上传 Google Drive 失败")

        self.report_state(
            status="completed",
            current_epoch=final_epoch,
            loss=final_loss,
            metrics=metrics,
        )
        print("TRAINING_COMPLETED", flush=True)


# ─────────────────────── 模式 B：HuggingFace / Lightning 框架回调工厂 ───────────────────────


def make_hf_trainer_callback(relay_ctx: ColabRelayContext):
    """
    生成可直接传入 HuggingFace `Trainer(callbacks=[...])` 的接力回调（H-10）：
    - `on_epoch_end` 仅设置 `control.should_save = True` 请求 Trainer 落盘 checkpoint，不在保存前空备份；
    - `on_save` 在 Trainer 已完成磁盘写入后同步 `args.output_dir` 与断点到 Google Drive，并输出 `CLOUD_SMOKE_PASSED` / `HANDOVER_REQUESTED`。
    """
    from transformers import TrainerCallback

    class _HFRelayCallback(TrainerCallback):
        def __init__(self) -> None:
            super().__init__()
            self._handover_pending = False

        @staticmethod
        def _latest_loss(state: Any) -> Optional[float]:
            if getattr(state, "log_history", None):
                for item in reversed(state.log_history):
                    if isinstance(item, dict) and "loss" in item:
                        return _safe_float(item["loss"])
            return None

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(round(state.epoch or 0))
            loss = self._latest_loss(state)
            control.should_save = True
            relay_ctx.report_state(
                status="training",
                current_epoch=epoch,
                loss=loss,
                current_step=state.global_step,
                total_steps=state.max_steps,
            )
            if (time.time() - relay_ctx.start_time) >= relay_ctx.max_session_seconds:
                self._handover_pending = True
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            epoch = int(round(state.epoch or 0))
            loss = self._latest_loss(state)
            out_dir = getattr(args, "output_dir", None)
            sync_ok = True
            if out_dir and os.path.isdir(out_dir):
                sync_ok = relay_ctx._run_rclone(
                    ["copy", "--", out_dir, f"{relay_ctx.gdrive_dir}/hf_checkpoints"],
                    timeout=600,
                ) and sync_ok
            sync_ok = relay_ctx.backup_to_gdrive(epoch=epoch, include_results=True) and sync_ok

            if not relay_ctx.smoke_test and not relay_ctx.cloud_smoke_verified:
                if not sync_ok:
                    relay_ctx.report_state(status="cloud_smoke_failed", current_epoch=epoch, loss=loss)
                    print("CLOUD_SMOKE_FAILED", flush=True)
                    raise RuntimeError("HuggingFace 云端首步冒烟网盘备份失败")
                relay_ctx.cloud_smoke_verified = True
                print(
                    f"CLOUD_SMOKE_PASSED [Step {state.global_step}] ✅ HuggingFace Trainer 云端断点已同步至 Google Drive！",
                    flush=True,
                )

            if self._handover_pending:
                if not sync_ok:
                    raise RuntimeError("HuggingFace 换棒前网盘同步失败")
                relay_ctx.report_state(status="handover", current_epoch=epoch, loss=loss)
                print("HANDOVER_REQUESTED", flush=True)
            else:
                relay_ctx.report_state(
                    status="training",
                    current_epoch=epoch,
                    loss=loss,
                    current_step=state.global_step,
                    total_steps=state.max_steps,
                )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if not self._handover_pending:
                relay_ctx.finish_training(final_epoch=int(round(state.epoch or relay_ctx.total_epochs)))

    return _HFRelayCallback()


# ─────────────────────── 模式 C：黑盒命令行包装器 (--wrap) ───────────────────────


def _graceful_interrupt_proc(proc: subprocess.Popen) -> None:
    """跨平台优雅中断训练子进程组（Windows 使用 CTRL_BREAK_EVENT / terminate，杜绝 ValueError）"""
    try:
        if os.name == "nt":
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    proc.send_signal(ctrl_break)
                    return
                except Exception:
                    pass
            proc.terminate()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                proc.send_signal(signal.SIGINT)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _force_kill_proc(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,
                timeout=15,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    except Exception:
        pass


def run_blackbox_wrapper(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 黑盒深度学习训练接力包装器")
    parser.add_argument("--wrap", required=True, help="原始训练命令，例如 'python main.py --epochs 100'")
    parser.add_argument("--model-name", default="custom-dl-job", help="模型或实验名称")
    parser.add_argument("--framework", default="pytorch-cli", help="训练框架标识")
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--gdrive-dir", default=None)
    parser.add_argument("--gdrive-dataset-dir", default=None)
    parser.add_argument("--local-dataset-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args(argv)

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
    loss_re = re.compile(r"(?:loss)[\s:=]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE)

    cur_epoch = 0
    cur_loss: Optional[float] = None
    handover_triggered = False
    watchdog: Optional[threading.Timer] = None

    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        args.wrap,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )

    try:
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
                cur_loss = _safe_float(m_loss.group(1))

            if m_ep or m_loss:
                ctx.report_state(status="training", current_epoch=cur_epoch, loss=cur_loss)

            now = time.time()
            if (
                not ctx.smoke_test
                and not ctx.cloud_smoke_verified
                and cur_epoch >= 1
                and (os.path.exists(ctx.checkpoint_path) or os.listdir(ctx.results_dir))
            ):
                if ctx.backup_to_gdrive(epoch=cur_epoch, include_results=True):
                    ctx.cloud_smoke_verified = True
                    print(f"CLOUD_SMOKE_PASSED [Epoch {cur_epoch}] ✅ 黑盒训练首步网盘备份核验通过！", flush=True)

            if not handover_triggered and (now - ctx.last_backup_time) >= ctx.backup_interval_seconds:
                ctx.backup_to_gdrive(epoch=cur_epoch, include_results=True)

            if (now - ctx.start_time) >= ctx.max_session_seconds and not handover_triggered:
                handover_triggered = True
                print("[RelayWrapper] 已达单实例最大安全时长，发送中断信号并继续排空日志等待安全落盘...", flush=True)
                _graceful_interrupt_proc(proc)
                watchdog = threading.Timer(60.0, _force_kill_proc, args=(proc,))
                watchdog.daemon = True
                watchdog.start()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

    rc = proc.wait()
    if handover_triggered:
        if not ctx.backup_to_gdrive(epoch=cur_epoch, include_results=True):
            return 1
        ctx.report_state(status="handover", current_epoch=cur_epoch, loss=cur_loss)
        print("HANDOVER_REQUESTED", flush=True)
        return 0
    if rc == 0:
        ctx.finish_training(final_epoch=cur_epoch or ctx.total_epochs, final_loss=cur_loss)
    return rc


if __name__ == "__main__":
    sys.exit(run_blackbox_wrapper())
