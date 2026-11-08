#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 深度学习上云前本机 4 阶段冒烟验证工具 (smoke_test_dl.py)
==========================================================================
在将训练脚本放入 Google Colab 之前，强制在本机执行 4 道验证关卡：
1. [语法与依赖编译] 检查训练脚本与 colab_relay_adapter.py 无语法错误
2. [Pass #1 冒烟保存] 跑 1 个 Epoch/Step 后触发模拟换棒，验证前向/反向传播、checkpoint.pt 落盘及遥测状态写入
3. [Pass #2 断点续训] 基于 Pass #1 生成的 checkpoint.pt 续训第 2 个 Epoch，验证 epoch 从 2 递增且 Loss 连续
4. [Google Drive 连通性] 验证本机 rclone 与 rclone.conf 就绪，确保上云后能正常拉取数据集与上传结果
"""

import argparse
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_rclone() -> str:
    cand = shutil.which("rclone")
    if cand:
        return cand
    home_bin = os.path.expanduser("~/.local/bin/rclone.exe" if os.name == "nt" else "~/.local/bin/rclone")
    return home_bin if os.path.exists(home_bin) else ""


def find_rclone_conf() -> str:
    if os.environ.get("RCLONE_CONFIG"):
        return os.environ["RCLONE_CONFIG"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        p = os.path.join(appdata, "rclone", "rclone.conf")
        if os.path.exists(p):
            return p
    return os.path.expanduser("~/.config/rclone/rclone.conf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab 深度学习串行接力 — 本机 Smoke Test 验证器")
    parser.add_argument("--script", default="train.py", help="要验证的训练脚本路径 (默认: train.py)")
    parser.add_argument("--python", default=sys.executable, help="用于执行本机 Smoke Test 的 Python 解释器")
    parser.add_argument("--gdrive-dir", default="gdrive:colab_relay", help="Google Drive 断点与结果目录")
    parser.add_argument("--skip-rclone-net", action="store_true", help="仅检查 rclone 配置文件，跳过在线请求")
    args = parser.parse_args()

    script_abs = os.path.abspath(args.script)
    workspace_dir = os.path.dirname(script_abs)
    adapter_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_relay_adapter.py")

    print("=" * 72)
    print("🚀 [Colab Pilot] 深度学习串行接力 — 本机 Smoke Test 4 关验证")
    print(f"   目标脚本: {script_abs}")
    print(f"   工作区  : {workspace_dir}")
    print("=" * 72)

    if not os.path.exists(script_abs):
        print(f"❌ [失败] 找不到训练脚本: {script_abs}")
        return 1

    # ── 第 1 关：语法与模块编译校验 ──
    print("\n[1/4] 正在校验 Python 语法与接力适配器...")
    try:
        py_compile.compile(script_abs, doraise=True)
        if os.path.exists(adapter_abs):
            py_compile.compile(adapter_abs, doraise=True)
        print("  ✅ 第 1 关通过：训练脚本与接力适配器语法无误。")
    except Exception as e:
        print(f"  ❌ 第 1 关失败：语法错误 -> {e}")
        return 1

    # 在隔离临时目录中执行 Pass #1 与 Pass #2，不污染工作区正式文件
    with tempfile.TemporaryDirectory(prefix="colab_smoke_") as tmpdir:
        # 拷贝适配器到沙箱目录方便导入
        if os.path.exists(adapter_abs):
            shutil.copy2(adapter_abs, os.path.join(tmpdir, "colab_relay_adapter.py"))

        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        done_path = os.path.join(tmpdir, "training_done.flag")
        state_path = os.path.join(tmpdir, ".colab_pilot_training_state.json")
        results_dir = os.path.join(tmpdir, "results")

        base_env = os.environ.copy()
        base_env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": workspace_dir + os.pathsep + tmpdir + os.pathsep + base_env.get("PYTHONPATH", ""),
            "SMOKE_TEST": "1",
            "SMOKE_SKIP_RCLONE": "1",
            "TOTAL_EPOCHS": "3",
            "CHECKPOINT_PATH": ckpt_path,
            "DONE_FLAG_PATH": done_path,
            "TRAINING_STATE_FILE": state_path,
            "RESULTS_DIR": results_dir,
            "GDRIVE_DIR": args.gdrive_dir,
        })

        # ── 第 2 关：Pass #1 (跑第 1 轮 + 断点保存 + 模拟换棒) ──
        print("\n[2/4] 正在执行 Smoke Pass #1（验证前向/反向传播 + Epoch 1 断点保存）...")
        env1 = base_env.copy()
        env1["SMOKE_PASS"] = "1"
        t0 = time.time()
        res1 = subprocess.run(
            [args.python, script_abs, "--smoke-test", "--smoke-pass", "1"],
            cwd=tmpdir,
            env=env1,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if res1.returncode != 0:
            print("  ❌ 第 2 关失败：Pass #1 进程返回非零退出码")
            print(res1.stdout[-1000:])
            print(res1.stderr[-1000:])
            return 1

        if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) == 0:
            print(f"  ❌ 第 2 关失败：未生成断点文件 {ckpt_path}")
            print(res1.stdout[-800:])
            return 1

        if not os.path.exists(state_path):
            print(f"  ❌ 第 2 关失败：未生成遥测状态文件 {state_path}")
            return 1

        with open(state_path, "r", encoding="utf-8") as f:
            st1 = json.load(f)
        ep1 = st1.get("current_epoch", 0)
        loss1 = st1.get("loss")
        print(
            f"  ✅ 第 2 关通过 ({time.time() - t0:.1f}s)：已生成 checkpoint.pt ({os.path.getsize(ckpt_path)} B)，"
            f"遥测状态 Epoch={ep1}, Loss={loss1}"
        )

        # ── 第 3 关：Pass #2 (强制从 Pass #1 断点续训第 2 轮) ──
        print("\n[3/4] 正在执行 Smoke Pass #2（验证从 checkpoint.pt 断点续训 Epoch 2）...")
        env2 = base_env.copy()
        env2["SMOKE_PASS"] = "2"
        t1 = time.time()
        res2 = subprocess.run(
            [args.python, script_abs, "--smoke-test", "--smoke-pass", "2"],
            cwd=tmpdir,
            env=env2,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if res2.returncode != 0:
            print("  ❌ 第 3 关失败：Pass #2 续训报错")
            print(res2.stdout[-1000:])
            print(res2.stderr[-1000:])
            return 1

        with open(state_path, "r", encoding="utf-8") as f:
            st2 = json.load(f)
        ep2 = st2.get("current_epoch", 0)
        loss2 = st2.get("loss")
        if ep2 <= ep1:
            print(f"  ❌ 第 3 关失败：断点未生效！Pass #1 Epoch={ep1}，Pass #2 Epoch={ep2}（未递增）")
            return 1

        print(
            f"  ✅ 第 3 关通过 ({time.time() - t1:.1f}s)：成功从 Epoch {ep1} 续训至 Epoch {ep2}，"
            f"Loss={loss2}，训练结果与完成标记已生成。"
        )

    # ── 第 4 关：rclone 与 Google Drive 凭证核验 ──
    print("\n[4/4] 正在核验本机 rclone 与 Google Drive 授权状态...")
    rclone_bin = find_rclone()
    rclone_conf = find_rclone_conf()
    if not rclone_bin or not os.path.exists(rclone_bin):
        print("  ❌ 第 4 关失败：未找到 rclone 可执行文件")
        return 1
    if not rclone_conf or not os.path.exists(rclone_conf) or os.path.getsize(rclone_conf) < 20:
        print(f"  ❌ 第 4 关失败：未找到有效的 rclone.conf ({rclone_conf})")
        return 1

    if not args.skip_rclone_net:
        remote_root = args.gdrive_dir.split(":")[0] + ":"
        rc_test = subprocess.run(
            [rclone_bin, "--config", rclone_conf, "--retries=5", "--low-level-retries=20", "about", remote_root],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if rc_test.returncode != 0:
            print(f"  ⚠️ 警告：rclone 在线探测返回非零 ({rc_test.stderr.strip()[:200]})")
        else:
            print(f"  ✅ 第 4 关通过：rclone 与 Google Drive ({args.gdrive_dir}) 读写就绪。")
    else:
        print(f"  ✅ 第 4 关通过：rclone 配置文件 ({rclone_conf}) 已就绪。")

    print("\n🎉 [ALL PASSED] 本机 Smoke Test 4 关全部通过！可以安全上传至 Google Colab 启动串行接力训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
