import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
DAEMON_PATH = REPO_ROOT / "src-tauri" / "remote" / "telemetry_daemon.py"


def load_daemon_module():
    spec = importlib.util.spec_from_file_location("telemetry_daemon_under_test", DAEMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TelemetryDaemonTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_daemon_module()

    def test_classify_role_shell_precedes_exec(self):
        # Regression for M-6: "bash -i" used to match `"bash " in low` as "exec"
        self.assertEqual(self.mod.classify_role("bash -i"), "shell")
        self.assertEqual(self.mod.classify_role("/bin/bash -i -l"), "shell")
        self.assertEqual(self.mod.classify_role("tmux new-session -s dev"), "shell")
        self.assertEqual(self.mod.classify_role("bash run_job.sh"), "exec")
        self.assertEqual(self.mod.classify_role("python3 train_resnet.py"), "train")
        self.assertEqual(self.mod.classify_role("python3 -m vllm.entrypoints.openai.api_server"), "llm")
        self.assertEqual(self.mod.classify_role("/usr/colab/bin/kernel_manager"), "system")

    def test_finite_sanitizes_nan_and_inf_for_strict_json(self):
        dirty = {
            "loss": float("nan"),
            "tps": float("inf"),
            "neg": float("-inf"),
            "nested": [1.5, float("nan"), {"ok": 42}],
        }
        cleaned = self.mod._finite(dirty)
        raw = json.dumps(cleaned, allow_nan=False)
        parsed = json.loads(raw)
        self.assertIsNone(parsed["loss"])
        self.assertIsNone(parsed["tps"])
        self.assertIsNone(parsed["neg"])
        self.assertEqual(parsed["nested"], [1.5, None, {"ok": 42}])

    def test_collect_once_survives_checkpoint_toctou_and_writes_atomic_json(self):
        with tempfile.TemporaryDirectory() as td:
            tel_file = os.path.join(td, "telemetry.json")
            tr_file = os.path.join(td, "training_state.json")
            with open(tr_file, "w", encoding="utf-8") as f:
                json.dump({"status": "training", "loss": 0.25}, f)

            self.mod.TELEMETRY_FILE = tel_file
            self.mod.TRAINING_STATE_FILE = tr_file
            self.mod.LLM_STATE_FILE = os.path.join(td, "missing_llm.json")
            self.mod.WORKSPACE_META_FILE = os.path.join(td, "missing_ws.json")

            real_exists = os.path.exists

            def fake_exists(p):
                if p == "/content/checkpoint.pt":
                    return True
                return real_exists(p)

            def fake_getsize(p):
                if p == "/content/checkpoint.pt":
                    raise FileNotFoundError("checkpoint vanished mid-replace")
                return os.stat(p).st_size

            with (
                mock.patch.object(self.mod.os.path, "exists", side_effect=fake_exists),
                mock.patch.object(self.mod.os.path, "getsize", side_effect=fake_getsize),
                mock.patch.object(
                    self.mod,
                    "get_processes_and_threads",
                    return_value={
                        "total_processes": 2,
                        "total_threads": 4,
                        "active_process": None,
                        "detected_workspace_env": None,
                        "processes": [],
                    },
                ),
            ):
                _, _, payload = self.mod.collect_once(100, 200, "Test CPU")

            self.assertFalse(payload["files"]["has_checkpoint"])
            self.assertEqual(payload["files"]["checkpoint_size_mb"], 0.0)
            self.assertTrue(os.path.exists(tel_file))
            self.assertFalse(os.path.exists(f"{tel_file}.{os.getpid()}.tmp"))
            with open(tel_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["training"]["status"], "training")
            self.assertFalse(math.isnan(saved["cpu"]["percent"]))

    def test_acquire_singleton_pid_lock_writes_pid(self):
        with tempfile.TemporaryDirectory() as td:
            self.mod.PID_FILE = os.path.join(td, "daemon.pid")
            fh = self.mod._acquire_singleton_pid_lock()
            self.assertIsNotNone(fh)
            try:
                with open(self.mod.PID_FILE, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read().strip(), str(os.getpid()))
            finally:
                fh.close()

    def test_is_live_telemetry_daemon_pid_rejects_recycled_zombie_and_bootstrap_pids(self):
        with tempfile.TemporaryDirectory() as proc_root:
            def mk_proc(pid: int, cmdline: bytes, state: str = "S"):
                pdir = os.path.join(proc_root, str(pid))
                os.makedirs(pdir, exist_ok=True)
                with open(os.path.join(pdir, "cmdline"), "wb") as f:
                    f.write(cmdline)
                with open(os.path.join(pdir, "stat"), "w", encoding="utf-8") as f:
                    f.write(f"{pid} (python3) {state} 1 1 1 0\n")

            # Recycled PID belonging to a user training job must NOT be considered a live telemetry daemon
            mk_proc(2001, b"python3\x00/content/train.py\x00", "R")
            # Zombie daemon PID must be rejected
            mk_proc(2002, b"python3\x00/content/.colab_pilot_daemon.py\x00--nonce\x00abc\x00", "Z")
            # Short-lived --bootstrap launcher PID must NOT count as the background daemon itself
            mk_proc(2003, b"python3\x00/content/.colab_pilot_daemon.py\x00--bootstrap\x00--nonce\x00abc\x00", "S")
            # Real live background daemon
            mk_proc(2004, b"python3\x00/content/.colab_pilot_daemon.py\x00--nonce\x00abc\x00", "S")

            self.assertFalse(self.mod.is_live_telemetry_daemon_pid(2001, proc_root=proc_root))
            self.assertFalse(self.mod.is_live_telemetry_daemon_pid(2002, proc_root=proc_root))
            self.assertFalse(self.mod.is_live_telemetry_daemon_pid(2003, proc_root=proc_root))
            self.assertFalse(self.mod.is_live_telemetry_daemon_pid(9999, proc_root=proc_root))
            self.assertTrue(self.mod.is_live_telemetry_daemon_pid(2004, proc_root=proc_root))

    def test_bootstrap_probe_never_reports_ok_from_stale_telemetry_or_recycled_pid(self):
        with tempfile.TemporaryDirectory() as td:
            proc_root = os.path.join(td, "proc")
            os.makedirs(os.path.join(proc_root, "3100"), exist_ok=True)
            with open(os.path.join(proc_root, "3100", "cmdline"), "wb") as f:
                f.write(b"python3\x00/content/train.py\x00")
            with open(os.path.join(proc_root, "3100", "stat"), "w", encoding="utf-8") as f:
                f.write("3100 (python3) R 1 1 1 0\n")

            self.mod.TELEMETRY_FILE = os.path.join(td, "telemetry.json")
            self.mod.PID_FILE = os.path.join(td, "daemon.pid")
            self.mod.PROBE_ACK_FILE = os.path.join(td, "ack.json")

            # Pre-populate stale files from a crashed old daemon + recycled PID 3100
            with open(self.mod.PID_FILE, "w", encoding="utf-8") as f:
                f.write("3100")
            with open(self.mod.TELEMETRY_FILE, "w", encoding="utf-8") as f:
                json.dump({"timestamp": 1700000000.0, "daemon_pid": 3100, "probe_nonce": "old-nonce"}, f)

            # Simulate new daemon failing immediately upon launch (writing nothing, or leaving stale sample)
            with (
                mock.patch.object(self.mod.subprocess, "Popen") as popen_mock,
                mock.patch.object(self.mod.os, "kill") as kill_mock,
            ):
                rc = self.mod.bootstrap_probe(
                    nonce="fresh-nonce-01",
                    timeout_sec=0.5,
                    proc_root=proc_root,
                )

            self.assertEqual(rc, 1, "bootstrap must fail when new daemon produces no fresh telemetry")
            popen_mock.assert_called_once()
            # Recycled PID 3100 (train.py) must NOT be killed!
            kill_mock.assert_not_called()
            self.assertFalse(os.path.exists(self.mod.PROBE_ACK_FILE))

    def test_bootstrap_probe_succeeds_only_after_live_daemon_writes_fresh_matching_sample(self):
        with tempfile.TemporaryDirectory() as td:
            proc_root = os.path.join(td, "proc")
            os.makedirs(proc_root, exist_ok=True)

            self.mod.TELEMETRY_FILE = os.path.join(td, "telemetry.json")
            self.mod.PID_FILE = os.path.join(td, "daemon.pid")
            self.mod.PROBE_ACK_FILE = os.path.join(td, "ack.json")

            # Seed stale telemetry before bootstrap
            with open(self.mod.TELEMETRY_FILE, "w", encoding="utf-8") as f:
                json.dump({"timestamp": 100.0, "daemon_pid": 4000, "probe_nonce": "stale"}, f)

            def fake_popen(cmd, **kwargs):
                nonce_idx = cmd.index("--nonce") + 1
                spawned_nonce = cmd[nonce_idx]
                pdir = os.path.join(proc_root, "4242")
                os.makedirs(pdir, exist_ok=True)
                with open(os.path.join(pdir, "cmdline"), "wb") as f:
                    f.write(f"python3\x00{cmd[1]}\x00--nonce\x00{spawned_nonce}\x00".encode("utf-8"))
                with open(os.path.join(pdir, "stat"), "w", encoding="utf-8") as f:
                    f.write("4242 (python3) S 1 1 1 0\n")
                with open(self.mod.PID_FILE, "w", encoding="utf-8") as f:
                    f.write("4242")
                with open(self.mod.TELEMETRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": self.mod.time.time() + 0.01,
                            "daemon_pid": 4242,
                            "probe_nonce": spawned_nonce,
                        },
                        f,
                    )
                return mock.MagicMock()

            with mock.patch.object(self.mod.subprocess, "Popen", side_effect=fake_popen):
                rc = self.mod.bootstrap_probe(
                    nonce="deploy-nonce-xyz",
                    timeout_sec=1.5,
                    proc_root=proc_root,
                )

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(self.mod.PROBE_ACK_FILE))
            with open(self.mod.PROBE_ACK_FILE, "r", encoding="utf-8") as f:
                ack = json.load(f)
            self.assertTrue(ack["ok"])
            self.assertEqual(ack["pid"], 4242)
            self.assertEqual(ack["nonce"], "deploy-nonce-xyz")


if __name__ == "__main__":
    unittest.main()
