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


if __name__ == "__main__":
    unittest.main()
