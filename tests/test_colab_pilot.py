import base64
import ctypes
import glob
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str):
    full_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


adapter_mod = load_module("colab_relay_adapter", ".agents/skills/colab-dl-relay/scripts/colab_relay_adapter.py")
runner_mod = load_module("dl_relay_runner", ".agents/skills/colab-dl-relay/scripts/dl_relay_runner.py")
smoke_mod = load_module("smoke_test_dl", ".agents/skills/colab-dl-relay/scripts/smoke_test_dl.py")
train_mod = load_module("pilot_train_script", "train.py")
gateway_mod = load_module("llm_local_gateway", ".agents/skills/colab-llm-bluegreen/scripts/llm_local_gateway.py")
sidecar_mod = load_module("llm_metrics_sidecar", ".agents/skills/colab-llm-bluegreen/scripts/llm_metrics_sidecar.py")
manager_mod = load_module("llm_bluegreen_manager", ".agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py")
qwen_mod = load_module("qwen_cli", "qwen_cli.py")


class _MaliciousPicklePayload:
    triggered = False

    def __reduce__(self):
        return (_mark_pickle_triggered, ())


def _mark_pickle_triggered():
    _MaliciousPicklePayload.triggered = True
    return {"epoch": 5}


class RelayAdapterAndTrainTests(unittest.TestCase):
    def test_c3_round2_fails_fast_when_remote_checkpoint_missing_or_rclone_errors(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = adapter_mod.ColabRelayContext(
                model_name="test-net",
                total_epochs=10,
                gdrive_dir="gdrive:colab_relay/exp1",
                results_dir=os.path.join(td, "results"),
                checkpoint_path=os.path.join(td, "checkpoint.pt"),
                done_flag_path=os.path.join(td, "training_done.flag"),
                smoke_test=False,
                round_num=2,
            )
            ctx.state_file = os.path.join(td, "state.json")

            with mock.patch.object(ctx, "_rclone_remote_file_exists", return_value=None):
                with self.assertRaises(RuntimeError):
                    ctx.prepare_cloud_workspace()

            with mock.patch.object(ctx, "_rclone_remote_file_exists", return_value=False):
                with self.assertRaises(RuntimeError):
                    ctx.prepare_cloud_workspace()

            with (
                mock.patch.object(ctx, "_rclone_remote_file_exists", return_value=True),
                mock.patch.object(ctx, "_run_rclone", return_value=False),
            ):
                with self.assertRaises(RuntimeError):
                    ctx.prepare_cloud_workspace()

    def test_c3_corrupt_local_checkpoint_raises_instead_of_resetting_to_epoch_1(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "checkpoint.pt")
            with open(ckpt, "wb") as f:
                f.write(b"NOT_A_VALID_TORCH_ZIP_FILE")

            ctx = adapter_mod.ColabRelayContext(
                model_name="corrupt-check",
                results_dir=os.path.join(td, "results"),
                checkpoint_path=ckpt,
                done_flag_path=os.path.join(td, "done.flag"),
                smoke_test=True,
            )
            ctx.state_file = os.path.join(td, "state.json")
            model = torch.nn.Linear(4, 2)
            with self.assertRaises(RuntimeError):
                ctx.load_torch_checkpoint(models=model)

    def test_m9_step_end_persists_current_epoch_best_metric_and_cloud_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "checkpoint.pt")
            ctx = adapter_mod.ColabRelayContext(
                model_name="metric-check",
                total_epochs=5,
                results_dir=os.path.join(td, "results"),
                checkpoint_path=ckpt,
                done_flag_path=os.path.join(td, "done.flag"),
                max_session_seconds=3600,
                smoke_test=False,
                round_num=1,
            )
            ctx.state_file = os.path.join(td, "state.json")
            model = torch.nn.Linear(4, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            buf = io.StringIO()
            with mock.patch.object(ctx, "backup_to_gdrive", return_value=True), redirect_stdout(buf):
                should_stop = ctx.step_end(
                    epoch=1,
                    loss=0.45,
                    models=model,
                    optimizers=opt,
                    metrics={"val_acc": 0.9125},
                )

            self.assertFalse(should_stop)
            self.assertIn("CLOUD_SMOKE_PASSED", buf.getvalue())
            saved = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.assertAlmostEqual(saved["best_metric"], 0.9125)

            # Verify fresh round-2 context restores epoch 2 + best_metric
            ctx2 = adapter_mod.ColabRelayContext(
                model_name="metric-check",
                total_epochs=5,
                results_dir=os.path.join(td, "results"),
                checkpoint_path=ckpt,
                done_flag_path=os.path.join(td, "done.flag"),
                smoke_test=False,
                round_num=2,
            )
            ctx2.state_file = os.path.join(td, "state.json")
            model2 = torch.nn.Linear(4, 2)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            next_epoch, _extra = ctx2.load_torch_checkpoint(models=model2, optimizers=opt2)
            self.assertEqual(next_epoch, 2)
            self.assertAlmostEqual(ctx2.best_metric, 0.9125)

    def test_h10_pattern_c_run_blackbox_wrapper_streams_and_emits_smoke_and_completion(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "checkpoint.pt")
            with open(ckpt, "wb") as f:
                f.write(b"dummy-ckpt")
            results_dir = os.path.join(td, "results")
            done_flag = os.path.join(td, "done.flag")
            state_file = os.path.join(td, "state.json")

            wrap_cmd = f'"{sys.executable}" -c "print(\'Epoch 1/2 loss=0.5\', flush=True); print(\'Epoch 2/2 loss=0.2\', flush=True)"'
            buf = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "TRAINING_STATE_FILE": state_file,
                        "DONE_FLAG_PATH": done_flag,
                    },
                    clear=False,
                ),
                mock.patch.object(adapter_mod.ColabRelayContext, "prepare_cloud_workspace"),
                mock.patch.object(adapter_mod.ColabRelayContext, "backup_to_gdrive", return_value=True),
                mock.patch.object(adapter_mod.ColabRelayContext, "_run_rclone", return_value=True),
                redirect_stdout(buf),
            ):
                rc = adapter_mod.run_blackbox_wrapper(
                    [
                        "--wrap",
                        wrap_cmd,
                        "--total-epochs",
                        "2",
                        "--results-dir",
                        results_dir,
                        "--checkpoint-path",
                        ckpt,
                    ]
                )

            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("CLOUD_SMOKE_PASSED", out)
            self.assertIn("TRAINING_COMPLETED", out)

    def test_h11_root_train_py_round2_missing_checkpoint_fails_fast(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env.update(
                {
                    "SMOKE_TEST": "0",
                    "RELAY_ROUND": "2",
                    "GDRIVE_DIR": "gdrive:nonexistent_relay_test",
                    "RCLONE_CONFIG": os.path.join(td, "missing_rclone.conf"),
                }
            )
            res = subprocess.run(
                [sys.executable, "-B", str(REPO_ROOT / "train.py")],
                cwd=td,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            self.assertNotEqual(res.returncode, 0, "train.py must fail fast when RELAY_ROUND=2 cannot pull checkpoint")


class GatewayAndSidecarTests(unittest.TestCase):
    def test_h7_gateway_refuses_startup_when_colab_cli_missing(self):
        with (
            mock.patch.dict(os.environ, {"PILOT_GATEWAY_ALLOW_MOCK": "0"}, clear=False),
            mock.patch.object(sys, "argv", ["llm_local_gateway.py", "--port", "18099"]),
            mock.patch("importlib.util.find_spec", return_value=None),
        ):
            rc = gateway_mod.main()
            self.assertEqual(rc, 2)

    def test_h3_cleanup_pid_file_only_removes_matching_pid(self):
        with tempfile.TemporaryDirectory() as td:
            pid_file = os.path.join(td, "llm_local_gateway.pid")
            with mock.patch.object(gateway_mod, "gateway_pid_path", return_value=pid_file):
                with open(pid_file, "w", encoding="utf-8") as f:
                    f.write("99999999")
                gateway_mod.cleanup_pid_file()
                self.assertTrue(os.path.exists(pid_file))

                with open(pid_file, "w", encoding="utf-8") as f:
                    f.write(str(os.getpid()))
                gateway_mod.cleanup_pid_file()
                self.assertFalse(os.path.exists(pid_file))

    def test_h5_decode_contents_payload_handles_base64_and_text(self):
        payload_obj = {"instant_tps": 48.5, "total_completion_tokens": 120}
        b64 = base64.b64encode(json.dumps(payload_obj).encode("utf-8")).decode("ascii")
        wrapped_b64 = b64[:12] + "\r\n" + b64[12:]
        decoded = gateway_mod.decode_contents_payload({"format": "base64", "content": wrapped_b64})
        self.assertEqual(decoded, payload_obj)

    def test_h2_h4_h5_gateway_http_cors_path_whitelist_and_deferred_sse_status(self):
        with tempfile.TemporaryDirectory() as td:
            st_file = os.path.join(td, "state.json")
            sess_file = os.path.join(td, "sessions.json")
            with open(st_file, "w", encoding="utf-8") as f:
                json.dump({"session_prefix": "qwen", "active_slot": "blue", "model_name": "Qwen-Test"}, f)
            with open(sess_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "qwen-blue": {
                            "endpoint": "m-s-qwenblue1",
                            "url": "https://example.colab.dev",
                            "token": "proxy-tok",
                            "kernel_id": "kid-1",
                        }
                    },
                    f,
                )

            fake_mode = {"kind": "error"}

            class FakeColabRuntime:
                def __init__(self, url, token, kernel_id=None, session_id=None):
                    self.url = url

                def execute_code(self, code, output_hook=None, timeout=300):
                    if fake_mode["kind"] == "kernel_error":
                        output_hook({"output_type": "error", "ename": "RuntimeError", "evalue": "CUDA OOM"})
                    elif fake_mode["kind"] == "upstream_400":
                        meta = gateway_mod.META_PREFIX + json.dumps({"status": 400, "ctype": "application/json"}) + "\n"
                        output_hook({"output_type": "stream", "text": meta + '{"error":"alternating roles required"}'})
                    elif fake_mode["kind"] == "sse_ok":
                        meta = gateway_mod.META_PREFIX + json.dumps({"status": 200, "ctype": "text/event-stream"}) + "\n"
                        chunk = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
                        output_hook({"output_type": "stream", "text": meta + chunk})

                def stop(self):
                    pass

            fake_colab_pkg = types.ModuleType("colab_cli")
            fake_rt_mod = types.ModuleType("colab_cli.runtime")
            fake_rt_mod.ColabRuntime = FakeColabRuntime

            with (
                mock.patch.object(gateway_mod, "state_path", return_value=st_file),
                mock.patch.object(gateway_mod, "sessions_path", return_value=sess_file),
                mock.patch.dict(sys.modules, {"colab_cli": fake_colab_pkg, "colab_cli.runtime": fake_rt_mod}),
            ):
                srv = gateway_mod.ThreadedHTTPServer(("127.0.0.1", 0), gateway_mod.make_gateway_handler("blue"))
                port = srv.server_address[1]
                t = threading.Thread(target=srv.serve_forever, daemon=True)
                t.start()
                try:
                    req_evil = urllib.request.Request(
                        f"http://127.0.0.1:{port}/pilot/gateway",
                        headers={"Origin": "https://evil.example"},
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_evil:
                        urllib.request.urlopen(req_evil, timeout=3)
                    self.assertEqual(ctx_evil.exception.code, 403)
                    self.assertIsNone(ctx_evil.exception.headers.get("Access-Control-Allow-Origin"))
                    ctx_evil.exception.close()

                    req_bad_path = urllib.request.Request(
                        f"http://127.0.0.1:{port}/admin/shutdown",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_path:
                        urllib.request.urlopen(req_bad_path, timeout=3)
                    self.assertEqual(ctx_path.exception.code, 403)
                    ctx_path.exception.close()

                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/pilot/health", timeout=3) as r_health:
                        health_data = json.loads(r_health.read().decode("utf-8"))
                    self.assertEqual(health_data["session_name"], "qwen-blue")
                    self.assertEqual(health_data["endpoint"], "m-s-qwenblue1")

                    fake_mode["kind"] = "kernel_error"
                    req_kerr = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        data=json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_kerr:
                        urllib.request.urlopen(req_kerr, timeout=3)
                    self.assertEqual(ctx_kerr.exception.code, 502)
                    ctx_kerr.exception.close()

                    fake_mode["kind"] = "upstream_400"
                    req_400 = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        data=json.dumps({"stream": True, "messages": []}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_400:
                        urllib.request.urlopen(req_400, timeout=3)
                    self.assertEqual(ctx_400.exception.code, 400)
                    ctx_400.exception.close()

                    fake_mode["kind"] = "sse_ok"
                    req_ok = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        data=json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_ok, timeout=3) as r_ok:
                        self.assertEqual(r_ok.status, 200)
                        body_txt = r_ok.read().decode("utf-8")
                    self.assertIn("Hi", body_txt)
                finally:
                    srv.shutdown()
                    srv.server_close()

    def test_m7_sidecar_check_backend_health_http_status(self):
        status_box = {"code": 503}

        class MockBackendHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                self.send_response(status_box["code"])
                self.end_headers()
                self.wfile.write(b"status")

        srv = HTTPServer(("127.0.0.1", 0), MockBackendHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tracker = sidecar_mod.LlmTelemetryTracker(
                    slot="blue",
                    role="active",
                    engine="vllm",
                    model_name="test",
                    gdrive_model_path="",
                    listen_port=8000,
                    backend_url=f"http://127.0.0.1:{port}",
                    state_file=os.path.join(td, "llm_state.json"),
                )
                self.assertFalse(tracker.check_backend_health())
                status_box["code"] = 200
                self.assertTrue(tracker.check_backend_health())
        finally:
            srv.shutdown()
            srv.server_close()


class ManagerAndQwenCliTests(unittest.TestCase):
    def test_i2_llama_cpp_shard_selection_and_m7_hf_streamer_finally(self):
        # Extract and run `find_primary_gguf` against a real 5-shard + mmproj directory
        m = re.search(
            r"(def find_primary_gguf\(root_dir: str\):[\s\S]*?return main_ggufs\[0\] if main_ggufs else None)",
            manager_mod.LLAMA_CPP_RUNNER_PY,
        )
        self.assertIsNotNone(m)
        ns = {"glob": glob, "os": os, "re": re}
        exec(m.group(1), ns)
        find_primary_gguf = ns["find_primary_gguf"]

        with tempfile.TemporaryDirectory() as td:
            for name in (
                "mmproj-Qwen-F16.gguf",
                "Qwen-00001-of-00005.gguf",
                "Qwen-00002-of-00005.gguf",
                "Qwen-00004-of-00005.gguf",
                "Qwen-00005-of-00005.gguf",
            ):
                Path(td, name).write_bytes(b"gguf")
            picked = find_primary_gguf(td)
            self.assertEqual(os.path.basename(picked), "Qwen-00001-of-00005.gguf")

        self.assertIn("finally:\n                try:\n                    streamer.end()", manager_mod.HF_OPENAI_SERVER_PY)

    def test_m11_qwen_cli_com_variant_size_matches_64bit_windows_abi(self):
        expected = 24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16
        self.assertEqual(ctypes.sizeof(qwen_mod.COMVariant), expected)

    def test_h2_qwen_web_mirror_csrf_origin_and_command_restrictions(self):
        received = []
        mirror = qwen_mod.LocalWebMirror(on_submit_text=received.append, max_events=40)
        url = mirror.start()
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        port = int(url.rsplit(":", 1)[1].rstrip("/"))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("POST", "/api/send", body=b'{"text":"hello"}', headers={"Content-Type": "application/json"})
            r1 = conn.getresponse()
            r1.read()
            conn.close()
            self.assertEqual(r1.status, 403)

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request(
                "POST",
                "/api/send",
                body=b'{"text":"hello"}',
                headers={
                    "Content-Type": "application/json",
                    "X-Qwen-Token": mirror.csrf_token,
                    "Origin": "https://attacker.example",
                },
            )
            r2 = conn.getresponse()
            r2.read()
            conn.close()
            self.assertEqual(r2.status, 403)

            for forbidden in ("/file C:/Windows/win.ini", "/switchover"):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                conn.request(
                    "POST",
                    "/api/send",
                    body=json.dumps({"text": forbidden}).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "X-Qwen-Token": mirror.csrf_token,
                    },
                )
                rf = conn.getresponse()
                rf.read()
                conn.close()
                self.assertEqual(rf.status, 403)

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request(
                "POST",
                "/api/send",
                body=json.dumps({"text": "你好 Qwen"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Qwen-Token": mirror.csrf_token,
                },
            )
            r4 = conn.getresponse()
            r4.read()
            conn.close()
            self.assertEqual(r4.status, 200)
            self.assertEqual(received, ["你好 Qwen"])

            for i in range(65):
                mirror.append(f"msg-{i}")
            self.assertLessEqual(len(mirror.events), 40)
        finally:
            mirror.stop()

    def test_m11_qwen_stream_chat_completion_handles_no_space_data_and_reasoning_content(self):
        class SseHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                lines = [
                    'data:{"choices":[{"delta":{"reasoning_content":"先思考"}}]}\n\n',
                    'data: {"choices":[{"delta":{"content":"再回答"}}]}\n\n',
                    "data:[DONE]\n\n",
                ]
                for ln in lines:
                    self.wfile.write(ln.encode("utf-8"))
                    self.wfile.flush()

        srv = HTTPServer(("127.0.0.1", 0), SseHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            tokens = []
            with mock.patch.object(qwen_mod, "ensure_gateway_running"):
                reply = qwen_mod.stream_chat_completion(
                    gateway_url=f"http://127.0.0.1:{port}",
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                    show_stats=False,
                    on_token=tokens.append,
                )
            self.assertEqual(reply, "先思考再回答")
            self.assertEqual("".join(tokens), "先思考再回答")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_h11_safe_torch_load_blocks_malicious_pickle_and_preserves_zero_best_metric(self):
        with tempfile.TemporaryDirectory() as td:
            evil_ckpt = os.path.join(td, "evil_checkpoint.pt")
            _MaliciousPicklePayload.triggered = False
            torch.save({"epoch": 3, "payload": _MaliciousPicklePayload()}, evil_ckpt)
            _MaliciousPicklePayload.triggered = False

            ctx = adapter_mod.ColabRelayContext(
                model_name="safe-load-test",
                results_dir=os.path.join(td, "results"),
                checkpoint_path=evil_ckpt,
                done_flag_path=os.path.join(td, "done.flag"),
                smoke_test=True,
            )
            ctx.state_file = os.path.join(td, "state.json")

            with mock.patch.dict(os.environ, {"ALLOW_UNSAFE_TORCH_LOAD": "0"}, clear=False):
                with self.assertRaises(RuntimeError):
                    ctx.load_torch_checkpoint()
                self.assertFalse(_MaliciousPicklePayload.triggered)

                with self.assertRaises(RuntimeError):
                    train_mod.load_checkpoint_safely(evil_ckpt)
                self.assertFalse(_MaliciousPicklePayload.triggered)

            # Legitimate checkpoint with best_metric == 0.0 loads via weights_only=True and preserves 0.0
            good_ckpt = os.path.join(td, "good_checkpoint.pt")
            good_ctx = adapter_mod.ColabRelayContext(
                model_name="zero-metric-test",
                results_dir=os.path.join(td, "results2"),
                checkpoint_path=good_ckpt,
                done_flag_path=os.path.join(td, "done2.flag"),
                smoke_test=True,
            )
            good_ctx.state_file = os.path.join(td, "state2.json")
            model = torch.nn.Linear(4, 2)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            good_ctx.best_metric = 0.0
            good_ctx.save_torch_checkpoint(epoch=4, loss=0.0, models=model, optimizers=opt, metrics={"loss": 0.0})

            resumed_ctx = adapter_mod.ColabRelayContext(
                model_name="zero-metric-test",
                results_dir=os.path.join(td, "results2"),
                checkpoint_path=good_ckpt,
                done_flag_path=os.path.join(td, "done2.flag"),
                smoke_test=True,
            )
            resumed_ctx.state_file = os.path.join(td, "state2.json")
            resumed_ctx.best_metric = 99.0
            start_ep, _ = resumed_ctx.load_torch_checkpoint(models=model, optimizers=opt)
            self.assertEqual(start_ep, 5)
            self.assertEqual(resumed_ctx.best_metric, 0.0)

    def test_c2_relay_rclone_sh_and_dl_relay_runner_round_classification(self):
        sh_bytes = (REPO_ROOT / "relay_rclone.sh").read_bytes()
        self.assertNotIn(b"\r", sh_bytes, "relay_rclone.sh must use LF line endings only")
        sh_text = sh_bytes.decode("utf-8")
        self.assertIn("-f train.py", sh_text)
        self.assertNotIn("runpy.run_path", sh_text)
        self.assertIn("Installation Complete", sh_text)
        self.assertIn("RCLONE_READY", sh_text)
        self.assertIn("FAST_HANDOVER_STREAK", sh_text)

        # Even if colab exec exits 0 and stale remote done flag exists, missing TRAIN_EXIT_CODE=0 or ANSI traceback MUST fail
        self.assertEqual(
            runner_mod.classify_relay_round_output(0, "TRAIN_EXIT_CODE=1\n", has_remote_done_flag=True),
            "failed",
        )
        self.assertEqual(
            runner_mod.classify_relay_round_output(
                0,
                "\x1b[0;31mSyntaxError\x1b[0m: invalid syntax\nTRAIN_EXIT_CODE=0\n",
                has_remote_done_flag=True,
            ),
            "failed",
        )
        self.assertEqual(
            runner_mod.classify_relay_round_output(
                0,
                "CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\nTRAIN_EXIT_CODE=0\n",
                has_remote_done_flag=False,
            ),
            "handover",
        )
        self.assertEqual(
            runner_mod.classify_relay_round_output(
                0,
                "TRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n",
                has_remote_done_flag=False,
            ),
            "completed",
        )

    def test_h10_sidecar_and_warmup_reject_sse_error_frames_and_empty_deltas(self):
        class BrokenEngineHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frames = [
                    'data: {"error":{"message":"CUDA out of memory"}}\n\n',
                    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    "data: [DONE]\n\n",
                ]
                for fr in frames:
                    self.wfile.write(fr.encode("utf-8"))
                    self.wfile.flush()

        eng_srv = HTTPServer(("127.0.0.1", 0), BrokenEngineHandler)
        eng_port = eng_srv.server_address[1]
        threading.Thread(target=eng_srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as td:
                st_path = os.path.join(td, "llm_state.json")
                tracker = sidecar_mod.LlmTelemetryTracker(
                    slot="green",
                    role="warming",
                    engine="transformers",
                    model_name="qwen",
                    gdrive_model_path="",
                    listen_port=0,
                    backend_url=f"http://127.0.0.1:{eng_port}",
                    state_file=st_path,
                )
                proxy_srv = sidecar_mod.ThreadingHTTPServer(("127.0.0.1", 0), sidecar_mod.make_handler(tracker))
                proxy_port = proxy_srv.server_address[1]
                threading.Thread(target=proxy_srv.serve_forever, daemon=True).start()
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        body=b'{"stream":true,"messages":[{"role":"user","content":"warmup"}]}',
                        headers={"Content-Type": "application/json"},
                    )
                    resp = conn.getresponse()
                    raw_warmup = resp.read().decode("utf-8", errors="ignore")
                    conn.close()
                    tracker.write_state_file()
                    saved_state = json.loads(Path(st_path).read_text(encoding="utf-8"))
                    self.assertEqual(saved_state["total_completion_tokens"], 0)
                    self.assertIn("CUDA out of memory", raw_warmup)
                finally:
                    proxy_srv.shutdown()
                    proxy_srv.server_close()
        finally:
            eng_srv.shutdown()
            eng_srv.server_close()

        boot_code = manager_mod.build_cloud_engine_boot_code(
            slot="green",
            role="warming",
            engine="transformers",
            model_name="qwen",
            gdrive_model_path="",
            workspace_abs=".",
        )
        self.assertIn('if _obj.get("error"):', boot_code)
        self.assertIn("CLOUD_WARMUP_FAILED: 云端 Warmup 流式生成返回异常", boot_code)

    def test_h10_switchover_preserves_saved_session_prefix_gpu_and_high_mem(self):
        parser = manager_mod.build_arg_parser()
        saved_st = {
            "session_prefix": "qwen-prod",
            "active_slot": "blue",
            "engine": "vllm",
            "model_name": "Qwen2.5-14B",
            "gdrive_model_path": "gdrive:models/qwen14b",
            "gpu": "G4",
            "high_mem": False,
        }
        args_default = parser.parse_args(["switchover"])
        cfg = manager_mod.resolve_switchover_settings(args_default, saved_st)
        self.assertEqual(cfg["session_prefix"], "qwen-prod")
        self.assertEqual(cfg["gpu"], "G4")
        self.assertFalse(cfg["high_mem"])
        self.assertEqual(cfg["old_slot"], "blue")
        self.assertEqual(cfg["new_slot"], "green")

        args_override = parser.parse_args(["switchover", "--session-prefix", "custom", "--gpu", "A100", "--high-mem"])
        cfg_ov = manager_mod.resolve_switchover_settings(args_override, saved_st)
        self.assertEqual(cfg_ov["session_prefix"], "custom")
        self.assertEqual(cfg_ov["gpu"], "A100")
        self.assertTrue(cfg_ov["high_mem"])

        bench_args = parser.parse_args(["benchmark"])
        self.assertIsNone(bench_args.session_prefix)
        stop_args = parser.parse_args(["stop"])
        self.assertIsNone(stop_args.session_prefix)

    def test_h5_qwen_cli_raises_on_sse_error_frame(self):
        class ErrSseHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"error":{"message":"KV cache exhausted"}}\n\n')
                self.wfile.flush()

        srv = HTTPServer(("127.0.0.1", 0), ErrSseHandler)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.object(qwen_mod, "ensure_gateway_running"):
                with self.assertRaises(RuntimeError) as ctx:
                    qwen_mod.stream_chat_completion(
                        gateway_url=f"http://127.0.0.1:{port}",
                        model="test-model",
                        messages=[{"role": "user", "content": "hi"}],
                        show_stats=False,
                    )
                self.assertIn("KV cache exhausted", str(ctx.exception))
        finally:
            srv.shutdown()
            srv.server_close()


class Phase2InstanceAndCliV080Tests(unittest.TestCase):
    def test_single_shape_accelerators_strip_high_mem_and_http_400_auto_retries(self):
        # G4, H100, L4, V5E1, V6E1 must NEVER pass --high-mem in either runner or manager
        for acc in ("G4", "g4", "H100", "h100", "L4", "V5E1", "V6E1"):
            cmd_dl = runner_mod.build_colab_new_cmd("colab", "exp-1", acc, high_mem=True)
            cmd_llm = manager_mod.build_colab_new_cmd("colab", "llm-blue", acc, high_mem=True)
            self.assertNotIn("--high-mem", cmd_dl, f"dl_relay_runner leaked --high-mem for {acc}")
            self.assertNotIn("--high-mem", cmd_llm, f"llm_bluegreen_manager leaked --high-mem for {acc}")
            if acc.upper() in ("V5E1", "V6E1"):
                self.assertIn("--tpu", cmd_dl)
                self.assertIn(acc.lower(), cmd_dl)
                self.assertIn("--tpu", cmd_llm)
                self.assertIn(acc.lower(), cmd_llm)
            else:
                self.assertIn("--gpu", cmd_dl)
                self.assertIn("--gpu", cmd_llm)

        # T4, A100, and CPU preserve --high-mem when requested
        for acc in ("T4", "A100", "CPU"):
            cmd_dl = runner_mod.build_colab_new_cmd("colab", "exp-2", acc, high_mem=True)
            cmd_llm = manager_mod.build_colab_new_cmd("colab", "llm-blue", acc, high_mem=True)
            self.assertIn("--high-mem", cmd_dl)
            self.assertIn("--high-mem", cmd_llm)
            if acc == "CPU":
                self.assertNotIn("--gpu", cmd_dl)
                self.assertNotIn("--gpu", cmd_llm)

        # Auto-fallback in dl_relay_runner when backend rejects --high-mem with HTTP 400
        dl_calls = []

        def fake_dl_run(cmd, timeout=300, stdin_text=None):
            dl_calls.append(list(cmd))
            if "--high-mem" in cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="[colab] A new version of Colab CLI is available: 0.7.5 -> 0.8.0\nError: Backend rejected accelerator: A100 (HTTP 400 Bad Request)",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="Created", stderr="")

        with mock.patch.object(runner_mod, "run_cmd", side_effect=fake_dl_run):
            res = runner_mod.allocate_colab_instance("colab", "train-a100", "A100", high_mem=True)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(len(dl_calls), 2)
        self.assertIn("--high-mem", dl_calls[0])
        self.assertNotIn("--high-mem", dl_calls[1])

    def test_v080_banner_stripping_keyring_fallback_and_400_retry_in_manager(self):
        v080_banner = (
            "[colab] A new version of Colab CLI is available: 0.7.5 -> 0.8.0\n"
            "[colab] Run 'colab update' to install the latest version.\n"
            "[colab] To silence this check, set enable_update_check = false in ~/.config/colab-cli/settings.json\n"
        )
        raw_status = v080_banner + "[llm-blue] Active (m-s-g4test)\nHardware: G4\n"
        cleaned = manager_mod.strip_colab_cli_banners(raw_status)
        self.assertTrue(cleaned.startswith("[llm-blue] Active"))
        self.assertNotIn("colab update", cleaned)

        self.assertEqual(
            runner_mod.classify_relay_round_output(
                0,
                v080_banner + "CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\nTRAIN_EXIT_CODE=0\n",
                has_remote_done_flag=False,
            ),
            "handover",
        )

        # Verify keyring fallback when v0.8.0 migrates & deletes token.json
        with tempfile.TemporaryDirectory() as td:
            fake_keyring = types.ModuleType("keyring")
            fake_keyring.get_password = lambda svc, usr: json.dumps(
                {"token": "redacted-access", "refresh_token": "redacted-refresh"}
            )
            with (
                mock.patch.object(manager_mod, "colab_cfg_dir", return_value=td),
                mock.patch.dict(sys.modules, {"keyring": fake_keyring}),
            ):
                ok, src = manager_mod.has_colab_oauth_credentials()
            self.assertTrue(ok)
            self.assertEqual(src, "keyring")

    def test_deploy_503_failure_state_and_keep_vm_on_fail_contract(self):
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "llm_bluegreen_state.json")
            keys_file = os.path.join(td, "llm_api_keys.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            parser = manager_mod.build_arg_parser()

            # 1) HTTP 503 capacity stockout -> writes phase="failed" & actionable Chinese last_error
            def fake_run_503(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                if cmd[1] == "new":
                    self.assertNotIn("--high-mem", cmd)  # H100 must strip --high-mem automatically
                    return subprocess.CompletedProcess(
                        cmd,
                        1,
                        stdout="",
                        stderr="POST /tun/m/assign -> 503 Service Unavailable",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            args_503 = parser.parse_args(
                [
                    "deploy",
                    "--session-name",
                    "my-h100",
                    "--slot",
                    "standalone",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "H100",
                    "--high-mem",
                    "--engine",
                    "llama.cpp",
                    "--model-name",
                    "Qwen3-Coder",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(manager_mod, "register_workspace"),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_503),
            ):
                rc_503 = manager_mod.cmd_deploy(args_503)

            self.assertEqual(rc_503, 1)
            st_503 = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertEqual(st_503["phase"], "failed")
            self.assertEqual(st_503["failed_session"], "my-h100")
            self.assertFalse(st_503["high_mem"])
            self.assertIn("503", st_503["last_error"])
            self.assertIn("G4 / L4 / T4", st_503["last_error"])

            # 2) Warmup failure on existing VM WITH --keep-vm-on-fail -> must NOT call `colab stop`
            stop_calls = []

            def fake_run_existing_fail(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "status":
                    return subprocess.CompletedProcess(cmd, 0, stdout="[existing-vm] Active\nHardware: T4\n", stderr="")
                if cmd[1] == "upload":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="upload error")
                if cmd[1] == "stop":
                    stop_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            args_keep = parser.parse_args(
                [
                    "deploy",
                    "--session-name",
                    "existing-vm",
                    "--slot",
                    "standalone",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "G4",
                    "--engine",
                    "builtin",
                    "--model-name",
                    "Qwen-Test",
                    "--api-key",
                    "sk-custom-existing",
                    "--keep-vm-on-fail",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(manager_mod, "register_workspace"),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_existing_fail),
            ):
                rc_keep = manager_mod.cmd_deploy(args_keep)

            self.assertEqual(rc_keep, 1)
            self.assertEqual(stop_calls, [], "--keep-vm-on-fail must preserve existing VM without calling colab stop")

            # Without --keep-vm-on-fail -> MUST call `colab stop -s existing-vm`
            args_stop = parser.parse_args(
                [
                    "deploy",
                    "--session-name",
                    "existing-vm",
                    "--slot",
                    "standalone",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "T4",
                    "--engine",
                    "builtin",
                    "--model-name",
                    "Qwen-Test",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(manager_mod, "register_workspace"),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_existing_fail),
            ):
                rc_stop = manager_mod.cmd_deploy(args_stop)

            self.assertEqual(rc_stop, 1)
            self.assertEqual(len(stop_calls), 1)
            self.assertEqual(stop_calls[0][-2:], ["-s", "existing-vm"])

            # 3) Successful standalone deploy writes llm_api_keys.json (18083+) & serving state
            args_ok = parser.parse_args(
                [
                    "deploy",
                    "--session-name",
                    "standalone-g4",
                    "--slot",
                    "standalone",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "G4",
                    "--high-mem",
                    "--engine",
                    "llama.cpp",
                    "--model-name",
                    "Qwen3-30B",
                    "--api-key",
                    "sk-colab-custom-999",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 62.4, "ttft_ms": 110.0, "role": "active"},
                ),
                mock.patch.object(manager_mod, "ensure_local_gateway"),
            ):
                rc_ok = manager_mod.cmd_deploy(args_ok)

            self.assertEqual(rc_ok, 0)
            st_ok = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            keys_ok = json.loads(Path(keys_file).read_text(encoding="utf-8"))
            self.assertEqual(st_ok["phase"], "serving")
            self.assertEqual(st_ok["active_session"], "standalone-g4")
            self.assertIsNone(st_ok["last_error"])
            self.assertFalse(st_ok["high_mem"])
            vm_entry = keys_ok["vms"]["standalone-g4"]
            self.assertEqual(vm_entry["api_key"], "sk-colab-custom-999")
            self.assertGreaterEqual(vm_entry["local_port"], 18083)

    def test_gateway_stale_state_self_healing_and_bearer_vm_routing(self):
        with tempfile.TemporaryDirectory() as td:
            st_file = os.path.join(td, "llm_bluegreen_state.json")
            sess_file = os.path.join(td, "sessions.json")
            keys_file = os.path.join(td, "llm_api_keys.json")

            # Stale llm-blue "serving" state while sessions.json is empty
            Path(st_file).write_text(
                json.dumps({"session_prefix": "llm", "active_slot": "blue", "phase": "serving", "model_name": "Stale"}),
                encoding="utf-8",
            )
            Path(sess_file).write_text("{}", encoding="utf-8")
            Path(keys_file).write_text(
                json.dumps(
                    {
                        "vms": {
                            "g4-node": {
                                "vm_name": "g4-node",
                                "slot": "standalone",
                                "api_key": "sk-secret-do-not-leak",
                                "local_port": 18083,
                                "remote_port": 8000,
                                "model_name": "Qwen-G4",
                                "engine": "llama.cpp",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(gateway_mod, "state_path", return_value=st_file),
                mock.patch.object(gateway_mod, "sessions_path", return_value=sess_file),
            ):
                srv = gateway_mod.ThreadedHTTPServer(("127.0.0.1", 0), gateway_mod.make_gateway_handler(None))
                port = srv.server_address[1]
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/pilot/gateway", timeout=3) as r_gw:
                        gw_data = json.loads(r_gw.read().decode("utf-8"))
                    self.assertFalse(gw_data["online"])
                    self.assertFalse(gw_data["tunnel_open"])
                    self.assertTrue(gw_data["stale_state_detected"])
                    self.assertEqual(gw_data["phase"], "offline")
                    self.assertNotIn("sk-secret-do-not-leak", json.dumps(gw_data))

                    with self.assertRaises(urllib.error.HTTPError) as ctx_503:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/pilot/health", timeout=3)
                    self.assertEqual(ctx_503.exception.code, 503)
                    ctx_503.exception.close()

                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r_models:
                        models_empty = json.loads(r_models.read().decode("utf-8"))
                    self.assertEqual(models_empty["data"], [])

                    # Now bring standalone `g4-node` online in sessions.json and route via Bearer API key
                    Path(sess_file).write_text(
                        json.dumps(
                            {
                                "g4-node": {
                                    "endpoint": "m-s-g4node99",
                                    "url": "https://g4.colab.dev",
                                    "token": "tok-g4",
                                    "kernel_id": "k-g4",
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    req_bearer = urllib.request.Request(
                        f"http://127.0.0.1:{port}/pilot/health",
                        headers={"Authorization": "Bearer sk-secret-do-not-leak"},
                    )
                    with urllib.request.urlopen(req_bearer, timeout=3) as r_b:
                        b_data = json.loads(r_b.read().decode("utf-8"))
                    self.assertTrue(b_data["ok"])
                    self.assertEqual(b_data["session_name"], "g4-node")
                    self.assertEqual(b_data["endpoint"], "m-s-g4node99")
                    self.assertEqual(b_data["model_name"], "Qwen-G4")
                    self.assertNotIn("sk-secret-do-not-leak", json.dumps(b_data))
                finally:
                    srv.shutdown()
                    srv.server_close()

    def test_real_v080_click_option_parsing_contract_for_all_python_cli_commands(self):
        if os.name == "nt":
            colab_py = (
                Path(os.environ.get("APPDATA", ""))
                / "uv"
                / "tools"
                / "google-colab-cli"
                / "Scripts"
                / "python.exe"
            )
        else:
            colab_py = (
                Path.home()
                / ".local"
                / "share"
                / "uv"
                / "tools"
                / "google-colab-cli"
                / "bin"
                / "python"
            )
        if not colab_py.exists():
            self.skipTest(f"google-colab-cli venv python not found at {colab_py}")

        valid_argvs = []
        for acc in ("CPU", "NONE", "T4", "L4", "G4", "A100", "H100", "V5E1", "V6E1"):
            for hm in (False, True):
                cmd_dl = runner_mod.build_colab_new_cmd("colab", "sess-dl", acc, hm)
                cmd_llm = manager_mod.build_colab_new_cmd("colab", "sess-llm", acc, hm)
                self.assertNotIn("--auth", cmd_dl[1:])
                self.assertNotIn("--auth", cmd_llm[1:])
                valid_argvs.append(cmd_dl[1:])
                valid_argvs.append(cmd_llm[1:])

        valid_argvs.extend(
            [
                ["upload", "-s", "sess-1", "/tmp/local.py", "/content/remote.py"],
                ["download", "-s", "sess-1", "/content/remote.json", "/tmp/local.json"],
                ["exec", "-s", "sess-1", "--timeout", "2400"],
                ["exec", "-s", "sess-1", "--timeout", "180"],
                ["exec", "-s", "sess-1", "--timeout", "30"],
                ["exec", "-s", "sess-1", "-f", "train.py", "--timeout", "86400"],
                ["install", "-s", "sess-1", "-r", "requirements.txt"],
                ["status", "-s", "sess-1"],
                ["sessions"],
                ["stop", "-s", "sess-1"],
                ["restart-kernel", "-s", "sess-1"],
                ["ssh", "-s", "sess-1"],
                ["ssh", "-i", "~/.ssh/colab_pilot_ed25519", "-s", "sess-1"],
                ["ssh", "--proxy-mode", "-s", "sess-1"],
                ["ssh", "-s", "sess-1", "--", "nvidia-smi"],
                ["--auth", "oauth2", "new", "-s", "sess-1", "--gpu", "g4"],
                ["--auth", "adc", "new", "-s", "sess-1", "--gpu", "t4", "--high-mem"],
            ]
        )

        invalid_argvs = [
            ["new", "-s", "sess-1", "--auth", "oauth2"],
            ["new", "-s", "sess-1", "--gpu", "g4", "--auth", "adc"],
            ["exec", "-s", "sess-1", "--auth", "oauth2"],
        ]

        validator_code = """
import json, sys
import click
import typer.main
from colab_cli.cli import app

payload = json.loads(sys.stdin.read())
root_cmd = typer.main.get_command(app)

def dry_parse(argv):
    root_ctx = root_cmd.make_context("colab", list(argv))
    prot = getattr(root_ctx, "_protected_args", None)
    if prot is None:
        prot = getattr(root_ctx, "protected_args", [])
    rem = list(prot) + list(root_ctx.args)
    assert rem, f"No subcommand resolved for {argv!r}"
    cmd_name, sub_cmd, sub_args = root_cmd.resolve_command(root_ctx, rem)
    assert sub_cmd is not None, f"Unknown subcommand {cmd_name!r} for {argv!r}"
    sub_ctx = sub_cmd.make_context(cmd_name, list(sub_args), parent=root_ctx)
    if not sub_cmd.allow_extra_args and not sub_cmd.ignore_unknown_options:
        assert not sub_ctx.args, f"Unconsumed extra args {sub_ctx.args!r} for {argv!r}"
    return root_ctx.params, sub_ctx.params

for argv in payload["valid"]:
    dry_parse(argv)

for bad in payload["invalid"]:
    try:
        dry_parse(bad)
    except Exception as exc:
        if exc.__class__.__name__ != "NoSuchOption" and "No such option" not in str(exc):
            raise
    else:
        raise AssertionError(f"Expected NoSuchOption for misplaced flag in {bad!r}")

print(f"PARSED_OK valid={len(payload['valid'])} invalid={len(payload['invalid'])}")
"""
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [str(colab_py), "-B", "-c", validator_code],
            input=json.dumps({"valid": valid_argvs, "invalid": invalid_argvs}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=env,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"v0.8.0 Click dry-parse contract failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}",
        )
        self.assertIn("PARSED_OK", proc.stdout)


class IssueRegressionTests(unittest.TestCase):
    def test_issue_44_default_gdrive_isolation_active_lock_and_train_temp_prefix(self):
        # 1) Bare default gdrive:colab_relay automatically isolates by session-prefix / session
        gd_a, iso_a = runner_mod.resolve_effective_gdrive_dir("gdrive:colab_relay", "task-alpha")
        gd_b, iso_b = runner_mod.resolve_effective_gdrive_dir("gdrive:colab_relay/", "worker-1", session_prefix="exp/beta 01")
        gd_explicit, iso_exp = runner_mod.resolve_effective_gdrive_dir("gdrive:colab_relay/custom_dir/", "task-alpha")
        self.assertTrue(iso_a)
        self.assertEqual(gd_a, "gdrive:colab_relay/task-alpha")
        self.assertTrue(iso_b)
        self.assertEqual(gd_b, "gdrive:colab_relay/exp-beta-01")
        self.assertFalse(iso_exp)
        self.assertEqual(gd_explicit, "gdrive:colab_relay/custom_dir")

        with tempfile.TemporaryDirectory() as td:
            locks_file = os.path.join(td, "dl_relay_locks.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(ws_dir, "train.py").write_text("print('ok')\n", encoding="utf-8")

            with mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file):
                ok1, _ = runner_mod.acquire_gdrive_dir_lock("gdrive:colab_relay/shared", "sess-a", ws_dir)
                self.assertTrue(ok1)
                # Second session targeting the same explicit directory without --allow-shared-gdrive-dir is blocked
                ok2, err2 = runner_mod.acquire_gdrive_dir_lock("gdrive:colab_relay/shared", "sess-b", ws_dir)
                self.assertFalse(ok2)
                self.assertIn("sess-a", err2)
                self.assertIn("--allow-shared-gdrive-dir", err2)

                rc_conflict = runner_mod.main(
                    ["--session", "sess-b", "--workspace", ws_dir, "--gdrive-dir", "gdrive:colab_relay/shared"]
                )
                self.assertEqual(rc_conflict, 1)

                # Explicit --allow-shared-gdrive-dir permits reuse
                ok3, _ = runner_mod.acquire_gdrive_dir_lock(
                    "gdrive:colab_relay/shared", "sess-b", ws_dir, allow_shared=True
                )
                self.assertTrue(ok3)
                runner_mod.release_gdrive_dir_lock("gdrive:colab_relay/shared", "sess-b")

            # 2) train.py honors RELAY_GDRIVE_DIR and adds session+pid prefix to hot-backup temp upload names
            ckpt_file = os.path.join(td, "checkpoint.pt")
            Path(ckpt_file).write_bytes(b"pt-weights")
            pushed_cmds_a = []
            pushed_cmds_b = []

            def make_fake_subproc(recorder):
                def _fake(cmd, *args, **kwargs):
                    recorder.append(list(cmd))
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                return _fake

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/sess-a",
                        "RELAY_SESSION": "sess-a",
                        "CHECKPOINT_PATH": ckpt_file,
                    },
                    clear=False,
                ),
                mock.patch.object(train_mod.subprocess, "run", side_effect=make_fake_subproc(pushed_cmds_a)),
            ):
                train_mod.rclone_push(ckpt_file)

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/sess-b",
                        "RELAY_SESSION": "sess-b",
                        "CHECKPOINT_PATH": ckpt_file,
                    },
                    clear=False,
                ),
                mock.patch.object(train_mod.subprocess, "run", side_effect=make_fake_subproc(pushed_cmds_b)),
            ):
                train_mod.rclone_push(ckpt_file)

            tmp_dest_a = pushed_cmds_a[0][-1]
            tmp_dest_b = pushed_cmds_b[0][-1]
            self.assertTrue(tmp_dest_a.startswith("gdrive:colab_relay/sess-a/.sess-a."))
            self.assertTrue(tmp_dest_a.endswith(".checkpoint.pt.uploading"))
            self.assertTrue(tmp_dest_b.startswith("gdrive:colab_relay/sess-b/.sess-b."))
            self.assertNotEqual(tmp_dest_a, tmp_dest_b)

    def test_issue_45_first_alloc_retry_keeps_relay_round_1_and_train_empty_remote_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(ws_dir, "train.py").write_text("print('train')\n", encoding="utf-8")
            rclone_conf = os.path.join(td, "rclone.conf")
            Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            locks_file = os.path.join(td, "locks.json")

            new_attempts = {"n": 0}
            captured_boot_scripts = []

            def fake_run_cmd(cmd, timeout=300, stdin_text=None):
                prog = os.path.basename(cmd[0]).lower()
                if "rclone" in prog:
                    if "copyto" in cmd:
                        Path(cmd[-1]).write_bytes(b"final-weights-ok")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                sub = cmd[1]
                if sub == "new":
                    new_attempts["n"] += 1
                    if new_attempts["n"] == 1:
                        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="503 Service Unavailable")
                    return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                if sub == "sessions":
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if sub == "upload":
                    return subprocess.CompletedProcess(cmd, 0, stdout="Uploaded", stderr="")
                if sub == "exec":
                    captured_boot_scripts.append(stdin_text or "")
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout="CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n",
                        stderr="",
                    )
                if sub == "stop":
                    return subprocess.CompletedProcess(cmd, 0, stdout="Stopped", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(runner_mod, "ALLOC_RETRY_SLEEP_SECONDS", 0.0),
                mock.patch.object(runner_mod, "HANDOVER_SLEEP_SECONDS", 0.0),
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run_cmd),
            ):
                rc = runner_mod.main(["--session", "exp-45", "--workspace", ws_dir, "--reset-done-flag"])

            self.assertEqual(rc, 0)
            self.assertEqual(new_attempts["n"], 2)
            self.assertEqual(len(captured_boot_scripts), 1)
            self.assertIn('"RELAY_ROUND": "1"', captured_boot_scripts[0])
            self.assertIn('"RESET_DONE_FLAG": "1"', captured_boot_scripts[0])

            # Issue #81: train.py with RELAY_ROUND=2 and an empty remote dir (no RESET_DONE_FLAG) MUST abort and raise RuntimeError
            missing_ckpt = os.path.join(td, "no_ckpt.pt")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "RELAY_ROUND": "2",
                        "RESET_DONE_FLAG": "0",
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/empty_new",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    train_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(["rclone", "lsf"], 0, stdout="", stderr=""),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx_empty:
                    train_mod.rclone_pull_checkpoint(missing_ckpt)
                self.assertIn("no_ckpt.pt", str(ctx_empty.exception))

            # Explicit RESET_DONE_FLAG=1 allows starting fresh when intentionally reset
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "RELAY_ROUND": "2",
                        "RESET_DONE_FLAG": "1",
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/empty_new",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    train_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(["rclone", "lsf"], 0, stdout="", stderr=""),
                ),
            ):
                self.assertFalse(train_mod.rclone_pull_checkpoint(missing_ckpt))

            # train.py: if remote HAS prior training artifacts (results/ or _prev.pt) but checkpoint.pt is missing, MUST fail fast
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "RELAY_ROUND": "2",
                        "RESET_DONE_FLAG": "0",
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/resuming_job",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    train_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(["rclone", "lsf"], 0, stdout="results/\n", stderr=""),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    train_mod.rclone_pull_checkpoint(missing_ckpt)

    def test_issue_46_relay_runner_aborts_when_old_instance_stop_fails_or_remains_on_server(self):
        for fail_mode in ("stop_nonzero", "still_in_billing_list"):
            with self.subTest(fail_mode=fail_mode), tempfile.TemporaryDirectory() as td:
                ws_dir = os.path.join(td, "ws")
                os.makedirs(ws_dir, exist_ok=True)
                Path(ws_dir, "train.py").write_text("print('ok')\n", encoding="utf-8")
                rclone_conf = os.path.join(td, "rclone.conf")
                Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
                locks_file = os.path.join(td, "locks.json")

                new_calls = []
                vm_allocated = {"live": False}

                def fake_run(cmd, timeout=300, stdin_text=None):
                    prog = os.path.basename(cmd[0]).lower()
                    if "rclone" in prog:
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    sub = cmd[1]
                    if sub == "new":
                        new_calls.append(list(cmd))
                        vm_allocated["live"] = True
                        return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")
                    if sub == "upload":
                        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
                    if sub == "exec":
                        return subprocess.CompletedProcess(
                            cmd,
                            0,
                            stdout="CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\nTRAIN_EXIT_CODE=0\n",
                            stderr="",
                        )
                    if sub == "stop":
                        if fail_mode == "stop_nonzero":
                            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Error: runtime unassign HTTP 500")
                        return subprocess.CompletedProcess(cmd, 0, stdout="Stop requested", stderr="")
                    if sub == "status":
                        if not vm_allocated["live"]:
                            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                        if fail_mode == "still_in_billing_list":
                            return subprocess.CompletedProcess(cmd, 0, stdout="[exp-46] Active (m-s-leak1)\nHardware: T4\n", stderr="")
                        return subprocess.CompletedProcess(cmd, 0, stdout="[exp-46] Active\n", stderr="")
                    if sub == "sessions":
                        if vm_allocated["live"] and fail_mode == "still_in_billing_list":
                            return subprocess.CompletedProcess(cmd, 0, stdout="exp-46  m-s-leak1  T4  Active\n", stderr="")
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                out_buf = io.StringIO()
                orig_stop = runner_mod.stop_and_verify_session
                with (
                    mock.patch.object(runner_mod, "HANDOVER_SLEEP_SECONDS", 0.0),
                    mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                    mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                    mock.patch.object(runner_mod, "register_workspace"),
                    mock.patch.object(
                        runner_mod,
                        "stop_and_verify_session",
                        side_effect=lambda cb, sn, **kw: orig_stop(cb, sn, max_retries=2, retry_delay=0.0, require_stop_rc_zero=kw.get("require_stop_rc_zero", True)),
                    ),
                    mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                    redirect_stdout(out_buf),
                ):
                    rc = runner_mod.main(["--session", "exp-46", "--workspace", ws_dir])

                printed = out_buf.getvalue()
                self.assertEqual(rc, 1)
                self.assertEqual(len(new_calls), 1, "Must NEVER allocate Round #2 VM when Round #1 VM stop is unverified")
                self.assertNotIn("本棒安全交接完毕", printed)

    def test_issue_47_hf_trainer_callback_backup_and_round2_resume_protocol(self):
        skill_doc = (REPO_ROOT / ".agents/skills/colab-dl-relay/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("resolve_hf_resume_checkpoint", skill_doc)
        self.assertIn("hf_checkpoints", skill_doc)

        fake_transformers = types.ModuleType("transformers")

        class DummyTrainerCallback:
            pass

        fake_transformers.TrainerCallback = DummyTrainerCallback

        with tempfile.TemporaryDirectory() as td:
            remote_store = os.path.join(td, "fake_gdrive_hf")
            r1_out = os.path.join(td, "round1_hf_checkpoints")
            r2_out = os.path.join(td, "round2_hf_checkpoints")
            os.makedirs(os.path.join(r1_out, "checkpoint-5"), exist_ok=True)
            os.makedirs(os.path.join(r1_out, "checkpoint-10"), exist_ok=True)
            Path(r1_out, "checkpoint-5", "trainer_state.json").write_text(
                json.dumps({"global_step": 5, "epoch": 0.5}), encoding="utf-8"
            )
            Path(r1_out, "checkpoint-10", "trainer_state.json").write_text(
                json.dumps({"global_step": 10, "epoch": 1.0}), encoding="utf-8"
            )

            ctx1 = adapter_mod.ColabRelayContext(
                model_name="hf-lora",
                framework="huggingface",
                total_epochs=3,
                gdrive_dir="gdrive:colab_relay/hf_exp",
                results_dir=os.path.join(td, "r1_results"),
                checkpoint_path=os.path.join(td, "r1_unused_checkpoint.pt"),
                hf_checkpoints_dir=r1_out,
                done_flag_path=os.path.join(td, "r1_done.flag"),
                smoke_test=False,
                round_num=1,
            )
            ctx1.state_file = os.path.join(td, "r1_state.json")

            def fake_rclone_r1(args, timeout=180):
                if args[0] == "copy" and args[-1].endswith("/hf_checkpoints"):
                    shutil.copytree(args[-2], remote_store, dirs_exist_ok=True)
                return True

            with (
                mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
                mock.patch.object(ctx1, "_run_rclone", side_effect=fake_rclone_r1),
            ):
                cb = adapter_mod.make_hf_trainer_callback(ctx1)
                fake_args = types.SimpleNamespace(output_dir=r1_out)
                fake_state = types.SimpleNamespace(epoch=1.0, global_step=10, max_steps=30, log_history=[{"loss": 0.32}])
                fake_ctrl = types.SimpleNamespace(should_save=False, should_training_stop=False)
                cb.on_save(fake_args, fake_state, fake_ctrl)

            self.assertTrue(os.path.isfile(os.path.join(remote_store, "checkpoint-10", "trainer_state.json")))

            # Round 2: NO checkpoint.pt exists anywhere, only hf_checkpoints/checkpoint-*/trainer_state.json on Drive
            ctx2 = adapter_mod.ColabRelayContext(
                model_name="hf-lora",
                framework="huggingface",
                total_epochs=3,
                gdrive_dir="gdrive:colab_relay/hf_exp",
                results_dir=os.path.join(td, "r2_results"),
                checkpoint_path=os.path.join(td, "r2_missing_checkpoint.pt"),
                hf_checkpoints_dir=r2_out,
                done_flag_path=os.path.join(td, "r2_done.flag"),
                smoke_test=False,
                round_num=2,
            )
            ctx2.state_file = os.path.join(td, "r2_state.json")

            def fake_exists_r2(remote_dir, filename, timeout=90):
                if filename == "hf_checkpoints":
                    return True
                return False

            def fake_rclone_r2(args, timeout=180):
                if args[0] == "copy" and args[3].endswith("/hf_checkpoints"):
                    shutil.copytree(remote_store, args[4], dirs_exist_ok=True)
                return True

            with (
                mock.patch.object(ctx2, "_rclone_remote_file_exists", side_effect=fake_exists_r2),
                mock.patch.object(
                    ctx2,
                    "_rclone_list_remote_hf_trainer_states",
                    return_value=["checkpoint-10/trainer_state.json", "checkpoint-5/trainer_state.json"],
                ),
                mock.patch.object(ctx2, "_run_rclone", side_effect=fake_rclone_r2),
            ):
                ctx2.prepare_cloud_workspace()
                resume_dir = ctx2.resolve_hf_resume_checkpoint(r2_out)

            self.assertIsNotNone(resume_dir)
            self.assertEqual(os.path.basename(resume_dir), "checkpoint-10")
            self.assertTrue(os.path.isfile(os.path.join(resume_dir, "trainer_state.json")))

    def test_issue_48_custom_and_auto_session_names_in_bluegreen_state_switchover_and_gateway(self):
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "llm_bluegreen_state.json")
            keys_file = os.path.join(td, "llm_api_keys.json")
            sess_file = os.path.join(td, "sessions.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            parser = manager_mod.build_arg_parser()

            # 1) Deploy blue slot with auto/custom session name `vm-custom-88aa` (not ending in -blue)
            args_dep = parser.parse_args(
                [
                    "deploy",
                    "--session-prefix",
                    "prod",
                    "--session-name",
                    "vm-custom-88aa",
                    "--slot",
                    "blue",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "G4",
                    "--engine",
                    "builtin",
                    "--model-name",
                    "Qwen-Custom",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 55.0, "ttft_ms": 80.0, "role": "active"},
                ),
                mock.patch.object(manager_mod, "ensure_local_gateway"),
            ):
                rc_dep = manager_mod.cmd_deploy(args_dep)

            self.assertEqual(rc_dep, 0)
            st_after_deploy = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertEqual(st_after_deploy["active_session"], "vm-custom-88aa")
            self.assertEqual(st_after_deploy["blue_session"], "vm-custom-88aa")
            self.assertEqual(st_after_deploy["slots"]["blue"]["session_name"], "vm-custom-88aa")

            # Gateway forced_slot="blue" and unified (None) both resolve to `vm-custom-88aa`
            Path(sess_file).write_text(
                json.dumps(
                    {
                        "vm-custom-88aa": {
                            "endpoint": "m-s-custom88",
                            "url": "https://custom.colab.dev",
                            "token": "tok-88",
                            "kernel_id": "k-88",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(gateway_mod, "state_path", return_value=bg_file),
                mock.patch.object(gateway_mod, "sessions_path", return_value=sess_file),
                mock.patch.object(gateway_mod, "api_keys_path", return_value=keys_file),
            ):
                sl_b, _, sess_b = gateway_mod.resolve_slot_session(forced_slot="blue")
                sl_u, _, sess_u = gateway_mod.resolve_slot_session(forced_slot=None)
            self.assertEqual(sl_b, "blue")
            self.assertEqual(sess_b["session_name"], "vm-custom-88aa")
            self.assertEqual(sess_u["session_name"], "vm-custom-88aa")

            # 2) Switchover from blue -> green with custom green name `vm-green-custom-99bb`
            role_calls = []
            stopped_sessions = []

            def fake_run_sw(cmd, timeout=300, stdin_text=None):
                sub = cmd[1]
                if sub == "stop":
                    stopped_sessions.append(cmd[-1])
                    return subprocess.CompletedProcess(cmd, 0, stdout="Stopped", stderr="")
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                if sub == "sessions":
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if sub == "exec":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout='DRAIN_STATUS {"active_streams": 0, "active_requests": 0}\n', stderr=""
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            args_sw = parser.parse_args(
                [
                    "switchover",
                    "--workspace",
                    ws_dir,
                    "--new-session-name",
                    "vm-green-custom-99bb",
                    "--drain-timeout",
                    "2",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 61.0, "ttft_ms": 75.0, "role": "warming"},
                ),
                mock.patch.object(manager_mod, "set_remote_slot_role", side_effect=lambda s, r: role_calls.append((s, r))),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_sw),
            ):
                rc_sw = manager_mod.cmd_switchover(args_sw)

            self.assertEqual(rc_sw, 0)
            self.assertIn(("vm-green-custom-99bb", "active"), role_calls)
            self.assertIn(("vm-custom-88aa", "draining"), role_calls)
            self.assertEqual(stopped_sessions, ["vm-custom-88aa"])
            st_after_sw = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertEqual(st_after_sw["active_slot"], "green")
            self.assertEqual(st_after_sw["active_session"], "vm-green-custom-99bb")
            self.assertEqual(st_after_sw["green_session"], "vm-green-custom-99bb")
            self.assertIsNone(st_after_sw["blue_session"])
            self.assertEqual(st_after_sw["slots"]["green"]["session_name"], "vm-green-custom-99bb")

    def test_issue_49_switchover_drains_in_flight_sse_streams_before_stopping_old_slot(self):
        # 1) Verify Sidecar tracks active_streams & active_requests > 0 throughout SSE streaming until EOF
        first_chunk_sent = threading.Event()
        allow_stream_finish = threading.Event()

        class SlowStreamingBackend(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"Part1"}}]}\n\n')
                self.wfile.flush()
                first_chunk_sent.set()
                allow_stream_finish.wait(timeout=5.0)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"Part2"}}]}\n\ndata: [DONE]\n\n')
                self.wfile.flush()

        backend_srv = sidecar_mod.ThreadingHTTPServer(("127.0.0.1", 0), SlowStreamingBackend)
        backend_port = backend_srv.server_address[1]
        threading.Thread(target=backend_srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as td:
                llm_state_file = os.path.join(td, "llm_state.json")
                tracker = sidecar_mod.LlmTelemetryTracker(
                    slot="blue",
                    role="active",
                    engine="vllm",
                    model_name="qwen",
                    listen_port=0,
                    backend_url=f"http://127.0.0.1:{backend_port}",
                    state_file=llm_state_file,
                )
                sidecar_srv = sidecar_mod.ThreadingHTTPServer(("127.0.0.1", 0), sidecar_mod.make_handler(tracker))
                sidecar_port = sidecar_srv.server_address[1]
                threading.Thread(target=sidecar_srv.serve_forever, daemon=True).start()
                try:
                    stream_done = threading.Event()
                    stream_body = []

                    def _client():
                        conn = http.client.HTTPConnection("127.0.0.1", sidecar_port, timeout=5)
                        conn.request(
                            "POST",
                            "/v1/chat/completions",
                            body=b'{"stream":true,"messages":[{"role":"user","content":"hi"}]}',
                            headers={"Content-Type": "application/json"},
                        )
                        resp = conn.getresponse()
                        stream_body.append(resp.read().decode("utf-8", errors="ignore"))
                        conn.close()
                        stream_done.set()

                    threading.Thread(target=_client, daemon=True).start()
                    self.assertTrue(first_chunk_sent.wait(timeout=3.0))

                    # While SSE is still open, state file and /pilot/health MUST report active_streams=1 & active_requests=1
                    mid_state = json.loads(Path(llm_state_file).read_text(encoding="utf-8"))
                    self.assertEqual(mid_state["active_streams"], 1)
                    self.assertEqual(mid_state["active_requests"], 1)

                    allow_stream_finish.set()
                    self.assertTrue(stream_done.wait(timeout=3.0))
                    end_state = json.loads(Path(llm_state_file).read_text(encoding="utf-8"))
                    self.assertEqual(end_state["active_streams"], 0)
                    self.assertEqual(end_state["active_requests"], 0)
                finally:
                    sidecar_srv.shutdown()
                    sidecar_srv.server_close()
        finally:
            allow_stream_finish.set()
            backend_srv.shutdown()
            backend_srv.server_close()

        # 2) Verify cmd_switchover polls draining slot until active_streams == 0 BEFORE calling `colab stop`
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "bg.json")
            keys_file = os.path.join(td, "keys.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(bg_file).write_text(
                json.dumps({"session_prefix": "llm", "active_slot": "blue", "active_session": "llm-blue", "gpu": "T4"}),
                encoding="utf-8",
            )

            events = []
            drain_counts = [2, 1, 0]

            def fake_run_drain(cmd, timeout=300, stdin_text=None):
                sub = cmd[1]
                if sub == "exec":
                    cnt = drain_counts.pop(0) if drain_counts else 0
                    events.append(f"drain_poll:{cnt}")
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=f'DRAIN_STATUS {{"active_streams": {cnt}, "active_requests": {cnt}}}\n',
                        stderr="",
                    )
                if sub == "stop":
                    events.append(f"stop:{cmd[-1]}")
                    return subprocess.CompletedProcess(cmd, 0, stdout="Stopped", stderr="")
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            parser = manager_mod.build_arg_parser()
            args_sw = parser.parse_args(
                [
                    "switchover",
                    "--workspace",
                    ws_dir,
                    "--drain-timeout",
                    "5",
                    "--drain-poll-interval",
                    "0.03",
                    "--gateway-port",
                    "18098",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 50.0, "ttft_ms": 90.0, "role": "warming"},
                ),
                mock.patch.object(
                    manager_mod,
                    "set_remote_slot_role",
                    side_effect=lambda s, r: events.append(f"role:{s}={r}"),
                ),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_drain),
            ):
                rc = manager_mod.cmd_switchover(args_sw)

            self.assertEqual(rc, 0)
            self.assertEqual(
                events,
                [
                    "role:llm-green=active",
                    "role:llm-blue=draining",
                    "drain_poll:2",
                    "drain_poll:1",
                    "drain_poll:0",
                    "stop:llm-blue",
                ],
            )

            # 3) Verify --drain-timeout expiration with stuck stream respects deadline & policy
            t_start = time.monotonic()
            stop_on_keep = []

            def fake_run_stuck(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "exec":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout='DRAIN_STATUS {"active_streams": 1, "active_requests": 1}\n', stderr=""
                    )
                if cmd[1] == "stop":
                    stop_on_keep.append(cmd[-1])
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            Path(bg_file).write_text(
                json.dumps({"session_prefix": "llm", "active_slot": "blue", "active_session": "llm-blue", "gpu": "T4"}),
                encoding="utf-8",
            )
            args_keep_policy = parser.parse_args(
                [
                    "switchover",
                    "--workspace",
                    ws_dir,
                    "--drain-timeout",
                    "0.12",
                    "--drain-poll-interval",
                    "0.04",
                    "--drain-timeout-policy",
                    "keep",
                    "--gateway-port",
                    "18098",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 50.0, "ttft_ms": 90.0, "role": "warming"},
                ),
                mock.patch.object(manager_mod, "set_remote_slot_role"),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_stuck),
                mock.patch.object(sys, "stderr", io.StringIO()),
            ):
                rc_keep_pol = manager_mod.cmd_switchover(args_keep_policy)

            self.assertEqual(rc_keep_pol, 1)
            self.assertGreaterEqual(time.monotonic() - t_start, 0.11)
            self.assertEqual(stop_on_keep, [])
            st_timeout = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertTrue(st_timeout.get("drain_timed_out"))
            self.assertIn("llm-blue", st_timeout.get("unreleased_sessions", []))

    def test_issue_50_switchover_old_slot_stop_failure_returns_nonzero_and_preserves_active_new_slot(self):
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "bg.json")
            keys_file = os.path.join(td, "keys.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(bg_file).write_text(
                json.dumps({"session_prefix": "llm", "active_slot": "blue", "active_session": "llm-blue", "gpu": "T4"}),
                encoding="utf-8",
            )

            def fake_run_stop_fail(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "exec":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout='DRAIN_STATUS {"active_streams": 0, "active_requests": 0}\n', stderr=""
                    )
                if cmd[1] == "stop":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 500 unassign failed")
                if cmd[1] == "status":
                    return subprocess.CompletedProcess(cmd, 0, stdout="[llm-blue] Active (m-s-blue1)\nHardware: T4\n", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            parser = manager_mod.build_arg_parser()
            args_sw = parser.parse_args(
                ["switchover", "--workspace", ws_dir, "--drain-timeout", "1", "--gateway-port", "18098"]
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            orig_stop = manager_mod.stop_and_verify_colab_session
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(
                    manager_mod,
                    "provision_and_warmup_slot",
                    return_value={"instant_tps": 48.0, "ttft_ms": 85.0, "role": "warming"},
                ),
                mock.patch.object(manager_mod, "set_remote_slot_role"),
                mock.patch.object(
                    manager_mod,
                    "stop_and_verify_colab_session",
                    side_effect=lambda cb, sn, **kw: orig_stop(cb, sn, max_retries=1, retry_delay=0.0, require_stop_rc_zero=True),
                ),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_stop_fail),
                redirect_stdout(out_buf),
                mock.patch.object(sys, "stderr", err_buf),
            ):
                rc = manager_mod.cmd_switchover(args_sw)

            self.assertEqual(rc, 1)
            combined_out = out_buf.getvalue() + "\n" + err_buf.getvalue()
            self.assertNotIn("已彻底释放", combined_out)
            saved_st = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertEqual(saved_st["active_slot"], "green")
            self.assertEqual(saved_st["active_session"], "llm-green")
            self.assertEqual(saved_st["phase"], "serving")
            self.assertFalse(saved_st["old_slot_released"])
            self.assertIn("llm-blue", saved_st["unreleased_sessions"])
            self.assertIn("llm-blue", saved_st["last_error"])

    def test_issue_51_stop_with_session_prefix_does_not_touch_other_active_prefix_or_gateway(self):
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "llm_bluegreen_state.json")
            prod_state = {
                "session_prefix": "production",
                "active_slot": "blue",
                "active_session": "production-blue",
                "blue_session": "production-blue",
                "slots": {"blue": {"session_name": "production-blue", "role": "active"}},
                "phase": "serving",
            }
            Path(bg_file).write_text(json.dumps(prod_state), encoding="utf-8")

            stopped_names = []
            gw_stop_calls = []

            def fake_run_exp_fail(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "stop":
                    stopped_names.append(cmd[-1])
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Error: failed to stop experiment VM")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            parser = manager_mod.build_arg_parser()
            args_stop_exp = parser.parse_args(["stop", "--session-prefix", "experiment"])
            orig_stop = manager_mod.stop_and_verify_colab_session
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "stop_local_gateway", side_effect=lambda: gw_stop_calls.append(1)),
                mock.patch.object(
                    manager_mod,
                    "stop_and_verify_colab_session",
                    side_effect=lambda cb, sn, **kw: orig_stop(cb, sn, max_retries=1, retry_delay=0.0, require_stop_rc_zero=False),
                ),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_exp_fail),
            ):
                rc_fail = manager_mod.cmd_stop(args_stop_exp)

            self.assertEqual(rc_fail, 1)
            self.assertNotIn("production-blue", stopped_names)
            self.assertEqual(stopped_names, ["experiment-blue", "experiment-green"])
            self.assertEqual(gw_stop_calls, [], "Must NOT stop shared local gateway used by production")
            self.assertTrue(os.path.exists(bg_file), "Must NOT delete production state file")
            preserved = json.loads(Path(bg_file).read_text(encoding="utf-8"))
            self.assertEqual(preserved["active_session"], "production-blue")

            # Even when experiment-* stops cleanly, production state & shared gateway remain untouched
            def fake_run_exp_ok(cmd, timeout=300, stdin_text=None):
                if cmd[1] == "stop":
                    return subprocess.CompletedProcess(cmd, 0, stdout="Stopped", stderr="")
                if cmd[1] == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "stop_local_gateway", side_effect=lambda: gw_stop_calls.append(1)),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fake_run_exp_ok),
            ):
                rc_ok = manager_mod.cmd_stop(args_stop_exp)

            self.assertEqual(rc_ok, 0)
            self.assertEqual(gw_stop_calls, [])
            self.assertTrue(os.path.exists(bg_file))

    def test_issue_53_final_weights_download_failure_returns_nonzero_and_rerun_redownloads(self):
        with tempfile.TemporaryDirectory() as td:
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(ws_dir, "train.py").write_text("print('ok')\n", encoding="utf-8")
            rclone_conf = os.path.join(td, "rclone.conf")
            Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            locks_file = os.path.join(td, "locks.json")
            final_ckpt_path = os.path.join(ws_dir, "checkpoint_final.pt")

            remote_completed = {"flag": False}
            allow_download = {"ok": False}
            colab_new_count = {"n": 0}

            def fake_run(cmd, timeout=300, stdin_text=None):
                prog = os.path.basename(cmd[0]).lower()
                if "rclone" in prog:
                    if "lsf" in cmd:
                        out = "TRAINING_COMPLETED.flag\ncheckpoint.pt\n" if remote_completed["flag"] else ""
                        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
                    if "copyto" in cmd:
                        if not allow_download["ok"]:
                            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="network reset during download")
                        Path(cmd[-1]).write_bytes(b"verified-final-model-weights")
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                sub = cmd[1]
                if sub == "new":
                    colab_new_count["n"] += 1
                    return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")
                if sub == "exec":
                    remote_completed["flag"] = True
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout="TRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n", stderr=""
                    )
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            buf1 = io.StringIO()
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(buf1),
            ):
                rc1 = runner_mod.main(["--session", "exp-53", "--workspace", ws_dir])

            self.assertEqual(rc1, 1)
            self.assertFalse(os.path.exists(final_ckpt_path))
            self.assertNotIn("最终模型已保存至", buf1.getvalue())
            receipt1 = json.loads(Path(ws_dir, ".colab_dl_relay_status.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt1["training_completed"])
            self.assertFalse(receipt1["weights_downloaded"])

            # Re-run on the same directory when remote flag already exists -> must re-download without allocating new VM
            allow_download["ok"] = True
            buf2 = io.StringIO()
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(buf2),
            ):
                rc2 = runner_mod.main(["--session", "exp-53", "--workspace", ws_dir])

            self.assertEqual(rc2, 0)
            self.assertEqual(colab_new_count["n"], 1, "Re-run must only backfill download without provisioning a new VM")
            self.assertTrue(os.path.isfile(final_ckpt_path))
            self.assertGreater(os.path.getsize(final_ckpt_path), 0)
            receipt2 = json.loads(Path(ws_dir, ".colab_dl_relay_status.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt2["weights_downloaded"])

    def test_issue_61_relay_rclone_sh_returns_nonzero_on_python_traceback_even_when_exec_rc_zero(self):
        sh_path = REPO_ROOT / "relay_rclone.sh"
        sh_bytes = sh_path.read_bytes()
        self.assertNotIn(b"\r", sh_bytes)
        sh_text = sh_bytes.decode("utf-8")
        self.assertNotIn('exit "${EXEC_RC:-1}"', sh_text)

        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        bash_bin = git_bash if (os.name == "nt" and os.path.exists(git_bash)) else shutil.which("bash")
        if not bash_bin:
            return
        probe = subprocess.run([bash_bin, "--version"], capture_output=True, text=True, timeout=10)
        if probe.returncode != 0:
            return

        with tempfile.TemporaryDirectory() as td:
            bin_dir = os.path.join(td, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            shutil.copy2(sh_path, os.path.join(td, "relay_rclone.sh"))
            Path(td, "train.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
            rclone_conf = os.path.join(td, "rclone.conf")
            Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            stop_marker = os.path.join(td, "stopped.txt")

            Path(bin_dir, "rclone").write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
            colab_mock = (
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "stop" ]]; then echo stopped >> "$STOP_MARKER"; exit 0; fi\n'
                'if [[ "$1" == "exec" && "$*" == *"--timeout 600"* ]]; then echo "RCLONE_READY"; exit 0; fi\n'
                'if [[ "$1" == "exec" ]]; then\n'
                '  echo "Traceback (most recent call last):"\n'
                '  echo "RuntimeError: CUDA OOM"\n'
                "  exit 0\n"
                "fi\n"
                "exit 0\n"
            )
            Path(bin_dir, "colab").write_bytes(colab_mock.encode("utf-8"))
            os.chmod(os.path.join(bin_dir, "rclone"), 0o755)
            os.chmod(os.path.join(bin_dir, "colab"), 0o755)

            runner_sh = (
                "#!/usr/bin/env bash\n"
                "chmod +x ./bin/rclone ./bin/colab 2>/dev/null || true\n"
                'export PATH="$PWD/bin:$PATH"\n'
                'export LOCAL_RCLONE_CONF="./rclone.conf"\n'
                'export STOP_MARKER="./stopped.txt"\n'
                'export SESSION="bash-test-61"\n'
                "exec bash ./relay_rclone.sh\n"
            )
            Path(td, "run_test.sh").write_bytes(runner_sh.encode("utf-8"))

            res = subprocess.run(
                [bash_bin, "./run_test.sh"],
                cwd=td,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            combined = (res.stdout or "") + "\n" + (res.stderr or "")
            self.assertEqual(res.returncode, 1, f"relay_rclone.sh must exit 1 on Traceback when EXEC_RC=0:\n{combined}")
            self.assertIn("熔断止损", combined)
            self.assertTrue(os.path.exists(stop_marker))

    def test_issue_62_benchmark_propagates_remote_exec_failure_and_missing_results(self):
        parser = manager_mod.build_arg_parser()
        args = parser.parse_args(["benchmark", "--session-name", "llm-blue", "--requests", "2"])

        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "bg.json")
            Path(bg_file).write_text(json.dumps({"model_name": "Qwen-Test"}), encoding="utf-8")

            # 1) Non-zero colab exec returncode -> non-zero exit
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(
                    manager_mod,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess(["colab", "exec"], 2, stdout="", stderr="exec error"),
                ),
            ):
                self.assertNotEqual(manager_mod.cmd_benchmark(args), 0)

            # 2) colab exec returns 0, but no BENCH_RESULT / Traceback -> non-zero exit
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(
                    manager_mod,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess(
                        ["colab", "exec"], 0, stdout="Traceback (most recent call last):\nConnectionRefusedError", stderr=""
                    ),
                ),
            ):
                self.assertEqual(manager_mod.cmd_benchmark(args), 1)

            # 3) colab exec returns 0 with 0 tokens -> non-zero exit
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(
                    manager_mod,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess(
                        ["colab", "exec"],
                        0,
                        stdout='BENCH_RESULT {"ok_requests": 0, "bench_tokens": 0, "state": {"total_completion_tokens": 0}}\n',
                        stderr="",
                    ),
                ),
            ):
                self.assertEqual(manager_mod.cmd_benchmark(args), 1)

            # 4) Valid benchmark result -> returns 0
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(
                    manager_mod,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess(
                        ["colab", "exec"],
                        0,
                        stdout='BENCH_RESULT {"ok_requests": 2, "bench_tokens": 36, "state": {"total_completion_tokens": 36, "instant_tps": 44.2}}\n',
                        stderr="",
                    ),
                ),
            ):
                self.assertEqual(manager_mod.cmd_benchmark(args), 0)

    def test_issue_63_smoke_test_dl_rejects_unread_checkpoint_and_missing_done_flag(self):
        with tempfile.TemporaryDirectory() as td:
            fake_rclone = os.path.join(td, "rclone.exe" if os.name == "nt" else "rclone")
            Path(fake_rclone).write_bytes(b"stub")
            fake_conf = os.path.join(td, "rclone.conf")
            Path(fake_conf).write_text("[gdrive]\ntype = drive\ntoken = {}\n", encoding="utf-8")

            # Case 1: Fake script writes dummy bytes in Pass #1, hardcodes epoch=2 in Pass #2 without reading ckpt or writing done flag
            fake_unread = os.path.join(td, "fake_unread.py")
            Path(fake_unread).write_text(
                "import json, os\n"
                "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
                "if p == 1:\n"
                "    open(os.environ['CHECKPOINT_PATH'], 'wb').write(b'dummy-bytes')\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
                "else:\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 2, 'loss': 0.5}))\n",
                encoding="utf-8",
            )

            # Case 2: Reads & parses checkpoint JSON, updates epoch=2, but omits done flag file
            fake_no_flag = os.path.join(td, "fake_no_flag.py")
            Path(fake_no_flag).write_text(
                "import json, os\n"
                "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
                "if p == 1:\n"
                "    open(os.environ['CHECKPOINT_PATH'], 'w', encoding='utf-8').write(json.dumps({'epoch': 1}))\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
                "else:\n"
                "    ck = json.loads(open(os.environ['CHECKPOINT_PATH'], 'r', encoding='utf-8').read())\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': ck['epoch'] + 1, 'loss': 0.4}))\n",
                encoding="utf-8",
            )

            # Case 3: Opens checkpoint in binary mode and ignores contents (passes read audit, caught by corrupt probe!)
            fake_ignored_read = os.path.join(td, "fake_ignored_read.py")
            Path(fake_ignored_read).write_text(
                "import json, os\n"
                "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
                "if p == 1:\n"
                "    open(os.environ['CHECKPOINT_PATH'], 'wb').write(b'anything')\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
                "else:\n"
                "    _ = open(os.environ['CHECKPOINT_PATH'], 'rb').read()\n"
                "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 2, 'loss': 0.4}))\n"
                "    open(os.environ['DONE_FLAG_PATH'], 'w').write('DONE')\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(smoke_mod, "find_rclone", return_value=fake_rclone),
                mock.patch.object(smoke_mod, "find_rclone_conf", return_value=fake_conf),
            ):
                self.assertEqual(smoke_mod.main(["--script", fake_unread, "--skip-rclone-net"]), 1)
                self.assertEqual(smoke_mod.main(["--script", fake_no_flag, "--skip-rclone-net"]), 1)
                self.assertEqual(smoke_mod.main(["--script", fake_ignored_read, "--skip-rclone-net"]), 1)
                self.assertEqual(smoke_mod.main(["--script", str(REPO_ROOT / "train.py"), "--skip-rclone-net"]), 0)

    def test_issue_68_cuda_engine_presets_rejected_on_cpu_and_tpu_before_provisioning(self):
        with tempfile.TemporaryDirectory() as td:
            bg_file = os.path.join(td, "bg.json")
            keys_file = os.path.join(td, "keys.json")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            parser = manager_mod.build_arg_parser()

            cli_calls = []

            def fail_if_called(cmd, *args, **kwargs):
                cli_calls.append(list(cmd))
                raise AssertionError(f"Cloud CLI must not be invoked for incompatible hardware/engine: {cmd}")

            for bad_gpu in ("CPU", "V5E1", "V6E1"):
                for cuda_eng in ("vllm", "llama.cpp", "transformers"):
                    with self.subTest(gpu=bad_gpu, engine=cuda_eng):
                        args_bad = parser.parse_args(
                            [
                                "deploy",
                                "--session-name",
                                f"bad-{bad_gpu.lower()}",
                                "--slot",
                                "standalone",
                                "--workspace",
                                ws_dir,
                                "--gpu",
                                bad_gpu,
                                "--engine",
                                cuda_eng,
                                "--model-name",
                                "Qwen-7B",
                            ]
                        )
                        err_buf = io.StringIO()
                        with (
                            mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                            mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                            mock.patch.object(manager_mod, "run_cmd", side_effect=fail_if_called),
                            mock.patch.object(sys, "stderr", err_buf),
                        ):
                            rc = manager_mod.cmd_deploy(args_bad)
                        self.assertEqual(rc, 1)
                        self.assertEqual(cli_calls, [])
                        self.assertIn("不兼容", err_buf.getvalue())

            # Explicit --llama-cpp-cpu ONLY exempts llama.cpp on CPU (NEVER transformers or vllm)
            self.assertIsNone(manager_mod.validate_hardware_engine_compatibility("CPU", "llama.cpp", allow_cpu_llama_cpp=True))
            self.assertIsNotNone(manager_mod.validate_hardware_engine_compatibility("CPU", "transformers", allow_cpu_llama_cpp=True))
            self.assertIsNotNone(manager_mod.validate_hardware_engine_compatibility("CPU", "vllm", allow_cpu_llama_cpp=True))
            args_tf_cpu = parser.parse_args(
                [
                    "deploy",
                    "--session-name",
                    "bad-tf-cpu",
                    "--slot",
                    "standalone",
                    "--workspace",
                    ws_dir,
                    "--gpu",
                    "CPU",
                    "--engine",
                    "transformers",
                    "--llama-cpp-cpu",
                    "--model-name",
                    "Qwen-7B",
                ]
            )
            with (
                mock.patch.object(manager_mod, "bg_state_path", return_value=bg_file),
                mock.patch.object(manager_mod, "api_keys_path", return_value=keys_file),
                mock.patch.object(manager_mod, "run_cmd", side_effect=fail_if_called),
                mock.patch.object(sys, "stderr", io.StringIO()),
            ):
                self.assertEqual(manager_mod.cmd_deploy(args_tf_cpu), 1)
            self.assertEqual(cli_calls, [])

            self.assertIsNone(manager_mod.validate_hardware_engine_compatibility("CPU", "builtin"))
            self.assertIsNone(manager_mod.validate_hardware_engine_compatibility("G4", "vllm"))
            self.assertIsNone(manager_mod.validate_hardware_engine_compatibility("T4", "llama.cpp"))

    def test_issue_47_53_hf_trainer_final_download_fallback_and_rerun_verification(self):
        with tempfile.TemporaryDirectory() as td:
            ws_dir = os.path.join(td, "ws_hf")
            os.makedirs(ws_dir, exist_ok=True)
            Path(ws_dir, "train.py").write_text("print('hf')\n", encoding="utf-8")
            rclone_conf = os.path.join(td, "rclone.conf")
            Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            locks_file = os.path.join(td, "locks.json")

            remote_hf = os.path.join(td, "remote_hf_checkpoints")
            os.makedirs(os.path.join(remote_hf, "checkpoint-20"), exist_ok=True)
            Path(remote_hf, "checkpoint-20", "trainer_state.json").write_text(
                json.dumps({"global_step": 20, "epoch": 2.0}), encoding="utf-8"
            )

            mode = {"lsf_err": False, "hf_copy_ok": True, "done": False}
            new_vm_calls = {"n": 0}
            hf_copy_calls = {"n": 0}

            def fake_run(cmd, timeout=300, stdin_text=None):
                prog = os.path.basename(cmd[0]).lower()
                if "rclone" in prog:
                    if "lsf" in cmd:
                        if mode["lsf_err"] and "--files-only" not in cmd:
                            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="googleapi: Error 401: Invalid Credentials")
                        if "--files-only" in cmd:
                            out = "training_done.flag\n" if mode["done"] else ""
                            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
                        out = "training_done.flag\nhf_checkpoints/\n" if mode["done"] else ""
                        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
                    if "copyto" in cmd:
                        # HF Trainer never produces checkpoint.pt on Drive
                        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="file not found")
                    if "copy" in cmd and cmd[-2].endswith("/hf_checkpoints"):
                        hf_copy_calls["n"] += 1
                        if not mode["hf_copy_ok"]:
                            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rclone hf copy failed")
                        shutil.copytree(remote_hf, cmd[-1], dirs_exist_ok=True)
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                sub = cmd[1]
                if sub == "new":
                    new_vm_calls["n"] += 1
                    return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")
                if sub == "exec":
                    mode["done"] = True
                    return subprocess.CompletedProcess(cmd, 0, stdout="TRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n", stderr="")
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            # 1) When rclone lsf fails with auth/network error (cannot confirm checkpoint.pt absent), MUST return 1
            mode["lsf_err"] = True
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner_mod.main(["--session", "hf-job", "--workspace", ws_dir]), 1)

            # 2) When remote has training_done.flag + hf_checkpoints/ (no checkpoint.pt), re-run backfills hf_checkpoints and exits 0
            mode["lsf_err"] = False
            mode["hf_copy_ok"] = True
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(io.StringIO()),
            ):
                rc_backfill = runner_mod.main(["--session", "hf-job", "--workspace", ws_dir])

            self.assertEqual(rc_backfill, 0)
            self.assertEqual(new_vm_calls["n"], 1, "Backfill re-run must not allocate a second VM")
            self.assertEqual(hf_copy_calls["n"], 1)
            local_ts = os.path.join(ws_dir, "hf_checkpoints", "checkpoint-20", "trainer_state.json")
            self.assertTrue(os.path.isfile(local_ts))
            receipt = json.loads(Path(ws_dir, ".colab_dl_relay_status.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["weights_downloaded"])
            self.assertTrue(receipt["local_weights_path"].endswith("checkpoint-20"))

            # 3) Subsequent re-run sees local hf_checkpoints/checkpoint-20 ready -> skips download and returns 0 immediately
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(io.StringIO()),
            ):
                rc_cached = runner_mod.main(["--session", "hf-job", "--workspace", ws_dir])

            self.assertEqual(rc_cached, 0)
            self.assertEqual(hf_copy_calls["n"], 1, "Local hf_checkpoints already valid; should not re-download")

    def test_issue_79_switchover_rejects_same_target_session_before_warmup_without_stopping_service(self):
        with tempfile.TemporaryDirectory() as td:
            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            cfg_dir = os.path.join(td, "colab-cli")
            os.makedirs(cfg_dir, exist_ok=True)
            bg_state_file = os.path.join(cfg_dir, "llm_bluegreen_state.json")
            initial_state = {
                "session_prefix": "llm",
                "active_slot": "blue",
                "active_session": "vm-shared",
                "blue_session": "vm-shared",
                "green_session": None,
                "slots": {
                    "blue": {"session_name": "vm-shared", "role": "active", "model_name": "Qwen/Qwen2.5-7B-Instruct"}
                },
                "phase": "serving",
                "engine": "builtin",
                "model_name": "builtin-mock",
                "gpu": "T4",
                "high_mem": True,
            }
            Path(bg_state_file).write_text(json.dumps(initial_state, ensure_ascii=False), encoding="utf-8")

            invoked_cmds = []

            def forbid_subproc(cmd, *args, **kwargs):
                invoked_cmds.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            parser = manager_mod.build_arg_parser()
            for candidate_argv in (
                ["switchover", "--workspace", ws_dir, "--session-name", "vm-shared"],
                ["switchover", "--workspace", ws_dir, "--session-name", "VM-SHARED"],
                ["switchover", "--workspace", ws_dir, "--session-prefix", "other-cluster", "--session-name", "vm-shared"],
            ):
                args_sw = parser.parse_args(candidate_argv)
                err_buf = io.StringIO()
                with (
                    mock.patch.object(manager_mod, "bg_state_path", return_value=bg_state_file),
                    mock.patch.object(
                        manager_mod,
                        "provision_and_warmup_slot",
                        side_effect=AssertionError("provision_and_warmup_slot must NOT be called"),
                    ),
                    mock.patch.object(manager_mod, "run_cmd", side_effect=forbid_subproc),
                    redirect_stderr(err_buf),
                ):
                    rc = manager_mod.cmd_switchover(args_sw)

                self.assertEqual(rc, 1, f"switchover must fail before warmup for {candidate_argv}")
                self.assertIn("vm-shared", err_buf.getvalue().lower())
                self.assertEqual(invoked_cmds, [], "must not invoke colab new/exec/stop when target matches active session")
                after_st = json.loads(Path(bg_state_file).read_text(encoding="utf-8"))
                self.assertEqual(after_st["phase"], "serving")
                self.assertEqual(after_st["active_session"], "vm-shared")

    def test_issue_80_legacy_bare_gdrive_checkpoint_detected_and_migrated_with_real_rclone(self):
        rclone_exe = runner_mod.find_bin("rclone")
        if not os.path.exists(rclone_exe) and not shutil.which(rclone_exe):
            self.skipTest("rclone binary not available")

        with tempfile.TemporaryDirectory() as td:
            drive_root = os.path.join(td, "drive_storage")
            legacy_dir = os.path.join(drive_root, "colab_relay")
            os.makedirs(legacy_dir, exist_ok=True)

            # Create valid Epoch 7 PyTorch checkpoint at legacy `gdrive:colab_relay/checkpoint.pt`
            legacy_ckpt = os.path.join(legacy_dir, "checkpoint.pt")
            import torch

            torch.save({"epoch": 7, "loss": 0.1234}, legacy_ckpt)

            rclone_conf_alias = os.path.join(td, "rclone_alias.conf")
            drive_norm = drive_root.replace("\\", "/")
            Path(rclone_conf_alias).write_text(
                f"[gdrive]\ntype = alias\nremote = {drive_norm}\n",
                encoding="utf-8",
            )

            ws_dir = os.path.join(td, "ws")
            os.makedirs(ws_dir, exist_ok=True)
            Path(ws_dir, "train.py").write_text("print('ok')\n", encoding="utf-8")
            locks_file = os.path.join(td, "locks.json")
            colab_calls = []

            def fake_colab_only(cmd, timeout=300, stdin_text=None):
                prog = os.path.basename(cmd[0]).lower()
                if "rclone" in prog:
                    return subprocess.run(
                        cmd,
                        input=stdin_text,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                    )
                colab_calls.append(list(cmd))
                if cmd[1] == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                if cmd[1] == "exec":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout="CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n", stderr=""
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="OK", stderr="")

            # 1) Default run without confirmation MUST stop before allocating VM and prompt about legacy migration
            out_unconfirmed = io.StringIO()
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf_alias),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_colab_only),
                redirect_stdout(out_unconfirmed),
            ):
                rc1 = runner_mod.main(["--session", "legacy-run", "--workspace", ws_dir])

            self.assertEqual(rc1, 1, "Must block silent from-scratch start when legacy root checkpoint exists")
            self.assertEqual(colab_calls, [], "Must abort BEFORE calling colab new")
            self.assertIn("gdrive:colab_relay", out_unconfirmed.getvalue())
            self.assertIn("gdrive:colab_relay/legacy-run", out_unconfirmed.getvalue())
            self.assertIn("--migrate-legacy-gdrive", out_unconfirmed.getvalue())
            self.assertTrue(os.path.isfile(legacy_ckpt), "Legacy checkpoint must remain untouched before confirmation")

            # 2) Passing --migrate-legacy-gdrive migrates checkpoint.pt via real rclone into gdrive:colab_relay/legacy-run
            out_migrated = io.StringIO()
            with (
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf_alias),
                mock.patch.object(runner_mod, "register_workspace"),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_colab_only),
                redirect_stdout(out_migrated),
            ):
                rc2 = runner_mod.main(["--session", "legacy-run", "--workspace", ws_dir, "--migrate-legacy-gdrive"])

            self.assertEqual(rc2, 0, f"Migrated run failed: {out_migrated.getvalue()}")
            migrated_ckpt = os.path.join(drive_root, "colab_relay", "legacy-run", "checkpoint.pt")
            self.assertTrue(os.path.isfile(migrated_ckpt), "Checkpoint must be migrated to isolated subdirectory")
            self.assertFalse(os.path.exists(legacy_ckpt), "Legacy root checkpoint moved cleanly")

            # Verify train.py pulls the Epoch 7 checkpoint from gdrive:colab_relay/legacy-run via real rclone
            pulled_local = os.path.join(td, "pulled", "checkpoint.pt")
            with (
                mock.patch.object(train_mod, "SMOKE_TEST", False),
                mock.patch.object(train_mod, "GDRIVE_DIR", "gdrive:colab_relay/legacy-run"),
                mock.patch.dict(
                    os.environ,
                    {
                        "SMOKE_TEST": "0",
                        "SMOKE_SKIP_RCLONE": "0",
                        "RELAY_ROUND": "1",
                        "RCLONE_CONFIG": rclone_conf_alias,
                        "RELAY_GDRIVE_DIR": "gdrive:colab_relay/legacy-run",
                        "PATH": os.path.dirname(rclone_exe) + os.pathsep + os.environ.get("PATH", ""),
                    },
                    clear=False,
                ),
            ):
                ok_pull = train_mod.rclone_pull_checkpoint(pulled_local)
            self.assertTrue(ok_pull)
            loaded = train_mod.load_checkpoint_safely(pulled_local)
            self.assertEqual(loaded["epoch"], 7)

    def test_issue_81_round2_empty_remote_directory_aborts_with_real_rclone(self):
        rclone_exe = runner_mod.find_bin("rclone")
        if not os.path.exists(rclone_exe) and not shutil.which(rclone_exe):
            self.skipTest("rclone binary not available")

        with tempfile.TemporaryDirectory() as td:
            drive_root = os.path.join(td, "drive_storage")
            empty_job_dir = os.path.join(drive_root, "colab_relay", "wiped-round2")
            os.makedirs(empty_job_dir, exist_ok=True)

            rclone_conf_alias = os.path.join(td, "rclone_alias.conf")
            drive_norm = drive_root.replace("\\", "/")
            Path(rclone_conf_alias).write_text(
                f"[gdrive]\ntype = alias\nremote = {drive_norm}\n",
                encoding="utf-8",
            )

            run_cwd = os.path.join(td, "run_cwd")
            os.makedirs(run_cwd, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "SMOKE_TEST": "0",
                    "RELAY_ROUND": "2",
                    "RESET_DONE_FLAG": "0",
                    "GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                    "RELAY_GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                    "RCLONE_CONFIG": rclone_conf_alias,
                    "PATH": os.path.dirname(rclone_exe) + os.pathsep + env.get("PATH", ""),
                }
            )
            res = subprocess.run(
                [sys.executable, "-B", str(REPO_ROOT / "train.py")],
                cwd=run_cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            self.assertNotEqual(res.returncode, 0, "train.py must abort when RELAY_ROUND=2 finds empty cloud directory")
            combined = (res.stdout or "") + "\n" + (res.stderr or "")
            self.assertIn("checkpoint.pt", combined)
            self.assertFalse(os.path.exists(os.path.join(run_cwd, "checkpoint.pt")), "Must not train from epoch 0")

    def test_issue_86_detached_relay_resolves_relative_workspace_and_reports_invalid_workspace(self):
        rclone_exe = runner_mod.find_bin("rclone")
        if not os.path.exists(rclone_exe) and not shutil.which(rclone_exe):
            self.skipTest("rclone binary not available")

        with tempfile.TemporaryDirectory() as td:
            parent_dir = os.path.join(td, "parent")
            proj_dir = os.path.join(parent_dir, "project")
            os.makedirs(proj_dir, exist_ok=True)
            Path(proj_dir, "train.py").write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")
            Path(proj_dir, "checkpoint_final.pt").write_bytes(b"READY_WEIGHTS")

            drive_root = os.path.join(td, "drive_storage")
            job_drive = os.path.join(drive_root, "colab_relay", "det-rel")
            os.makedirs(job_drive, exist_ok=True)
            Path(job_drive, "training_done.flag").write_text("done\n", encoding="utf-8")

            rclone_conf_alias = os.path.join(td, "rclone_alias.conf")
            drive_norm = drive_root.replace("\\", "/")
            Path(rclone_conf_alias).write_text(
                f"[gdrive]\ntype = alias\nremote = {drive_norm}\n",
                encoding="utf-8",
            )

            home_dir = os.path.join(td, "fake_home")
            os.makedirs(home_dir, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "RCLONE_CONFIG": rclone_conf_alias,
                    "USERPROFILE": home_dir,
                    "HOME": home_dir,
                    "PATH": os.path.dirname(rclone_exe) + os.pathsep + env.get("PATH", ""),
                }
            )
            runner_script = str(REPO_ROOT / ".agents" / "skills" / "colab-dl-relay" / "scripts" / "dl_relay_runner.py")

            res = subprocess.run(
                [sys.executable, runner_script, "--session", "det-rel", "--workspace", "project", "--detach"],
                cwd=parent_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            self.assertEqual(res.returncode, 0, f"Detach parent failed: {res.stdout}\n{res.stderr}")
            self.assertIn("已在后台启动", res.stdout)

            # Wait up to 5s for the detached child to write `.colab_dl_relay_status.json` inside `parent/project/`
            receipt_file = os.path.join(proj_dir, ".colab_dl_relay_status.json")
            deadline = time.time() + 5.0
            while time.time() < deadline and not os.path.exists(receipt_file):
                time.sleep(0.1)

            log_path = os.path.join(proj_dir, ".colab_dl_relay.log")
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace") if os.path.exists(log_path) else ""
            self.assertNotIn("找不到训练脚本", log_text, f"Detached child resolved wrong path:\n{log_text}")
            self.assertNotIn(os.path.join("project", "project"), log_text)
            self.assertTrue(os.path.isfile(receipt_file), f"Detached child did not complete; log:\n{log_text}")
            receipt = json.loads(Path(receipt_file).read_text(encoding="utf-8"))
            self.assertTrue(receipt["training_completed"])
            self.assertTrue(receipt["weights_downloaded"])

            # Verify detached child registered relative `train.py` (not absolute path) in workspaces.json
            ws_reg_path = os.path.join(home_dir, ".config", "colab-cli", "workspaces.json")
            if os.path.exists(ws_reg_path):
                ws_reg = json.loads(Path(ws_reg_path).read_text(encoding="utf-8"))
                self.assertEqual(ws_reg.get("det-rel", {}).get("active_script"), "train.py")

            # Verify pre-detach GDrive lock conflict fails fast in parent without overwriting existing .colab_dl_relay.pid
            locks_path = os.path.join(home_dir, ".config", "colab-cli", "dl_relay_locks.json")
            os.makedirs(os.path.dirname(locks_path), exist_ok=True)
            Path(locks_path).write_text(
                json.dumps(
                    {
                        "gdrive:colab_relay/shared-lock": {
                            "session": "sess-live",
                            "pid": os.getpid(),
                            "workspace": proj_dir,
                            "updated_at": time.time(),
                        }
                    }
                ),
                encoding="utf-8",
            )
            pid_file = os.path.join(proj_dir, ".colab_dl_relay.pid")
            Path(pid_file).write_text("999999", encoding="utf-8")
            res_locked = subprocess.run(
                [
                    sys.executable,
                    runner_script,
                    "--session",
                    "sess-intruder",
                    "--workspace",
                    "project",
                    "--gdrive-dir",
                    "gdrive:colab_relay/shared-lock",
                    "--detach",
                ],
                cwd=parent_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            self.assertNotEqual(res_locked.returncode, 0, "Conflicting GDrive lock must fail in --detach parent")
            self.assertIn("sess-live", res_locked.stdout + res_locked.stderr)
            self.assertEqual(Path(pid_file).read_text(encoding="utf-8"), "999999", "Must not overwrite .colab_dl_relay.pid")

    def test_instance_keep_alive_profile_registered_by_cli_scripts(self):
        runner_mod = load_module(
            "dl_relay_runner_ka_test",
            REPO_ROOT / ".agents" / "skills" / "colab-dl-relay" / "scripts" / "dl_relay_runner.py",
        )
        mgr_mod = load_module(
            "llm_bluegreen_manager_ka_test",
            REPO_ROOT / ".agents" / "skills" / "colab-llm-bluegreen" / "scripts" / "llm_bluegreen_manager.py",
        )
        with tempfile.TemporaryDirectory() as td:
            cfg_dir = os.path.join(td, "cfg")
            ws_dir = os.path.join(td, "ws")
            os.makedirs(cfg_dir, exist_ok=True)
            os.makedirs(ws_dir, exist_ok=True)
            with (
                mock.patch.object(runner_mod, "colab_cfg_dir", return_value=cfg_dir),
                mock.patch.object(mgr_mod, "colab_cfg_dir", return_value=cfg_dir),
            ):
                runner_mod.register_workspace("dl-vm-1", ws_dir, "train.py", "DL training")
                mgr_mod.register_workspace("llm-vm-1", ws_dir, "LLM deploy")
                prof_file = os.path.join(cfg_dir, "pilot_session_profiles.json")
                data = json.loads(Path(prof_file).read_text(encoding="utf-8"))
                self.assertEqual(
                    data["dl-vm-1"]["keep_alive"],
                    {"enabled": True, "interval_seconds": 90, "kernel_ping": True},
                )
                self.assertEqual(
                    data["llm-vm-1"]["keep_alive"],
                    {"enabled": True, "interval_seconds": 90, "kernel_ping": True},
                )
                # User customization must NOT be overwritten by subsequent register_workspace calls
                data["dl-vm-1"]["keep_alive"]["interval_seconds"] = 300
                data["dl-vm-1"]["keep_alive"]["enabled"] = False
                Path(prof_file).write_text(json.dumps(data), encoding="utf-8")
                runner_mod.register_workspace("dl-vm-1", ws_dir, "train.py", "DL training")
                after = json.loads(Path(prof_file).read_text(encoding="utf-8"))
                self.assertEqual(after["dl-vm-1"]["keep_alive"]["interval_seconds"], 300)
                self.assertEqual(after["dl-vm-1"]["keep_alive"]["enabled"], False)

    def test_dl_relay_grab_modes_fallback_gpus_and_env_overrides(self):
        runner_mod = load_module(
            "dl_relay_runner_grab_test",
            REPO_ROOT / ".agents" / "skills" / "colab-dl-relay" / "scripts" / "dl_relay_runner.py",
        )
        # 1. parse_fallback_gpus deduplicates and rejects invalid models
        self.assertEqual(
            runner_mod.parse_fallback_gpus("A100", "g4, L4, a100, G4"),
            ["A100", "G4", "L4"],
        )
        with self.assertRaises(ValueError):
            runner_mod.parse_fallback_gpus("T4", "G4,RTX4090")

        # 2. build_detached_child_cmd forwards grab policy flags
        parsed = runner_mod.build_parser().parse_args(
            [
                "--session",
                "grab-exp",
                "--workspace",
                ".",
                "--grab-mode",
                "budget",
                "--grab-timeout-minutes",
                "90",
                "--grab-retry-seconds",
                "12",
                "--fallback-gpus",
                "G4,L4",
            ]
        )
        child_cmd = runner_mod.build_detached_child_cmd(
            parsed, "/abs/ws", "/abs/ws/train.py", "gdrive:colab_relay/grab-exp"
        )
        self.assertIn("--grab-mode", child_cmd)
        self.assertIn("budget", child_cmd)
        self.assertIn("90.0", child_cmd)
        self.assertIn("--grab-retry-seconds", child_cmd)
        self.assertIn("12.0", child_cmd)
        self.assertIn("--fallback-gpus", child_cmd)
        self.assertIn("G4,L4", child_cmd)

        # 3. run_cmd injects COLAB_CLI_ASSIGN_RETRIES=0 and GPU(60s)/TPU(300s) COLAB_CLI_REQUEST_TIMEOUT
        captured_envs = []

        def fake_subproc_run(cmd, **kwargs):
            captured_envs.append((list(cmd), dict(kwargs.get("env") or {})))
            return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")

        with mock.patch.object(runner_mod.subprocess, "run", side_effect=fake_subproc_run):
            runner_mod.run_cmd(["colab", "new", "-s", "vm-gpu", "--gpu", "A100"])
            runner_mod.run_cmd(["colab", "new", "-s", "vm-tpu", "--tpu", "v6e1"])

        self.assertEqual(captured_envs[0][1].get("COLAB_CLI_ASSIGN_RETRIES"), "0")
        self.assertEqual(captured_envs[0][1].get("COLAB_CLI_REQUEST_TIMEOUT"), "60")
        self.assertEqual(captured_envs[1][1].get("COLAB_CLI_ASSIGN_RETRIES"), "0")
        self.assertEqual(captured_envs[1][1].get("COLAB_CLI_REQUEST_TIMEOUT"), "300")

        # 4. --grab-mode fast + --fallback-gpus G4 rotates immediately on A100 503 and completes on G4
        with tempfile.TemporaryDirectory() as td:
            ws_dir = os.path.join(td, "ws")
            cfg_dir = os.path.join(td, "cfg")
            os.makedirs(ws_dir, exist_ok=True)
            os.makedirs(cfg_dir, exist_ok=True)
            Path(os.path.join(ws_dir, "train.py")).write_text("print('ok')\n", encoding="utf-8")
            rclone_conf = os.path.join(td, "rclone.conf")
            Path(rclone_conf).write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            locks_file = os.path.join(td, "locks.json")
            tried_gpus = []

            def fake_run(cmd, timeout=300, stdin_text=None):
                prog = os.path.basename(cmd[0]).lower()
                if "rclone" in prog:
                    if "copyto" in cmd:
                        Path(cmd[-1]).write_bytes(b"weights-ok")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                sub = cmd[1]
                if sub == "new":
                    gpu_idx = cmd.index("--gpu") + 1
                    gpu_name = cmd[gpu_idx]
                    tried_gpus.append(gpu_name)
                    if gpu_name == "A100":
                        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 503 Service Unavailable")
                    return subprocess.CompletedProcess(cmd, 0, stdout="Session READY", stderr="")
                if sub == "status":
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No active session")
                if sub == "sessions":
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if sub == "upload":
                    return subprocess.CompletedProcess(cmd, 0, stdout="Uploaded", stderr="")
                if sub == "exec":
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout="CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n",
                        stderr="",
                    )
                if sub == "stop":
                    return subprocess.CompletedProcess(cmd, 0, stdout="Stopped", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(runner_mod, "colab_cfg_dir", return_value=cfg_dir),
                mock.patch.object(runner_mod, "relay_locks_path", return_value=locks_file),
                mock.patch.object(runner_mod, "find_rclone_conf", return_value=rclone_conf),
                mock.patch.object(runner_mod, "run_cmd", side_effect=fake_run),
                redirect_stdout(io.StringIO()),
            ):
                rc = runner_mod.main(
                    [
                        "--session",
                        "exp-fallback",
                        "--workspace",
                        ws_dir,
                        "--gpu",
                        "A100",
                        "--grab-mode",
                        "fast",
                        "--fallback-gpus",
                        "G4",
                        "--reset-done-flag",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(tried_gpus, ["A100", "G4"])


if __name__ == "__main__":
    unittest.main()
