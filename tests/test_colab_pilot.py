import base64
import ctypes
import glob
import http.client
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
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


if __name__ == "__main__":
    unittest.main()
