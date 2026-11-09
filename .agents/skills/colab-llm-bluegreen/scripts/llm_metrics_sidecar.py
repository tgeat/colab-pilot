#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 大语言模型统一 OpenAI 计量边车与状态守护进程 (llm_metrics_sidecar.py)
================================================================================
适配任意底层大模型推理引擎（vLLM / Ollama / llama.cpp / TGI / SGLang / Transformers）：
- 底层推理引擎监听内部端口（默认 `127.0.0.1:8001`）
- 本计量边车监听对外端口（默认 `0.0.0.0:8000`），透明代理 `/v1/chat/completions`、`/v1/completions`、`/v1/models`
- 在转发 SSE 流式响应 (`data: {...}`) 时，逐 Chunk 精确统计：
  * 首字延迟 `ttft_ms` (Time To First Token)
  * 瞬时 Token 输出频率 `instant_tps` (tok/s)
  * 60 秒滑动平均速率 `avg_tps_60s` (tok/s) 与峰值 `peak_tps`
  * 最近 24 个采样点的频率波形 `tps_history`
  * 当前并发流数 `active_streams`、累计请求数与累计生成 Token 总数
  * 蓝绿部署槽位 (`blue` / `green`) 与角色状态 (`active` / `warming` / `draining` / `standby`)
- 每 1.5 秒原子更新 `/content/.colab_pilot_llm_state.json`，由 Colab Pilot 仪表盘实时展示。
"""

import argparse
import collections
import http.client
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _default_llm_state_file() -> str:
    if os.path.isdir("/content"):
        return "/content/.colab_pilot_llm_state.json"
    return os.path.abspath(".colab_pilot_llm_state.json")


class LlmTelemetryTracker:
    """线程安全的 LLM Token 输出频率与蓝绿在线状态统计器"""

    def __init__(
        self,
        slot: str = "blue",
        role: str = "active",
        engine: str = "vllm",
        model_name: str = "qwen2.5-7b-instruct",
        gdrive_model_path: str = "",
        listen_port: int = 8000,
        backend_url: str = "http://127.0.0.1:8001",
        builtin_engine: bool = False,
        max_session_seconds: float = 21.5 * 3600,
        state_file: Optional[str] = None,
    ):
        self.lock = threading.Lock()
        self.slot = slot.lower()
        self.role = role.lower()
        self.engine = engine
        self.model_name = model_name
        self.gdrive_model_path = gdrive_model_path
        self.listen_port = listen_port
        self.backend_url = backend_url.rstrip("/")
        self.builtin_engine = builtin_engine
        self.max_session_seconds = max_session_seconds
        self.state_file = state_file or os.environ.get("LLM_STATE_FILE", _default_llm_state_file())

        self.start_time = time.time()
        self.online = True
        self.backend_healthy = True if builtin_engine else False
        self.active_streams = 0
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.error_count = 0

        self.instant_tps: float = 0.0
        self.last_request_tps: float = 0.0
        self.peak_tps: float = 0.0
        self.last_ttft_ms: Optional[float] = None
        self.avg_ttft_ms: Optional[float] = None
        self.last_request_at: Optional[float] = None

        # 记录过去 60 秒内的 (timestamp, token_count) 事件用于计算滑动 TPS
        self._token_events: Deque[Tuple[float, int]] = collections.deque()
        self._ttft_samples: Deque[float] = collections.deque(maxlen=30)
        self.tps_history: Deque[float] = collections.deque([0.0] * 20, maxlen=24)

    def check_backend_health(self) -> bool:
        if self.builtin_engine:
            self.backend_healthy = True
            return True
        parsed = urlparse(self.backend_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=1.5):
                self.backend_healthy = True
                return True
        except Exception:
            self.backend_healthy = False
            return False

    def on_request_start(self, prompt_tokens_est: int = 0) -> int:
        with self.lock:
            self.active_streams += 1
            self.total_requests += 1
            self.total_prompt_tokens += max(0, prompt_tokens_est)
            self.last_request_at = time.time()
            return self.total_requests

    def on_first_token(self, ttft_ms: float) -> None:
        with self.lock:
            self.last_ttft_ms = round(ttft_ms, 1)
            self._ttft_samples.append(ttft_ms)
            self.avg_ttft_ms = round(sum(self._ttft_samples) / len(self._ttft_samples), 1)

    def on_tokens_emitted(self, count: int, stream_elapsed_sec: float, stream_total_tokens: int) -> None:
        now = time.time()
        with self.lock:
            self.total_completion_tokens += count
            self._token_events.append((now, count))
            if stream_elapsed_sec > 0.02:
                cur_tps = round(stream_total_tokens / stream_elapsed_sec, 2)
                self.instant_tps = cur_tps
                self.last_request_tps = cur_tps
                if cur_tps > self.peak_tps:
                    self.peak_tps = cur_tps

    def on_request_end(self, ok: bool = True) -> None:
        with self.lock:
            self.active_streams = max(0, self.active_streams - 1)
            if not ok:
                self.error_count += 1
        self.write_state_file()

    def _compute_rolling_tps_locked(self, now: float) -> float:
        cutoff_60 = now - 60.0
        while self._token_events and self._token_events[0][0] < cutoff_60:
            self._token_events.popleft()
        if not self._token_events:
            return 0.0
        total_60 = sum(c for _, c in self._token_events)
        span = max(1.0, min(60.0, now - self._token_events[0][0] + 0.5))
        return round(total_60 / span, 2)

    def _compute_instant_tps_locked(self, now: float) -> float:
        cutoff_3 = now - 3.0
        recent = [c for ts, c in self._token_events if ts >= cutoff_3]
        if recent:
            span = max(0.5, now - next(ts for ts, _ in self._token_events if ts >= cutoff_3))
            val = round(sum(recent) / span, 2)
            self.instant_tps = val
            if val > self.peak_tps:
                self.peak_tps = val
            return val
        if self.active_streams == 0:
            # 无活跃流时保留最近一次请求的解码速率在 last_request_tps，但实时表回落
            self.instant_tps = round(self.last_request_tps, 2) if (
                self.last_request_at and (now - self.last_request_at) < 8.0
            ) else 0.0
        return self.instant_tps

    def snapshot(self, record_history: bool = False) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            inst_tps = self._compute_instant_tps_locked(now)
            avg_60 = self._compute_rolling_tps_locked(now)
            if record_history:
                sample_val = inst_tps if inst_tps > 0 else (self.last_request_tps if self.last_request_at and (now - self.last_request_at) < 4.0 else 0.0)
                self.tps_history.append(round(sample_val, 1))
            elapsed = round(now - self.start_time, 1)
            handover_in = max(0.0, round(self.max_session_seconds - elapsed, 1))
            status_str = "online" if self.backend_healthy else "starting_backend"
            if self.role == "warming":
                status_str = "warming_up"
            elif self.role == "draining":
                status_str = "draining"

            return {
                "mode": "llm_bluegreen",
                "status": status_str,
                "online": self.online and self.backend_healthy,
                "backend_healthy": self.backend_healthy,
                "slot": self.slot,
                "role": self.role,
                "engine": self.engine,
                "model_name": self.model_name,
                "gdrive_model_path": self.gdrive_model_path,
                "port": self.listen_port,
                "backend_url": self.backend_url,
                "instant_tps": inst_tps,
                "last_request_tps": self.last_request_tps,
                "avg_tps_60s": avg_60,
                "peak_tps": self.peak_tps,
                "ttft_ms": self.last_ttft_ms,
                "avg_ttft_ms": self.avg_ttft_ms,
                "active_streams": self.active_streams,
                "total_requests": self.total_requests,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "error_count": self.error_count,
                "tps_history": list(self.tps_history),
                "last_request_at": self.last_request_at,
                "elapsed_seconds": elapsed,
                "handover_in_seconds": handover_in,
                "timestamp": now,
            }

    def write_state_file(self, record_history: bool = False) -> Dict[str, Any]:
        self.check_backend_health()
        snap = self.snapshot(record_history=record_history)
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_file)
        except Exception:
            pass
        return snap


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # 兼顾中英文 Token 估算：每个 CJK 字符约 1 token，每 4 个 ASCII 字符约 1 token
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk = len(text) - cjk
    return max(1, cjk + max(1, non_cjk // 4) if non_cjk > 0 else cjk)


def make_handler(tracker: LlmTelemetryTracker):
    class SidecarHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            self.close_connection = True

        def do_GET(self) -> None:
            if self.path.startswith("/pilot/health"):
                snap = tracker.write_state_file(record_history=False)
                self._send_json(
                    200 if snap["backend_healthy"] else 503,
                    {
                        "ok": snap["backend_healthy"],
                        "slot": snap["slot"],
                        "role": snap["role"],
                        "engine": snap["engine"],
                        "model_name": snap["model_name"],
                        "instant_tps": snap["instant_tps"],
                        "active_streams": snap["active_streams"],
                    },
                )
                return

            if self.path.startswith("/pilot/metrics"):
                snap = tracker.write_state_file(record_history=False)
                self._send_json(200, snap)
                return

            if self.path.startswith("/v1/models") and tracker.builtin_engine:
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": tracker.model_name,
                                "object": "model",
                                "owned_by": f"colab-pilot-{tracker.slot}",
                            }
                        ],
                    },
                )
                return

            self._proxy_request("GET")

        def do_POST(self) -> None:
            if self.path.startswith("/pilot/role"):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                with tracker.lock:
                    if "role" in body:
                        tracker.role = str(body["role"]).lower()
                    if "slot" in body:
                        tracker.slot = str(body["slot"]).lower()
                snap = tracker.write_state_file(record_history=False)
                self._send_json(200, {"ok": True, "slot": snap["slot"], "role": snap["role"]})
                return

            if tracker.builtin_engine and (
                self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions")
            ):
                self._handle_builtin_completion()
                return

            self._proxy_request("POST")

        def _handle_builtin_completion(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
            try:
                req_json = json.loads(raw_body)
            except Exception:
                req_json = {}

            is_stream = bool(req_json.get("stream", False))
            max_tokens = int(req_json.get("max_tokens", 48))
            prompt_est = max(8, len(raw_body) // 6)

            req_start = time.time()
            tracker.on_request_start(prompt_tokens_est=prompt_est)

            tokens_to_emit = [
                f"[{tracker.slot.upper()}/{tracker.engine}] ",
                "Colab ",
                "Pilot ",
                "LLM ",
                "蓝绿热切换与 ",
                "Token ",
                "频率监督边车 ",
                "运行正常。",
                "当前槽位=",
                f"{tracker.slot}，",
                "模型=",
                f"{tracker.model_name}，",
                "支持零中断 ",
                "Make-Before-Break ",
                "平滑切流！",
            ]
            while len(tokens_to_emit) < min(max_tokens, 36):
                tokens_to_emit.append("⚡")

            try:
                if is_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    first_token_ts: Optional[float] = None
                    emitted = 0
                    for idx, piece in enumerate(tokens_to_emit):
                        time.sleep(0.022)  # ~45 tok/s
                        now = time.time()
                        if first_token_ts is None:
                            first_token_ts = now
                            tracker.on_first_token((now - req_start) * 1000.0)
                        emitted += 1
                        tracker.on_tokens_emitted(1, max(0.02, now - first_token_ts), emitted)

                        chunk = {
                            "id": f"chatcmpl-{int(req_start * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(now),
                            "model": tracker.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None if idx < len(tokens_to_emit) - 1 else "stop",
                                }
                            ],
                        }
                        line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(line)
                        self.wfile.flush()

                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    tracker.on_request_end(ok=True)
                else:
                    time.sleep(0.04)
                    tracker.on_first_token((time.time() - req_start) * 1000.0)
                    full_text = "".join(tokens_to_emit)
                    time.sleep(0.25)
                    tracker.on_tokens_emitted(len(tokens_to_emit), 0.25, len(tokens_to_emit))
                    tracker.on_request_end(ok=True)
                    self._send_json(
                        200,
                        {
                            "id": f"chatcmpl-{int(req_start * 1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": tracker.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": full_text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_est,
                                "completion_tokens": len(tokens_to_emit),
                                "total_tokens": prompt_est + len(tokens_to_emit),
                            },
                        },
                    )
            except Exception:
                tracker.on_request_end(ok=False)

        def _proxy_request(self, method: str) -> None:
            parsed = urlparse(tracker.backend_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8001

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else None

            is_completion = "/completions" in self.path or "/api/generate" in self.path or "/api/chat" in self.path
            req_start = time.time()
            if is_completion:
                tracker.on_request_start(prompt_tokens_est=max(4, length // 6))

            try:
                conn = http.client.HTTPConnection(host, port, timeout=600)
                fwd_headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "connection", "transfer-encoding")
                }
                conn.request(method, self.path, body=body, headers=fwd_headers)
                resp = conn.getresponse()

                ctype = resp.getheader("Content-Type", "")
                is_sse = "text/event-stream" in ctype.lower()

                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                first_token_ts: Optional[float] = None
                stream_tokens = 0

                if is_sse:
                    while True:
                        line_bytes = resp.fp.readline()
                        if not line_bytes:
                            break
                        self.wfile.write(line_bytes)
                        self.wfile.flush()

                        line_str = line_bytes.decode("utf-8", errors="ignore").strip()
                        if line_str.startswith("data:") and is_completion:
                            data_part = line_str[5:].strip()
                            if data_part and data_part != "[DONE]":
                                now = time.time()
                                if first_token_ts is None:
                                    first_token_ts = now
                                    tracker.on_first_token((now - req_start) * 1000.0)
                                tok_inc = 1
                                try:
                                    j = json.loads(data_part)
                                    choices = j.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        txt = (
                                            delta.get("content")
                                            or delta.get("reasoning_content")
                                            or choices[0].get("text")
                                            or ""
                                        )
                                        if txt:
                                            tok_inc = _estimate_tokens_from_text(txt)
                                except Exception:
                                    pass
                                stream_tokens += tok_inc
                                tracker.on_tokens_emitted(tok_inc, max(0.02, now - first_token_ts), stream_tokens)
                else:
                    raw_resp = resp.read()
                    if is_completion and raw_resp:
                        now = time.time()
                        tracker.on_first_token((now - req_start) * 1000.0)
                        tok_cnt = 16
                        try:
                            j = json.loads(raw_resp.decode("utf-8", errors="ignore"))
                            usage = j.get("usage") or {}
                            if "completion_tokens" in usage:
                                tok_cnt = int(usage["completion_tokens"])
                        except Exception:
                            pass
                        tracker.on_tokens_emitted(tok_cnt, max(0.05, now - req_start), tok_cnt)
                    self.wfile.write(raw_resp)
                    self.wfile.flush()

                conn.close()
                if is_completion:
                    tracker.on_request_end(ok=(200 <= resp.status < 400))
            except Exception as e:
                if is_completion:
                    tracker.on_request_end(ok=False)
                try:
                    self._send_json(502, {"error": f"Backend engine unreachable: {e}"})
                except Exception:
                    pass

    return SidecarHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot LLM 统一计量边车与状态上报服务")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIDECAR_PORT", "8000")))
    parser.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--slot", default=os.environ.get("LLM_SLOT", "blue"), help="蓝绿槽位: blue 或 green")
    parser.add_argument("--role", default=os.environ.get("LLM_ROLE", "active"), help="active / warming / draining / standby")
    parser.add_argument("--engine", default=os.environ.get("LLM_ENGINE", "vllm"), help="vllm / ollama / llama.cpp / transformers")
    parser.add_argument("--model-name", default=os.environ.get("LLM_MODEL_NAME", "qwen2.5-7b-instruct"))
    parser.add_argument("--gdrive-model-path", default=os.environ.get("GDRIVE_MODEL_PATH", ""))
    parser.add_argument("--builtin-engine", action="store_true", help="使用内置轻量流式生成引擎（用于本机 Smoke Test 或快速验证）")
    parser.add_argument("--state-file", default=os.environ.get("LLM_STATE_FILE", _default_llm_state_file()))
    args = parser.parse_args()

    tracker = LlmTelemetryTracker(
        slot=args.slot,
        role=args.role,
        engine=args.engine,
        model_name=args.model_name,
        gdrive_model_path=args.gdrive_model_path,
        listen_port=args.port,
        backend_url=args.backend_url,
        builtin_engine=args.builtin_engine,
        state_file=args.state_file,
    )

    def _heartbeat_loop() -> None:
        while True:
            try:
                tracker.write_state_file(record_history=True)
            except Exception:
                pass
            time.sleep(1.5)

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(tracker))
    print(
        f"🚀 [LLM Sidecar] 已启动: 0.0.0.0:{args.port} -> {args.backend_url} "
        f"(slot={args.slot}, role={args.role}, engine={args.engine}, model={args.model_name})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
