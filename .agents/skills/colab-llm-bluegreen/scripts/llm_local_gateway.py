#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 本机蓝绿零中断 OpenAI 网关守护进程 (llm_local_gateway.py)
====================================================================================
运行于本机 (默认 http://127.0.0.1:18080，同时监听 18081/blue 与 18082/green)，负责：
1. 自动读取 `~/.config/colab-cli/llm_bluegreen_state.json` 与 `sessions.json`
2. 采用纯 Python 进程内直连桥接（Contents API + ColabRuntime WebSocket），
   不拉起任何外部 `ssh.exe` / `colab.exe` 子进程，彻底杜绝 Windows 控制台窗口弹窗闪烁。
3. 对本机所有 AI 客户端与 Colab Pilot 面板暴露统一固定的本地端点：`http://127.0.0.1:18080/v1`
"""

import argparse
import base64
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional, Tuple

SLOT_PORTS = {
    "blue": 18081,
    "green": 18082,
}


def cfg_dir() -> str:
    d = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(d, exist_ok=True)
    return d


def state_path() -> str:
    return os.path.join(cfg_dir(), "llm_bluegreen_state.json")


def sessions_path() -> str:
    return os.path.join(cfg_dir(), "sessions.json")


def gateway_pid_path() -> str:
    return os.path.join(cfg_dir(), "llm_local_gateway.pid")


def read_state() -> Dict[str, Any]:
    p = state_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def read_sessions() -> Dict[str, Any]:
    p = sessions_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def resolve_slot_session(forced_slot: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    st = read_state()
    prefix = st.get("session_prefix") or "llm"
    slot = (forced_slot or st.get("active_slot") or "blue").lower()
    sess_name = f"{prefix}-{slot}"
    sessions = read_sessions()
    sess = sessions.get(sess_name)
    if not sess and not forced_slot and sessions:
        sess = next(iter(sessions.values()))
    return slot, st, sess


def fetch_remote_llm_state(sess: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sess:
        return None
    url = (sess.get("url") or "").rstrip("/")
    token = sess.get("token") or ""
    if not url or not token:
        return None
    req_url = f"{url}/api/contents/content/.colab_pilot_llm_state.json?content=1"
    req = urllib.request.Request(
        req_url,
        headers={
            "X-Colab-Runtime-Proxy-Token": token,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            content = data.get("content")
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, dict):
                return content
    except Exception:
        pass
    return None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def make_gateway_handler(forced_slot: Optional[str] = None):
    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            slot, st, sess = resolve_slot_session(forced_slot)
            target_port = SLOT_PORTS.get(slot, 18081)

            if self.path.startswith("/pilot/gateway"):
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "active_slot": slot,
                        "target_port": target_port,
                        "tunnel_open": sess is not None,
                        "model_name": st.get("model_name"),
                        "engine": st.get("engine"),
                        "gpu": st.get("gpu"),
                    },
                )
                return

            if self.path.startswith("/pilot/health"):
                remote_st = fetch_remote_llm_state(sess)
                healthy = bool(remote_st and remote_st.get("backend_healthy", True)) if remote_st else (sess is not None)
                self._send_json(
                    200 if healthy else 503,
                    {
                        "ok": healthy,
                        "slot": (remote_st or {}).get("slot", slot),
                        "role": (remote_st or {}).get("role", "active"),
                        "engine": (remote_st or {}).get("engine", st.get("engine", "llama.cpp")),
                        "model_name": (remote_st or {}).get("model_name", st.get("model_name", "")),
                        "instant_tps": (remote_st or {}).get("instant_tps", 0.0),
                        "active_streams": (remote_st or {}).get("active_streams", 0),
                    },
                )
                return

            if self.path.startswith("/pilot/metrics"):
                remote_st = fetch_remote_llm_state(sess) or {
                    "mode": "llm_bluegreen",
                    "status": "online" if sess else "offline",
                    "online": sess is not None,
                    "backend_healthy": sess is not None,
                    "slot": slot,
                    "role": "active",
                    "engine": st.get("engine", "llama.cpp"),
                    "model_name": st.get("model_name", ""),
                    "instant_tps": st.get("warmup_tps", 0.0),
                }
                self._send_json(200, remote_st)
                return

            if self.path.startswith("/v1/models"):
                model_id = st.get("model_name") or "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": int(st.get("updated_at") or time.time()),
                                "owned_by": f"colab-pilot-{slot}",
                            }
                        ],
                    },
                )
                return

            self._send_json(404, {"error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:
            slot, st, sess = resolve_slot_session(forced_slot)
            if not sess:
                self._send_json(502, {"error": {"message": f"No active Colab session found for slot [{slot}]"}})
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req_json = json.loads(raw_body.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                req_json = {}

            is_stream = bool(req_json.get("stream", False))
            model_id = st.get("model_name") or req_json.get("model") or "default"

            req_b64 = base64.b64encode(raw_body).decode("ascii")
            remote_code = f"""
import base64, http.client, json, sys
raw_body = base64.b64decode({json.dumps(req_b64)})
conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=300)
conn.request("POST", {json.dumps(self.path)}, body=raw_body, headers={{"Content-Type": "application/json"}})
resp = conn.getresponse()
ctype = (resp.getheader("Content-Type") or "").lower()
if "text/event-stream" in ctype:
    while True:
        line_b = resp.fp.readline()
        if not line_b:
            break
        sys.stdout.write(line_b.decode("utf-8", "ignore"))
        sys.stdout.flush()
else:
    sys.stdout.write(resp.read().decode("utf-8", "ignore"))
    sys.stdout.flush()
conn.close()
"""
            try:
                from colab_cli.runtime import ColabRuntime

                rt = ColabRuntime(
                    sess["url"],
                    sess["token"],
                    kernel_id=sess.get("kernel_id"),
                    session_id=sess.get("session_id"),
                )
                if is_stream:
                    self.close_connection = True
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def on_out(out: Dict[str, Any]) -> None:
                        if out.get("output_type") == "stream":
                            txt = out.get("text", "")
                            if txt:
                                self.wfile.write(txt.encode("utf-8"))
                                self.wfile.flush()

                    try:
                        rt.execute_code(remote_code, output_hook=on_out, timeout=300)
                    finally:
                        try:
                            rt.stop()
                        except Exception:
                            pass
                else:
                    chunks = []

                    def on_out_sync(out: Dict[str, Any]) -> None:
                        if out.get("output_type") == "stream":
                            txt = out.get("text", "")
                            if txt:
                                chunks.append(txt)

                    try:
                        rt.execute_code(remote_code, output_hook=on_out_sync, timeout=300)
                    finally:
                        try:
                            rt.stop()
                        except Exception:
                            pass

                    raw_resp = "".join(chunks)
                    if raw_resp.lstrip().startswith("data:"):
                        pieces = []
                        for ln in raw_resp.splitlines():
                            ln_s = ln.strip()
                            if ln_s.startswith("data:") and ln_s[5:].strip() != "[DONE]":
                                try:
                                    obj = json.loads(ln_s[5:].strip())
                                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                                    if piece:
                                        pieces.append(piece)
                                except Exception:
                                    pass
                        self._send_json(
                            200,
                            {
                                "id": f"chatcmpl-{int(time.time()*1000)}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": "".join(pieces)},
                                        "finish_reason": "stop",
                                    }
                                ],
                            },
                        )
                    else:
                        try:
                            self._send_json(200, json.loads(raw_resp))
                        except Exception:
                            self._send_json(
                                200,
                                {
                                    "id": f"chatcmpl-{int(time.time()*1000)}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {"role": "assistant", "content": raw_resp},
                                            "finish_reason": "stop",
                                        }
                                    ],
                                },
                            )
            except Exception as e:
                self._send_json(502, {"error": {"message": f"Gateway runtime bridge error: {e}"}})

    return GatewayHandler


def _start_auxiliary_slot_server(port: int, slot: str) -> None:
    try:
        srv = ThreadedHTTPServer(("127.0.0.1", port), make_gateway_handler(forced_slot=slot))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
    except OSError:
        pass


def _keepalive_worker() -> None:
    """
    后台保活与代理令牌自动续期线程（每 90 秒执行一次）：
    1. 调用 Colab `/tun/m/assignments` 自动刷新 `sessions.json` 中的 `runtimeProxyInfo.token`（防止 3600s 令牌过期）
    2. 调用 `/tun/m/keep-alive/{endpoint}` (带 `X-Colab-Tunnel: Google` 头) 维持 Google Colab 隧道租约
    3. 通过 `ColabRuntime` 向云端 Jupyter Kernel 发送轻量心跳 (`pass`)，刷新 Kernel `last_activity`，
       防止因 `llama-server` 独立后台运行而导致 Jupyter Kernel 被 Google Colab 判定为空闲 (Idle Timeout) 回收实例。
    """
    while True:
        time.sleep(90.0)
        try:
            sessions = read_sessions()
            if not sessions:
                continue

            # 1. 刷新 assignments 与 runtimeProxyInfo.token，并发送 /tun/m/keep-alive/{endpoint}
            try:
                from colab_cli.common import state as colab_state

                assignments = colab_state.client.list_assignments()
                ep_map = {a.endpoint: a for a in assignments}
                updated = False
                for s_name, s_data in list(sessions.items()):
                    if not isinstance(s_data, dict):
                        continue
                    ep = s_data.get("endpoint")
                    if ep and ep in ep_map:
                        a = ep_map[ep]
                        new_tok = a.runtime_proxy_info.token
                        new_url = a.runtime_proxy_info.url
                        if new_tok and s_data.get("token") != new_tok:
                            s_data["token"] = new_tok
                            s_data["url"] = new_url
                            updated = True
                        try:
                            ka_url = f"{colab_state.client.colab_domain}/tun/m/keep-alive/{ep}"
                            colab_state.client.session.request(
                                "GET",
                                ka_url,
                                params={"authuser": "0"},
                                headers={
                                    "Accept": "application/json",
                                    "X-Colab-Client-Agent": "colab-cli",
                                    "X-Colab-Tunnel": "Google",
                                },
                                timeout=10,
                            )
                        except Exception:
                            pass
                if updated:
                    p = sessions_path()
                    tmp = p + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(sessions, f, ensure_ascii=False, indent=2)
                    os.replace(tmp, p)
            except Exception:
                pass

            # 2. 对当前活跃的 LLM 会话发送 Jupyter Kernel 轻量心跳，更新 Kernel last_activity
            _, _, active_sess = resolve_slot_session(None)
            if active_sess and active_sess.get("url") and active_sess.get("token"):
                try:
                    from colab_cli.runtime import ColabRuntime

                    rt = ColabRuntime(
                        active_sess["url"],
                        active_sess["token"],
                        kernel_id=active_sess.get("kernel_id"),
                        session_id=active_sess.get("session_id"),
                    )
                    try:
                        rt.execute_code("pass", timeout=15)
                    finally:
                        rt.stop()
                except Exception:
                    pass
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 本机蓝绿 OpenAI 网关 (纯 Python 无弹窗版)")
    parser.add_argument("--port", type=int, default=18080, help="本机网关监听端口 (默认 18080)")
    args = parser.parse_args()

    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=0.5):
            return 0
    except Exception:
        pass

    try:
        srv = ThreadedHTTPServer(("127.0.0.1", args.port), make_gateway_handler(forced_slot=None))
    except OSError:
        return 0

    with open(gateway_pid_path(), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    for slot, port in SLOT_PORTS.items():
        _start_auxiliary_slot_server(port, slot)

    ka_thread = threading.Thread(target=_keepalive_worker, daemon=True)
    ka_thread.start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
