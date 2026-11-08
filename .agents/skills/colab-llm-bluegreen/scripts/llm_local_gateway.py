#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 本机蓝绿零中断 OpenAI 网关与 SSH 隧道守护进程 (llm_local_gateway.py)
====================================================================================
运行于本机 (默认 http://127.0.0.1:18080)，负责：
1. 自动读取 `~/.config/colab-cli/llm_bluegreen_state.json` 中的 `active_slot` (blue / green) 与 `standby_slot`
2. 自动拉起并守护通往 Colab 云端实例 `:8000` (计量边车) 的后台 SSH 隧道：
   - blue 槽位本地映射端口: 18081 -> Colab (llm-blue) 127.0.0.1:8000
   - green 槽位本地映射端口: 18082 -> Colab (llm-green) 127.0.0.1:8000
3. 对本机所有 AI 客户端 (OpenAI SDK / Cherry Studio / Chatbox / Open WebUI / Cursor / curl)
   暴露唯一固定的本地端点：`http://127.0.0.1:18080/v1`
4. 当触发蓝绿热切换 (`switchover`) 时，网关自动感知 `active_slot` 变更，新请求毫秒级无感切流至新槽位，
   在途流式响应在原槽位平滑输出完毕，本机客户端无需修改端口或重启。
"""

import argparse
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Dict, Optional

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


def gateway_pid_path() -> str:
    return os.path.join(cfg_dir(), "llm_local_gateway.pid")


def read_state() -> dict:
    p = state_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def ensure_ssh_key() -> str:
    key_path = os.path.join(cfg_dir(), "id_ed25519")
    if not os.path.exists(key_path):
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key_path],
            capture_output=True,
            check=False,
        )
    return key_path.replace("\\", "/")


def find_colab_bin() -> str:
    ext = ".exe" if os.name == "nt" else ""
    p = os.path.expanduser(f"~/.local/bin/colab{ext}")
    if os.path.exists(p):
        return p.replace("\\", "/")
    cand = shutil.which("colab")
    return (cand or "colab").replace("\\", "/")


def is_port_open(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
        conn.request("GET", "/pilot/health")
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


class TunnelSupervisor:
    def __init__(self) -> None:
        self.procs: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def ensure_slot_tunnel(self, session_prefix: str, slot: str) -> None:
        port = SLOT_PORTS.get(slot)
        if not port:
            return
        proc = self.procs.get(slot)
        if proc is not None and proc.poll() is None:
            return

        session_name = f"{session_prefix}-{slot}"
        key_path = ensure_ssh_key()
        colab_bin = find_colab_bin()
        proxy_cmd = f"{colab_bin} ssh --proxy-mode -s {session_name} -i {key_path}"
        null_dev = "NUL" if os.name == "nt" else "/dev/null"
        cmd = [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:8000",
            "-o",
            f"ProxyCommand={proxy_cmd}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={null_dev}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-i",
            key_path,
            "root@colab-runtime",
        ]
        kwargs = {"stdin": subprocess.DEVNULL, "close_fds": True}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00000200 | 0x00000008
        else:
            kwargs["start_new_session"] = True

        self.procs[slot] = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )

    def stop_slot_tunnel(self, slot: str) -> None:
        proc = self.procs.pop(slot, None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop_all(self) -> None:
        self.stop_event.set()
        with self.lock:
            for slot in list(self.procs.keys()):
                self.stop_slot_tunnel(slot)

    def loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                st = read_state()
                prefix = st.get("session_prefix") or "llm"
                active = (st.get("active_slot") or "blue").lower()
                standby = (st.get("standby_slot") or "").lower()
                wanted = {s for s in (active, standby) if s in SLOT_PORTS}
                with self.lock:
                    for s in wanted:
                        self.ensure_slot_tunnel(prefix, s)
                    for s in list(self.procs.keys()):
                        if s not in wanted:
                            self.stop_slot_tunnel(s)
            except Exception:
                pass
            time.sleep(2.0)


SUPERVISOR = TunnelSupervisor()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        pass

    def _active_target_port(self) -> tuple[str, int, dict]:
        st = read_state()
        slot = (st.get("active_slot") or "blue").lower()
        port = SLOT_PORTS.get(slot, 18081)
        return slot, port, st

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        slot, target_port, st = self._active_target_port()
        if self.path.startswith("/pilot/gateway"):
            body = json.dumps(
                {
                    "ok": True,
                    "active_slot": slot,
                    "target_port": target_port,
                    "tunnel_open": is_port_open(target_port),
                    "model_name": st.get("model_name"),
                    "engine": st.get("engine"),
                    "gpu": st.get("gpu"),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/v1/models"):
            model_id = st.get("model_name") or "Qwen/Qwen3.8-27B"
            body = json.dumps(
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
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=15)
            conn.request("GET", self.path)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            self._send_cors()
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            err = json.dumps({"error": f"Colab slot [{slot}] tunnel connecting: {e}"}).encode("utf-8")
            self.send_response(502)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_POST(self) -> None:
        slot, target_port, st = self._active_target_port()
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            req_json = json.loads(raw_body.decode("utf-8", errors="ignore") or "{}")
        except Exception:
            req_json = {}

        is_stream = bool(req_json.get("stream", False))

        # 确保目标槽位隧道已打开
        if not is_port_open(target_port):
            SUPERVISOR.ensure_slot_tunnel(st.get("session_prefix") or "llm", slot)
            for _ in range(16):
                if is_port_open(target_port):
                    break
                time.sleep(0.5)

        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=300)
            conn.request(
                "POST",
                self.path,
                body=raw_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()

            if is_stream:
                self.close_connection = True
                self.send_response(resp.status)
                self._send_cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    line = resp.fp.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                # 某些底层引擎始终返回 SSE 流，若客户端请求非流式 (stream=False)，自动聚合为标准 OpenAI 响应
                ctype = (resp.getheader("Content-Type") or "").lower()
                raw_resp = resp.read()
                if "text/event-stream" in ctype or raw_resp.startswith(b"data: "):
                    pieces = []
                    model_id = st.get("model_name") or "Qwen/Qwen3.8-27B"
                    for ln in raw_resp.decode("utf-8", errors="ignore").splitlines():
                        if ln.startswith("data: ") and ln[6:].strip() != "[DONE]":
                            try:
                                obj = json.loads(ln[6:])
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    pieces.append(delta["content"])
                            except Exception:
                                pass
                    full_text = "".join(pieces)
                    out = json.dumps(
                        {
                            "id": f"chatcmpl-{int(time.time()*1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": full_text},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    self.send_response(resp.status)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw_resp)))
                    self.end_headers()
                    self.wfile.write(raw_resp)
            conn.close()
        except Exception as e:
            err = json.dumps({"error": {"message": f"Gateway proxy error on slot [{slot}]: {e}"}}).encode("utf-8")
            self.send_response(502)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 本机蓝绿 OpenAI 网关")
    parser.add_argument("--port", type=int, default=18080, help="本机网关监听端口 (默认 18080)")
    args = parser.parse_args()

    with open(gateway_pid_path(), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    t = threading.Thread(target=SUPERVISOR.loop, daemon=True)
    t.start()

    srv = ThreadedHTTPServer(("127.0.0.1", args.port), GatewayHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SUPERVISOR.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
