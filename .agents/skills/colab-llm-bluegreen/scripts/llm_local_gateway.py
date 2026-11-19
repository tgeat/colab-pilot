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
import atexit
import base64
import importlib.util
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

SLOT_PORTS = {
    "blue": 18081,
    "green": 18082,
}

ALLOWED_POST_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
}

META_PREFIX = "__PILOT_META__:"

_METRICS_CACHE: Dict[Tuple[str, str], Tuple[Dict[str, Any], float]] = {}
_METRICS_LOCK = threading.Lock()
_ACTIVE_STREAMS = 0
_ACTIVE_BY_SESSION: Dict[str, int] = {}
_ACTIVE_BY_SLOT: Dict[str, int] = {}
_ACTIVE_LOCK = threading.Lock()

# 保活线程刷新得到的代理令牌仅保存在内存中（endpoint -> (url, token)），
# 不再回写 sessions.json，避免与 colab-cli 并发写同一文件产生竞争/截断。
_TOKEN_OVERRIDES: Dict[str, Tuple[str, str]] = {}
_TOKEN_LOCK = threading.Lock()
_REFRESH_LOCK = threading.Lock()
_LAST_ON_DEMAND_REFRESH: float = 0.0
_DYNAMIC_VM_SERVERS: Dict[int, str] = {}
_DYNAMIC_LOCK = threading.Lock()


def cfg_dir() -> str:
    d = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(d, exist_ok=True)
    return d


def state_path() -> str:
    return os.path.join(cfg_dir(), "llm_bluegreen_state.json")


def sessions_path() -> str:
    return os.path.join(cfg_dir(), "sessions.json")


def api_keys_path() -> str:
    return os.path.join(os.path.dirname(state_path()), "llm_api_keys.json")


def gateway_pid_path() -> str:
    return os.path.join(cfg_dir(), "llm_local_gateway.pid")


def cleanup_pid_file() -> None:
    """退出时仅在 PID 文件仍指向当前进程时清理，防止残留旧 PID 导致误杀无关进程 (H-3)。"""
    p = gateway_pid_path()
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                recorded = f.read().strip()
            if recorded == str(os.getpid()):
                os.remove(p)
    except OSError:
        pass


def is_allowed_origin(origin: Optional[str]) -> bool:
    """仅允许无 Origin (CLI/Tauri reqwest) 或本机回环 / Tauri WebView Origin，拒绝任意外部网页跨域调用 (H-2)。"""
    if not origin:
        return True
    raw = origin.strip()
    if raw in ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"):
        return True
    try:
        parsed = urlsplit(raw)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return (parsed.hostname or "").lower() in ("localhost", "127.0.0.1", "tauri.localhost")


def read_state() -> Dict[str, Any]:
    p = state_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def read_sessions() -> Dict[str, Any]:
    p = sessions_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def read_api_keys() -> Dict[str, Any]:
    p = api_keys_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def resolve_vm_by_bearer_or_header(auth_header: Optional[str], vm_header: Optional[str]) -> Optional[str]:
    """支持通过 `X-Colab-Vm` 或每台虚拟机专属 `Authorization: Bearer sk-colab-...` 路由到对应实例。"""
    if vm_header and vm_header.strip():
        return vm_header.strip()
    if not auth_header:
        return None
    token = auth_header.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        return None
    vms = read_api_keys().get("vms")
    if isinstance(vms, dict):
        for vm_name, info in vms.items():
            if isinstance(info, dict) and info.get("api_key") == token:
                return vm_name
    return None


def _merge_vm_meta(base_st: Dict[str, Any], vm_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """合并单机配置元信息（绝不合入敏感 `api_key`，防止未鉴权 GET 端点泄露密钥）。"""
    out = dict(base_st)
    if isinstance(vm_meta, dict):
        for k in ("vm_name", "slot", "local_port", "remote_port", "model_name", "engine"):
            if vm_meta.get(k) is not None:
                out[k] = vm_meta[k]
    return out


def _apply_token_override(sess: Dict[str, Any], session_name: str) -> Dict[str, Any]:
    """若保活线程已为该 endpoint 刷新过令牌，返回覆盖了 url/token 且附带 session_name 的副本（不修改原 dict）。"""
    out = dict(sess)
    out["session_name"] = session_name
    ep = out.get("endpoint")
    if not ep:
        return out
    with _TOKEN_LOCK:
        override = _TOKEN_OVERRIDES.get(ep)
    if not override:
        return out
    new_url, new_tok = override
    if new_url:
        out["url"] = new_url
    if new_tok:
        out["token"] = new_tok
    return out


def get_active_gateway_counts(
    session_name: Optional[str] = None,
    slot: Optional[str] = None,
) -> Dict[str, Any]:
    with _ACTIVE_LOCK:
        by_sess = {k: v for k, v in _ACTIVE_BY_SESSION.items() if v > 0}
        by_slot = {k: v for k, v in _ACTIVE_BY_SLOT.items() if v > 0}
        sess_cnt = by_sess.get(session_name or "", 0) if session_name else 0
        slot_cnt = by_slot.get((slot or "").lower(), 0) if slot else 0
        return {
            "total": _ACTIVE_STREAMS,
            "session_active": sess_cnt,
            "slot_active": slot_cnt,
            "by_session": by_sess,
            "by_slot": by_slot,
        }


def _resolve_slot_name_from_state(st: Dict[str, Any], slot: str, prefix: str) -> str:
    """优先从 `slots[slot].session_name`、`{slot}_session`、`active_session`/`standby_session` 读取真实会话名，最后回退 `{prefix}-{slot}`。"""
    slot_norm = (slot or "blue").strip().lower()
    slots = st.get("slots")
    if isinstance(slots, dict) and isinstance(slots.get(slot_norm), dict):
        sn = str(slots[slot_norm].get("session_name") or "").strip()
        if sn:
            return sn
    direct = st.get(f"{slot_norm}_session")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if str(st.get("active_slot") or "").lower() == slot_norm:
        act = st.get("active_session")
        if isinstance(act, str) and act.strip():
            return act.strip()
    if str(st.get("standby_slot") or "").lower() == slot_norm:
        stb = st.get("standby_session")
        if isinstance(stb, str) and stb.strip():
            return stb.strip()
    return prefix if slot_norm == "standalone" else f"{prefix}-{slot_norm}"


def resolve_slot_session(
    forced_slot: Optional[str] = None,
    forced_vm: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    st = read_state()
    sessions = read_sessions()
    vms_raw = read_api_keys().get("vms")
    vms_cfg: Dict[str, Any] = vms_raw if isinstance(vms_raw, dict) else {}

    if forced_vm:
        vm_meta = vms_cfg.get(forced_vm) if isinstance(vms_cfg.get(forced_vm), dict) else None
        inferred_slot = (
            str((vm_meta or {}).get("slot") or "").lower()
            or ("blue" if forced_vm.endswith("-blue") else "green" if forced_vm.endswith("-green") else "standalone")
        )
        merged_st = _merge_vm_meta(st, vm_meta)
        sess = sessions.get(forced_vm)
        if not isinstance(sess, dict):
            return inferred_slot, merged_st, None
        return inferred_slot, merged_st, _apply_token_override(sess, forced_vm)

    prefix = (st.get("session_prefix") or "llm").strip() or "llm"
    slot = (forced_slot or st.get("active_slot") or "blue").lower()

    if forced_slot:
        preferred_name = _resolve_slot_name_from_state(st, slot, prefix)
        fallback_name = f"{prefix}-{slot}"
        cand_names = [preferred_name] + ([fallback_name] if fallback_name != preferred_name else [])
        sess = None
        resolved_name = preferred_name
        for cand_name in cand_names:
            cand_sess = sessions.get(cand_name)
            if isinstance(cand_sess, dict):
                sess = cand_sess
                resolved_name = cand_name
                break
        if not isinstance(sess, dict):
            for cand_vm, cand_meta in vms_cfg.items():
                if (
                    isinstance(cand_meta, dict)
                    and str(cand_meta.get("slot") or "").lower() == slot
                    and isinstance(sessions.get(cand_vm), dict)
                ):
                    sess = sessions[cand_vm]
                    resolved_name = cand_vm
                    break
        merged_st = _merge_vm_meta(st, vms_cfg.get(resolved_name))
        if not isinstance(sess, dict):
            return slot, merged_st, None
        return slot, merged_st, _apply_token_override(sess, resolved_name)

    active_session = st.get("active_session")
    sess_name = (
        active_session.strip()
        if (isinstance(active_session, str) and active_session.strip())
        else _resolve_slot_name_from_state(st, slot, prefix)
    )
    sess = sessions.get(sess_name)
    resolved_name = sess_name
    if not isinstance(sess, dict) and sessions:
        # 仅回退到同前缀记录的真实槽位名/默认蓝绿名或已在 llm_api_keys.json 显式注册的大模型会话，绝不误用普通训练会话
        for cand_slot in ("blue", "green"):
            for cand_name in (
                _resolve_slot_name_from_state(st, cand_slot, prefix),
                f"{prefix}-{cand_slot}",
            ):
                cand_sess = sessions.get(cand_name)
                if isinstance(cand_sess, dict):
                    sess = cand_sess
                    resolved_name = cand_name
                    slot = cand_slot
                    break
            if isinstance(sess, dict):
                break
        if not isinstance(sess, dict):
            for cand_vm, cand_meta in vms_cfg.items():
                cand_sess = sessions.get(cand_vm)
                if isinstance(cand_sess, dict):
                    sess = cand_sess
                    resolved_name = cand_vm
                    if isinstance(cand_meta, dict) and cand_meta.get("slot"):
                        slot = str(cand_meta["slot"]).lower()
                    break

    merged_st = _merge_vm_meta(st, vms_cfg.get(resolved_name))
    if not isinstance(sess, dict):
        return slot, merged_st, None
    return slot, merged_st, _apply_token_override(sess, resolved_name)


def decode_contents_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """解析 Jupyter Contents API 返回的 JSON / text / base64 内容 (H-5)。"""
    content = data.get("content")
    fmt = str(data.get("format") or "").lower()
    if isinstance(content, dict):
        return dict(content)
    if isinstance(content, str):
        raw_text = content
        if fmt == "base64":
            try:
                raw_text = base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")
            except Exception:
                return None
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            if fmt != "base64":
                try:
                    decoded = base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")
                    parsed = json.loads(decoded)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return None


def _enrich_metrics_with_session(metrics: Dict[str, Any], sess: Dict[str, Any], slot: str) -> Dict[str, Any]:
    out = dict(metrics)
    out.setdefault("slot", slot)
    if sess.get("session_name"):
        out["session_name"] = sess["session_name"]
    if sess.get("endpoint"):
        out["endpoint"] = sess["endpoint"]
    return out


def fetch_remote_llm_state(
    sess: Optional[Dict[str, Any]],
    slot: str = "blue",
    max_cache_age: float = 120.0,
) -> Optional[Dict[str, Any]]:
    if not sess:
        return None
    url = (sess.get("url") or "").rstrip("/")
    token = sess.get("token") or ""
    ep = str(sess.get("endpoint") or "")
    if not url or not token:
        return None
    cache_key = (slot, ep)
    req_url = f"{url}/api/contents/content/.colab_pilot_llm_state.json?content=1"
    req = urllib.request.Request(
        req_url,
        headers={
            "X-Colab-Runtime-Proxy-Token": token,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            parsed = decode_contents_payload(data) if isinstance(data, dict) else None
            if isinstance(parsed, dict) and ("total_completion_tokens" in parsed or "instant_tps" in parsed):
                enriched = _enrich_metrics_with_session(parsed, sess, slot)
                with _METRICS_LOCK:
                    _METRICS_CACHE[cache_key] = (enriched, time.time())
                return enriched
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            maybe_refresh_tokens_now(min_interval=5.0)
    except Exception:
        pass
    with _METRICS_LOCK:
        cached = _METRICS_CACHE.get(cache_key)
        if cached and (time.time() - cached[1]) <= max_cache_age:
            return dict(cached[0])
    return None


def get_cached_llm_state(slot: str, sess: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sess:
        return None
    ep = str(sess.get("endpoint") or "")
    cache_key = (slot, ep)
    with _METRICS_LOCK:
        cached = _METRICS_CACHE.get(cache_key)
        if cached and (time.time() - cached[1]) <= 60.0:
            return _enrich_metrics_with_session(cached[0], sess, slot)
    return None


def refresh_tokens_and_keepalive(send_kernel_ping: bool = True) -> bool:
    """刷新 Colab `runtimeProxyInfo.token` 至内存 `_TOKEN_OVERRIDES` 并维持隧道/Kernel 活跃 (H-6)。"""
    sessions = read_sessions()
    if not sessions:
        return False

    refreshed_any = False
    try:
        from colab_cli.common import state as colab_state

        assignments = colab_state.client.list_assignments()
        ep_map = {a.endpoint: a for a in assignments}
        for _s_name, s_data in list(sessions.items()):
            if not isinstance(s_data, dict):
                continue
            ep = s_data.get("endpoint")
            if ep and ep in ep_map:
                a = ep_map[ep]
                new_tok = a.runtime_proxy_info.token
                new_url = a.runtime_proxy_info.url
                with _TOKEN_LOCK:
                    if new_tok and s_data.get("token") != new_tok:
                        _TOKEN_OVERRIDES[ep] = (new_url or s_data.get("url") or "", new_tok)
                        refreshed_any = True
                    elif new_tok:
                        _TOKEN_OVERRIDES.pop(ep, None)
                        refreshed_any = True
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
        with _TOKEN_LOCK:
            for stale_ep in [e for e in _TOKEN_OVERRIDES if e not in ep_map]:
                _TOKEN_OVERRIDES.pop(stale_ep, None)
    except Exception:
        return False

    if send_kernel_ping:
        with _ACTIVE_LOCK:
            busy = _ACTIVE_STREAMS > 0
        if not busy:
            _, _, active_sess = resolve_slot_session(None)
            if active_sess and active_sess.get("url") and active_sess.get("token"):
                try:
                    from colab_cli.runtime import ColabRuntime

                    rt = ColabRuntime(
                        active_sess["url"],
                        active_sess["token"],
                        kernel_id=active_sess.get("kernel_id"),
                        session_id=None,
                    )
                    try:
                        rt.execute_code("pass", timeout=15)
                    finally:
                        rt.stop()
                except Exception:
                    pass

    return refreshed_any


def maybe_refresh_tokens_now(min_interval: float = 5.0) -> bool:
    """遇到 401/403 或连接鉴权异常时按需即时刷新代理令牌（带节流保护，防止并发风暴）。"""
    global _LAST_ON_DEMAND_REFRESH
    with _REFRESH_LOCK:
        now = time.time()
        if (now - _LAST_ON_DEMAND_REFRESH) < min_interval:
            return False
        _LAST_ON_DEMAND_REFRESH = now
    return refresh_tokens_and_keepalive(send_kernel_ping=False)


def build_remote_proxy_code(remote_port: int, req_path: str, req_b64: str) -> str:
    return f"""
import base64, http.client, json, sys
raw_body = base64.b64decode({json.dumps(req_b64)})
conn = http.client.HTTPConnection("127.0.0.1", {int(remote_port)}, timeout=300)
try:
    conn.request("POST", {json.dumps(req_path)}, body=raw_body, headers={{"Content-Type": "application/json"}})
    resp = conn.getresponse()
    ctype = (resp.getheader("Content-Type") or "").lower()
    sys.stdout.write({json.dumps(META_PREFIX)} + json.dumps({{"status": int(resp.status), "ctype": ctype}}) + "\\n")
    sys.stdout.flush()
    if "text/event-stream" in ctype and resp.status < 400:
        while True:
            line_b = resp.fp.readline()
            if not line_b:
                break
            sys.stdout.write(line_b.decode("utf-8", "replace"))
            sys.stdout.flush()
    else:
        sys.stdout.write(resp.read().decode("utf-8", "replace"))
        sys.stdout.flush()
finally:
    conn.close()
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def make_gateway_handler(forced_slot: Optional[str] = None, forced_vm: Optional[str] = None):
    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin and is_allowed_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin.strip())
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Colab-Vm")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_headers_done = True
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def _resolve_request_target(self) -> Tuple[Optional[str], str, Dict[str, Any], Optional[Dict[str, Any]]]:
            effective_vm = forced_vm or resolve_vm_by_bearer_or_header(
                self.headers.get("Authorization"),
                self.headers.get("X-Colab-Vm"),
            )
            slot, st, sess = resolve_slot_session(forced_slot=forced_slot, forced_vm=effective_vm)
            return effective_vm, slot, st, sess

        def do_OPTIONS(self) -> None:
            origin = self.headers.get("Origin")
            if origin and not is_allowed_origin(origin):
                self._send_json(403, {"error": {"message": "Forbidden cross-origin preflight"}})
                return
            self.send_response(204)
            self._send_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            origin = self.headers.get("Origin")
            if origin and not is_allowed_origin(origin):
                self._send_json(403, {"error": {"message": "Forbidden cross-origin request"}})
                return

            effective_vm, slot, st, sess = self._resolve_request_target()
            target_port = int(st.get("local_port") or SLOT_PORTS.get(slot, 18081))
            session_name = (sess or {}).get("session_name") or effective_vm
            endpoint = (sess or {}).get("endpoint")
            raw_phase = st.get("phase")
            effective_phase = raw_phase if sess is not None else ("failed" if raw_phase == "failed" else "offline")
            stale_state = bool(st and raw_phase == "serving" and sess is None)

            gw_counts = get_active_gateway_counts(session_name=session_name, slot=slot)

            if self.path.startswith("/pilot/gateway"):
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "online": sess is not None,
                        "session_active": sess is not None,
                        "stale_state_detected": stale_state,
                        "phase": effective_phase,
                        "last_error": st.get("last_error"),
                        "session_prefix": st.get("session_prefix", "llm"),
                        "active_slot": slot,
                        "session_name": session_name,
                        "vm_name": session_name,
                        "endpoint": endpoint,
                        "target_port": target_port,
                        "tunnel_open": sess is not None,
                        "model_name": st.get("model_name") if sess is not None or raw_phase == "failed" else None,
                        "engine": st.get("engine"),
                        "gpu": st.get("gpu"),
                        "active_streams": gw_counts["total"],
                        "active_requests": gw_counts["total"],
                        "active_requests_by_session": gw_counts["by_session"],
                        "active_requests_by_slot": gw_counts["by_slot"],
                    },
                )
                return

            if self.path.startswith("/pilot/health"):
                remote_st = get_cached_llm_state(slot, sess)
                healthy = (
                    bool(remote_st and remote_st.get("backend_healthy", True))
                    if (remote_st and sess is not None)
                    else (sess is not None)
                )
                rem_streams = int((remote_st or {}).get("active_streams") or 0)
                rem_reqs = int((remote_st or {}).get("active_requests") or rem_streams)
                self._send_json(
                    200 if healthy else 503,
                    {
                        "ok": healthy,
                        "phase": effective_phase,
                        "last_error": st.get("last_error"),
                        "slot": (remote_st or {}).get("slot", slot),
                        "session_name": session_name,
                        "vm_name": session_name,
                        "endpoint": endpoint,
                        "role": (remote_st or {}).get("role", "active" if healthy else "offline"),
                        "engine": (remote_st or {}).get("engine", st.get("engine", "llama.cpp")),
                        "model_name": (remote_st or {}).get("model_name", st.get("model_name", "")),
                        "instant_tps": (remote_st or {}).get("instant_tps", 0.0),
                        "active_streams": max(rem_streams, gw_counts["session_active"]),
                        "active_requests": max(rem_reqs, gw_counts["session_active"]),
                    },
                )
                return

            if self.path.startswith("/pilot/metrics"):
                remote_st = get_cached_llm_state(slot, sess)
                if sess is not None and remote_st and isinstance(remote_st, dict) and "total_completion_tokens" in remote_st:
                    self._send_json(200, remote_st)
                else:
                    # 绝不返回残缺的 fallback 字典覆盖 Colab Pilot 的完整遥测快照
                    self._send_json(
                        503,
                        {
                            "ok": False,
                            "slot": slot,
                            "session_name": session_name,
                            "vm_name": session_name,
                            "endpoint": endpoint,
                            "error": "session_offline" if sess is None else "metrics_cache_warming",
                        },
                    )
                return

            if self.path.startswith("/v1/models"):
                if sess is None:
                    self._send_json(200, {"object": "list", "data": []})
                    return
                model_id = st.get("model_name") or "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
                owner_tag = session_name or slot
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": int(st.get("updated_at") or time.time()),
                                "owned_by": f"colab-pilot-{owner_tag}",
                            }
                        ],
                    },
                )
                return

            self._send_json(404, {"error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if origin and not is_allowed_origin(origin):
                self._send_json(403, {"error": {"message": "Forbidden cross-origin request"}})
                return

            ctype_hdr = (self.headers.get("Content-Type") or "").lower()
            if any(
                ctype_hdr.startswith(bad)
                for bad in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data")
            ):
                self._send_json(415, {"error": {"message": "Content-Type must be application/json"}})
                return

            req_path = urlsplit(self.path).path.rstrip("/") or "/"
            if req_path not in ALLOWED_POST_PATHS:
                self._send_json(403, {"error": {"message": f"Forbidden gateway path: {req_path}"}})
                return

            effective_vm, slot, st, sess = self._resolve_request_target()
            if not sess:
                target_label = effective_vm or slot
                err_hint = f" ({st.get('last_error')})" if st.get("last_error") else ""
                self._send_json(
                    502,
                    {"error": {"message": f"No active Colab session found for target [{target_label}]{err_hint}"}},
                )
                return

            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length < 0 or length > 16 * 1024 * 1024:
                    raise ValueError("invalid Content-Length")
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Invalid request body: {e}"}})
                return
            try:
                req_json = json.loads(raw_body.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                req_json = {}
            if not isinstance(req_json, dict):
                req_json = {}

            is_stream = bool(req_json.get("stream", False))
            model_id = st.get("model_name") or req_json.get("model") or "default"

            try:
                remote_port = int(st.get("remote_port") or st.get("port") or 8000)
            except (TypeError, ValueError):
                remote_port = 8000

            req_b64 = base64.b64encode(raw_body).decode("ascii")
            remote_code = build_remote_proxy_code(remote_port, req_path, req_b64)

            tracked_session = str(
                sess.get("session_name")
                or effective_vm
                or _resolve_slot_name_from_state(st, slot, (st.get("session_prefix") or "llm").strip() or "llm")
            )
            tracked_slot = str(slot or "blue").lower()

            global _ACTIVE_STREAMS
            with _ACTIVE_LOCK:
                _ACTIVE_STREAMS += 1
                _ACTIVE_BY_SESSION[tracked_session] = _ACTIVE_BY_SESSION.get(tracked_session, 0) + 1
                _ACTIVE_BY_SLOT[tracked_slot] = _ACTIVE_BY_SLOT.get(tracked_slot, 0) + 1
            try:
                from colab_cli.runtime import ColabRuntime

                self.send_headers_done = False

                def _run_once(cur_sess: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str], List[str], bool]:
                    meta_info: Dict[str, Any] = {}
                    meta_ready = {"done": False, "buf": ""}
                    body_chunks: List[str] = []
                    kernel_errors: List[str] = []
                    client_gone = {"v": False}

                    rt = ColabRuntime(
                        cur_sess["url"],
                        cur_sess["token"],
                        kernel_id=cur_sess.get("kernel_id"),
                        session_id=None,
                    )

                    def _begin_sse_headers() -> None:
                        if self.send_headers_done:
                            return
                        self.close_connection = True
                        self.send_response(200)
                        self._send_cors()
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_headers_done = True
                        self.end_headers()

                    def _emit_body_piece(piece: str) -> None:
                        if not piece or client_gone["v"]:
                            return
                        is_upstream_sse = (
                            is_stream
                            and meta_info.get("status", 200) < 400
                            and "text/event-stream" in str(meta_info.get("ctype") or "")
                        )
                        if is_upstream_sse:
                            _begin_sse_headers()
                            try:
                                self.wfile.write(piece.encode("utf-8"))
                                self.wfile.flush()
                            except OSError:
                                client_gone["v"] = True
                                try:
                                    rt.stop()
                                except Exception:
                                    pass
                        else:
                            body_chunks.append(piece)

                    def on_out(out: Dict[str, Any]) -> None:
                        if client_gone["v"]:
                            return
                        otype = out.get("output_type")
                        if otype == "error":
                            ename = out.get("ename") or "KernelError"
                            evalue = out.get("evalue") or "Unknown remote execution error"
                            kernel_errors.append(f"{ename}: {evalue}")
                            return
                        if otype != "stream":
                            return
                        txt = out.get("text", "")
                        if not txt:
                            return
                        if not meta_ready["done"]:
                            meta_ready["buf"] += txt
                            if "\n" not in meta_ready["buf"]:
                                if len(meta_ready["buf"]) < 4096:
                                    return
                                # 超长首行无元信息前缀：按普通响应体回退处理
                                meta_ready["done"] = True
                                _emit_body_piece(meta_ready["buf"])
                                meta_ready["buf"] = ""
                                return
                            first_line, rest = meta_ready["buf"].split("\n", 1)
                            meta_ready["buf"] = ""
                            meta_ready["done"] = True
                            if first_line.startswith(META_PREFIX):
                                try:
                                    parsed_m = json.loads(first_line[len(META_PREFIX) :])
                                    if isinstance(parsed_m, dict):
                                        meta_info.update(parsed_m)
                                except Exception:
                                    pass
                                if (
                                    is_stream
                                    and meta_info.get("status", 200) < 400
                                    and "text/event-stream" in str(meta_info.get("ctype") or "")
                                ):
                                    _begin_sse_headers()
                                _emit_body_piece(rest)
                            else:
                                _emit_body_piece(first_line + "\n" + rest)
                        else:
                            _emit_body_piece(txt)

                    try:
                        rt.execute_code(remote_code, output_hook=on_out, timeout=300)
                    finally:
                        try:
                            rt.stop()
                        except Exception:
                            pass

                    if not meta_ready["done"] and meta_ready["buf"]:
                        body_chunks.append(meta_ready["buf"])

                    return (meta_info if meta_info else None), body_chunks, kernel_errors, client_gone["v"]

                try:
                    meta, chunks, k_errors, gone = _run_once(sess)
                except Exception as first_exc:
                    # 若尚未发送响应头且可能是代理 Token 过期，立即刷新一次并重试 (H-6)
                    if not self.send_headers_done and maybe_refresh_tokens_now(min_interval=5.0):
                        _, _, refreshed_sess = resolve_slot_session(forced_slot=forced_slot, forced_vm=effective_vm)
                        if refreshed_sess:
                            meta, chunks, k_errors, gone = _run_once(refreshed_sess)
                        else:
                            raise first_exc
                    else:
                        raise first_exc

                if gone:
                    return

                if self.send_headers_done:
                    if k_errors:
                        err_frame = json.dumps({"error": {"message": "; ".join(k_errors)}}, ensure_ascii=False)
                        try:
                            self.wfile.write(f"data: {err_frame}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except OSError:
                            pass
                    return

                raw_resp = "".join(chunks)
                upstream_status = int((meta or {}).get("status") or 0)

                if k_errors and not upstream_status:
                    self._send_json(
                        502,
                        {"error": {"message": f"Remote Colab runtime error: {'; '.join(k_errors)}"}},
                    )
                    return

                if not meta and not raw_resp.strip():
                    maybe_refresh_tokens_now(min_interval=5.0)
                    self._send_json(
                        502,
                        {"error": {"message": "Empty response from remote Colab runtime"}},
                    )
                    return

                if upstream_status >= 400:
                    try:
                        err_obj = json.loads(raw_resp)
                        if not isinstance(err_obj, dict):
                            err_obj = {"error": {"message": raw_resp}}
                    except Exception:
                        err_obj = {"error": {"message": raw_resp.strip() or f"Upstream HTTP {upstream_status}"}}
                    self._send_json(upstream_status, err_obj)
                    return

                if raw_resp.lstrip().startswith("data:"):
                    if is_stream:
                        self.close_connection = True
                        self.send_response(200)
                        self._send_cors()
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_headers_done = True
                        self.end_headers()
                        self.wfile.write(raw_resp.encode("utf-8"))
                        self.wfile.flush()
                        return
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
                        parsed_json = json.loads(raw_resp)
                        if isinstance(parsed_json, dict):
                            self._send_json(upstream_status or 200, parsed_json)
                            return
                    except Exception:
                        pass
                    self._send_json(
                        upstream_status or 200,
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
                if not getattr(self, "send_headers_done", False):
                    self._send_json(502, {"error": {"message": f"Gateway runtime bridge error: {e}"}})
            finally:
                with _ACTIVE_LOCK:
                    _ACTIVE_STREAMS = max(0, _ACTIVE_STREAMS - 1)
                    rem_sess = max(0, _ACTIVE_BY_SESSION.get(tracked_session, 0) - 1)
                    if rem_sess > 0:
                        _ACTIVE_BY_SESSION[tracked_session] = rem_sess
                    else:
                        _ACTIVE_BY_SESSION.pop(tracked_session, None)
                    rem_slot = max(0, _ACTIVE_BY_SLOT.get(tracked_slot, 0) - 1)
                    if rem_slot > 0:
                        _ACTIVE_BY_SLOT[tracked_slot] = rem_slot
                    else:
                        _ACTIVE_BY_SLOT.pop(tracked_slot, None)

    return GatewayHandler


def _start_auxiliary_slot_server(port: int, slot: str) -> None:
    try:
        srv = ThreadedHTTPServer(("127.0.0.1", port), make_gateway_handler(forced_slot=slot))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
    except OSError:
        pass


def _sync_dynamic_vm_servers() -> None:
    """根据 `llm_api_keys.json` 动态监听每台独立大模型虚拟机的专属本地端口 (18083+)。"""
    vms_raw = read_api_keys().get("vms")
    if not isinstance(vms_raw, dict):
        return
    for vm_name, info in vms_raw.items():
        if not isinstance(info, dict):
            continue
        try:
            port = int(info.get("local_port") or 0)
        except (TypeError, ValueError):
            continue
        if port < 18083 or port > 65535:
            continue
        with _DYNAMIC_LOCK:
            if port in _DYNAMIC_VM_SERVERS:
                continue
        try:
            srv = ThreadedHTTPServer(("127.0.0.1", port), make_gateway_handler(forced_slot=None, forced_vm=vm_name))
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            with _DYNAMIC_LOCK:
                _DYNAMIC_VM_SERVERS[port] = vm_name
        except OSError:
            pass


def _keepalive_worker() -> None:
    """
    后台保活与代理令牌自动续期线程（启动即刻执行首轮，随后每 90 秒循环，H-6）：
    1. 调用 Colab `/tun/m/assignments` 自动刷新 `runtimeProxyInfo.token`（防止 3600s 令牌过期），
       刷新结果仅保存在进程内 `_TOKEN_OVERRIDES`，不回写 `sessions.json`（避免与 colab-cli 竞争写入）
    2. 调用 `/tun/m/keep-alive/{endpoint}` (带 `X-Colab-Tunnel: Google` 头) 维持 Google Colab 隧道租约
    3. 通过 `ColabRuntime` 向云端 Jupyter Kernel 发送轻量心跳 (`pass`)，刷新 Kernel `last_activity`。
    """
    while True:
        try:
            refresh_tokens_and_keepalive(send_kernel_ping=True)
        except Exception:
            pass
        time.sleep(90.0)


def _metrics_poller_worker() -> None:
    """后台每 2.0 秒异步预拉取云端 `.colab_pilot_llm_state.json`，使本机 `/pilot/metrics` 达到 <1ms 内存级响应。"""
    while True:
        try:
            _sync_dynamic_vm_servers()
            polled_eps = set()
            for sl in ("blue", "green"):
                _, _, sess = resolve_slot_session(forced_slot=sl)
                if sess:
                    ep = sess.get("endpoint")
                    if ep:
                        polled_eps.add(ep)
                    fetch_remote_llm_state(sess, slot=sl, max_cache_age=120.0)
            vms_raw = read_api_keys().get("vms")
            if isinstance(vms_raw, dict):
                for vm_name in list(vms_raw.keys()):
                    sl, _, sess = resolve_slot_session(forced_vm=vm_name)
                    if sess and sess.get("endpoint") not in polled_eps:
                        polled_eps.add(sess.get("endpoint"))
                        fetch_remote_llm_state(sess, slot=sl, max_cache_age=120.0)
        except Exception:
            pass
        time.sleep(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 本机蓝绿 OpenAI 网关 (纯 Python 无弹窗版)")
    parser.add_argument("--port", type=int, default=18080, help="本机网关监听端口 (默认 18080)")
    args = parser.parse_args()

    # 绑定端口前强制检查 colab_cli 依赖，杜绝缺少依赖的解释器霸占 :18080 端口并对所有请求返回 502 (H-7)
    if os.environ.get("PILOT_GATEWAY_ALLOW_MOCK") != "1" and importlib.util.find_spec("colab_cli") is None:
        print(
            "ERROR: 当前 Python 环境缺少 `colab_cli` 模块，拒绝绑定本机网关端口。\n"
            "请使用 `google-colab-cli` 工具虚拟环境中的 Python 启动本脚本。",
            file=sys.stderr,
            flush=True,
        )
        return 2

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
    atexit.register(cleanup_pid_file)

    def _on_sig(_signum: int, _frame: Any) -> None:
        cleanup_pid_file()
        raise SystemExit(0)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig_obj = getattr(signal, sig_name, None)
        if sig_obj is not None:
            try:
                signal.signal(sig_obj, _on_sig)
            except Exception:
                pass

    for slot, port in SLOT_PORTS.items():
        _start_auxiliary_slot_server(port, slot)
    _sync_dynamic_vm_servers()

    ka_thread = threading.Thread(target=_keepalive_worker, daemon=True)
    ka_thread.start()

    mp_thread = threading.Thread(target=_metrics_poller_worker, daemon=True)
    mp_thread.start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        cleanup_pid_file()
    return 0


if __name__ == "__main__":
    sys.exit(main())
