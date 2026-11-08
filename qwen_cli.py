#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — Qwen/Qwen3.8-27B 本机客户端 (qwen_cli.py)
====================================================================
直接连接本机蓝绿零中断网关 (默认 http://127.0.0.1:18080/v1)，调用部署在
Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96GB) 上的 Qwen/Qwen3.8-27B。

支持四种使用方式：
1. 原生 Win32 RichEdit50W 中文 GUI + 内置网页端 + 终端三端实时同步模式 (默认模式):
   qwen
2. 纯命令行交互模式 (支持 /zh 弹出原生 RichEdit 中文输入框 或 /web 打开网页端):
   qwen --cli
3. 单次快速提问模式:
   qwen "用 Python 实现一个 LRU Cache"
4. 附加本地代码/文件分析模式:
   qwen -f train.py "请总结这个脚本的核心逻辑"
"""

import argparse
import http.client
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 启用 Windows 终端 ANSI 颜色支持，并恢复代码页为 936
if os.name == "nt":
    os.system("")
    try:
        import ctypes
        _k32 = ctypes.windll.kernel32
        _k32.SetConsoleCP(936)
        _k32.SetConsoleOutputCP(936)
    except Exception:
        pass


# ====================================================================
# Windows 11 原生 TSF (Text Services Framework) + IMM32 + RichEdit50W 支持
# ====================================================================
_WIN32_INITIALIZED = False
_TSF_CLIENT_ID = 0
_TSF_THREAD_MGR = None


def _init_win32_ctypes() -> None:
    global _WIN32_INITIALIZED
    if _WIN32_INITIALIZED or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        u32 = ctypes.windll.user32
        imm = ctypes.windll.imm32
        ole32 = ctypes.windll.ole32
        msctf = ctypes.windll.msctf

        ole32.OleInitialize.argtypes = [wintypes.LPVOID]
        ole32.OleInitialize.restype = ctypes.c_long
        ole32.OleInitialize(None)

        ctypes.windll.LoadLibrary("Msftedit.dll")

        u32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        u32.CreateWindowExW.restype = wintypes.HWND
        u32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u32.SendMessageW.restype = wintypes.LPARAM
        u32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u32.PostMessageW.restype = wintypes.BOOL
        u32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
        u32.MoveWindow.restype = wintypes.BOOL
        u32.SetFocus.argtypes = [wintypes.HWND]
        u32.SetFocus.restype = wintypes.HWND
        u32.GetFocus.argtypes = []
        u32.GetFocus.restype = wintypes.HWND
        u32.GetParent.argtypes = [wintypes.HWND]
        u32.GetParent.restype = wintypes.HWND
        u32.SetForegroundWindow.argtypes = [wintypes.HWND]
        u32.SetForegroundWindow.restype = wintypes.BOOL
        u32.BringWindowToTop.argtypes = [wintypes.HWND]
        u32.BringWindowToTop.restype = wintypes.BOOL
        u32.ActivateKeyboardLayout.argtypes = [wintypes.HKL, wintypes.UINT]
        u32.ActivateKeyboardLayout.restype = wintypes.HKL
        u32.GetKeyState.argtypes = [ctypes.c_int]
        u32.GetKeyState.restype = ctypes.c_short
        u32.CallWindowProcW.argtypes = [
            ctypes.c_void_p,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        u32.CallWindowProcW.restype = wintypes.LPARAM

        imm.ImmGetContext.argtypes = [wintypes.HWND]
        imm.ImmGetContext.restype = wintypes.HANDLE
        imm.ImmSetOpenStatus.argtypes = [wintypes.HANDLE, wintypes.BOOL]
        imm.ImmSetOpenStatus.restype = wintypes.BOOL
        imm.ImmGetOpenStatus.argtypes = [wintypes.HANDLE]
        imm.ImmGetOpenStatus.restype = wintypes.BOOL
        imm.ImmSetConversionStatus.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        imm.ImmSetConversionStatus.restype = wintypes.BOOL
        imm.ImmGetConversionStatus.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        imm.ImmGetConversionStatus.restype = wintypes.BOOL
        imm.ImmReleaseContext.argtypes = [wintypes.HWND, wintypes.HANDLE]
        imm.ImmReleaseContext.restype = wintypes.BOOL
        imm.ImmAssociateContextEx.argtypes = [wintypes.HWND, wintypes.HANDLE, wintypes.DWORD]
        imm.ImmAssociateContextEx.restype = wintypes.BOOL
        imm.ImmGetCompositionStringW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        imm.ImmGetCompositionStringW.restype = ctypes.c_long

        msctf.TF_CreateThreadMgr.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        msctf.TF_CreateThreadMgr.restype = ctypes.c_long
        msctf.TF_CreateInputProcessorProfiles.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        msctf.TF_CreateInputProcessorProfiles.restype = ctypes.c_long

        _WIN32_INITIALIZED = True
    except Exception:
        pass


def _activate_tsf_chinese_compartments() -> None:
    """通过 Windows TSF (MSCTF.dll) 激活当前线程 ITfThreadMgr 并将输入法状态设为中文+中文标点"""
    global _TSF_CLIENT_ID, _TSF_THREAD_MGR
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        _init_win32_ctypes()
        msctf = ctypes.windll.msctf

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

            @classmethod
            def from_uuid(cls, u: uuid.UUID) -> "GUID":
                d1, d2, d3 = u.fields[:3]
                arr = (ctypes.c_ubyte * 8)(*u.bytes[8:])
                return cls(d1, d2, d3, arr)

        class VARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", wintypes.WORD),
                ("wReserved1", wintypes.WORD),
                ("wReserved2", wintypes.WORD),
                ("wReserved3", wintypes.WORD),
                ("lVal", wintypes.LONG),
                ("pad", wintypes.LONG),
            ]

        # 1. 切换语言到简体中文 (0x0804)
        pipp = ctypes.c_void_p()
        if msctf.TF_CreateInputProcessorProfiles(ctypes.byref(pipp)) == 0 and pipp.value:
            ipp_vtbl = ctypes.cast(
                ctypes.cast(pipp.value, ctypes.POINTER(ctypes.c_void_p))[0],
                ctypes.POINTER(ctypes.c_void_p),
            )
            ChangeLangProto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.WORD)
            ChangeLangProto(ipp_vtbl[14])(pipp.value, 0x0804)

        # 2. 创建并激活 ITfThreadMgr
        if _TSF_THREAD_MGR is None:
            ptm = ctypes.c_void_p()
            if msctf.TF_CreateThreadMgr(ctypes.byref(ptm)) == 0 and ptm.value:
                _TSF_THREAD_MGR = ptm
        if _TSF_THREAD_MGR is None or not _TSF_THREAD_MGR.value:
            return

        tm_vtbl = ctypes.cast(
            ctypes.cast(_TSF_THREAD_MGR.value, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        if not _TSF_CLIENT_ID:
            ActivateProto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
            cid = wintypes.DWORD()
            if ActivateProto(tm_vtbl[3])(_TSF_THREAD_MGR.value, ctypes.byref(cid)) == 0:
                _TSF_CLIENT_ID = cid.value

        # 3. 设置 TSF Keyboard OpenClose = 1 及 Conversion = 0x401 (中文 + 中文标点)
        iid_cm = GUID.from_uuid(uuid.UUID("{7DCF57AC-18AD-438B-824D-979BFFB74B7C}"))
        guid_open = GUID.from_uuid(uuid.UUID("{58273AAD-01BB-4164-95C6-755BA0B5162D}"))
        guid_conv = GUID.from_uuid(uuid.UUID("{CCF05DD8-4A87-11D7-A6E2-00065B84435C}"))

        QIProto = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
        )
        pcm = ctypes.c_void_p()
        if QIProto(tm_vtbl[0])(_TSF_THREAD_MGR.value, ctypes.byref(iid_cm), ctypes.byref(pcm)) == 0 and pcm.value:
            cm_vtbl = ctypes.cast(
                ctypes.cast(pcm.value, ctypes.POINTER(ctypes.c_void_p))[0],
                ctypes.POINTER(ctypes.c_void_p),
            )
            GetCompProto = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
            )
            SetValProto = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(VARIANT)
            )
            for g_obj, val in ((guid_open, 1), (guid_conv, 0x0401)):
                pcomp = ctypes.c_void_p()
                if GetCompProto(cm_vtbl[3])(pcm.value, ctypes.byref(g_obj), ctypes.byref(pcomp)) == 0 and pcomp.value:
                    comp_vtbl = ctypes.cast(
                        ctypes.cast(pcomp.value, ctypes.POINTER(ctypes.c_void_p))[0],
                        ctypes.POINTER(ctypes.c_void_p),
                    )
                    v = VARIANT(vt=3, lVal=val)
                    SetValProto(comp_vtbl[3])(pcomp.value, _TSF_CLIENT_ID or 1, ctypes.byref(v))
    except Exception:
        pass


def force_enable_chinese_ime(
    hwnd: int,
    extra_hwnds: Optional[List[int]] = None,
    set_win_focus: bool = True,
) -> None:
    """在目标窗口句柄上同时通过 Win32 IMM32 与 Windows 11 TSF 强制挂载并开启中文输入状态"""
    if os.name != "nt" or not hwnd:
        return
    try:
        import ctypes

        _init_win32_ctypes()
        u32 = ctypes.windll.user32
        imm = ctypes.windll.imm32

        # 1. 激活简体中文键盘布局 (0x08040804) 与 TSF 中文 Compartments
        u32.ActivateKeyboardLayout(0x08040804, 0)
        _activate_tsf_chinese_compartments()

        # 2. 如果需要，直接将 Win32 键盘焦点赋予目标输入控件
        if set_win_focus:
            u32.SetFocus(hwnd)

        # 3. 对目标控件、当前焦点窗口及父级窗口全部关联默认 IME 上下文并开启中文模式
        targets = [hwnd]
        cur_focus = u32.GetFocus()
        if cur_focus and cur_focus not in targets:
            targets.append(cur_focus)
        if extra_hwnds:
            for eh in extra_hwnds:
                if eh and eh not in targets:
                    targets.append(eh)

        for h in targets:
            imm.ImmAssociateContextEx(h, None, 0x0010)  # IACE_DEFAULT
            himc = imm.ImmGetContext(h)
            if himc:
                imm.ImmSetOpenStatus(himc, 1)
                # IME_CMODE_NATIVE (0x0001) | IME_CMODE_SYMBOL (0x0400)
                imm.ImmSetConversionStatus(himc, 0x0001 | 0x0400, 0)
                imm.ImmReleaseContext(h, himc)
    except Exception:
        pass


class NativeWin32RichEdit:
    """
    在 Tkinter 容器内嵌入真正的 Windows 11 原生 RICHEDIT50W (Msftedit.dll) 富文本输入控件。
    相比 Tcl/Tk 自绘的 TkChild (无 ITfTextStoreACP / TSF 文档管理器支持，TSF pdoc=0x0)，
    RICHEDIT50W 启用 SES_USECTF 后具备微软官方原生 TSF (ITfDocumentMgr) 与 IMM32 双重支持，
    100% 兼容 Windows 11 微软拼音 (TextInputHost.exe)、搜狗拼音、微信输入法等中文输入法。
    """

    def __init__(
        self,
        parent_frame: Any,
        root_tk: Any,
        on_submit: Callable[[], None],
        height_px: int = 64,
        single_line: bool = False,
    ) -> None:
        self.parent_frame = parent_frame
        self.root_tk = root_tk
        self.on_submit = on_submit
        self.single_line = single_line
        self.hwnd: int = 0
        self._fallback_text: Any = None
        self._wndproc_ref: Any = None
        self._orig_wndproc: Any = None

        if os.name == "nt":
            try:
                self._create_win32_richedit(height_px)
                return
            except Exception:
                self.hwnd = 0

        self._create_fallback_tk(height_px)

    def _apply_charformat(self) -> None:
        if not self.hwnd or os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        u32 = ctypes.windll.user32

        class CHARFORMAT2W(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwMask", wintypes.DWORD),
                ("dwEffects", wintypes.DWORD),
                ("yHeight", wintypes.LONG),
                ("yOffset", wintypes.LONG),
                ("crTextColor", wintypes.DWORD),
                ("bCharSet", wintypes.BYTE),
                ("bPitchAndFamily", wintypes.BYTE),
                ("szFaceName", wintypes.WCHAR * 32),
                ("wWeight", wintypes.WORD),
                ("sSpacing", wintypes.SHORT),
                ("crBackColor", wintypes.DWORD),
                ("lcid", wintypes.LCID),
                ("dwReserved", wintypes.DWORD),
                ("sStyle", wintypes.SHORT),
                ("wKerning", wintypes.WORD),
                ("bUnderlineType", wintypes.BYTE),
                ("bAnimation", wintypes.BYTE),
                ("bRevAuthor", wintypes.BYTE),
                ("bUnderlineColor", wintypes.BYTE),
            ]

        cf = CHARFORMAT2W()
        cf.cbSize = ctypes.sizeof(CHARFORMAT2W)
        # CFM_COLOR (0x40000000) | CFM_FACE (0x20000000) | CFM_SIZE (0x80000000) | CFM_CHARSET (0x08000000)
        cf.dwMask = 0x40000000 | 0x20000000 | 0x80000000 | 0x08000000
        cf.dwEffects = 0
        cf.yHeight = 12 * 20  # 12pt in twips
        cf.crTextColor = 0x00FFFFFF  # 纯白文字 (COLORREF 0x00bbggrr)
        cf.bCharSet = 134  # GB2312_CHARSET
        cf.szFaceName = "Microsoft YaHei UI"

        u32.SendMessageW(self.hwnd, 0x0444, 0, ctypes.addressof(cf))  # EM_SETCHARFORMAT, SCF_DEFAULT
        u32.SendMessageW(self.hwnd, 0x0444, 4, ctypes.addressof(cf))  # EM_SETCHARFORMAT, SCF_ALL

    def _create_win32_richedit(self, height_px: int) -> None:
        import ctypes
        from ctypes import wintypes

        _init_win32_ctypes()
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        imm = ctypes.windll.imm32

        self.parent_frame.update_idletasks()
        w = max(240, self.parent_frame.winfo_width())
        h = max(32, height_px)

        WS_CHILD = 0x40000000
        WS_VISIBLE = 0x10000000
        WS_TABSTOP = 0x00010000
        ES_MULTILINE = 0x0004
        ES_AUTOVSCROLL = 0x0040
        ES_AUTOHSCROLL = 0x0080
        ES_WANTRETURN = 0x1000

        style = WS_CHILD | WS_VISIBLE | WS_TABSTOP
        if self.single_line:
            style |= ES_AUTOHSCROLL
        else:
            style |= ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN

        hwnd = u32.CreateWindowExW(
            0,
            "RICHEDIT50W",
            "",
            style,
            0,
            0,
            w,
            h,
            self.parent_frame.winfo_id(),
            0,
            k32.GetModuleHandleW(None),
            None,
        )
        if not hwnd:
            raise RuntimeError("CreateWindowExW(RICHEDIT50W) failed")
        self.hwnd = hwnd

        # 1. 纯文本模式 (防止粘贴富文本时改变背景或字色) + 多级撤销 + 多代码页支持
        u32.SendMessageW(hwnd, 0x0459, 1 | 4 | 32, 0)  # EM_SETTEXTMODE: TM_PLAINTEXT | TM_MULTILEVELUNDO | TM_MULTICODEPAGE

        # 2. 开启 Windows TSF (Cicero Text Services Framework) 原生支持
        # SES_USECTF (0x10000) | SES_CTFALLOWEMBED (0x200000) | SES_CTFALLOWSMARTTAG (0x400000) | SES_CTFALLOWPROOFING (0x800000)
        ctf_flags = 0x00010000 | 0x00200000 | 0x00400000 | 0x00800000
        u32.SendMessageW(hwnd, 0x04CC, ctf_flags, ctf_flags)  # EM_SETEDITSTYLE

        # 3. 深色背景 (#0f172a -> COLORREF 0x002a170f) 与雅黑白色字体
        u32.SendMessageW(hwnd, 0x0443, 0, 0x002A170F)  # EM_SETBKGNDCOLOR
        self._apply_charformat()

        # 4. 子类化窗口过程：处理 Enter 发送 vs Shift+Enter 换行 vs 输入法候选词上屏
        WNDPROC = ctypes.WINFUNCTYPE(
            wintypes.LPARAM,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        u32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, WNDPROC]
        u32.SetWindowLongPtrW.restype = ctypes.c_void_p

        def _wndproc(h_wnd: int, msg: int, wp: int, lp: int) -> int:
            # WM_KEYDOWN (0x0100) + VK_RETURN (0x0D)
            if msg == 0x0100 and wp == 0x0D:
                shift_down = (u32.GetKeyState(0x10) & 0x8000) != 0
                composing = False
                himc = imm.ImmGetContext(h_wnd)
                if himc:
                    comp_len = imm.ImmGetCompositionStringW(himc, 0x0008, None, 0)  # GCS_COMPSTR
                    imm.ImmReleaseContext(h_wnd, himc)
                    if comp_len and comp_len > 0:
                        composing = True
                if not composing and (self.single_line or not shift_down):
                    self.root_tk.after(10, self.on_submit)
                    return 0
            # 拦截非 Shift+Enter 产生的 WM_CHAR(\r / \n)，避免提示音或残留空行
            elif msg == 0x0102 and wp in (0x0D, 0x0A):
                shift_down = (u32.GetKeyState(0x10) & 0x8000) != 0
                if self.single_line or not shift_down:
                    return 0
            # WM_KEYDOWN + VK_ESCAPE (0x1B) 在单行弹窗模式下关闭窗口
            elif msg == 0x0100 and wp == 0x1B and self.single_line:
                self.root_tk.after(10, self.root_tk.destroy)
                return 0
            # WM_SETFOCUS (0x0007): 先执行 RichEdit 原生 WM_SETFOCUS 激活 TSF DocumentMgr，再确保开启中文模式
            elif msg == 0x0007:
                res = u32.CallWindowProcW(self._orig_wndproc, h_wnd, msg, wp, lp)
                self.root_tk.after(
                    10,
                    lambda: force_enable_chinese_ime(
                        h_wnd,
                        extra_hwnds=[self.root_tk.winfo_id(), u32.GetParent(self.root_tk.winfo_id())],
                        set_win_focus=False,
                    ),
                )
                return res

            return u32.CallWindowProcW(self._orig_wndproc, h_wnd, msg, wp, lp)

        self._wndproc_ref = WNDPROC(_wndproc)
        self._orig_wndproc = u32.SetWindowLongPtrW(hwnd, -4, self._wndproc_ref)

        def _on_configure(event: Any) -> None:
            if self.hwnd:
                u32.MoveWindow(self.hwnd, 0, 0, max(60, event.width), max(28, event.height), True)

        self.parent_frame.bind("<Configure>", _on_configure)
        self.parent_frame.bind("<Button-1>", lambda _e: self.focus_ime())

    def _create_fallback_tk(self, height_px: int) -> None:
        import tkinter as tk

        txt = tk.Text(
            self.parent_frame,
            height=max(1, height_px // 22),
            bg="#0f172a",
            fg="#ffffff",
            insertbackground="#4ade80",
            font=("Microsoft YaHei UI", 12),
            relief="flat",
            wrap="word",
        )
        txt.pack(fill="both", expand=True)

        def _on_ret(_e: Any = None) -> str:
            self.root_tk.after(15, self.on_submit)
            return "break"

        txt.bind("<Return>", _on_ret)
        self._fallback_text = txt
        self.hwnd = txt.winfo_id()

    def get_text(self) -> str:
        if self._fallback_text is not None:
            return self._fallback_text.get("1.0", "end-1c")
        if not self.hwnd or os.name != "nt":
            return ""
        import ctypes

        u32 = ctypes.windll.user32
        length = u32.SendMessageW(self.hwnd, 0x000E, 0, 0)  # WM_GETTEXTLENGTH
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 4)
        u32.SendMessageW(self.hwnd, 0x000D, length + 2, ctypes.addressof(buf))  # WM_GETTEXT
        return buf.value

    def set_text(self, text: str) -> None:
        if self._fallback_text is not None:
            self._fallback_text.delete("1.0", "end")
            if text:
                self._fallback_text.insert("1.0", text)
            return
        if not self.hwnd or os.name != "nt":
            return
        import ctypes

        u32 = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(text)
        u32.SendMessageW(self.hwnd, 0x000C, 0, ctypes.addressof(buf))  # WM_SETTEXT
        self._apply_charformat()
        if text:
            u32.SendMessageW(self.hwnd, 0x00B1, -1, -1)  # EM_SETSEL to end

    def focus_ime(self) -> None:
        if self._fallback_text is not None:
            self._fallback_text.focus_force()
            force_enable_chinese_ime(self.hwnd)
            return
        if not self.hwnd or os.name != "nt":
            return
        import ctypes

        u32 = ctypes.windll.user32
        top_hwnd = u32.GetParent(self.root_tk.winfo_id())
        force_enable_chinese_ime(
            self.hwnd,
            extra_hwnds=[self.root_tk.winfo_id(), top_hwnd],
            set_win_focus=True,
        )


def read_user_input(prompt: str) -> str:
    """读取终端用户输入"""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if sys.stdin.isatty():
        return input()
    raw = sys.stdin.buffer.readline()
    if not raw:
        raise EOFError
    try:
        return raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace").rstrip("\r\n")


C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_BLUE = "\033[94m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"


def ensure_gateway_running(gateway_url: str) -> None:
    parsed = urlparse(gateway_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18080
    if host not in ("127.0.0.1", "localhost"):
        return
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return
    except Exception:
        pass

    gw_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".agents",
        "skills",
        "colab-llm-bluegreen",
        "scripts",
        "llm_local_gateway.py",
    )
    if not os.path.exists(gw_script):
        return

    print(f"{C_YELLOW}⚡ 正在唤醒本机蓝绿网关 ({host}:{port}) 与 Colab SSH 隧道...{C_RESET}", flush=True)
    kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000 | 0x00000200 | 0x00000008
    else:
        kwargs["start_new_session"] = True
    log_path = os.path.join(os.path.expanduser("~/.config/colab-cli"), "llm_local_gateway.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    subprocess.Popen(
        [sys.executable, gw_script, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    for _ in range(20):
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/pilot/gateway")
            st = json.loads(conn.getresponse().read().decode("utf-8"))
            conn.close()
            if st.get("tunnel_open"):
                break
        except Exception:
            pass
        time.sleep(0.6)


def fetch_gateway_status(gateway_url: str) -> Dict[str, Any]:
    parsed = urlparse(gateway_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18080
    try:
        conn = http.client.HTTPConnection(host, port, timeout=4)
        conn.request("GET", "/pilot/gateway")
        gw = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    try:
        conn = http.client.HTTPConnection(host, port, timeout=4)
        conn.request("GET", "/pilot/metrics")
        remote = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        gw["remote"] = remote
    except Exception:
        pass
    return gw


def stream_chat_completion(
    gateway_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    show_stats: bool = True,
    on_token: Optional[Callable[[str], None]] = None,
    on_stats: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    ensure_gateway_running(gateway_url)
    parsed = urlparse(gateway_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18080

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    chunks_count = 0
    collected: List[str] = []

    conn = http.client.HTTPConnection(host, port, timeout=300)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer colab-pilot"},
    )
    resp = conn.getresponse()
    if resp.status >= 400:
        err_text = resp.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {err_text}")

    while True:
        raw_line = resp.fp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            obj = json.loads(data_str)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                chunks_count += 1
                collected.append(piece)
                print(piece, end="", flush=True)
                if on_token is not None:
                    on_token(piece)
        except Exception:
            pass

    conn.close()
    t_end = time.perf_counter()
    print(flush=True)

    if show_stats:
        total_s = max(0.001, t_end - t_start)
        ttft_ms = (t_first - t_start) * 1000.0 if t_first is not None else total_s * 1000.0
        gen_s = max(0.001, t_end - (t_first or t_start))
        tps = chunks_count / gen_s if chunks_count > 1 else chunks_count / total_s
        st = fetch_gateway_status(gateway_url)
        slot = (st.get("active_slot") or "blue").upper()
        slot_color = C_BLUE if slot == "BLUE" else C_GREEN
        remote = st.get("remote") or {}
        sidecar_tps = remote.get("last_request_tps") or remote.get("instant_tps") or round(tps, 2)
        print(
            f"{C_DIM}────────────────────────────────────────────────────────────────────\n"
            f"⚡ 槽位: {slot_color}[{slot}]{C_RESET}{C_DIM} | "
            f"输出分段: {chunks_count} | "
            f"首字延迟 (TTFT): {ttft_ms:.0f} ms | "
            f"实测速率: {C_CYAN}{sidecar_tps} tok/s{C_RESET}{C_DIM} | "
            f"总耗时: {total_s:.2f}s{C_RESET}",
            flush=True,
        )
        if on_stats is not None:
            stats_plain = (
                f"⚡ 槽位: [{slot}] | 输出分段: {chunks_count} | "
                f"首字延迟 (TTFT): {ttft_ms:.0f} ms | 实测速率: {sidecar_tps} tok/s | 总耗时: {total_s:.2f}s"
            )
            on_stats(stats_plain, st)

    return "".join(collected)


def print_banner(gateway_url: str, model: str, max_tokens: int, web_url: str = "") -> None:
    st = fetch_gateway_status(gateway_url)
    slot = (st.get("active_slot") or "blue").upper()
    slot_badge = f"{C_BLUE}[BLUE · ACTIVE]{C_RESET}" if slot == "BLUE" else f"{C_GREEN}[GREEN · ACTIVE]{C_RESET}"
    gpu = st.get("gpu") or "G4 (96GB)"
    tunnel_ok = f"{C_GREEN}● 已连通{C_RESET}" if st.get("tunnel_open") else f"{C_YELLOW}○ 连接中{C_RESET}"

    print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║{C_RESET}  ✦ {C_BOLD}Colab Pilot · Qwen/Qwen3.8-27B 本机终端客户端{C_RESET}                     {C_BOLD}{C_CYAN}║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"  • 本机网关 : {C_BOLD}{gateway_url}{C_RESET} ({tunnel_ok})")
    print(f"  • 云端模型 : {C_BOLD}{model}{C_RESET} @ Colab {C_MAGENTA}{gpu}{C_RESET}")
    print(f"  • 当前槽位 : {slot_badge}   | 最大输出: {max_tokens} tokens")
    if web_url:
        print(f"  • 网页面板 : {C_BOLD}{C_GREEN}{web_url}{C_RESET} (输入 /web 可直接唤起浏览器中文对话窗)")
    print(f"  • 快捷指令 : {C_DIM}/status 状态 | /web 网页版 | /file <路径> 读文件 | /max <N> 长度 | /clear 清空 | /switchover 蓝绿热切 | /exit 退出{C_RESET}")
    print(f"{C_DIM}────────────────────────────────────────────────────────────────────────{C_RESET}")


def format_status_text(st: Dict[str, Any]) -> str:
    if not st.get("ok"):
        return f"✗ 网关未响应: {st.get('error')}"
    remote = st.get("remote") or {}
    slot = (st.get("active_slot") or "blue").upper()
    lines = [
        "▸ [Colab Pilot 蓝绿网关与云端边车实时遥测]",
        f"  • 当前承载槽位 : [{slot}] (本地隧道端口 :{st.get('target_port')}, 连通={st.get('tunnel_open')})",
        f"  • 模型与引擎   : {st.get('model_name')} ({st.get('engine')}) | GPU: {st.get('gpu')}",
    ]
    if remote:
        lines.append(
            f"  • Token 速率   : 实时={remote.get('instant_tps')} tok/s | "
            f"上次请求={remote.get('last_request_tps')} tok/s | "
            f"60s均值={remote.get('avg_tps_60s')} tok/s | 峰值={remote.get('peak_tps')} tok/s"
        )
        lines.append(
            f"  • 延迟与计数   : 最新 TTFT={remote.get('ttft_ms')} ms | "
            f"累计请求={remote.get('total_requests')} | "
            f"累计生成={remote.get('total_completion_tokens')} tokens | 错误={remote.get('error_count')}"
        )
    return "\n".join(lines)


def print_status_panel(gateway_url: str) -> None:
    st = fetch_gateway_status(gateway_url)
    if not st.get("ok"):
        print(f"{C_RED}✗ 网关未响应: {st.get('error')}{C_RESET}")
        return
    remote = st.get("remote") or {}
    slot = (st.get("active_slot") or "blue").upper()
    print(f"\n{C_BOLD}▸ [Colab Pilot 蓝绿网关与云端边车实时遥测]{C_RESET}")
    print(f"  • 当前承载槽位 : [{slot}] (本地隧道端口 :{st.get('target_port')}, 连通={st.get('tunnel_open')})")
    print(f"  • 模型与引擎   : {st.get('model_name')} ({st.get('engine')}) | GPU: {st.get('gpu')}")
    if remote:
        print(
            f"  • Token 速率   : 实时={C_CYAN}{remote.get('instant_tps')} tok/s{C_RESET} | "
            f"上次请求={C_GREEN}{remote.get('last_request_tps')} tok/s{C_RESET} | "
            f"60s均值={remote.get('avg_tps_60s')} tok/s | 峰值={remote.get('peak_tps')} tok/s"
        )
        print(
            f"  • 延迟与计数   : 最新 TTFT={remote.get('ttft_ms')} ms | "
            f"累计请求={remote.get('total_requests')} | "
            f"累计生成={remote.get('total_completion_tokens')} tokens | 错误={remote.get('error_count')}"
        )
    print()


# ====================================================================
# 内置本地网页对话服务 (100% 浏览器级输入法支持 + 三端实时同步)
# ====================================================================
_WEB_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Colab Pilot — Qwen/Qwen3.8-27B 本地中文对话面板</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; color: #f8fafc; font-family: "Microsoft YaHei UI", system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
  header { background: #090d16; padding: 12px 18px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; }
  .title { color: #38bdf8; font-weight: 700; font-size: 16px; }
  .badge { color: #4ade80; font-size: 13px; }
  .toolbar { background: #1e293b; padding: 8px 18px; display: flex; gap: 8px; flex-wrap: wrap; }
  .toolbar button { background: #334155; color: #e2e8f0; border: none; border-radius: 4px; padding: 5px 12px; cursor: pointer; font-size: 12px; font-family: inherit; }
  .toolbar button:hover { background: #475569; color: #fff; }
  #chat { flex: 1; overflow-y: auto; padding: 16px 20px; white-space: pre-wrap; line-height: 1.6; font-size: 15px; }
  .msg-user { color: #4ade80; font-weight: 700; margin-top: 12px; }
  .msg-user-body { color: #ffffff; margin-bottom: 6px; }
  .msg-bot { color: #38bdf8; font-weight: 700; margin-top: 6px; }
  .msg-bot-body { color: #e2e8f0; }
  .msg-stats { color: #94a3b8; font-family: Consolas, monospace; font-size: 12px; margin: 6px 0 12px 0; border-top: 1px solid #1e293b; padding-top: 4px; }
  .msg-sys { color: #fbbf24; margin: 8px 0; font-size: 13px; }
  footer { background: #1e293b; padding: 12px 18px; display: flex; gap: 10px; align-items: stretch; }
  textarea { flex: 1; background: #0f172a; color: #fff; border: 1px solid #334155; border-radius: 6px; padding: 10px 12px; font-size: 15px; font-family: inherit; resize: none; height: 68px; outline: none; }
  textarea:focus { border-color: #38bdf8; }
  #send { background: #0284c7; color: #fff; border: none; border-radius: 6px; padding: 0 22px; font-weight: 700; font-size: 14px; cursor: pointer; }
  #send:hover { background: #0369a1; }
</style>
</head>
<body>
<header>
  <div class="title">🚀 Colab Pilot · Qwen/Qwen3.8-27B 本地中文客户端</div>
  <div class="badge" id="badge">● 已连接本机蓝绿网关 (终端/GUI/网页三端实时同步)</div>
</header>
<div class="toolbar">
  <button onclick="sendCmd('/status')">📊 实时状态 (/status)</button>
  <button onclick="sendCmd('/clear')">🧹 清空对话 (/clear)</button>
  <button onclick="sendCmd('/switchover')">🔄 蓝绿热切 (/switchover)</button>
  <button onclick="document.getElementById('inp').focus()">🀄 聚焦中文输入框</button>
</div>
<div id="chat"></div>
<footer>
  <textarea id="inp" placeholder="请直接使用中文输入法打字提问 (Enter 发送，Shift+Enter 换行)..." autofocus></textarea>
  <button id="send" onclick="submitInput()">发送<br>(Enter)</button>
</footer>
<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const badge = document.getElementById('badge');
let isComposing = false;
let lastSeq = 0;

inp.addEventListener('compositionstart', () => { isComposing = true; });
inp.addEventListener('compositionend', () => { isComposing = false; });
inp.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !isComposing && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    submitInput();
  }
});

async function submitInput() {
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  inp.focus();
  await fetch('/api/send', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
}

async function sendCmd(cmd) {
  await fetch('/api/send', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: cmd})
  });
  inp.focus();
}

async function poll() {
  try {
    const r = await fetch('/api/events?since=' + lastSeq);
    const data = await r.json();
    if (data.badge) badge.textContent = data.badge;
    if (data.reset) { chat.innerHTML = ''; }
    for (const ev of (data.events || [])) {
      lastSeq = ev.seq;
      const span = document.createElement('span');
      span.className = ev.cls || '';
      span.textContent = ev.text;
      chat.appendChild(span);
      chat.scrollTop = chat.scrollHeight;
    }
  } catch (e) {}
  setTimeout(poll, 120);
}
poll();
</script>
</body>
</html>
"""


class LocalWebMirror:
    """内置轻量级本地网页端镜像服务，提供 100% 浏览器原生中文输入法支持并与 GUI/CLI 实时同步"""

    def __init__(self, on_submit_text: Callable[[str], None]) -> None:
        self.on_submit_text = on_submit_text
        self.lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self.seq = 0
        self.badge_text = "● 已连接本机蓝绿网关"
        self.url = ""
        self._server: Optional[ThreadingHTTPServer] = None

    def append(self, text: str, tag: str = "bot_text") -> None:
        cls_map = {
            "banner": "msg-sys",
            "user_prefix": "msg-user",
            "user_text": "msg-user-body",
            "bot_prefix": "msg-bot",
            "bot_text": "msg-bot-body",
            "stats": "msg-stats",
            "sys_info": "msg-sys",
            "error": "msg-sys",
        }
        with self.lock:
            self.seq += 1
            self.events.append({"seq": self.seq, "text": text, "cls": cls_map.get(tag, "msg-bot-body")})

    def clear(self) -> None:
        with self.lock:
            self.events.clear()
            self.seq += 1
            self.events.append({"seq": self.seq, "text": "🧹 已清空多轮对话历史。\n", "cls": "msg-sys", "reset": True})

    def set_badge(self, text: str) -> None:
        with self.lock:
            self.badge_text = text

    def start(self) -> str:
        mirror = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/" or self.path.startswith("/index"):
                    body = _WEB_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/api/events"):
                    since = 0
                    if "since=" in self.path:
                        try:
                            since = int(self.path.split("since=", 1)[1].split("&")[0])
                        except ValueError:
                            since = 0
                    with mirror.lock:
                        evs = [e for e in mirror.events if e["seq"] > since]
                        reset = any(e.get("reset") for e in evs)
                        payload = json.dumps(
                            {"events": evs, "badge": mirror.badge_text, "reset": reset},
                            ensure_ascii=False,
                        ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                if self.path == "/api/send":
                    ln = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(ln).decode("utf-8", errors="replace")
                    try:
                        obj = json.loads(raw)
                        txt = (obj.get("text") or "").strip()
                        if txt:
                            mirror.on_submit_text(txt)
                    except Exception:
                        pass
                    resp = b'{"ok":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return
                self.send_response(404)
                self.end_headers()

        for port in range(18088, 18099):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                self._server = srv
                self.url = f"http://127.0.0.1:{port}"
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                return self.url
            except Exception:
                continue
        return ""

    def open_browser(self) -> None:
        if not self.url:
            return
        if os.name == "nt":
            for edge_path in (
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ):
                if os.path.exists(edge_path):
                    try:
                        subprocess.Popen([edge_path, f"--app={self.url}"])
                        return
                    except Exception:
                        pass
        webbrowser.open(self.url)


def popup_chinese_input_dialog() -> Optional[str]:
    """弹出原生 Win32 RichEdit50W 中文输入框（供纯命令行模式下需要中文输入法时调用）"""
    try:
        import tkinter as tk
    except Exception:
        return None

    result: List[Optional[str]] = [None]
    root = tk.Tk()
    root.title("Qwen3.8-27B 原生中文输入法输入框 (Enter 发送 / Esc 取消)")
    root.configure(bg="#0f172a")
    root.geometry("640x135")
    root.attributes("-topmost", True)

    lbl = tk.Label(
        root,
        text="💬 请使用中文输入法输入问题 (原生 Win32 RichEdit50W 控件，按 Enter 直接发送):",
        bg="#0f172a",
        fg="#4ade80",
        font=("Microsoft YaHei UI", 10, "bold"),
        anchor="w",
    )
    lbl.pack(fill="x", padx=14, pady=(12, 4))

    box_border = tk.Frame(root, bg="#38bdf8", padx=1, pady=1)
    box_border.pack(fill="x", padx=14, pady=6)
    box_inner = tk.Frame(box_border, bg="#0f172a", padx=8, pady=6)
    box_inner.pack(fill="x")
    rich_holder = tk.Frame(box_inner, bg="#0f172a", height=36)
    rich_holder.pack(fill="x")

    rich_input: List[Optional[NativeWin32RichEdit]] = [None]

    def _submit() -> None:
        if rich_input[0] is not None:
            result[0] = rich_input[0].get_text().strip()
        root.destroy()

    rich_input[0] = NativeWin32RichEdit(
        parent_frame=rich_holder,
        root_tk=root,
        on_submit=_submit,
        height_px=36,
        single_line=True,
    )
    root.bind("<Escape>", lambda _e: root.destroy())

    def _init_ime() -> None:
        root.deiconify()
        root.lift()
        root.focus_force()
        if rich_input[0] is not None:
            rich_input[0].focus_ime()

    root.after(50, _init_ime)
    root.mainloop()
    return result[0]


def run_gui_repl(gateway_url: str, model: str, system_prompt: str, max_tokens: int) -> int:
    """
    启动原生 Win32 RichEdit50W 中文 GUI 窗口 + 内置网页端 + 命令行终端三端实时同步模式。
    解决 Windows 11 终端下 TSF/IME 无法弹出中文候选框以及 Tkinter 自绘控件无 TSF 文档管理器的问题：
    1. 在输入框区域嵌入真正的 Windows 11 RICHEDIT50W (Msftedit.dll) 原生控件并启用 SES_USECTF。
    2. 通过 MSCTF (ITfThreadMgr / ITfCompartmentMgr) 与 IMM32 双通道强制开启中文输入状态。
    3. 内置本地网页镜像面板 (http://127.0.0.1:18088)，支持在 GUI、网页端、命令行三端同步输入与输出。
    """
    try:
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        import tkinter as tk
        from tkinter import filedialog, scrolledtext
    except Exception as e:
        print(f"{C_YELLOW}! 无法初始化图形窗口 ({e})，回退至纯终端模式...{C_RESET}")
        return run_cli_repl(gateway_url, model, system_prompt, max_tokens)

    ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    web_mirror = LocalWebMirror(on_submit_text=lambda txt: ui_queue.put(("stdin_line", txt)))
    web_url = web_mirror.start()

    ensure_gateway_running(gateway_url)
    print_banner(gateway_url, model, max_tokens, web_url=web_url)
    print(
        f"{C_GREEN}✓ 已弹出【Qwen3.8-27B 原生中文对话窗口 (Win32 RichEdit50W + TSF)】！{C_RESET}\n"
        f"{C_DIM}  • 方式 A (推荐): 直接在弹出的原生对话窗口底部输入框中使用中文输入法打字；\n"
        f"  • 方式 B (网页): 访问 {web_url} 或输入 /web 唤起 Edge/浏览器中文对话面板；\n"
        f"  • 方式 C (终端): 在下方 You > 提示符直接输入（直接按回车可将中文窗口置顶聚焦）。{C_RESET}\n",
        flush=True,
    )
    sys.stdout.write(f"{C_BOLD}{C_GREEN}You > {C_RESET}")
    sys.stdout.flush()

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    busy_lock = threading.Lock()
    state: Dict[str, Any] = {"max_tokens": max_tokens, "system_prompt": system_prompt, "busy": False}

    root = tk.Tk()
    root.title(f"Colab Pilot — {model} 本机中文交互客户端")
    root.geometry("920x680")
    root.minsize(700, 500)
    root.configure(bg="#0f172a")

    # 居中显示
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = max(0, (sw - 920) // 2)
    y = max(0, (sh - 680) // 2)
    root.geometry(f"920x680+{x}+{y}")
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))

    # 顶部状态栏
    header = tk.Frame(root, bg="#090d16", padx=14, pady=10)
    header.pack(fill="x")

    title_lbl = tk.Label(
        header,
        text=f"🚀 Colab Pilot · {model}",
        bg="#090d16",
        fg="#38bdf8",
        font=("Microsoft YaHei UI", 12, "bold"),
    )
    title_lbl.pack(side="left")

    status_badge = tk.Label(
        header,
        text="● 正在连接本机蓝绿网关...",
        bg="#090d16",
        fg="#4ade80",
        font=("Microsoft YaHei UI", 9),
    )
    status_badge.pack(side="right")

    # 快捷功能栏
    toolbar = tk.Frame(root, bg="#1e293b", padx=10, pady=6)
    toolbar.pack(fill="x")

    def _btn(parent: Any, text: str, cmd: Callable[[], None], fg: str = "#e2e8f0") -> Any:
        b = tk.Button(
            parent,
            text=text,
            command=cmd,
            bg="#334155",
            fg=fg,
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=3,
            font=("Microsoft YaHei UI", 9),
            cursor="hand2",
        )
        b.pack(side="left", padx=4)
        return b

    # 聊天历史区域
    chat_frame = tk.Frame(root, bg="#0f172a", padx=12, pady=8)
    chat_frame.pack(fill="both", expand=True)

    chat_box = scrolledtext.ScrolledText(
        chat_frame,
        bg="#0f172a",
        fg="#f8fafc",
        insertbackground="#38bdf8",
        font=("Microsoft YaHei UI", 11),
        relief="flat",
        wrap="word",
        padx=10,
        pady=10,
        state="disabled",
    )
    chat_box.pack(fill="both", expand=True)

    chat_box.tag_config("banner", foreground="#38bdf8", font=("Microsoft YaHei UI", 10, "bold"))
    chat_box.tag_config("user_prefix", foreground="#4ade80", font=("Microsoft YaHei UI", 11, "bold"))
    chat_box.tag_config("user_text", foreground="#ffffff", font=("Microsoft YaHei UI", 11))
    chat_box.tag_config("bot_prefix", foreground="#38bdf8", font=("Microsoft YaHei UI", 11, "bold"))
    chat_box.tag_config("bot_text", foreground="#e2e8f0", font=("Microsoft YaHei UI", 11))
    chat_box.tag_config("stats", foreground="#94a3b8", font=("Consolas", 9))
    chat_box.tag_config("sys_info", foreground="#fbbf24", font=("Microsoft YaHei UI", 10))
    chat_box.tag_config("error", foreground="#f87171", font=("Microsoft YaHei UI", 10, "bold"))

    # 底部输入区域 (嵌入原生 Win32 RICHEDIT50W 控件)
    bottom = tk.Frame(root, bg="#1e293b", padx=12, pady=10)
    bottom.pack(fill="x", side="bottom")

    hint_bar = tk.Frame(bottom, bg="#1e293b")
    hint_bar.pack(fill="x", pady=(0, 6))
    tk.Label(
        hint_bar,
        text="💬 原生 Win32 RichEdit50W 中文输入区 (已开启 TSF/IME 中文模式 | Enter 发送 | Shift+Enter 换行):",
        bg="#1e293b",
        fg="#94a3b8",
        font=("Microsoft YaHei UI", 9),
    ).pack(side="left")

    input_row = tk.Frame(bottom, bg="#1e293b")
    input_row.pack(fill="x")

    input_border = tk.Frame(input_row, bg="#38bdf8", padx=1, pady=1)
    input_border.pack(side="left", fill="x", expand=True, padx=(0, 8))
    input_padding = tk.Frame(input_border, bg="#0f172a", padx=10, pady=8)
    input_padding.pack(fill="both", expand=True)
    rich_holder = tk.Frame(input_padding, bg="#0f172a", height=62)
    rich_holder.pack(fill="both", expand=True)

    send_btn = tk.Button(
        input_row,
        text="发送\n(Enter)",
        bg="#0284c7",
        fg="#ffffff",
        activebackground="#0369a1",
        activeforeground="#ffffff",
        relief="flat",
        width=10,
        height=3,
        font=("Microsoft YaHei UI", 10, "bold"),
        cursor="hand2",
    )
    send_btn.pack(side="right", fill="y")

    def _append_chat(text: str, tag: str = "bot_text") -> None:
        chat_box.configure(state="normal")
        chat_box.insert("end", text, tag)
        chat_box.see("end")
        chat_box.configure(state="disabled")
        web_mirror.append(text, tag)

    def _update_badge(st: Dict[str, Any]) -> None:
        slot = (st.get("active_slot") or "blue").upper()
        gpu = st.get("gpu") or "G4 (96GB)"
        remote = st.get("remote") or {}
        tps = remote.get("last_request_tps") or remote.get("instant_tps") or "--"
        ok = st.get("tunnel_open")
        dot = "● 已连通" if ok else "○ 连接中"
        b_txt = f"[{slot} · ACTIVE] | Colab {gpu} | {dot} | 速率: {tps} tok/s"
        status_badge.configure(
            text=b_txt,
            fg="#38bdf8" if slot == "BLUE" else "#4ade80",
        )
        web_mirror.set_badge(b_txt)

    rich_box_ref: List[Optional[NativeWin32RichEdit]] = [None]

    def _activate_ime(_event: Any = None) -> None:
        if rich_box_ref[0] is not None:
            rich_box_ref[0].focus_ime()

    def _bring_window_to_front() -> None:
        try:
            root.deiconify()
            root.lift()
            root.attributes("-topmost", True)
            root.focus_force()
            if os.name == "nt":
                import ctypes
                u32 = ctypes.windll.user32
                top_hwnd = u32.GetParent(root.winfo_id())
                if top_hwnd:
                    u32.BringWindowToTop(top_hwnd)
                    u32.SetForegroundWindow(top_hwnd)
            _activate_ime()
            root.after(400, lambda: root.attributes("-topmost", False))
        except Exception:
            pass

    def _reprint_cli_prompt() -> None:
        sys.stdout.write(f"\n{C_BOLD}{C_GREEN}You > {C_RESET}")
        sys.stdout.flush()

    def _handle_command_or_chat(raw_text: str) -> None:
        user_input = raw_text.strip()
        if not user_input:
            return

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"\n{C_DIM}✓ 已退出 Qwen3.8-27B 客户端（云端实例保持在线）。{C_RESET}", flush=True)
            root.destroy()
            return

        if user_input.lower() in ("/zh", "/ime", "/gui"):
            _bring_window_to_front()
            _reprint_cli_prompt()
            return

        if user_input.lower() == "/web":
            web_mirror.open_browser()
            msg = f"🌐 已在浏览器中打开中文对话面板: {web_url}\n"
            _append_chat(f"\n{msg}", "sys_info")
            print(f"\n{C_GREEN}{msg.strip()}{C_RESET}", flush=True)
            _reprint_cli_prompt()
            return

        if user_input.lower() == "/help":
            help_txt = (
                "内置指令列表：\n"
                "  /status              查看本机网关、SSH 隧道与云端实时 tok/s 指标\n"
                f"  /web                 在浏览器/Edge 独立窗口打开中文对话面板 ({web_url})\n"
                "  /file <路径> [提问]  读取本地文件并发送给 Qwen3.8-27B 分析\n"
                "  /max <数量>          修改单次回复最大 Token 数 (当前 "
                f"{state['max_tokens']})\n"
                "  /system <提示词>     设置或修改 System Prompt\n"
                "  /clear               清空多轮对话上下文历史\n"
                "  /switchover          在后台触发 Make-Before-Break 蓝绿热切换\n"
                "  /exit                退出客户端\n"
            )
            _append_chat(f"\n{help_txt}\n", "sys_info")
            print(f"\n{C_YELLOW}{help_txt}{C_RESET}", flush=True)
            _reprint_cli_prompt()
            return

        if user_input.lower() == "/status":
            def _bg_status() -> None:
                st = fetch_gateway_status(gateway_url)
                ui_queue.put(("status", st))
            threading.Thread(target=_bg_status, daemon=True).start()
            return

        if user_input.lower() == "/clear":
            messages.clear()
            if state["system_prompt"]:
                messages.append({"role": "system", "content": state["system_prompt"]})
            chat_box.configure(state="normal")
            chat_box.delete("1.0", "end")
            chat_box.configure(state="disabled")
            web_mirror.clear()
            _append_chat("✓ 已清空多轮对话历史。\n", "sys_info")
            print(f"\n{C_YELLOW}✓ 已清空多轮对话历史。{C_RESET}", flush=True)
            _reprint_cli_prompt()
            return

        if user_input.lower().startswith("/max "):
            try:
                state["max_tokens"] = max(16, int(user_input.split(None, 1)[1].strip()))
                msg = f"✓ 最大输出长度已更新为: {state['max_tokens']} tokens\n"
                _append_chat(msg, "sys_info")
                print(f"\n{C_GREEN}{msg.strip()}{C_RESET}", flush=True)
            except ValueError:
                _append_chat("✗ 请输入有效的整数，例如: /max 1024\n", "error")
            _reprint_cli_prompt()
            return

        if user_input.lower().startswith("/system "):
            state["system_prompt"] = user_input.split(None, 1)[1].strip()
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = state["system_prompt"]
            else:
                messages.insert(0, {"role": "system", "content": state["system_prompt"]})
            _append_chat("✓ System Prompt 已更新。\n", "sys_info")
            print(f"\n{C_GREEN}✓ System Prompt 已更新。{C_RESET}", flush=True)
            _reprint_cli_prompt()
            return

        if user_input.lower() == "/switchover":
            mgr = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".agents",
                "skills",
                "colab-llm-bluegreen",
                "scripts",
                "llm_bluegreen_manager.py",
            )
            st = fetch_gateway_status(gateway_url)
            gpu = st.get("gpu") or "G4"
            msg = (
                f"⚡ 已在后台启动 Make-Before-Break 并行蓝绿热切换 (GPU={gpu})...\n"
                "   新槽位完成模型加载与真机 Warmup 冒烟前，您可继续无中断与当前槽位对话！\n"
            )
            _append_chat(f"\n{msg}", "sys_info")
            print(f"\n{C_YELLOW}{msg.strip()}{C_RESET}", flush=True)
            subprocess.Popen([sys.executable, mgr, "switchover", "--session-prefix", "llm", "--gpu", gpu])
            _reprint_cli_prompt()
            return

        display_input = user_input
        if user_input.lower().startswith("/file "):
            rest = user_input[6:].strip()
            parts = rest.split(None, 1)
            fpath = parts[0].strip("\"'")
            question = parts[1] if len(parts) > 1 else "请分析并总结该文件的内容："
            if not os.path.exists(fpath):
                _append_chat(f"✗ 找不到文件: {fpath}\n", "error")
                print(f"\n{C_RED}✗ 找不到文件: {fpath}{C_RESET}", flush=True)
                _reprint_cli_prompt()
                return
            try:
                content = open(fpath, "r", encoding="utf-8", errors="replace").read()
            except Exception as e:
                _append_chat(f"✗ 读取文件失败: {e}\n", "error")
                _reprint_cli_prompt()
                return
            display_input = f"[附加文件: {os.path.basename(fpath)}] {question}"
            user_input = f"{question}\n\n文件名: `{os.path.basename(fpath)}`\n```\n{content[:24000]}\n```"

        if not busy_lock.acquire(blocking=False):
            _append_chat("⏳ 上一条回复正在生成中，请稍候...\n", "sys_info")
            return

        state["busy"] = True
        send_btn.configure(state="disabled", text="生成中...")
        _append_chat("\nYou > ", "user_prefix")
        _append_chat(f"{display_input}\n", "user_text")
        _append_chat("Qwen3.8-27B > ", "bot_prefix")

        print(f"\n{C_BOLD}{C_GREEN}You > {C_RESET}{display_input}", flush=True)
        print(f"\n{C_BOLD}{C_CYAN}Qwen3.8-27B > {C_RESET}", end="", flush=True)

        def _worker() -> None:
            messages.append({"role": "user", "content": user_input})
            try:
                reply = stream_chat_completion(
                    gateway_url=gateway_url,
                    model=model,
                    messages=messages,
                    max_tokens=state["max_tokens"],
                    show_stats=True,
                    on_token=lambda piece: ui_queue.put(("token", piece)),
                    on_stats=lambda s_txt, st_obj: ui_queue.put(("stats", (s_txt, st_obj))),
                )
                messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                messages.pop()
                ui_queue.put(("error", str(e)))
                print(f"\n{C_RED}✗ 请求失败: {e}{C_RESET}", flush=True)
            finally:
                ui_queue.put(("done", None))
                state["busy"] = False
                busy_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _on_send(_event: Any = None) -> None:
        if rich_box_ref[0] is None:
            return
        raw = rich_box_ref[0].get_text()
        if not raw.strip():
            _activate_ime()
            return
        rich_box_ref[0].set_text("")
        _activate_ime()
        _handle_command_or_chat(raw)

    rich_box_ref[0] = NativeWin32RichEdit(
        parent_frame=rich_holder,
        root_tk=root,
        on_submit=_on_send,
        height_px=62,
        single_line=False,
    )
    input_padding.bind("<Button-1>", lambda _e: _activate_ime())
    send_btn.configure(command=_on_send)

    # 当主窗口获得焦点但焦点落在 root 容器上时，自动把焦点交还给原生 RichEdit50W 输入框
    def _on_root_focus_in(event: Any) -> None:
        if event.widget is root:
            root.after(15, _activate_ime)

    root.bind("<FocusIn>", _on_root_focus_in)

    def _pick_file() -> None:
        fpath = filedialog.askopenfilename(title="选择要发送给 Qwen3.8-27B 分析的文件")
        if fpath and rich_box_ref[0] is not None:
            cur = rich_box_ref[0].get_text().strip()
            q = cur if cur else "请分析并总结这个文件的内容"
            rich_box_ref[0].set_text(f'/file "{fpath}" {q}')
            _activate_ime()

    _btn(toolbar, "📊 实时状态 (/status)", lambda: _handle_command_or_chat("/status"))
    _btn(toolbar, "📂 附加文件 (/file)", _pick_file)
    _btn(toolbar, "🧹 清空对话 (/clear)", lambda: _handle_command_or_chat("/clear"))
    _btn(toolbar, "🔄 蓝绿热切 (/switchover)", lambda: _handle_command_or_chat("/switchover"))
    _btn(toolbar, "🌐 网页版打开 (/web)", web_mirror.open_browser, fg="#38bdf8")
    _btn(toolbar, "🀄 激活中文输入法", _activate_ime, fg="#4ade80")

    # 后台同时监听命令行标准输入，实现命令行、GUI 与网页三端均可输入
    def _stdin_listener() -> None:
        while True:
            try:
                if sys.stdin.isatty():
                    line = input()
                else:
                    raw = sys.stdin.buffer.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.strip():
                    ui_queue.put(("stdin_line", line.strip()))
                else:
                    ui_queue.put(("focus_gui", None))
            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                break

    threading.Thread(target=_stdin_listener, daemon=True).start()

    # 初始欢迎提示与网关状态获取
    _append_chat(
        f"✨ 欢迎使用 Colab Pilot · {model} 本地客户端！\n"
        "• 已启用原生 Win32 RichEdit50W + Windows 11 TSF 中文输入法引擎：请直接在下方输入框中使用拼音打字，按 Enter 发送。\n"
        f"• 同时已启动内置网页端 ({web_url})，您也可点击上方【🌐 网页版打开】在浏览器中对话。\n"
        "────────────────────────────────────────────────────────────────────\n",
        "banner",
    )

    def _init_status_and_ime() -> None:
        _bring_window_to_front()

        def _bg() -> None:
            st = fetch_gateway_status(gateway_url)
            ui_queue.put(("init_badge", st))

        threading.Thread(target=_bg, daemon=True).start()

    def _poll_with_init() -> None:
        try:
            while True:
                kind, payload = ui_queue.get_nowait()
                if kind == "init_badge":
                    _update_badge(payload)
                elif kind == "focus_gui":
                    _bring_window_to_front()
                    _reprint_cli_prompt()
                elif kind == "token":
                    _append_chat(payload, "bot_text")
                elif kind == "stats":
                    s_txt, st_obj = payload
                    _append_chat(
                        f"\n────────────────────────────────────────────────────────────────────\n{s_txt}\n",
                        "stats",
                    )
                    _update_badge(st_obj)
                elif kind == "status":
                    _update_badge(payload)
                    txt = format_status_text(payload)
                    _append_chat(f"\n{txt}\n", "sys_info")
                    print_status_panel(gateway_url)
                    _reprint_cli_prompt()
                elif kind == "error":
                    _append_chat(f"\n✗ 请求失败: {payload}\n", "error")
                elif kind == "stdin_line":
                    _handle_command_or_chat(payload)
                elif kind == "done":
                    send_btn.configure(state="normal", text="发送\n(Enter)")
                    _activate_ime()
                    _reprint_cli_prompt()
        except queue.Empty:
            pass
        root.after(20, _poll_with_init)

    root.after(80, _init_status_and_ime)
    root.after(20, _poll_with_init)
    root.mainloop()
    return 0


def run_cli_repl(gateway_url: str, model: str, system_prompt: str, max_tokens: int) -> int:
    ensure_gateway_running(gateway_url)
    print_banner(gateway_url, model, max_tokens)
    print(f"  • 中文输入 : {C_GREEN}若当前终端无法唤起中文输入法，输入 /zh 或直接回车可弹出原生 Win32 RichEdit 中文输入框{C_RESET}")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = read_user_input(f"\n{C_BOLD}{C_GREEN}You > {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_DIM}✓ 已退出 Qwen3.8-27B 客户端（云端实例保持在线）。{C_RESET}")
            break

        if not user_input or user_input.lower() in ("/zh", "/ime", "/input"):
            popped = popup_chinese_input_dialog()
            if not popped:
                continue
            user_input = popped
            print(f"{C_BOLD}{C_GREEN}You (中文输入) > {C_RESET}{user_input}", flush=True)

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"{C_DIM}✓ 已退出 Qwen3.8-27B 客户端（云端实例保持在线）。{C_RESET}")
            break

        if user_input.lower() in ("/gui", "/web"):
            return run_gui_repl(gateway_url, model, system_prompt, max_tokens)

        if user_input.lower() == "/help":
            print(
                f"\n{C_BOLD}内置指令列表：{C_RESET}\n"
                f"  {C_CYAN}/zh{C_RESET}                  弹出原生 Win32 RichEdit 中文输入框 (解决终端无法打中文汉字问题)\n"
                f"  {C_CYAN}/gui{C_RESET}                 切换至完整原生中文图形 + 网页双栈对话窗口\n"
                f"  {C_CYAN}/status{C_RESET}              查看本机网关、SSH 隧道与云端实时 tok/s 指标\n"
                f"  {C_CYAN}/file <路径> [提问]{C_RESET}  读取本地文件并发送给 Qwen3.8-27B 分析\n"
                f"  {C_CYAN}/max <数量>{C_RESET}          修改单次回复最大 Token 数 (当前 {max_tokens})\n"
                f"  {C_CYAN}/system <提示词>{C_RESET}     设置或修改 System Prompt\n"
                f"  {C_CYAN}/clear{C_RESET}               清空多轮对话上下文历史\n"
                f"  {C_CYAN}/switchover{C_RESET}          在后台触发 Make-Before-Break 蓝绿热切换\n"
                f"  {C_CYAN}/exit{C_RESET}                退出 CLI"
            )
            continue

        if user_input.lower() == "/status":
            print_status_panel(gateway_url)
            continue

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            print(f"{C_YELLOW}✓ 已清空多轮对话历史。{C_RESET}")
            continue

        if user_input.lower().startswith("/max "):
            try:
                max_tokens = max(16, int(user_input.split(None, 1)[1].strip()))
                print(f"{C_GREEN}✓ 最大输出长度已更新为: {max_tokens} tokens{C_RESET}")
            except ValueError:
                print(f"{C_RED}✗ 请输入有效的整数，例如: /max 1024{C_RESET}")
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input.split(None, 1)[1].strip()
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
            print(f"{C_GREEN}✓ System Prompt 已更新。{C_RESET}")
            continue

        if user_input.lower() == "/switchover":
            mgr = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".agents",
                "skills",
                "colab-llm-bluegreen",
                "scripts",
                "llm_bluegreen_manager.py",
            )
            st = fetch_gateway_status(gateway_url)
            gpu = st.get("gpu") or "G4"
            print(f"{C_YELLOW}⚡ 已在后台启动 Make-Before-Break 并行蓝绿热切换 (GPU={gpu})...{C_RESET}")
            print(f"{C_DIM}   新槽位完成模型加载与真机 Warmup 冒烟前，您可继续无中断与当前槽位对话！{C_RESET}")
            subprocess.Popen([sys.executable, mgr, "switchover", "--session-prefix", "llm", "--gpu", gpu])
            continue

        if user_input.lower().startswith("/file "):
            rest = user_input[6:].strip()
            parts = rest.split(None, 1)
            fpath = parts[0].strip("\"'")
            question = parts[1] if len(parts) > 1 else "请分析并总结该文件的内容："
            if not os.path.exists(fpath):
                print(f"{C_RED}✗ 找不到文件: {fpath}{C_RESET}")
                continue
            try:
                content = open(fpath, "r", encoding="utf-8", errors="replace").read()
            except Exception as e:
                print(f"{C_RED}✗ 读取文件失败: {e}{C_RESET}")
                continue
            user_input = f"{question}\n\n文件名: `{os.path.basename(fpath)}`\n```\n{content[:24000]}\n```"

        messages.append({"role": "user", "content": user_input})
        print(f"\n{C_BOLD}{C_CYAN}Qwen3.8-27B > {C_RESET}", end="", flush=True)
        try:
            reply = stream_chat_completion(
                gateway_url=gateway_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                show_stats=True,
            )
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            messages.pop()
            print(f"\n{C_RED}✗ 请求失败: {e}{C_RESET}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Colab Pilot — Qwen/Qwen3.8-27B 本机客户端 (支持原生 Win32 RichEdit50W 中文 GUI + 内置网页端 + 终端三端同步)"
    )
    parser.add_argument("prompt", nargs="*", help="直接提问的内容（留空则进入交互式多轮对话模式）")
    parser.add_argument("--file", "-f", default="", help="附加分析的本地文件路径")
    parser.add_argument("--gateway", "-g", default="http://127.0.0.1:18080/v1", help="本机蓝绿网关地址")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3.8-27B", help="模型名称")
    parser.add_argument(
        "--system",
        "-s",
        default="你是由阿里云通义实验室研发的 Qwen3.8-27B 大语言模型，回答准确、精炼、专业。请默认使用中文回答。",
        help="系统提示词",
    )
    parser.add_argument("--max-tokens", "-n", type=int, default=512, help="单次回复最大 Token 数 (默认 512)")
    parser.add_argument("--stdin", action="store_true", help="从标准输入管道读取附加内容")
    parser.add_argument("--status", action="store_true", help="仅查看当前本机网关与云端槽位状态")
    parser.add_argument("--no-stats", action="store_true", help="不打印底部的 TTFT 与 tok/s 统计栏")
    parser.add_argument("--cli", action="store_true", help="使用纯命令行模式（不自动弹出原生中文对话窗口）")
    parser.add_argument("--gui", action="store_true", help="强制开启原生中文 GUI + 网页端 + 终端三端同步窗口")
    args = parser.parse_args()

    if args.status:
        ensure_gateway_running(args.gateway)
        print_status_panel(args.gateway)
        return 0

    prompt_text = " ".join(args.prompt).strip()

    piped_text = ""
    if args.stdin:
        try:
            piped_text = sys.stdin.read().strip()
        except Exception:
            piped_text = ""

    if args.file:
        if not os.path.exists(args.file):
            print(f"{C_RED}✗ 找不到文件: {args.file}{C_RESET}", file=sys.stderr)
            return 1
        file_content = open(args.file, "r", encoding="utf-8", errors="replace").read()
        q = prompt_text or "请分析并解释这个文件："
        prompt_text = f"{q}\n\n文件名: `{os.path.basename(args.file)}`\n```\n{file_content[:24000]}\n```"

    if piped_text:
        q = prompt_text or "请分析以下输入内容："
        prompt_text = f"{q}\n\n```\n{piped_text[:24000]}\n```"

    if prompt_text:
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": prompt_text})
        stream_chat_completion(
            gateway_url=args.gateway,
            model=args.model,
            messages=msgs,
            max_tokens=args.max_tokens,
            show_stats=not args.no_stats,
        )
        return 0

    if args.cli and not args.gui:
        return run_cli_repl(
            gateway_url=args.gateway,
            model=args.model,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
        )

    return run_gui_repl(
        gateway_url=args.gateway,
        model=args.model,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
