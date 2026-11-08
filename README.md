# Colab Pilot

[![Built with Antigravity](https://img.shields.io/badge/Built%20with-Google%20Antigravity-4285F4?logo=google&logoColor=white)](https://antigravity.google)
[![Powered by Gemini](https://img.shields.io/badge/AI-Google%20Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://deepmind.google/technologies/gemini/)

Google Colab 虚拟机管理与 24h+ 全自动接力训练桌面应用。技术栈：**Tauri 2 + React 18 + TypeScript + Rust**。

## 运行

```powershell
npm install
npm run tauri dev      # 开发模式
npm run tauri build    # 打包安装程序（输出到 src-tauri/target/release/bundle）
```

需要先准备：
- Rust（stable, MSVC）和 Node 18+
- `colab-cli`（`~/.local/bin/colab.exe`），并已运行过一次 `colab status` 完成 Google 登录
- `rclone`（`~/.local/bin/rclone.exe`），并已用 `rclone config` 配好 `gdrive:` 远程

## 架构

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Rust · Colab REST | `src-tauri/src/colab.rs` | 复用 `~/.config/colab-cli/token.json` 的 OAuth 凭证；ccu-info / assignments / unassign；Jupyter Contents API 读写文件 |
| Rust · 子进程 | `src-tauri/src/cli.rs` | 调用 colab / rclone（无控制台窗口、UTF-8、可取消、流式输出、整树结束进程） |
| Rust · 接力 | `src-tauri/src/relay.rs` | 每台虚拟机一个接力监工任务：分配 → 上传 rclone.conf/train.py → 装环境 → 训练 → 销毁 → 换棒 |
| Rust · 主体 | `src-tauri/src/lib.rs` | 每 4 秒后台轮询 + 推送 `overview` 事件；彻底销毁逻辑；Tauri commands |
| 云端探针 | `src-tauri/remote/telemetry_daemon.py` | 每 2.5 秒把 CPU/GPU/内存/磁盘/进程/线程快照写入 `/content/.colab_pilot_telemetry.json`（通过 Contents API 读取，不占用 Kernel） |
| 前端 | `src/` | 浅色主题界面：每台虚拟机一张卡片，右上角「新增虚拟机」 |
| 训练脚本 | `train.py`, `requirements.txt` | 接力训练脚本（可在应用内编辑） |

## 删除即停费

删除虚拟机会依次执行：停止接力监工 → `colab stop` → 直接调用 `/tun/m/unassign` 兜底 → 轮询 `/tun/m/assignments`。
**只有在服务端计费列表里确认该 endpoint 已经消失，才算删除成功**；否则界面会报错并保留卡片，不会出现界面删了、云端还在扣费的情况。
服务端存在但本地没有记录的实例，会以「外部实例」卡片显示出来，同样可以删除。

## 诊断

```powershell
src-tauri\target\debug\colab-pilot.exe --print-overview        # 无界面输出后端总览 JSON
src-tauri\target\debug\colab-pilot.exe --selftest-lifecycle    # 仅调试构建可用：真实创建 CPU 实例 → 探针 → 销毁核验
src-tauri\target\debug\colab-pilot.exe --selftest-relay        # 仅调试构建可用：真实执行跨实例接力训练端到端核验
```

旧版 Python/Tornado 实现保留在 `legacy/` 目录中。

## 致谢 (Acknowledgments)

本项目由 [Google Antigravity](https://antigravity.google) 与 [Google Gemini](https://deepmind.google/technologies/gemini/) 辅助开发完成。
