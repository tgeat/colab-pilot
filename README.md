# Colab Pilot

[![Built with Antigravity](https://img.shields.io/badge/Built%20with-Google%20Antigravity-4285F4?logo=google&logoColor=white)](https://antigravity.google)
[![Powered by Gemini](https://img.shields.io/badge/AI-Google%20Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://deepmind.google/technologies/gemini/)

Google Colab 深度学习 24h+ 串行接力训练、大语言模型（LLM）蓝绿无损热切换与实时云端算力监控桌面平台。技术栈：**Tauri 2 + React 18 + TypeScript + Rust + Python Skills**。

## 运行

```powershell
npm install
npm run tauri dev      # 开发模式
npm run tauri build    # 打包安装程序（输出到 src-tauri/target/release/bundle）
```

需要先准备：
- Rust（stable, MSVC）、Node 18+ 与 Python 3.10+
- `colab-cli`（`~/.local/bin/colab.exe`），并已运行过一次 `colab status` 完成 Google 登录
- `rclone`（`~/.local/bin/rclone.exe`），并已用 `rclone config` 配好 `gdrive:` 远程

## 架构

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Rust · Colab REST | `src-tauri/src/colab.rs` | 复用 `~/.config/colab-cli/token.json` 的 OAuth 凭证（支持过期自动续期与 401 重试）；`ccu-info` / `assignments` / `unassign`；Jupyter Contents API 读写 |
| Rust · 子进程 | `src-tauri/src/cli.rs` | 调用 `colab` / `rclone`（无控制台弹窗、按字节行缓冲防半截 UTF-8 乱码与 stderr 管道死锁、可取消并整树结束进程） |
| Rust · 接力 | `src-tauri/src/relay.rs` | 跨实例串行接力监工：分配 → 上传 `rclone.conf`/`train.py` → 安装环境 → 流式训练 → 停费核验 → 换棒与异常熔断 |
| Rust · 主体 | `src-tauri/src/lib.rs` | 后台定时轮询推送 `overview` 事件；单实例唤醒握手（`:18079`）；删除即停费核验；内置 LLM 流式对话桥接与取消控制 |
| 云端探针 | `src-tauri/remote/telemetry_daemon.py` | 每 2.5 秒将 CPU/GPU/VRAM/内存/磁盘/进程及训练/推理状态快照写入 `/content/.colab_pilot_telemetry.json`（经 Contents API 非阻塞读取） |
| 前端监控与对话台 | `src/` | 实时实例仪表盘（工作区卡片、硬件利用率、接力进度、蓝绿槽位 tok/s 曲线）、内置流式 `LlmChatModal` 对话窗与停费核验交互 |
| Skill · 深度学习接力 | `.agents/skills/colab-dl-relay/` | `colab_relay_adapter.py`、`smoke_test_dl.py`、`dl_relay_runner.py`：本机双轮断点冒烟 + 云端真机首步网盘写入核验 + 串行换棒守护 |
| Skill · LLM 蓝绿部署 | `.agents/skills/colab-llm-bluegreen/` | `llm_bluegreen_manager.py`、`llm_local_gateway.py`、`llm_metrics_sidecar.py`：云端多引擎部署、实时 tok/s 边车统计、固定本地 OpenAI 网关（`127.0.0.1:18080/v1`）与 Make-Before-Break 并行热切换 |
| 终端 / GUI 对话客户端 | `qwen_cli.py`, `qwen.cmd` | 命令行与原生中文输入/网页镜像客户端，直连本地蓝绿网关进行多轮流式对话与一键后台热切换 |
| 示例训练脚本 | `train.py`, `relay_rclone.sh` | PyTorch 断点续训与 `rclone` 多线程 Google Drive 热备参考实现 |

## 核心工作流

### 1. 删除即停费核验
点击实例卡片的「释放停费」会依次执行：停止本机接力任务与子进程树 → `colab stop` → 直接调用 `/tun/m/unassign` 兜底 → 轮询 `/tun/m/assignments`。
**只有在服务端计费列表确认该 `endpoint` 已消失后才返回核验通过**；若网络波动或未分配完成导致残留未关联实例，界面会明确告警并保留卡片，杜绝假性删除与后台隐形扣费。

### 2. 深度学习 24h+ 串行接力（`colab-dl-relay`）
```powershell
# 1. 上云前执行本机双轮断点续训冒烟核验
python .agents/skills/colab-dl-relay/scripts/smoke_test_dl.py --script train.py --skip-rclone-net

# 2. 启动跨实例串行接力守护进程（自动注册工作区供 Colab Pilot 实时监控）
python .agents/skills/colab-dl-relay/scripts/dl_relay_runner.py --workspace . --script train.py --session dl-relay --gpu T4 --gdrive-dir gdrive:colab_relay
```

### 3. 大语言模型蓝绿无损热切换（`colab-llm-bluegreen`）
```powershell
# 1. 部署初始槽位并在云端真机完成 Warmup 冒烟测速，自动拉起本机网关 127.0.0.1:18080
python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py deploy --session-prefix llm --slot blue --gpu T4 --engine builtin --model-name qwen2.5-7b-instruct

# 2. 执行 Make-Before-Break 并行蓝绿热切换（新槽位冒烟通过后原子切流并自动释放旧槽位）
python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py switchover

# 3. 在 Colab Pilot 桌面端直接点击实例卡片「立即对话」，或启动独立中文客户端
.\qwen.cmd
```

## 测试与诊断

```powershell
# 自动化测试套件（前端逻辑 / Rust 后端 / Python 适配器与网关）
npm test
cargo test --manifest-path src-tauri/Cargo.toml
python -B -m unittest discover -s tests -p "test_*.py" -v

# 调试构建无头诊断命令
src-tauri\target\debug\colab-pilot.exe --print-overview        # 无界面输出后端总览 JSON
```

旧版 Python/Tornado 实现保留在 `legacy/` 目录中。

## 致谢 (Acknowledgments)

本项目由 [Google Antigravity](https://antigravity.google) 与 [Google Gemini](https://deepmind.google/technologies/gemini/) 辅助开发完成。
