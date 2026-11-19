# Google Colab CLI + Rclone 超过 24 小时全自动接力训练方案

本方案针对 Google Colab 单次会话最长 24 小时的硬性限制，结合 **`google-colab-cli`**（本地命令行编排）与 **`rclone`**（免弹窗多线程云盘同步），实现 **100% 无人值守的跨实例接力训练（Relay Training）**。

---

## 一、方案核心优势

1. **彻底绕开 Google Drive 授权弹窗坑**：在本机生成一次 `rclone.conf` 持久化令牌，每次开新实例自动上传该文件即可静默读写私有网盘，半夜切换机器绝不卡死。
2. **谷歌内网 16 线程满速传输**：所有数据集拉取与 Checkpoint 备份均由 Colab 通过谷歌机房内网直连 Google Drive 多线程完成，不消耗本机宽带流量。
3. **双重防丢与防回退覆盖机制**：
   * **单棒限时优雅退出**：Python 脚本内置 22.5 小时安全计时器，在触及 24 小时强制断电前主动保存并输出 `HANDOVER_REQUESTED` 安全退出。
   * **中途定时热备份 + 上一轮快照保留**：训练过程中每隔 30 分钟自动通过原子写入 + `rclone` 同步最新权重，并在接力棒切换时保留 `checkpoint_prev.pt`；当 `RELAY_ROUND > 1` 且远端断点缺失或损坏时立即 Fail-Fast 报错停机，绝不从 Epoch 1 静默覆盖已训权重。
4. **服务端硬核验防漏关机**：每棒结束后、异常快速熔断或手动销毁时，调用 Google Colab `/tun/m/unassign/{endpoint}` 并通过 `/tun/m/assignments` 二次核验服务端计费列表清零，杜绝空转扣费。

---

## 二、前置配置步骤

1. **安装并登录 `colab-cli`**：
   ```bash
   uv tool install google-colab-cli
   colab auth login
   ```
2. **配置免交互 `rclone` Google Drive 远端（命名为 `gdrive`）**：
   ```bash
   rclone config
   # 新建 remote: 名称填 gdrive，类型选 drive，按浏览器授权完成后保存
   rclone lsd gdrive:
   ```

---

## 三、启动方式

- **方式 A（推荐 — Colab Pilot 桌面管控台）**：直接启动 Colab Pilot Tauri 客户端，在目标实例卡片中配置 GPU 型号、总轮数与网盘目录（如 `gdrive:colab_relay`），点击 **开启接力** 即可实时监控 GPU 显存、Loss 曲线、接力棒次与计费核验状态。
- **方式 B（命令行无人值守脚本）**：
  ```bash
  python -u .agents/skills/colab-dl-relay/scripts/dl_relay_runner.py \
    --session relay-worker \
    --workspace . \
    --script train.py \
    --gpu T4 \
    --high-mem \
    --gdrive-dir gdrive:colab_relay \
    --total-epochs 500
  ```
  或在 Bash 环境下执行 `./relay_rclone.sh`。
