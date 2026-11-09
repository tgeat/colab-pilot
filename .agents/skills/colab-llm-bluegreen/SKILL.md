---
name: colab-llm-bluegreen
description: >-
  Pre-configure, deploy, and supervise Large Language Models (LLMs) on Google Colab
  with Cloud Real-Machine Warmup Smoke verification, real-time token output
  frequency (tok/s) supervision, online health monitoring, and zero-downtime
  Make-Before-Break parallel Blue-Green hot switchover via colab-cli and Google
  Drive. Use this skill when the user wants to deploy an LLM (vLLM, Ollama,
  llama.cpp, GGUF, AWQ, LoRA, or HuggingFace weights) on Colab, monitor token
  generation speed (tok/s) and TTFT in Colab Pilot, or perform parallel blue-green
  hot switchover between Colab instances.
---

# Colab LLM Parallel Blue-Green Deployment & Token Supervision Skill (`colab-llm-bluegreen`)

This skill standardizes Large Language Model (LLM) deployment on Google Colab across arbitrary inference engines (`vllm`, `ollama`, `llama.cpp`, `transformers`), injects a unified **OpenAI SSE Token Frequency Sidecar** ([`llm_metrics_sidecar.py`](./scripts/llm_metrics_sidecar.py)), runs a **Windowless Pure-Python Local OpenAI Gateway** ([`llm_local_gateway.py`](./scripts/llm_local_gateway.py)), and executes **Make-Before-Break Parallel Blue-Green Hot Switchover** ([`llm_bluegreen_manager.py`](./scripts/llm_bluegreen_manager.py)) via `colab-cli` with **Colab Cloud Real-Machine Warmup Smoke** verification.

## Core Design Principles

1. **Cloud-Native Real-Machine Warmup Smoke (No Fake Local LLM Smoke)**:
   - Because LLM inference engines (`vLLM`, `llama.cpp`, `FlashAttention`, `CUDA`) and 7B–90B weights require Colab Linux + GPU VRAM, **smoke testing is performed directly on the Colab GPU instance during slot warmup** before switching any traffic.
   - Before provisioning Colab, `precheck` only validates `colab-cli`, `rclone.conf`, and verifies `--gdrive-model-path` (if provided).
   - Once the Colab VM boots and loads the model (`GET /health == 200`), it sends a real streaming `POST /v1/chat/completions` Warmup request on the cloud GPU and verifies `instant_tps > 0` and `total_completion_tokens > 0`.
   - **Fail-Fast Auto-Stop**: If the Colab VM hits CUDA OOM or fails the Warmup Smoke test, `llm_bluegreen_manager.py` **immediately executes `colab stop`** on the failed slot to prevent idle billing, while keeping the existing active slot serving traffic uninterrupted.
2. **Native Multi-Shard GGUF & `llama.cpp` Support (`--engine llama.cpp`)**:
   - Automatically downloads and extracts the official prebuilt CUDA `llama-server` binary (`b12219-bin-ubuntu-cuda-12.8-x64`) into `/content/llama_bin` (~14s, zero compilation overhead).
   - Automatically detects single or multi-tier/multi-shard GGUF repositories on HuggingFace (e.g. `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, or specify a tier via `REPO_ID:SUBDIR` such as `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF:Q2_0`, defaulting to `IQ3_S`) and pulls shards at ~320 MB/s using `HF_HUB_ENABLE_HF_TRANSFER=1`.
   - Launches `llama-server` with `-lm mmap --lazy-mode on -ngl 99 -c 32768 --reasoning-budget 0` so primary transformer/MoE weights stay resident in GPU VRAM (`-ngl 99`, e.g. 55 GB inside G4's 96 GB VRAM) while large per-layer n-gram lookup tables (`per_layer_token_embd`, 28.8 GB Shard 2) remain memory-mapped on disk/RAM without blocking startup.
   - Waits for `GET /health` to return HTTP `200` (not `503 Loading model`) before triggering the Warmup Smoke request.
3. **Unified Sidecar (`:8000 -> :8001`) & Windowless Local Gateway (`:18080`)**:
   - The backend engine (`vLLM`, `Ollama`, `llama-server`, `transformers`) binds to internal port `127.0.0.1:8001`.
   - [`llm_metrics_sidecar.py`](./scripts/llm_metrics_sidecar.py) binds to `0.0.0.0:8000`, proxies `/v1/chat/completions` (counting both `delta.content` and `delta.reasoning_content`), and writes `/content/.colab_pilot_llm_state.json` every 1.5s.
   - [`llm_local_gateway.py`](./scripts/llm_local_gateway.py) runs on `127.0.0.1:18080` (plus `18081`/`18082`) using **pure-Python in-process bridging** (`ColabRuntime` WebSocket + Contents API) without spawning any external `ssh.exe` or `colab.exe` subprocesses—completely eliminating flashing console windows on Windows.
4. **Make-Before-Break Parallel Hot Switchover (`<prefix>-blue` ↔ `<prefix>-green`)**:
   - While `llm-blue` serves traffic, `llm-green` is provisioned in parallel, pulls weights, and runs the **Cloud Warmup Smoke Test**.
   - Only after `llm-green` passes the warmup check (`SLOT_READY`), traffic is atomically switched (`llm-green` → `ACTIVE`, `llm-blue` → `DRAINING`) and `llm-blue` is terminated (`colab stop -s llm-blue`).

---

## Fast-Path Execution & Anti-Pitfall Rules (Must Follow)

1. **Go Straight to `precheck` → `deploy` (No Manual Piecemeal Probing)**:
   - Do **not** run manual `colab new` + multiple interactive `colab exec` commands to inspect `nvidia-smi`, download `llama-server`, or check `--help` flags—`llm_bluegreen_manager.py deploy` already automates the entire pipeline in a single pass.
2. **Windows PowerShell Quoting & Argument Rules**:
   - **Never pass empty `--gdrive-model-path ""` in PowerShell** (Windows PowerShell 5.1 strips empty quotes, causing `argparse` to fail with `expected one argument`). Omit `--gdrive-model-path` when pulling directly from HuggingFace or Ollama.
   - **Never construct complex `python -c "..."` one-liners with nested quotes `\"` or pipes `||` in PowerShell**. Always invoke `llm_bluegreen_manager.py` directly with `python -u`.
   - On Windows, background processes must use `creationflags = 0x08000000` (`CREATE_NO_WINDOW`) **only**—never combine `0x08000000` with `DETACHED_PROCESS (0x00000008)`, which detaches the console and causes child console binaries to pop up visible flashing windows.
3. **Deployment Time Profile (Reference on Colab G4 · 96 GB VRAM)**:
   - **Colab G4 VM provisioning + sidecar upload**: ~12s
   - **Prebuilt CUDA `llama-server` download & extraction**: ~14s
   - **HuggingFace GGUF weight download (`hf_transfer` @ ~320 MB/s)**: ~250s for 83.86 GB (`IQ3_S`), ~190s for 66.82 GB (`Q2_0`) — **this is ~85% of total deployment time**; schedule a 60s timer to report live `model_dir=` GB progress to the user during `deploy`.
   - **VRAM loading (`-ngl 99 -lm mmap --lazy-mode on`) + Warmup Smoke**: ~23s

---

## Step-by-Step Workflow

### Step 1: Run Pre-Upload Configuration & Precheck

Verify `colab-cli`, `rclone.conf`, and sidecar syntax (< 2s):

```powershell
python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" precheck `
  --engine llama.cpp `
  --model-name "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
```

---

### Step 2: Deploy the Primary Slot (`blue`) & Execute Cloud Real-Machine Warmup Smoke

Use `deploy` to provision `<prefix>-blue` on Colab, pull weights, start the inference engine + sidecar, and run the **Colab Cloud Real-Machine Warmup Smoke Test**:

```powershell
# Example A: Deploy a multi-shard GGUF model from HuggingFace on Colab G4 (96 GB VRAM)
python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" deploy `
  --session-prefix llm `
  --slot blue `
  --workspace "C:\Users\86177\Desktop\colab-pilot" `
  --gpu G4 `
  --engine llama.cpp `
  --model-name "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"

# Example B: Deploy a HuggingFace Transformers / vLLM model or pull from Google Drive
python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" deploy `
  --session-prefix llm `
  --slot blue `
  --workspace "C:\Users\86177\Desktop\colab-pilot" `
  --gpu G4 `
  --engine transformers `
  --model-name "Qwen/Qwen3.8-27B"
```

- If the Cloud Warmup Smoke test fails (e.g. CUDA OOM), the script automatically runs `colab stop -s llm-blue` so zero CU is wasted.

---

### Step 3: Supervise Token Output Frequency & Trigger Blue-Green Hot Switchover

1. **Generate Streaming Load / Verify Live `tok/s` in Colab Pilot**:
   ```powershell
   python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" benchmark `
     --session-prefix llm `
     --requests 5
   ```
2. **Execute Make-Before-Break Parallel Hot Switchover (`blue -> green` or `green -> blue`)**:
   ```powershell
   python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" switchover `
     --session-prefix llm `
     --workspace "C:\Users\86177\Desktop\colab-pilot" `
     --gpu G4
   ```
   - Optional: pass `--hold-overlap-seconds 15` to observe both `llm-blue` and `llm-green` online simultaneously in **Colab Pilot** before the old slot is terminated.
3. **Stop All Slots & Release Billing**:
   ```powershell
   python -u "C:\Users\86177\.gemini\config\skills\colab-llm-bluegreen\scripts\llm_bluegreen_manager.py" stop `
     --session-prefix llm
   ```
