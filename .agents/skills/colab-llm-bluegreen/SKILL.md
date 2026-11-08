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

This skill standardizes Large Language Model (LLM) deployment on Google Colab across arbitrary inference engines (`vllm`, `ollama`, `llama.cpp`, `transformers`), injects a unified **OpenAI SSE Token Frequency Sidecar** ([`llm_metrics_sidecar.py`](./scripts/llm_metrics_sidecar.py)), and executes **Make-Before-Break Parallel Blue-Green Hot Switchover** ([`llm_bluegreen_manager.py`](./scripts/llm_bluegreen_manager.py)) via `colab-cli` with **Colab Cloud Real-Machine Warmup Smoke** verification.

## Core Design Principles

1. **Cloud-Native Real-Machine Warmup Smoke (No Fake Local LLM Smoke)**:
   - Because LLM inference engines (`vLLM`, `FlashAttention`, `CUDA`) and 7B–70B weights require Colab Linux + GPU VRAM, **smoke testing is performed directly on the Colab GPU instance during slot warmup** before switching any traffic.
   - Before provisioning Colab, `precheck` only validates `colab-cli`, `rclone.conf`, and verifies that `--gdrive-model-path` exists on Google Drive.
   - Once the Colab VM boots and loads the model, it sends a real streaming `POST /v1/chat/completions` Warmup request on the cloud GPU and verifies `instant_tps > 0` and `total_completion_tokens > 0`.
   - **Fail-Fast Auto-Stop**: If the Colab VM hits CUDA OOM or fails the Warmup Smoke test, `llm_bluegreen_manager.py` **immediately executes `colab stop`** on the failed slot to prevent idle billing, while keeping the existing active slot serving traffic uninterrupted.
2. **Unified Sidecar for All LLM Engines (`:8000 -> :8001`)**:
   - The backend engine (`vLLM`, `Ollama`, `llama-server`) binds to internal port `127.0.0.1:8001`.
   - [`llm_metrics_sidecar.py`](./scripts/llm_metrics_sidecar.py) binds to `0.0.0.0:8000`, proxies `/v1/chat/completions`, and measures per-chunk SSE token emission (`instant_tps`, `avg_tps_60s`, `peak_tps`, `ttft_ms`, `active_streams`, and `tps_history`), writing `/content/.colab_pilot_llm_state.json` every 1.5s.
3. **Make-Before-Break Parallel Hot Switchover (`<prefix>-blue` ↔ `<prefix>-green`)**:
   - While `llm-blue` serves traffic, `llm-green` is provisioned in parallel, pulls weights from Google Drive, and runs the **Cloud Warmup Smoke Test**.
   - Only after `llm-green` passes the warmup check (`SLOT_READY`), traffic is atomically switched (`llm-green` → `ACTIVE`, `llm-blue` → `DRAINING`) and `llm-blue` is terminated (`colab stop -s llm-blue`).

---

## Step-by-Step Workflow

### Step 1: Run Pre-Upload Configuration & Google Drive Weight Precheck

Verify `colab-cli`, `rclone.conf`, and Google Drive model weight path readiness (< 2s):

```bash
python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py precheck \
  --engine vllm \
  --model-name qwen2.5-7b-instruct \
  --gdrive-model-path ""
```

---

### Step 2: Deploy the Primary Slot (`blue`) & Execute Cloud Real-Machine Warmup Smoke

Use `deploy` to provision `<prefix>-blue` on Colab, pull weights from Google Drive, start the inference engine + sidecar, and run the **Colab Cloud Real-Machine Warmup Smoke Test**:

```bash
python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py deploy \
  --session-prefix llm \
  --slot blue \
  --workspace . \
  --gpu T4 \
  --high-mem \
  --engine vllm \
  --model-name qwen2.5-7b-instruct \
  --gdrive-model-path "gdrive:colab_relay/models/qwen2.5-7b"
```

- If the Cloud Warmup Smoke test fails (e.g. CUDA OOM), the script automatically runs `colab stop -s llm-blue` so zero CU is wasted.

---

### Step 3: Supervise Token Output Frequency & Trigger Blue-Green Hot Switchover

1. **Generate Streaming Load / Verify Live `tok/s` in Colab Pilot**:
   ```bash
   python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py benchmark \
     --session-prefix llm \
     --requests 5
   ```
2. **Execute Make-Before-Break Parallel Hot Switchover (`blue -> green` or `green -> blue`)**:
   ```bash
   python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py switchover \
     --session-prefix llm \
     --workspace . \
     --gpu T4 \
     --high-mem
   ```
   - Optional: pass `--hold-overlap-seconds 15` to observe both `llm-blue` and `llm-green` online simultaneously in **Colab Pilot** before the old slot is terminated.
3. **Stop All Slots & Release Billing**:
   ```bash
   python .agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py stop \
     --session-prefix llm
   ```
