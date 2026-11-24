import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

function loadTypeScript() {
  const candidates = [
    path.join(repoRoot, "node_modules", "typescript"),
    path.join(os.tmpdir(), "colab_pilot_frontend_verify", "node_modules", "typescript"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) {
      const req = createRequire(pathToFileURL(path.join(repoRoot, "package.json")).href);
      return req(p);
    }
  }
  throw new Error(
    "Cannot find `typescript` module. Run `npm install` before running `npm test`.",
  );
}

const ts = loadTypeScript();

function transpileToDataUrl(relFilePath, importRewrites = {}) {
  let src = fs.readFileSync(path.join(repoRoot, relFilePath), "utf8");
  for (const [spec, replacement] of Object.entries(importRewrites)) {
    src = src.replaceAll(`from "${spec}"`, `from "${replacement}"`);
  }
  const { outputText } = ts.transpileModule(src, {
    fileName: relFilePath,
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      strict: true,
    },
  });
  return `data:text/javascript;base64,${Buffer.from(outputText, "utf8").toString("base64")}`;
}

const formatUrl = transpileToDataUrl("src/format.ts");
const chatUtilsUrl = transpileToDataUrl("src/chatUtils.ts", {
  "./format": formatUrl,
});
const mockTauriCoreSource = [
  "export const invokeCalls = [];",
  "export async function invoke(cmd, args) {",
  "  invokeCalls.push({ cmd, args });",
  "  return null;",
  "}",
].join("\n");
const mockTauriCoreUrl = `data:text/javascript;base64,${Buffer.from(
  mockTauriCoreSource,
  "utf8",
).toString("base64")}`;
const apiUrl = transpileToDataUrl("src/api.ts", {
  "@tauri-apps/api/core": mockTauriCoreUrl,
});

const formatMod = await import(formatUrl);
const chatUtils = await import(chatUtilsUrl);
const mockTauriCore = await import(mockTauriCoreUrl);
const apiMod = await import(apiUrl);

test("fmtBytes handles sub-byte, zero, negative, NaN, and extreme sizes without undefined unit (L-2)", () => {
  assert.equal(formatMod.fmtBytes(0), "0 B");
  assert.equal(formatMod.fmtBytes(-42), "0 B");
  assert.equal(formatMod.fmtBytes(Number.NaN), "0 B");
  assert.equal(formatMod.fmtBytes(0.5), "< 1 B");
  assert.equal(formatMod.fmtBytes(1), "1 B");
  assert.equal(formatMod.fmtBytes(1536), "1.5 KB");
  assert.ok(!formatMod.fmtBytes(1e30).includes("undefined"));
});

test("num, numOrNull, pct, and str guard malformed telemetry values (L-2)", () => {
  assert.equal(formatMod.num(undefined, 7), 7);
  assert.equal(formatMod.num("  ", 5), 5);
  assert.equal(formatMod.num("12.5"), 12.5);
  assert.equal(formatMod.num(Number.POSITIVE_INFINITY, 3), 3);
  assert.equal(formatMod.numOrNull(null), null);
  assert.equal(formatMod.numOrNull("NaN"), null);
  assert.equal(formatMod.numOrNull("42"), 42);
  assert.equal(formatMod.pct(-15), 0);
  assert.equal(formatMod.pct(180), 100);
  assert.equal(formatMod.str(null, "fallback"), "fallback");
  assert.equal(formatMod.str(99), "99");
});

test("errText safely formats circular objects, empty strings, and Error instances (L-2)", () => {
  assert.equal(apiMod.errText(null), "未知错误");
  assert.equal(apiMod.errText(undefined), "未知错误");
  assert.equal(apiMod.errText(""), "未知错误");
  assert.equal(apiMod.errText("boom"), "boom");
  assert.equal(apiMod.errText(new Error("rpc failed")), "rpc failed");
  assert.equal(apiMod.errText({ message: undefined, code: 500 }), '{"code":500}');
  const circular = {};
  circular.self = circular;
  assert.equal(typeof apiMod.errText(circular), "string");
});

test("isSafeMarkdownUrl blocks dangerous URI schemes and allows http/https/mailto (H-1)", () => {
  assert.equal(chatUtils.isSafeMarkdownUrl("javascript:alert(1)"), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("  JaVaScRiPt:alert(document.cookie) "), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("data:text/html,<script>alert(1)</script>"), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("vbscript:msgbox(1)"), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("file:///C:/Windows/win.ini"), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("//evil.example/payload"), false);
  assert.equal(chatUtils.isSafeMarkdownUrl("https://colab.research.google.com/"), true);
  assert.equal(chatUtils.isSafeMarkdownUrl("http://127.0.0.1:18080/v1/models"), true);
  assert.equal(chatUtils.isSafeMarkdownUrl("mailto:support@example.com"), true);
});

test("extractThinkParts and stripThinkBlocks handle partial streaming tags and multi-blocks (L-5, H-9)", () => {
  const partial = chatUtils.extractThinkParts("<th", true);
  assert.equal(partial.mainText, "");

  const unclosed = chatUtils.extractThinkParts("<think>step 1", true);
  assert.equal(unclosed.thinkText, "step 1");
  assert.equal(unclosed.mainText, "");

  const multi = chatUtils.extractThinkParts("<think>first</think>Hello <think>second</think>world!");
  assert.equal(multi.thinkText, "first\n\nsecond");
  assert.equal(multi.mainText, "Hello world!");

  assert.equal(chatUtils.stripThinkBlocks("<think>only reasoning</think>"), "");
  assert.equal(chatUtils.stripThinkBlocks("<think>plan</think>\nFinal answer"), "Final answer");
});

test("parseMarkdownSegments supports c#, c++, f#, info strings, and unclosed streaming fences (M-1, L-5)", () => {
  const md = [
    "Intro",
    "```c#",
    "Console.WriteLine(1);",
    "```",
    "Middle",
    "```c++ title=\"main.cpp\"",
    "int main() {}",
    "```",
    "```python",
    "print('streaming')",
  ].join("\n");

  const segs = chatUtils.parseMarkdownSegments(md);
  const codeSegs = segs.filter((s) => s.type === "code");
  assert.equal(codeSegs.length, 3);
  assert.equal(codeSegs[0].lang, "c#");
  assert.equal(codeSegs[0].text, "Console.WriteLine(1);");
  assert.equal(codeSegs[1].lang, "c++");
  assert.equal(codeSegs[2].lang, "python");
  assert.equal(codeSegs[2].text, "print('streaming')");
});

test("buildHistoryMessages drops failed/cancelled/unpaired turns so consecutive user roles never occur (H-9)", () => {
  const turns = [
    { id: "1", role: "user", content: "First question" },
    { id: "2", role: "assistant", content: "<think>secret</think>First answer" },
    { id: "3", role: "user", content: "Failed question" },
    { id: "4", role: "assistant", content: "", error: "HTTP 502" },
    { id: "5", role: "user", content: "Cancelled question" },
    { id: "6", role: "assistant", content: "<think>incomplete</think>", cancelled: true },
    { id: "7", role: "user", content: "Dangling user without assistant" },
  ];

  const msgs = chatUtils.buildHistoryMessages(turns, "You are helpful.", "Latest question");
  assert.deepEqual(msgs, [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "First question" },
    { role: "assistant", content: "First answer" },
    { role: "user", content: "Latest question" },
  ]);

  for (let i = 1; i < msgs.length; i += 1) {
    assert.notEqual(msgs[i - 1].role, msgs[i].role, "roles must strictly alternate");
  }
});

test("resolveBlueGreenStatus reports offline when 0 VMs are on_server and isolates session_prefix (H-4, M-12)", () => {
  const staleBg = {
    session_prefix: "qwen",
    active_slot: "blue",
    phase: "serving",
    model_name: "Qwen3-14B",
  };
  const emptyStatus = chatUtils.resolveBlueGreenStatus(staleBg, []);
  assert.equal(emptyStatus.clusterState, "offline");
  assert.equal(emptyStatus.anyOnlineVm, false);

  const unrelatedVm = {
    name: "other-blue",
    on_server: true,
    telemetry: { llm: { online: true, slot: "blue", role: "active" } },
  };
  const isolatedStatus = chatUtils.resolveBlueGreenStatus(staleBg, [unrelatedVm]);
  assert.equal(isolatedStatus.blueVm, undefined);
  assert.equal(isolatedStatus.clusterState, "offline");
  assert.equal(isolatedStatus.anyOnlineVm, true);

  const matchingVm = {
    name: "qwen-blue",
    on_server: true,
    telemetry: { llm: { online: true, slot: "blue", role: "active" } },
  };
  const liveStatus = chatUtils.resolveBlueGreenStatus(staleBg, [matchingVm]);
  assert.equal(liveStatus.blueVm?.name, "qwen-blue");
  assert.equal(liveStatus.clusterState, "online");
  assert.equal(liveStatus.anyOnlineVm, true);
});

test("single-shape accelerators (G4, H100, L4, V5E1, V6E1) force effectiveHighMem=false while T4/A100/CPU preserve toggle", () => {
  for (const acc of ["G4", "H100", "L4", "V5E1", "V6E1", "g4", " h100 "]) {
    assert.equal(chatUtils.isSingleShapeAccelerator(acc), true, `${acc} must be single-shape`);
    assert.equal(chatUtils.resolveEffectiveHighMem(acc, true), false, `${acc} must not pass high_mem=true`);
    assert.equal(chatUtils.resolveEffectiveHighMem(acc, false), false);
  }
  for (const acc of ["T4", "A100", "CPU"]) {
    assert.equal(chatUtils.isSingleShapeAccelerator(acc), false, `${acc} must allow dual shape`);
    assert.equal(chatUtils.resolveEffectiveHighMem(acc, true), true);
    assert.equal(chatUtils.resolveEffectiveHighMem(acc, false), false);
  }

  const createDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "CreateVmDialog.tsx"),
    "utf8",
  );
  assert.doesNotMatch(createDlgSrc, /selectedAcc\.singleShape\s*\?\s*true/);
  for (const acc of ["G4", "H100", "L4", "V5E1", "V6E1"]) {
    assert.match(
      createDlgSrc,
      new RegExp(`v:\\s*"${acc}"[^\\n]*singleShape:\\s*true`),
      `${acc} in CreateVmDialog ACCELERATORS must have singleShape: true`,
    );
  }
});

test("isLlmVmReady and hasAnyReadyLlmVm reject bare, deploying, warming, or failed VMs (RC-FE-3, RC-FE-5)", () => {
  const bareVm = { name: "gpu-vm", on_server: true, busy: null, telemetry: null };
  const trainVm = {
    name: "train-01",
    on_server: true,
    busy: null,
    telemetry: { training: { status: "training" }, threads: { llm_process: null } },
  };
  const deployingLlmVm = {
    name: "llm-blue",
    on_server: true,
    busy: "deploying_llm",
    llm_api: { api_key: "sk-colab-test" },
    telemetry: null,
  };
  const failedLlmVm = {
    name: "llm-blue",
    on_server: true,
    busy: null,
    deploy_error: "model pull failed",
    llm_api: { api_key: "sk-colab-test" },
    telemetry: null,
  };
  const warmingLlmVm = {
    name: "llm-blue",
    on_server: true,
    busy: null,
    telemetry: { llm: { online: false, backend_healthy: false } },
  };
  const healthyLlmVm = {
    name: "llm-blue",
    on_server: true,
    busy: null,
    telemetry: { llm: { online: true, backend_healthy: true } },
  };
  const manualSshLlmVm = {
    name: "custom-gpu",
    on_server: true,
    busy: null,
    telemetry: { llm: null, threads: { llm_process: { pid: 4321 } } },
  };

  assert.equal(chatUtils.isLlmVm(bareVm), false);
  assert.equal(chatUtils.isLlmVm(trainVm), false);
  assert.equal(chatUtils.isLlmVm(deployingLlmVm), true);

  assert.equal(chatUtils.isLlmVmReady(bareVm), false);
  assert.equal(chatUtils.isLlmVmReady(trainVm), false);
  assert.equal(chatUtils.isLlmVmReady(deployingLlmVm), false);
  assert.equal(chatUtils.isLlmVmReady(failedLlmVm), false);
  assert.equal(chatUtils.isLlmVmReady(warmingLlmVm), false);
  assert.equal(chatUtils.isLlmVmReady(healthyLlmVm), true);
  assert.equal(chatUtils.isLlmVmReady(manualSshLlmVm), true);

  const staleBg = { session_prefix: "llm", active_slot: "blue", phase: "serving" };
  assert.equal(chatUtils.hasAnyReadyLlmVm([bareVm, trainVm, deployingLlmVm, failedLlmVm, warmingLlmVm]), false);
  const bgWithOnlyBareVm = chatUtils.resolveBlueGreenStatus(staleBg, [bareVm, trainVm]);
  assert.equal(bgWithOnlyBareVm.anyOnlineVm, false);
  assert.equal(bgWithOnlyBareVm.clusterState, "offline");

  assert.equal(chatUtils.hasAnyReadyLlmVm([bareVm, healthyLlmVm]), true);
});

test("all 16 IPC methods are exported, zero colab console remnants remain, and new-VM controls are wired", () => {
  const expectedIpcMethods = [
    "getOverview",
    "refreshNow",
    "createVm",
    "deleteVm",
    "deployProbe",
    "startRelay",
    "stopRelay",
    "readWorkspaceFiles",
    "saveWorkspaceFiles",
    "streamLlmChat",
    "cancelLlmChat",
    "openVmSshTerminal",
    "rotateLlmApiKey",
    "deployLlmOnVm",
    "confirmAppClose",
    "dismissRelayStatus",
  ];
  for (const fn of expectedIpcMethods) {
    assert.equal(typeof apiMod.api[fn], "function", `api.${fn} must be a function`);
  }

  const appSrc = fs.readFileSync(path.join(repoRoot, "src", "App.tsx"), "utf8");
  const vmCardSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "VmCard.tsx"), "utf8");
  assert.match(appSrc, /<CreateVmDialog/);
  assert.match(appSrc, /<ScriptDialog/);
  assert.match(appSrc, /<StopRelayDialog/);
  assert.match(appSrc, /<CloseAppRelayDialog/);
  assert.match(vmCardSrc, /vm\.telemetry_error/);
  assert.match(vmCardSrc, /vm\.deploy_error/);
  assert.match(vmCardSrc, /<RelayPanel/);

  const srcFiles = [
    ...fs.readdirSync(path.join(repoRoot, "src")).map((f) => path.join(repoRoot, "src", f)),
    ...fs
      .readdirSync(path.join(repoRoot, "src", "components"))
      .map((f) => path.join(repoRoot, "src", "components", f)),
  ].filter((p) => fs.statSync(p).isFile());
  for (const file of srcFiles) {
    const content = fs.readFileSync(file, "utf8");
    assert.doesNotMatch(content, /colab\s+console/i, `${path.basename(file)} must not reference removed colab console`);
    // #75: must never recommend invalid `colab auth login`
    assert.doesNotMatch(content, /colab\s+auth\s+login/i, `${path.basename(file)} must not contain invalid colab auth login`);
  }
});

test("tauri.conf.json enforces non-null CSP and Cargo.toml has empty default features (H-1, M-4)", () => {
  const tauriConf = JSON.parse(
    fs.readFileSync(path.join(repoRoot, "src-tauri", "tauri.conf.json"), "utf8"),
  );
  const csp = tauriConf?.app?.security?.csp;
  assert.equal(typeof csp, "string");
  assert.ok(csp.includes("default-src 'self'"));
  assert.ok(csp.includes("script-src 'self'"));

  const cargoToml = fs.readFileSync(path.join(repoRoot, "src-tauri", "Cargo.toml"), "utf8");
  assert.match(cargoToml, /default\s*=\s*\[\s*\]/);

  const scriptDlg = fs.readFileSync(
    path.join(repoRoot, "src", "components", "ScriptDialog.tsx"),
    "utf8",
  );
  assert.match(scriptDlg, /closable=\{!saving\}/);
});

// ───────────────────── Issues #40 & #41 ─────────────────────
test("Issues #40 & #41: evaluateStopRelayOutcome prevents false destroy success and reports kernel interrupt failures", () => {
  // #40: destroy=true with verified_unassigned=false MUST NOT claim VM was destroyed
  const failedDestroy = apiMod.evaluateStopRelayOutcome("train-01", true, {
    supervisor_stopped: true,
    destroy_vm: true,
    kernel_interrupted: false,
    kernel_interrupt_error: null,
    terminate_receipt: {
      name: "train-01",
      endpoint: "ep-b",
      verified_unassigned: false,
      remaining_endpoints: ["ep-b"],
      steps: ["unassign returned 500"],
      ccu: null,
      error: "服务端仍存在活跃端点 ep-b",
    },
  });
  assert.equal(failedDestroy.ok, false);
  assert.equal(failedDestroy.toastKind, "error");
  assert.equal(failedDestroy.keepDialogOpen, true);
  assert.doesNotMatch(failedDestroy.message, /已关闭.*并销毁云端实例/);
  assert.match(failedDestroy.message, /停费核验未通过/);
  assert.match(failedDestroy.message, /ep-b/);

  // #40: destroy=true with verified_unassigned=true succeeds
  const verifiedDestroy = apiMod.evaluateStopRelayOutcome("train-01", true, {
    supervisor_stopped: true,
    destroy_vm: true,
    kernel_interrupted: true,
    kernel_interrupt_error: null,
    terminate_receipt: {
      name: "train-01",
      endpoint: "ep-b",
      verified_unassigned: true,
      remaining_endpoints: [],
      steps: ["verified"],
      ccu: { balance: 80, burn_rate_hourly: 0, assignments_count: 0 },
      error: null,
    },
  });
  assert.equal(verifiedDestroy.ok, true);
  assert.equal(verifiedDestroy.toastKind, "success");
  assert.equal(verifiedDestroy.keepDialogOpen, false);
  assert.match(verifiedDestroy.message, /已关闭 \[train-01\] 接力并销毁云端实例/);

  // #41: destroy=false with kernel_interrupted=false warns training may still run
  const failedKeep = apiMod.evaluateStopRelayOutcome("train-01", false, {
    supervisor_stopped: true,
    destroy_vm: false,
    kernel_interrupted: false,
    kernel_interrupt_error: "colab restart-kernel 超时 (90s)",
    terminate_receipt: null,
  });
  assert.equal(failedKeep.ok, false);
  assert.equal(failedKeep.toastKind, "warn");
  assert.equal(failedKeep.keepDialogOpen, true);
  assert.match(failedKeep.message, /未能确认中断.*云端 Kernel/);
  assert.match(failedKeep.message, /colab restart-kernel 超时/);
  assert.match(failedKeep.message, /继续运行/);

  // #41: destroy=false with kernel_interrupted=true confirms interruption
  const okKeep = apiMod.evaluateStopRelayOutcome("train-01", false, {
    supervisor_stopped: true,
    destroy_vm: false,
    kernel_interrupted: true,
    kernel_interrupt_error: null,
    terminate_receipt: null,
  });
  assert.equal(okKeep.ok, true);
  assert.equal(okKeep.keepDialogOpen, false);
  assert.match(okKeep.message, /云端训练已中断/);

  // Legacy TerminateReceipt compatibility
  const legacyFail = apiMod.evaluateStopRelayOutcome("train-01", true, {
    name: "train-01",
    endpoint: "ep-legacy",
    verified_unassigned: false,
    remaining_endpoints: ["ep-legacy"],
    steps: [],
    ccu: null,
    error: "still assigned",
  });
  assert.equal(legacyFail.ok, false);
  assert.equal(legacyFail.keepDialogOpen, true);

  // App.tsx must not unconditionally toast "已关闭...并销毁云端实例" inside handleStopRelay
  const appSrc = fs.readFileSync(path.join(repoRoot, "src", "App.tsx"), "utf8");
  assert.doesNotMatch(appSrc, /已关闭\s*\[\$\{vm\.name\}\]\s*接力并销毁云端实例/);
});

// ───────────────────── Issue #42 & #72 & #73 IPC Contracts ─────────────────────
test("Issues #42, #72, #73: api passes workspacePath, vmName, confirmAppClose mode, and dismissRelayStatus name", async () => {
  mockTauriCore.invokeCalls.length = 0;

  await apiMod.api.readWorkspaceFiles("D:/ws_b", "vm-custom");
  await apiMod.api.saveWorkspaceFiles("print(1)", "torch", "D:/ws_b", "vm-custom");
  await apiMod.api.startRelay("vm-custom", { gpu_type: "T4", high_mem: true, gdrive_dir: "gdrive:colab_relay", max_session_hours: 20, backup_interval_minutes: 30, total_epochs: 10, reset_done_flag: false }, "D:/ws_b");
  await apiMod.api.stopRelay("vm-custom", false);
  await apiMod.api.confirmAppClose("shutdown_vms");
  await apiMod.api.dismissRelayStatus("vm-offline");

  assert.deepEqual(mockTauriCore.invokeCalls[0], {
    cmd: "read_workspace_files",
    args: { workspacePath: "D:/ws_b", vmName: "vm-custom" },
  });
  assert.deepEqual(mockTauriCore.invokeCalls[1], {
    cmd: "save_workspace_files",
    args: { trainPy: "print(1)", requirementsTxt: "torch", workspacePath: "D:/ws_b", vmName: "vm-custom" },
  });
  assert.equal(mockTauriCore.invokeCalls[2].cmd, "start_relay");
  assert.equal(mockTauriCore.invokeCalls[2].args.workspacePath, "D:/ws_b");
  assert.deepEqual(mockTauriCore.invokeCalls[3], {
    cmd: "stop_relay",
    args: { name: "vm-custom", destroyVm: false },
  });
  assert.deepEqual(mockTauriCore.invokeCalls[4], {
    cmd: "confirm_app_close",
    args: { mode: "shutdown_vms" },
  });
  assert.deepEqual(mockTauriCore.invokeCalls[5], {
    cmd: "dismiss_relay_status",
    args: { name: "vm-offline" },
  });
});

// ───────────────────── Issues #43 & #75 ─────────────────────
test("Issues #43 & #75: ADC without OAuth token blocks blind billing VM creation and guides colab auth / colab status", () => {
  const blockedAdc = chatUtils.checkCreateVmAuthGuard("adc", { token_ok: false, adc_ok: true });
  assert.equal(blockedAdc.blocked, true);
  assert.equal(blockedAdc.level, "danger");
  assert.match(blockedAdc.message, /colab auth/);
  assert.match(blockedAdc.message, /colab status/);
  assert.doesNotMatch(blockedAdc.message, /colab auth login/);

  const warnOauthMissing = chatUtils.checkCreateVmAuthGuard("oauth2", { token_ok: false, adc_ok: false });
  assert.equal(warnOauthMissing.blocked, false);
  assert.equal(warnOauthMissing.level, "warn");
  assert.match(warnOauthMissing.message, /colab auth/);

  const warnAdcWithOauth = chatUtils.checkCreateVmAuthGuard("adc", { token_ok: true, adc_ok: true });
  assert.equal(warnAdcWithOauth.blocked, false);
  assert.equal(warnAdcWithOauth.level, "warn");
  assert.match(warnAdcWithOauth.message, /监控与关机停费核验仍依赖.*Colab OAuth/);

  const cleanOauth = chatUtils.checkCreateVmAuthGuard("oauth2", { token_ok: true, adc_ok: false });
  assert.equal(cleanOauth.blocked, false);
  assert.equal(cleanOauth.message, null);
});

// ───────────────────── Issue #44 ─────────────────────
test("Issue #44: default gdrive:colab_relay isolates per-VM subdirectory and detects running custom-dir collision", () => {
  assert.equal(chatUtils.isDefaultBareGdriveDir("gdrive:colab_relay"), true);
  assert.equal(chatUtils.isDefaultBareGdriveDir("  gdrive:colab_relay/// "), true);
  assert.equal(chatUtils.isDefaultBareGdriveDir("gdrive:colab_relay/exp1"), false);

  const dirA = chatUtils.resolveEffectiveRelayGdriveDir("gdrive:colab_relay", "exp-a");
  const dirB = chatUtils.resolveEffectiveRelayGdriveDir("gdrive:colab_relay", "exp-b");
  assert.equal(dirA, "gdrive:colab_relay/exp-a");
  assert.equal(dirB, "gdrive:colab_relay/exp-b");
  assert.notEqual(dirA, dirB);

  const runningVms = [
    {
      name: "exp-a",
      on_server: true,
      relay: {
        running: true,
        config: { gdrive_dir: "gdrive:shared_yolo/" },
      },
    },
    {
      name: "exp-default",
      on_server: true,
      relay: {
        running: true,
        config: { gdrive_dir: "gdrive:colab_relay" },
      },
    },
  ];

  // Different VM using default bare gdrive:colab_relay will NOT collide with exp-default
  assert.equal(
    chatUtils.findConflictingRelayVm("gdrive:colab_relay", "exp-new", runningVms),
    undefined,
  );
  // Different VM manually pointing to occupied gdrive:shared_yolo triggers collision warning
  const conflict = chatUtils.findConflictingRelayVm("gdrive:shared_yolo", "exp-new", runningVms);
  assert.equal(conflict?.name, "exp-a");
  // Same VM checking its own dir is not a conflict
  assert.equal(
    chatUtils.findConflictingRelayVm("gdrive:shared_yolo", "exp-a", runningVms),
    undefined,
  );
});

// ───────────────────── Issues #48 & #64 ─────────────────────
test("Issues #48 & #64: blue-green resolves real session names, blocks warming targets, and syncs vanished targets", () => {
  const autoNamedBlue = {
    name: "colab-t4-auto-01",
    accelerator: "T4",
    on_server: true,
    busy: null,
    telemetry: {
      llm: { online: true, backend_healthy: true, slot: "blue", role: "draining", model_name: "Qwen2.5-7B" },
    },
  };
  const customNamedGreen = {
    name: "prod-deepseek-node",
    accelerator: "A100",
    on_server: true,
    busy: null,
    telemetry: {
      llm: { online: true, backend_healthy: true, slot: "green", role: "active", model_name: "DeepSeek-R1-14B" },
    },
  };
  const warmingNode = {
    name: "standby-node",
    accelerator: "L4",
    on_server: true,
    busy: null,
    telemetry: {
      llm: { online: true, backend_healthy: true, slot: "blue", role: "warming", model_name: "Qwen3-8B" },
    },
  };

  const bgState = {
    session_prefix: "llm",
    active_slot: "green",
    active_session: "prod-deepseek-node",
    blue_session: "colab-t4-auto-01",
    green_session: "prod-deepseek-node",
    slots: {
      blue: { session_name: "colab-t4-auto-01", model_name: "Qwen2.5-7B" },
      green: { session_name: "prod-deepseek-node", model_name: "DeepSeek-R1-14B" },
    },
    model_name: "Fallback-Model",
  };

  // #48: matches real session names without requiring -blue / -green suffixes
  const status = chatUtils.resolveBlueGreenStatus(bgState, [autoNamedBlue, customNamedGreen]);
  assert.equal(status.blueVm?.name, "colab-t4-auto-01");
  assert.equal(status.greenVm?.name, "prod-deepseek-node");
  assert.equal(status.activeSlot, "green");
  assert.equal(status.activeVm?.name, "prod-deepseek-node");
  assert.equal(status.clusterState, "online");

  // #48: auto routing submits real active session name and its real model name
  const autoTarget = chatUtils.resolveChatTargetState("auto", bgState, [autoNamedBlue, customNamedGreen]);
  assert.equal(autoTarget.canSend, true);
  assert.equal(autoTarget.submitTargetVm, "prod-deepseek-node");
  assert.equal(autoTarget.submitModel, "DeepSeek-R1-14B");

  // #64: warming node is NOT ready, labeled with warming notice, and blocked from sending
  assert.equal(chatUtils.isLlmVmReady(warmingNode), false);
  assert.match(chatUtils.formatChatOptionLabel(warmingNode, bgState), /预热中，暂不可对话/);
  const warmingSelected = chatUtils.resolveChatTargetState("standby-node", bgState, [
    warmingNode,
    customNamedGreen,
  ]);
  assert.equal(warmingSelected.canSend, false);
  assert.equal(warmingSelected.submitTargetVm, null);
  assert.match(warmingSelected.disabledReason, /预热中/);

  // #64: vanished target VM resets selection to "auto" and never submits stale disappeared VM name
  const vanishedTarget = chatUtils.resolveChatTargetState("deleted-blue-vm", bgState, [customNamedGreen]);
  assert.equal(vanishedTarget.targetDisappeared, true);
  assert.equal(vanishedTarget.disappearedVmName, "deleted-blue-vm");
  assert.equal(vanishedTarget.effectiveSelectValue, "auto");
  assert.equal(vanishedTarget.submitTargetVm, "prod-deepseek-node");
});

// ───────────────────── Issue #52, #53, #54 ─────────────────────
test("Issues #52, #53, #54: CPU relay preserved, download failure distinguished from completed, and session hours bounded 0.1~23.5", () => {
  const createDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "CreateVmDialog.tsx"),
    "utf8",
  );
  const relayPanelSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "RelayPanel.tsx"),
    "utf8",
  );

  // #52: CPU must never be rewritten to T4
  assert.doesNotMatch(createDlgSrc, /acc\s*===\s*"CPU"\s*\?\s*"T4"/);
  assert.doesNotMatch(relayPanelSrc, /vm\.accelerator\s*!==\s*"CPU"/);
  assert.match(relayPanelSrc, /"CPU"/);

  // #53: training_completed=true + weights_downloaded=false (or download_failed) shows error + retry, NEVER green checkpoint recovered
  const dlFailed = chatUtils.resolveRelayCompletionView({
    vm_name: "exp-1",
    running: false,
    round: 2,
    phase: "download_failed",
    phase_label: "权重下载失败",
    started_at: null,
    round_started_at: null,
    config: {},
    logs: [],
    child_pid: null,
    training_completed: true,
    weights_downloaded: false,
    download_error: "rclone copyto exited 1",
  });
  assert.equal(dlFailed.isDownloadFailed, true);
  assert.equal(dlFailed.isCompletedAndDownloaded, false);
  assert.equal(dlFailed.canRetryDownload, true);
  assert.doesNotMatch(dlFailed.sub, /已从 Google Drive 回收最终 checkpoint/);
  assert.match(dlFailed.sub, /下载到本地失败/);

  const dlSuccess = chatUtils.resolveRelayCompletionView({
    vm_name: "exp-1",
    running: false,
    round: 2,
    phase: "completed",
    phase_label: "已完成",
    started_at: null,
    round_started_at: null,
    config: {},
    logs: [],
    child_pid: null,
    training_completed: true,
    weights_downloaded: true,
    download_error: null,
    local_weights_path: "D:/ws/checkpoint_final.pt",
  });
  assert.equal(dlSuccess.isDownloadFailed, false);
  assert.equal(dlSuccess.isCompletedAndDownloaded, true);
  assert.match(dlSuccess.sub, /已从 Google Drive 回收最终 checkpoint/);

  // #54: session hours strictly bounded to [0.1, 23.5]
  assert.equal(chatUtils.isValidRelaySessionHours(0.1), true);
  assert.equal(chatUtils.isValidRelaySessionHours(22.5), true);
  assert.equal(chatUtils.isValidRelaySessionHours(23.5), true);
  assert.equal(chatUtils.isValidRelaySessionHours(0), false);
  assert.equal(chatUtils.isValidRelaySessionHours(0.05), false);
  assert.equal(chatUtils.isValidRelaySessionHours(24), false);
  assert.equal(chatUtils.isValidRelaySessionHours(36), false);
  assert.equal(chatUtils.isValidRelaySessionHours(48), false);
  assert.equal(chatUtils.isValidRelaySessionHours(""), false);
  assert.doesNotMatch(relayPanelSrc, /Math\.min\(48/);
});

// ───────────────────── Issues #56 & #57 & #59 ─────────────────────
test("Issues #56, #57, #59: missing CCU shows 未知 (never green 0.00 CU/h), stale cache banner & toast wired, API key rotation guarded", () => {
  // #56: missing ccu -> "未知", never "0.00 CU/h" or "ok"
  const missingWithActive = formatMod.formatBurnRate(null, true);
  assert.equal(missingWithActive.text, "未知");
  assert.equal(missingWithActive.cls, "warn");
  assert.equal(missingWithActive.known, false);

  const missingIdle = formatMod.formatBurnRate(undefined, false);
  assert.equal(missingIdle.text, "未知");
  assert.equal(missingIdle.cls, "dim");
  assert.notEqual(missingIdle.cls, "ok");

  const missingNan = formatMod.formatBurnRate({ balance: 10, burn_rate_hourly: Number.NaN, assignments_count: 1 }, true);
  assert.equal(missingNan.text, "未知");

  const validZero = formatMod.formatBurnRate({ balance: 50, burn_rate_hourly: 0, assignments_count: 0 }, false);
  assert.equal(validZero.text, "0.00 CU/h");
  assert.equal(validZero.cls, "ok");

  const validActive = formatMod.formatBurnRate({ balance: 50, burn_rate_hourly: 1.96, assignments_count: 1 }, true);
  assert.equal(validActive.text, "1.96 CU/h");
  assert.equal(validActive.cls, "warn");

  const headerSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "Header.tsx"), "utf8");
  const deleteDlgSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "DeleteVmDialog.tsx"), "utf8");
  assert.match(headerSrc, /formatBurnRate/);
  assert.match(deleteDlgSrc, /formatBurnRate/);

  // #57 & #59 in App.tsx
  const appSrc = fs.readFileSync(path.join(repoRoot, "src", "App.tsx"), "utf8");
  assert.match(appSrc, /ov\.is_stale/);
  assert.match(appSrc, /ov\.last_ok_timestamp/);
  assert.match(appSrc, /最新查询失败/);
  assert.match(appSrc, /轮换 API Key 失败/);
});

// ───────────────────── Issues #58 & #71 ─────────────────────
test("Issues #58 & #71: ScriptDialog disables save on read failure and guards unsaved edits before closing", () => {
  const scriptDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "ScriptDialog.tsx"),
    "utf8",
  );
  assert.match(scriptDlgSrc, /loadedOk/);
  assert.match(scriptDlgSrc, /loadError/);
  assert.match(scriptDlgSrc, /const canSave = !saving && !loading && loadedOk && !loadError/);
  assert.match(scriptDlgSrc, /origTrain/);
  assert.match(scriptDlgSrc, /origReq/);
  assert.match(scriptDlgSrc, /isDirty/);
  assert.match(scriptDlgSrc, /confirmDiscard/);
  assert.match(scriptDlgSrc, /有未保存的修改，确定放弃编辑并关闭吗/);
});

// ───────────────────── Issue #65 ─────────────────────
test("Issue #65: idle LLM displays 0.0 tok/s realtime rate and labels last_request_tps separately", () => {
  const idleTps = formatMod.formatRealtimeTps({
    instantTps: 0,
    lastRequestTps: 42.6,
    peakTps: 50.0,
    activeStreams: 0,
  });
  assert.equal(idleTps.realtimeTps, 0);
  assert.equal(idleTps.realtimeText, "0.0 tok/s");
  assert.equal(idleTps.isIdle, true);
  assert.equal(idleTps.lastRequestText, "上次请求: 42.6 tok/s");

  const activeTps = formatMod.formatRealtimeTps({
    instantTps: 35.2,
    lastRequestTps: 42.6,
    activeStreams: 1,
  });
  assert.equal(activeTps.realtimeTps, 35.2);
  assert.equal(activeTps.isIdle, false);

  const idlePill = chatUtils.resolveVmStatusPillSpec(
    {
      name: "llm-blue",
      on_server: true,
      busy: null,
      telemetry: { llm: { online: true, backend_healthy: true, instant_tps: 0, last_request_tps: 42.6 } },
    },
    false,
  );
  assert.equal(idlePill.text, "LLM · 0.0 tok/s (空闲)");

  for (const rel of ["src/App.tsx", "src/components/LlmChatModal.tsx", "src/components/VmCard.tsx"]) {
    const content = fs.readFileSync(path.join(repoRoot, rel), "utf8");
    assert.doesNotMatch(content, /instant_tps\)\s*\|\|\s*num\([^)]*last_request_tps/);
    assert.doesNotMatch(content, /instantTps\s*>\s*0\s*\?\s*instantTps\s*:\s*lastTps/);
  }
});

// ───────────────────── Issues #66, #67, #68, #69, #70 ─────────────────────
test("Issues #66, #67, #68, #69, #70: modal scroll layout, general GDrive opt-in, CUDA hardware matrix, multiline envVars, and focus trap", () => {
  // #66: CSS modal 3-part layout & adv-config-box no overflow:hidden squash
  const css = fs.readFileSync(path.join(repoRoot, "src", "styles.css"), "utf8");
  assert.match(css, /\.modal-body\s*>\s*\*\s*\{\s*flex-shrink:\s*0;\s*\}/);
  const advBoxMatch = css.match(/\.adv-config-box\s*\{([^}]*)\}/);
  assert.ok(advBoxMatch, ".adv-config-box rule must exist");
  assert.match(advBoxMatch[1], /flex-shrink:\s*0/);
  assert.doesNotMatch(advBoxMatch[1], /overflow:\s*hidden/);

  // #67: general mode does not require GDrive by default
  assert.equal(
    chatUtils.requiresGdriveConfig({
      bootMode: "general",
      syncRclone: false,
      mountGdrive: false,
      gdriveModelPath: "",
    }),
    false,
  );
  assert.equal(
    chatUtils.requiresGdriveConfig({
      bootMode: "relay",
      syncRclone: false,
      mountGdrive: false,
    }),
    true,
  );
  assert.equal(
    chatUtils.requiresGdriveConfig({
      bootMode: "llm",
      syncRclone: false,
      mountGdrive: false,
      gdriveModelPath: "gdrive:models/qwen",
    }),
    true,
  );

  // #68: CUDA engines (vllm, llama.cpp, transformers) rejected on CPU and TPU, allowed on NVIDIA GPUs
  for (const cudaEng of ["vllm", "llama.cpp", "transformers"]) {
    for (const nonCudaHw of ["CPU", "V5E1", "V6E1"]) {
      assert.equal(
        chatUtils.isLlmEngineCompatibleWithAccelerator(cudaEng, nonCudaHw),
        false,
        `${cudaEng} must be incompatible with ${nonCudaHw}`,
      );
    }
    for (const nvidiaHw of ["T4", "L4", "G4", "A100", "H100"]) {
      assert.equal(
        chatUtils.isLlmEngineCompatibleWithAccelerator(cudaEng, nvidiaHw),
        true,
        `${cudaEng} must be compatible with ${nvidiaHw}`,
      );
    }
  }
  assert.equal(chatUtils.isLlmEngineCompatibleWithAccelerator("ollama", "CPU"), true);
  assert.equal(chatUtils.isLlmEngineCompatibleWithAccelerator("builtin", "V5E1"), true);

  // #69: envVars uses multiline <textarea>
  const createDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "CreateVmDialog.tsx"),
    "utf8",
  );
  assert.match(createDlgSrc, /v:\s*"transformers"[^\n]*requiresCuda:\s*true/);
  assert.match(createDlgSrc, /const \[syncRclone, setSyncRclone\] = useState\(false\)/);
  assert.match(createDlgSrc, /<textarea[\s\S]{0,120}value=\{envVars\}/);

  // #70: Focus trap cycling & dialog ARIA attributes
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, 3, false), 0);
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, 0, true), 3);
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, -1, false), 0);
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, -1, true), 3);
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, 1, false), 2);
  assert.equal(chatUtils.computeNextFocusTrapIndex(4, 2, true), 1);

  const uiSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "ui.tsx"), "utf8");
  const chatModalSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "LlmChatModal.tsx"),
    "utf8",
  );
  assert.match(uiSrc, /role="dialog"/);
  assert.match(uiSrc, /aria-modal="true"/);
  assert.match(uiSrc, /useModalFocusTrap/);
  assert.match(chatModalSrc, /role="dialog"/);
  assert.match(chatModalSrc, /aria-modal="true"/);
  assert.match(chatModalSrc, /useModalFocusTrap/);
});

// ───────────────────── Issues #73, #74, #76 ─────────────────────
test("Issues #73, #74, #76: offline relay card preserved, recent session avoids prefix guessing, and relay status/drawer toggle works", () => {
  // #74: unverified history session with `llm-` prefix MUST NOT guess "llm" boot_mode
  const unverifiedRecent = chatUtils.buildPresetFromRecentSession({
    session_name: "llm-debug-shell",
    workspace_name: "ws",
    workspace_path: "D:/ws",
    endpoint: null,
    accelerator: "T4",
    machine_shape: "HIGH_RAM",
    created_at: null,
    terminated_at: null,
    last_event_at: "2026-11-19T12:00:00Z",
    exec_count: 1,
    file_op_count: 0,
    last_summary: "exec",
    profile_saved: false,
  });
  assert.equal(unverifiedRecent.boot_mode, "general");
  assert.equal(unverifiedRecent.unverified_history_profile, true);

  // #74: saved profile restores exact boot_mode & relay_config
  const savedRelayRecent = chatUtils.buildPresetFromRecentSession({
    session_name: "cpu-train-job",
    workspace_name: "ws",
    workspace_path: "D:/ws2",
    endpoint: null,
    accelerator: "CPU",
    machine_shape: "STANDARD",
    created_at: null,
    terminated_at: null,
    last_event_at: "2026-11-19T12:00:00Z",
    exec_count: 3,
    file_op_count: 2,
    last_summary: "train",
    profile_saved: true,
    boot_mode: "relay",
    relay_config: {
      gpu_type: "CPU",
      high_mem: false,
      gdrive_dir: "gdrive:colab_relay/cpu-train-job",
      max_session_hours: 18,
      backup_interval_minutes: 20,
      total_epochs: 80,
      reset_done_flag: false,
    },
  });
  assert.equal(savedRelayRecent.boot_mode, "relay");
  assert.equal(savedRelayRecent.accelerator, "CPU");
  assert.equal(savedRelayRecent.relay_config?.gpu_type, "CPU");
  assert.equal(savedRelayRecent.unverified_history_profile, false);

  // #76: active relay without remote telemetry yet MUST NOT report "空闲"
  const activeRelayPill = chatUtils.resolveVmStatusPillSpec(
    {
      name: "relay-vm",
      on_server: true,
      busy: null,
      running_task: null,
      telemetry: null,
      relay: {
        vm_name: "relay-vm",
        running: true,
        round: 1,
        phase: "installing_env",
        phase_label: "安装环境",
        started_at: 1000,
        round_started_at: 1000,
        config: {},
        logs: [],
        child_pid: 123,
      },
    },
    false,
  );
  assert.notEqual(activeRelayPill.text, "空闲");
  assert.equal(activeRelayPill.text, "安装环境");

  // #73: offline relay record card status & clear action
  const offlineFailedPill = chatUtils.resolveVmStatusPillSpec(
    {
      name: "relay-failed-vm",
      on_server: false,
      busy: null,
      running_task: null,
      telemetry: null,
      relay: {
        vm_name: "relay-failed-vm",
        running: false,
        round: 1,
        phase: "error",
        phase_label: "训练异常",
        started_at: 1000,
        round_started_at: 1000,
        config: {},
        logs: ["Traceback: RuntimeError"],
        child_pid: null,
        last_error: "RuntimeError",
      },
    },
    false,
  );
  assert.equal(offlineFailedPill.text, "接力失败 · 实例已下线");

  const vmCardSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "VmCard.tsx"), "utf8");
  assert.doesNotMatch(vmCardSrc, /Boolean\(vm\.relay\?\.running\)\s*\|\|\s*relayExpanded/);
  assert.match(vmCardSrc, /const showRelayDrawer = relayExpanded;/);
  assert.match(vmCardSrc, /清除记录/);
});

// ───────────────────── Issues #78, #80, #82, #83, #87 ─────────────────────
test("Issue #78: server list failure keeps offline relay card in billing-unknown state until confirmed unassigned", () => {
  // 1) Overview unknown detection works for both explicit server_list_known=false and simulated error without cache
  assert.equal(chatUtils.isOverviewServerListUnknown(null), true);
  assert.equal(
    chatUtils.isOverviewServerListUnknown({
      timestamp: 2000,
      server_list_known: false,
      error: "503 Service Unavailable",
      is_stale: false,
    }),
    true,
  );
  assert.equal(
    chatUtils.isOverviewServerListUnknown({
      timestamp: 2000,
      error: "network error",
      is_stale: false,
    }),
    true,
  );
  assert.equal(
    chatUtils.isOverviewServerListUnknown({
      timestamp: 2000,
      server_list_known: true,
      error: null,
      is_stale: false,
    }),
    false,
  );
  assert.equal(
    chatUtils.isOverviewServerListUnknown({
      timestamp: 2000,
      server_list_known: true,
      error: null,
      is_stale: true,
      stale_reason: "timeout within 60s cache",
    }),
    false,
  );

  // 2) Offline relay pill reports 计费状态未知 when server list failed, and 实例已下线/已释放 only after confirmation
  const offlineRelayVm = {
    name: "relay-job",
    on_server: false,
    busy: null,
    running_task: null,
    telemetry: null,
    relay: {
      vm_name: "relay-job",
      running: false,
      round: 2,
      phase: "error",
      phase_label: "OOM",
      started_at: 1000,
      round_started_at: 1000,
      config: {},
      logs: ["err"],
      child_pid: null,
      last_error: "OOM",
    },
  };
  const pillUnknown = chatUtils.resolveVmStatusPillSpec(offlineRelayVm, false, true);
  assert.match(pillUnknown.text, /计费状态未知/);
  assert.doesNotMatch(pillUnknown.text, /已下线|已释放/);

  const pillConfirmed = chatUtils.resolveVmStatusPillSpec(offlineRelayVm, false, false);
  assert.equal(pillConfirmed.text, "接力失败 · 实例已下线");

  const headerSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "Header.tsx"), "utf8");
  const vmCardSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "VmCard.tsx"), "utf8");
  assert.match(headerSrc, /serverUnknown\s*\?\s*"未知"\s*:\s*ov\.server_count/);
  assert.match(vmCardSrc, /isBillingUnknown[\s\S]{0,400}云端实例计费状态未知/);
});

test("Issue #82: mergeOverviewSnapshot ignores late older is_stale snapshots and preserves newer online VMs", () => {
  const onlineVm = { name: "vm-online", on_server: true };
  const freshSnap = {
    timestamp: 2000,
    seq: 5,
    last_ok_timestamp: 2000,
    is_stale: false,
    stale_reason: null,
    server_list_known: true,
    auth_ok: true,
    error: null,
    ccu: { balance: 50, burn_rate_hourly: 1.96, assignments_count: 1 },
    server_count: 1,
    gdrive: null,
    env: {},
    vms: [onlineVm],
    recent_sessions: [],
  };

  // 1) Late-arriving older stale snapshot (timestamp 1900 < 2000) MUST be completely ignored
  const lateStale = {
    ...freshSnap,
    timestamp: 1900,
    seq: undefined,
    last_ok_timestamp: 1900,
    is_stale: true,
    stale_reason: "delayed timeout",
    server_count: 0,
    vms: [],
  };
  const afterLate = chatUtils.mergeOverviewSnapshot(freshSnap, lateStale);
  assert.equal(afterLate, freshSnap);
  assert.equal(afterLate.vms.length, 1);
  assert.equal(afterLate.is_stale, false);

  // 2) Same timestamp (2000) with older last_ok_timestamp (1950) MUST also be ignored
  const sameSecOlderData = {
    ...lateStale,
    timestamp: 2000,
    last_ok_timestamp: 1950,
  };
  assert.equal(chatUtils.mergeOverviewSnapshot(freshSnap, sameSecOlderData), freshSnap);

  // 3) Out-of-order lower monotonic seq (seq 4 < 5) MUST be ignored even if timestamp matches
  const lowerSeq = {
    ...freshSnap,
    seq: 4,
    is_stale: true,
    stale_reason: "out-of-order seq",
    vms: [],
  };
  assert.equal(chatUtils.mergeOverviewSnapshot(freshSnap, lowerSeq), freshSnap);

  // 4) Genuinely later query failure (timestamp 2010 > 2000) surfaces independent stale banner WITHOUT wiping online VMs
  const laterFailure = {
    ...freshSnap,
    timestamp: 2010,
    seq: 6,
    last_ok_timestamp: 1900,
    is_stale: true,
    stale_reason: "503 upstream timeout",
    server_count: 0,
    vms: [],
  };
  const mergedLaterErr = chatUtils.mergeOverviewSnapshot(freshSnap, laterFailure);
  assert.equal(mergedLaterErr.vms.length, 1);
  assert.equal(mergedLaterErr.vms[0].name, "vm-online");
  assert.equal(mergedLaterErr.last_ok_timestamp, 2000);
  assert.equal(mergedLaterErr.is_stale, true);
  assert.equal(mergedLaterErr.stale_reason, "503 upstream timeout");

  // 5) Regression (#57 + #82): newer seq stale fallback whose timestamp was pinned to prev_ok_ts (2000)
  //    after an 8s cached_fresh poll (timestamp 2003, seq 6, last_ok_timestamp 2000) MUST NOT be dropped!
  const afterCachedPoll = {
    ...freshSnap,
    timestamp: 2003,
    seq: 6,
    last_ok_timestamp: 2000,
    is_stale: false,
    stale_reason: null,
  };
  const manualRefreshStaleFallback = {
    ...freshSnap,
    timestamp: 2000,
    seq: 7,
    last_ok_timestamp: 2000,
    is_stale: true,
    stale_reason: "502 Bad Gateway",
  };
  const mergedManualFail = chatUtils.mergeOverviewSnapshot(afterCachedPoll, manualRefreshStaleFallback);
  assert.equal(mergedManualFail.is_stale, true);
  assert.equal(mergedManualFail.stale_reason, "502 Bad Gateway");
  assert.equal(mergedManualFail.seq, 7);
  assert.equal(mergedManualFail.vms.length, 1);

  // 6) #78 sanitization: resolveRelayCompletionView strips prior 停费核验通过 label when billingUnknown=true
  const sanitizedView = chatUtils.resolveRelayCompletionView(
    {
      vm_name: "stopped-job",
      running: false,
      round: 1,
      phase: "stopped",
      phase_label: "已关闭接力并销毁云端实例（停费核验通过）",
      started_at: 1000,
      round_started_at: 1000,
      config: {},
      logs: [],
      child_pid: null,
      last_error: null,
    },
    true,
  );
  assert.doesNotMatch(sanitizedView.stateLabel, /停费核验通过|销毁云端实例|已释放/);
  assert.match(sanitizedView.stateLabel, /云端计费状态未知/);
});

test("Issues #80, #83, #87: setScriptDirty IPC, legacy GDrive confirmation UI, and topmost window-close modal stacking", async () => {
  mockTauriCore.invokeCalls.length = 0;
  await apiMod.api.setScriptDirty(true);
  await apiMod.api.confirmAppClose("force_exit");
  await apiMod.api.startRelay(
    "legacy-run",
    {
      gpu_type: "T4",
      high_mem: true,
      gdrive_dir: "gdrive:colab_relay",
      max_session_hours: 20,
      backup_interval_minutes: 30,
      total_epochs: 10,
      reset_done_flag: false,
    },
    "D:/ws",
    "migrate",
  );
  assert.deepEqual(mockTauriCore.invokeCalls[0], {
    cmd: "set_script_dirty",
    args: { dirty: true },
  });
  assert.deepEqual(mockTauriCore.invokeCalls[1], {
    cmd: "confirm_app_close",
    args: { mode: "force_exit" },
  });
  assert.equal(mockTauriCore.invokeCalls[2].cmd, "start_relay");
  assert.equal(mockTauriCore.invokeCalls[2].args.legacyGdriveAction, "migrate");

  // #87: `.overlay.overlay-topmost` z-index strictly exceeds `.overlay` (100) and stays below `.toasts` (200)
  const css = fs.readFileSync(path.join(repoRoot, "src", "styles.css"), "utf8");
  const topmostMatch = css.match(/\.overlay\.overlay-topmost\s*\{\s*z-index:\s*(\d+)\s*;\s*\}/);
  assert.ok(topmostMatch, ".overlay.overlay-topmost z-index rule must exist");
  const topmostZ = Number(topmostMatch[1]);
  assert.ok(topmostZ > 100 && topmostZ < 200, `Expected 100 < topmostZ < 200, got ${topmostZ}`);

  // #87 & #83: CloseAppRelayDialog and CloseAppUnsavedScriptDialog must be rendered AFTER LlmChatModal & DeleteVmDialog in App.tsx
  const appSrc = fs.readFileSync(path.join(repoRoot, "src", "App.tsx"), "utf8");
  const chatIdx = appSrc.indexOf("<LlmChatModal");
  const deleteIdx = appSrc.indexOf("<DeleteVmDialog");
  const closeRelayIdx = appSrc.indexOf("<CloseAppRelayDialog");
  const closeScriptIdx = appSrc.indexOf("<CloseAppUnsavedScriptDialog");
  assert.ok(chatIdx > 0 && deleteIdx > 0 && closeRelayIdx > 0 && closeScriptIdx > 0);
  assert.ok(
    closeRelayIdx > chatIdx && closeRelayIdx > deleteIdx,
    "CloseAppRelayDialog must appear after LlmChatModal and DeleteVmDialog in DOM order",
  );
  assert.ok(
    closeScriptIdx > chatIdx && closeScriptIdx > deleteIdx,
    "CloseAppUnsavedScriptDialog must appear after LlmChatModal and DeleteVmDialog in DOM order",
  );
  assert.match(appSrc, /close-requested-unsaved-script/);

  const stopDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "StopRelayDialog.tsx"),
    "utf8",
  );
  assert.match(stopDlgSrc, /export function CloseAppRelayDialog[\s\S]*?<Modal[\s\S]{0,80}topPriority/);
  assert.match(stopDlgSrc, /export function CloseAppUnsavedScriptDialog[\s\S]*?<Modal[\s\S]{0,80}topPriority/);

  const chatModalSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "LlmChatModal.tsx"),
    "utf8",
  );
  assert.match(chatModalSrc, /if\s*\(modalEl\s*&&\s*!isTopmostModal\(modalEl\)\)\s*return;/);

  const relayPanelSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "RelayPanel.tsx"),
    "utf8",
  );
  assert.match(relayPanelSrc, /迁移旧断点并续训/);
  assert.match(relayPanelSrc, /忽略旧断点全新开始/);
  assert.match(relayPanelSrc, /legacyConfirmPending/);
});

test("Per-instance Keep-Alive: default creation config, interval validation, badge states, and IPC wiring", async () => {
  assert.equal(chatUtils.isValidKeepAliveInterval(30), true);
  assert.equal(chatUtils.isValidKeepAliveInterval(90), true);
  assert.equal(chatUtils.isValidKeepAliveInterval(600), true);
  assert.equal(chatUtils.isValidKeepAliveInterval(29), false);
  assert.equal(chatUtils.isValidKeepAliveInterval(601), false);
  assert.equal(chatUtils.isValidKeepAliveInterval(Number.NaN), false);

  const badgeOff = chatUtils.resolveKeepAliveBadge(null);
  assert.equal(badgeOff.tone, "muted");
  assert.equal(badgeOff.label, "保活未开启");

  const badgeBusy = chatUtils.resolveKeepAliveBadge({
    enabled: true,
    interval_seconds: 90,
    kernel_ping: true,
    state: "active",
    last_ok_at: 1795400000,
    kernel_ping_state: "skipped_busy",
    success_count: 5,
    last_error: null,
  });
  assert.equal(badgeBusy.tone, "ok");
  assert.match(badgeBusy.label, /保活运行中 \(90s\)/);
  assert.match(badgeBusy.detail, /忙时已自动避让/);
  assert.match(badgeBusy.detail, /累计 5 次/);

  const badgeErr = chatUtils.resolveKeepAliveBadge({
    enabled: true,
    interval_seconds: 60,
    kernel_ping: true,
    state: "error",
    last_ok_at: null,
    kernel_ping_state: "error",
    success_count: 0,
    last_error: "隧道保活失败: HTTP 503",
  });
  assert.equal(badgeErr.tone, "bad");
  assert.equal(badgeErr.label, "保活异常");
  assert.match(badgeErr.detail, /HTTP 503/);

  const restored = chatUtils.buildPresetFromRecentSession({
    session_name: "a100-work",
    workspace_name: "proj",
    workspace_path: "C:/proj",
    endpoint: "m-s-ka01",
    accelerator: "A100",
    machine_shape: "HIGH_RAM",
    created_at: null,
    terminated_at: null,
    last_event_at: "2026-11-24T00:00:00Z",
    exec_count: 1,
    file_op_count: 0,
    last_summary: "done",
    boot_mode: "general",
    keep_alive: { enabled: true, interval_seconds: 180, kernel_ping: false },
    profile_saved: true,
  });
  assert.deepEqual(restored.keep_alive, {
    enabled: true,
    interval_seconds: 180,
    kernel_ping: false,
  });

  mockTauriCore.invokeCalls.length = 0;
  await apiMod.api.setVmKeepAlive("vm-a100", {
    enabled: true,
    interval_seconds: 90,
    kernel_ping: true,
  });
  await apiMod.api.pingVmKeepAlive("vm-a100");
  assert.deepEqual(mockTauriCore.invokeCalls[0], {
    cmd: "set_vm_keep_alive",
    args: {
      name: "vm-a100",
      config: { enabled: true, interval_seconds: 90, kernel_ping: true },
    },
  });
  assert.deepEqual(mockTauriCore.invokeCalls[1], {
    cmd: "ping_vm_keep_alive",
    args: { name: "vm-a100" },
  });

  const createDlgSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "CreateVmDialog.tsx"),
    "utf8",
  );
  assert.match(createDlgSrc, /DEFAULT_KEEP_ALIVE/);
  assert.match(createDlgSrc, /实例保活策略/);
  assert.match(createDlgSrc, /keep_alive:\s*\{/);

  const vmCardSrc = fs.readFileSync(
    path.join(repoRoot, "src", "components", "VmCard.tsx"),
    "utf8",
  );
  assert.match(vmCardSrc, /function KeepAliveBar/);
  assert.match(vmCardSrc, /立即保活/);
});
