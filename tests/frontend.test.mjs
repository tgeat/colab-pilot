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
const mockTauriCoreUrl = `data:text/javascript;base64,${Buffer.from(
  "export async function invoke() { return null; }\n",
  "utf8",
).toString("base64")}`;
const apiUrl = transpileToDataUrl("src/api.ts", {
  "@tauri-apps/api/core": mockTauriCoreUrl,
});

const formatMod = await import(formatUrl);
const chatUtils = await import(chatUtilsUrl);
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
  // Partial `<th` while streaming must not flash raw tag in mainText
  const partial = chatUtils.extractThinkParts("<th", true);
  assert.equal(partial.mainText, "");

  // Unclosed `<think>` while streaming
  const unclosed = chatUtils.extractThinkParts("<think>step 1", true);
  assert.equal(unclosed.thinkText, "step 1");
  assert.equal(unclosed.mainText, "");

  // Multiple `<think>` blocks
  const multi = chatUtils.extractThinkParts("<think>first</think>Hello <think>second</think>world!");
  assert.equal(multi.thinkText, "first\n\nsecond");
  assert.equal(multi.mainText, "Hello world!");

  // stripThinkBlocks returns empty string when only thinking exists
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

  // Unrelated VM with slot="blue" must NOT be hijacked into `qwen` blue-green cluster (H-4)
  const unrelatedVm = {
    name: "other-blue",
    on_server: true,
    telemetry: { llm: { online: true, slot: "blue", role: "active" } },
  };
  const isolatedStatus = chatUtils.resolveBlueGreenStatus(staleBg, [unrelatedVm]);
  assert.equal(isolatedStatus.blueVm, undefined);
  assert.equal(isolatedStatus.clusterState, "offline");
  assert.equal(isolatedStatus.anyOnlineVm, true);

  // Matching `qwen-blue` VM online -> clusterState === "online"
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

test("all 14 IPC methods are exported, zero colab console remnants remain, and new-VM controls are wired", () => {
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
  ];
  for (const fn of expectedIpcMethods) {
    assert.equal(typeof apiMod.api[fn], "function", `api.${fn} must be a function`);
  }

  const appSrc = fs.readFileSync(path.join(repoRoot, "src", "App.tsx"), "utf8");
  const vmCardSrc = fs.readFileSync(path.join(repoRoot, "src", "components", "VmCard.tsx"), "utf8");
  assert.match(appSrc, /<CreateVmDialog/);
  assert.match(appSrc, /<ScriptDialog/);
  assert.match(appSrc, /<StopRelayDialog/);
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
