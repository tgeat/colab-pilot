import { useEffect, useRef, useState } from "react";
import {
  Activity, Clock, Cpu, HardDrive, Layers, Loader2, MemoryStick, Network, Radar,
  Repeat2, ScrollText, Trash2, Zap,
} from "lucide-react";
import type { RelayConfig, VmView } from "../types";
import { fmtDuration, shortEndpoint } from "../format";
import { ColabIcon, KV, Meter } from "./ui";
import { RelayPanel } from "./RelayPanel";

function statusPill(vm: VmView) {
  if (vm.busy === "creating") return <span className="pill pill-blue"><span className="dot pulse" />创建中</span>;
  if (vm.busy === "deleting") return <span className="pill pill-red"><span className="dot pulse" />销毁中</span>;
  if (vm.busy === "probing") return <span className="pill pill-blue"><span className="dot pulse" />部署探针</span>;
  if (vm.relay?.running && !vm.on_server) return <span className="pill pill-amber"><span className="dot pulse" />换棒中（未计费）</span>;
  if (vm.relay?.running) return <span className="pill pill-blue"><span className="dot pulse" />接力训练中</span>;
  if (vm.running_task) return <span className="pill pill-amber"><span className="dot pulse" />执行中</span>;
  if (vm.on_server) return <span className="pill pill-green"><span className="dot" />在线空闲 · 计费中</span>;
  return <span className="pill pill-gray"><span className="dot" />离线</span>;
}

function trainingStatusText(s?: string) {
  const map: Record<string, string> = {
    pulling_checkpoint: "拉取断点",
    training: "训练中",
    syncing_gdrive: "同步网盘",
    handover_sync: "交接同步",
    handover_exit: "已交接退出",
    completed: "已完成",
  };
  return s ? map[s] ?? s : "未启动";
}

export function VmCard(props: {
  vm: VmView;
  onDelete: (vm: VmView) => void;
  onProbe: (vm: VmView) => void;
  onStartRelay: (vm: VmView, cfg: RelayConfig) => void;
  onStopRelay: (vm: VmView) => void;
  pending: boolean;
}) {
  const { vm } = props;
  const t = vm.telemetry;
  const tr = t?.training ?? null;
  const relay = vm.relay;
  const [tab, setTab] = useState<"threads" | "logs">(relay?.running ? "logs" : "threads");
  const logRef = useRef<HTMLDivElement>(null);
  const stale = vm.telemetry_age !== null && vm.telemetry_age > 20;

  useEffect(() => {
    if (tab === "logs" && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [tab, relay?.logs?.length]);

  const acc = vm.accelerator.toUpperCase();
  const avatarCls = acc.startsWith("V") ? "tpu" : acc === "CPU" ? "cpu" : "";
  const gpu = t?.gpu;
  const procs = t?.threads?.processes ?? [];
  const kernel = procs.find((p) => p.role === "kernel" || p.role === "train") ?? null;
  const rclone = t?.threads?.rclone_process ?? null;
  const deleting = vm.busy === "deleting";

  return (
    <section className={`vm-card ${relay?.running ? "relay-on" : ""} ${deleting ? "deleting" : ""}`}>
      {/* 头部 */}
      <div className="vm-head">
        <div className={`vm-avatar ${avatarCls}`} title={`Google Colab · ${vm.accelerator}`}>
          <ColabIcon size={26} />
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="vm-title">
            <span className="vm-name">{vm.name}</span>
            <span className={`chip ${avatarCls === "tpu" ? "chip-purple" : "chip-blue"}`}>{vm.accelerator}</span>
            <span className="chip">{vm.high_ram ? "High-RAM" : "Standard"}</span>
            {vm.orphan && <span className="chip chip-red" title="服务端存在但不在 colab-cli 会话表中的实例，已自动纳入管控以防隐形计费">外部实例</span>}
            {statusPill(vm)}
          </div>
          <div className="vm-meta">
            <span title={vm.endpoint ?? ""}><Network size={12} />{shortEndpoint(vm.endpoint)}</span>
            <span><Clock size={12} />已运行 {fmtDuration(t?.uptime_seconds)}</span>
            {t && (
              <span className={stale ? "warn" : ""}>
                <Radar size={12} />遥测 {vm.telemetry_age !== null ? `${vm.telemetry_age.toFixed(0)}s 前` : "—"}
                {stale ? "（探针可能已停止）" : ""}
              </span>
            )}
          </div>
        </div>
        <div className="vm-head-actions">
          <button
            className="btn btn-sm"
            onClick={() => props.onProbe(vm)}
            disabled={!vm.on_server || !vm.managed || !!vm.busy || relay?.phase === "training"}
            title="在云端部署/重启资源与线程遥测探针"
          >
            {vm.busy === "probing" ? <Loader2 size={14} className="spin" /> : <Activity size={14} />}
            {t ? "重启探针" : "部署探针"}
          </button>
          <button
            className="btn btn-sm btn-danger-ghost"
            onClick={() => props.onDelete(vm)}
            disabled={deleting || vm.busy === "creating"}
            title="unassign 并核验服务端计费列表，彻底停止计费"
          >
            {deleting ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />}
            删除虚拟机
          </button>
        </div>
      </div>

      {/* 主体三栏 */}
      <div className="vm-body">
        {/* ① 运行状态 */}
        <div className="vm-col">
          <div className="col-title">
            <span className="left"><ColabIcon size={15} />运行状态</span>
            <span className="hint">{vm.variant}</span>
          </div>
          <div className="kv">
            <KV label="服务端计费" value={vm.on_server ? "计费中" : "未计费"} cls={vm.on_server ? "warn" : "ok"} />
            <KV label="Kernel 任务" value={relay?.phase === "training" ? "train.py" : vm.running_task ?? "空闲"} mono />
            <KV label="训练状态" value={trainingStatusText(tr?.status)} />
            <KV label="当前 Loss" value={tr?.loss !== undefined ? tr.loss.toFixed(5) : "—"} mono />
            <KV label="云端 rclone.conf" value={t?.files?.has_rclone_conf ? "已就绪" : "未上传"} cls={t?.files?.has_rclone_conf ? "ok" : "dim"} />
            <KV
              label="本地断点 checkpoint.pt"
              value={t?.files?.has_checkpoint ? `${t.files.checkpoint_size_mb} MB` : "无"}
              cls={t?.files?.has_checkpoint ? "ok" : "dim"}
            />
            <KV label="下次网盘热备" value={tr?.next_backup_in_seconds !== undefined ? fmtDuration(tr.next_backup_in_seconds) : "—"} mono />
            <KV label="最近热备 Epoch" value={tr?.last_backup_epoch ?? "—"} mono />
            <KV
              wide
              label="最近执行"
              mono
              value={vm.last_execution ? `${vm.last_execution[0]} @ ${vm.last_execution[2]}` : "—"}
              title={vm.last_execution ? vm.last_execution.join(" ") : ""}
            />
          </div>
        </div>

        {/* ② 资源状态 */}
        <div className="vm-col">
          <div className="col-title">
            <span className="left"><Layers size={14} />资源状态</span>
            <span className="hint" title={t?.cpu?.model}>{t ? `${t.cpu.cores} vCPU` : ""}</span>
          </div>
          {t ? (
            <div className="meters">
              <Meter
                icon={<Zap size={13} />}
                name={gpu?.available ? `GPU · ${gpu.name}` : "GPU"}
                value={gpu?.available ? gpu.gpu_util_percent ?? 0 : 0}
                display={gpu?.available ? undefined : "N/A"}
                sub={gpu?.available
                  ? `${gpu.temperature_c ?? 0}°C · ${gpu.power_draw_w ?? 0}/${gpu.power_limit_w ?? 0} W`
                  : "CPU 运行时或未检测到 nvidia-smi"}
              />
              <Meter
                icon={<MemoryStick size={13} />}
                name="显存 VRAM"
                value={gpu?.available ? gpu.vram_percent ?? 0 : 0}
                display={gpu?.available ? undefined : "N/A"}
                sub={gpu?.available ? `${((gpu.vram_used_mb ?? 0) / 1024).toFixed(1)} / ${((gpu.vram_total_mb ?? 0) / 1024).toFixed(1)} GB` : "—"}
              />
              <Meter
                icon={<Cpu size={13} />}
                name="CPU"
                value={t.cpu.percent}
                sub={`负载 ${t.cpu.load_avg.join(" / ")} · ${t.cpu.model}`}
              />
              <Meter
                icon={<MemoryStick size={13} />}
                name="内存 RAM"
                value={t.memory.percent}
                sub={`${t.memory.used_gb} / ${t.memory.total_gb} GB`}
              />
              <Meter
                icon={<HardDrive size={13} />}
                name="磁盘 /content"
                value={t.disk.percent}
                sub={`${t.disk.used_gb} / ${t.disk.total_gb} GB · 剩余 ${t.disk.free_gb} GB`}
              />
            </div>
          ) : (
            <div className="probe-missing">
              <ColabIcon size={28} />
              <div>
                {vm.busy === "creating"
                  ? "正在向 Google Colab 申请实例…"
                  : !vm.on_server
                    ? "实例尚未在服务端就绪"
                    : vm.orphan
                      ? "外部创建的实例不在 colab-cli 会话表中，无法部署探针"
                      : vm.telemetry_error ?? "尚未部署遥测探针"}
              </div>
              {vm.on_server && vm.managed && !vm.busy && (
                <button className="btn btn-sm btn-primary" onClick={() => props.onProbe(vm)}>
                  <Activity size={14} />部署遥测探针
                </button>
              )}
            </div>
          )}
        </div>

        {/* ③ 接力训练 */}
        <div className="vm-col relay-col">
          <div className="col-title">
            <span className="left"><Repeat2 size={14} />24h+ 接力训练</span>
            <span className="hint">rclone 16 线程 · 原子热备</span>
          </div>
          <RelayPanel
            vm={vm}
            pending={props.pending}
            onStart={(cfg) => props.onStartRelay(vm, cfg)}
            onStopRequest={() => props.onStopRelay(vm)}
          />
        </div>
      </div>

      {/* 底部：线程状态 / 接力日志 */}
      <div className="vm-foot">
        <div className="tabs">
          <button className={`tab ${tab === "threads" ? "active" : ""}`} onClick={() => setTab("threads")}>
            <Activity size={14} />线程与进程
            <span className="count">{t?.threads?.total_threads ?? 0}</span>
          </button>
          <button className={`tab ${tab === "logs" ? "active" : ""}`} onClick={() => setTab("logs")}>
            <ScrollText size={14} />接力日志
            <span className="count">{relay?.logs?.length ?? 0}</span>
          </button>
        </div>
        <div className="tab-body">
          {tab === "threads" ? (
            <>
              <div className="thread-chips">
                <div className="tchip">云端进程 <b>{t?.threads?.total_processes ?? 0}</b></div>
                <div className="tchip">云端线程 <b>{t?.threads?.total_threads ?? 0}</b></div>
                <div className="tchip">
                  训练内核 <b>{kernel ? `PID ${kernel.pid} · ${kernel.threads} 线程 · ${kernel.cpu_percent}%` : "—"}</b>
                </div>
                <div className="tchip">rclone 同步 <b>{rclone ? `PID ${rclone.pid} · ${rclone.threads} 线程` : "空闲"}</b></div>
                <div className="tchip">
                  本机监工 <b>{relay?.running ? `运行中${relay.child_pid ? ` · 子进程 ${relay.child_pid}` : ""}` : "待命"}</b>
                </div>
              </div>
              {procs.length ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr><th>PID</th><th>角色</th><th>线程</th><th>CPU%</th><th>MEM%</th><th>RSS</th><th>状态</th><th>运行时长</th><th>命令</th></tr>
                    </thead>
                    <tbody>
                      {procs.map((p) => (
                        <tr key={p.pid}>
                          <td>{p.pid}</td>
                          <td><span className={`role role-${p.role}`}>{p.role}</span></td>
                          <td><b>{p.threads}</b></td>
                          <td>{p.cpu_percent}</td>
                          <td>{p.mem_percent}</td>
                          <td>{p.rss_mb} MB</td>
                          <td>{p.state}</td>
                          <td>{p.elapsed}</td>
                          <td className="cmd" title={p.command}>{p.command}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="dim" style={{ fontSize: 12.5 }}>部署遥测探针后即可查看云端进程与线程明细。</div>
              )}
            </>
          ) : (
            <div className="log" ref={logRef}>
              {relay?.logs?.length
                ? relay.logs.map((l, i) => (
                    <div key={i} className={/❌|⚠️|Error|Traceback/.test(l) ? "ln-err" : /✅|🎉/.test(l) ? "ln-ok" : ""}>{l}</div>
                  ))
                : <span className="dim">暂无接力日志。打开右上方「接力训练」开关后，这里会实时显示每一棒的分配、上传、训练与销毁过程。</span>}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

