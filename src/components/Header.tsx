import { MessageSquare, Power, RefreshCw } from "lucide-react";
import type { Overview } from "../types";
import { fmtBytes, num, str } from "../format";
import { hasAnyReadyLlmVm, isLlmVm } from "../chatUtils";
import { ColabIcon } from "./ui";

export function Header(props: {
  ov: Overview | null;
  refreshing: boolean;
  onRefresh: () => void;
  onOpenLlmChat: (target: string) => void;
  onOpenCreateVm: () => void;
}) {
  const { ov } = props;
  const ccu = ov?.ccu;
  const rate = num(ccu?.burn_rate_hourly);
  const gd = ov?.gdrive;
  const vms = ov?.vms ?? [];
  const anyReadyLlm = hasAnyReadyLlmVm(vms);
  const hasLlm = !!ov?.llm_bluegreen || vms.some((v) => isLlmVm(v));
  const activeSlot = str(ov?.llm_bluegreen?.active_slot).toUpperCase() || undefined;

  // 网盘状态仅在有意义时显示：未授权 / 有断点 / 训练完成
  let gdNote: { text: string; cls: string } | null = null;
  if (ov && !ov.env.rclone_conf_ok) gdNote = { text: "未授权", cls: "bad" };
  else if (gd?.has_done_flag) gdNote = { text: "训练完成", cls: "ok" };
  else if (gd?.has_checkpoint) gdNote = { text: `断点 ${fmtBytes(gd.checkpoint_size)}`, cls: "" };

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <ColabIcon size={24} />
          <span className="brand-title">Colab Pilot</span>
        </div>

        <div className="stats">
          <div className="stat" title="服务端计费实例数">
            <span className="stat-label">实例</span>
            <span className="stat-value">{ov ? ov.server_count : "—"}</span>
          </div>
          <div className="stat" title="当前每小时计算单元消耗速率">
            <span className="stat-label">计费</span>
            <span className={`stat-value ${rate > 0 ? "warn" : "ok"}`}>{rate.toFixed(2)} CU/h</span>
          </div>
          <div className="stat" title="Google Colab 账户剩余计算单元">
            <span className="stat-label">余额</span>
            <span className="stat-value">{ccu ? `${num(ccu.balance).toFixed(1)} CU` : "—"}</span>
          </div>
          {gdNote && (
            <div className="stat" title={gd?.error ?? gd?.dir ?? ""}>
              <span className="stat-label">网盘</span>
              <span className={`stat-value ${gdNote.cls}`}>{gdNote.text}</span>
            </div>
          )}
        </div>

        <div className="top-actions">
          <button
            className="add-btn"
            onClick={props.onOpenCreateVm}
            title="选择 GPU/TPU/CPU 加速器与全量 Colab 配置并开机启动新虚拟机（自动配置免密 SSH 与大模型 API）"
          >
            <Power size={13} />
            <span>开机 / 新建实例</span>
          </button>

          {hasLlm && (
            <button
              className="btn btn-primary btn-sm chat-trigger-btn"
              onClick={() => props.onOpenLlmChat("auto")}
              disabled={!anyReadyLlm}
              title={
                anyReadyLlm
                  ? "打开云端大模型交互对话窗口"
                  : "当前没有在线且健康就绪的大模型实例（若正在部署请等待预热完成）"
              }
            >
              <MessageSquare size={14} />
              <span>大模型对话</span>
              {anyReadyLlm && activeSlot && <span className="chat-trigger-slot">{activeSlot}</span>}
            </button>
          )}

          <button
            className="icon-btn"
            onClick={props.onRefresh}
            disabled={props.refreshing}
            title="立即刷新（每 1.5 秒自动高频同步）"
          >
            <RefreshCw size={15} className={props.refreshing ? "spin" : ""} />
          </button>
        </div>
      </div>
    </header>
  );
}
