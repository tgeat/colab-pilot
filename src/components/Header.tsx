import { Cloud, FileCode2, HardDrive, Plus, RefreshCw } from "lucide-react";
import type { Overview } from "../types";
import { fmtBytes } from "../format";
import { ColabIcon } from "./ui";

export function Header(props: {
  ov: Overview | null;
  refreshing: boolean;
  onRefresh: () => void;
  onAdd: () => void;
  onScripts: () => void;
}) {
  const { ov } = props;
  const ccu = ov?.ccu;
  const rate = ccu?.burn_rate_hourly ?? 0;
  const gd = ov?.gdrive;
  const gdOk = !!ov?.env.rclone_conf_ok && (gd?.ok ?? false);

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <div className="brand-mark" title="Google Colab">
            <ColabIcon size={28} />
          </div>
          <div>
            <div className="brand-title">Colab Pilot</div>
            <div className="brand-sub">Colab 虚拟机 · 24h+ 接力训练管控台</div>
          </div>
        </div>

        <div className="stats">
          <div className="stat" title="Google Colab 账户剩余计算单元">
            <div className="stat-icon" style={{ background: "#fff7e6" }}>
              <ColabIcon size={18} />
            </div>
            <div>
              <div className="stat-label">剩余计算单元</div>
              <div className="stat-value">{ccu ? `${ccu.balance.toFixed(2)} CU` : "—"}</div>
            </div>
          </div>
          <div className="stat" title="当前每小时计算单元消耗速率">
            <div
              className="stat-icon"
              style={{
                background: rate > 0 ? "var(--warning-50)" : "#fff7e6",
              }}
            >
              <ColabIcon size={18} />
            </div>
            <div>
              <div className="stat-label">计费速率</div>
              <div className="stat-value" style={{ color: rate > 0 ? "var(--warning)" : "var(--success)" }}>
                {rate.toFixed(2)} CU/h
              </div>
            </div>
          </div>
          <div className="stat" title="Colab 服务端 /tun/m/assignments 真实计费实例数">
            <div className="stat-icon" style={{ background: "#fff7e6" }}>
              <ColabIcon size={18} />
            </div>
            <div>
              <div className="stat-label">服务端计费实例</div>
              <div className="stat-value">{ov ? `${ov.server_count} 台` : "—"}</div>
            </div>
          </div>
          <div className="stat" title={gd?.error ?? ov?.env.rclone_conf ?? ""}>
            <div
              className="stat-icon"
              style={{
                background: gdOk ? "var(--success-50)" : "var(--danger-50)",
                color: gdOk ? "var(--success)" : "var(--danger)",
              }}
            >
              {gdOk ? <Cloud size={16} /> : <HardDrive size={16} />}
            </div>
            <div>
              <div className="stat-label">Google Drive · {gd?.dir ?? "gdrive:colab_relay"}</div>
              <div className="stat-value" style={{ fontSize: 12.5 }}>
                {!ov?.env.rclone_conf_ok
                  ? "未授权"
                  : !gd
                    ? "检测中…"
                    : gd.has_done_flag
                      ? "训练已完成"
                      : gd.has_checkpoint
                        ? `断点 ${fmtBytes(gd.checkpoint_size)}`
                        : "已授权 · 无断点"}
              </div>
            </div>
          </div>
        </div>

        <div className="top-actions">
          <button className="btn" onClick={props.onScripts}>
            <FileCode2 size={15} /> 训练脚本
          </button>
          <button className="icon-btn" onClick={props.onRefresh} disabled={props.refreshing} title="立即刷新">
            <RefreshCw size={16} className={props.refreshing ? "spin" : ""} />
          </button>
          <button className="add-btn" onClick={props.onAdd} title="新增 Colab 虚拟机">
            <span className="plus"><Plus size={16} strokeWidth={2.6} /></span>
            新增虚拟机
          </button>
        </div>
      </div>
    </header>
  );
}
