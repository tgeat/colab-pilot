import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { AlertTriangle, Plus } from "lucide-react";
import { api, errText } from "./api";
import type { Overview, RelayConfig, VmView } from "./types";
import { Header } from "./components/Header";
import { VmCard } from "./components/VmCard";
import { CreateVmDialog } from "./components/CreateVmDialog";
import { DeleteVmDialog } from "./components/DeleteVmDialog";
import { StopRelayDialog } from "./components/StopRelayDialog";
import { ScriptDialog } from "./components/ScriptDialog";
import { ColabIcon, ToastProvider, useToast } from "./components/ui";

function Dashboard() {
  const toast = useToast();
  const [ov, setOv] = useState<Overview | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [showScripts, setShowScripts] = useState(false);
  const [deleting, setDeleting] = useState<VmView | null>(null);
  const [stopping, setStopping] = useState<VmView | null>(null);
  const [pending, setPending] = useState<Record<string, boolean>>({});

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      setOv(await api.refreshNow());
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setRefreshing(false);
    }
  }, [toast]);

  useEffect(() => {
    api.getOverview().then((o) => o.timestamp && setOv(o)).catch(() => {});
    refresh();
    const un = listen<Overview>("overview", (e) => setOv(e.payload));
    return () => {
      un.then((f) => f());
    };
  }, []);

  const withPending = async (name: string, fn: () => Promise<void>) => {
    setPending((p) => ({ ...p, [name]: true }));
    try {
      await fn();
    } finally {
      setPending((p) => ({ ...p, [name]: false }));
    }
  };

  const onCreate = async (p: { name: string | null; accelerator: string; high_mem: boolean; auto_relay: boolean }) => {
    toast(`正在向 Google Colab 申请 ${p.accelerator} 实例…`, "info");
    try {
      const name = await api.createVm({ ...p, relay_config: null });
      toast(`虚拟机 ${name} 已就绪${p.auto_relay ? "，接力训练已开启" : ""}`, "success");
    } catch (e) {
      toast(`创建失败：${errText(e)}`, "error");
      throw e;
    }
  };

  const onProbe = (vm: VmView) =>
    withPending(vm.name, async () => {
      try {
        await api.deployProbe(vm.name);
        toast(`${vm.name} 遥测探针已部署`, "success");
      } catch (e) {
        toast(errText(e), "error");
      }
    });

  const onStartRelay = (vm: VmView, cfg: RelayConfig) =>
    withPending(vm.name, async () => {
      try {
        await api.startRelay(vm.name, cfg);
        toast(`${vm.name} 已开启 24h+ 接力训练`, "success");
      } catch (e) {
        toast(errText(e), "error");
      }
    });

  const onStopRelay = async (vm: VmView, destroy: boolean) => {
    try {
      const r = await api.stopRelay(vm.name, destroy);
      if (destroy) {
        r?.verified_unassigned
          ? toast(`${vm.name} 接力已关闭，实例已彻底销毁并停止计费`, "success")
          : toast(`接力已关闭，但实例销毁未通过核验：${r?.error ?? ""}`, "error");
      } else {
        toast(`${vm.name} 接力已关闭，实例保留（仍在计费）`, "warn");
      }
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const env = ov?.env;
  const vms = ov?.vms ?? [];

  return (
    <>
      <Header
        ov={ov}
        refreshing={refreshing}
        onRefresh={refresh}
        onAdd={() => setShowCreate(true)}
        onScripts={() => setShowScripts(true)}
      />
      <main className="main">
        {ov?.error && (
          <div className="banner banner-danger"><AlertTriangle size={16} />{ov.error}</div>
        )}
        {env && (!env.colab_ok || !env.rclone_ok || !env.rclone_conf_ok) && (
          <div className="banner banner-warn">
            <AlertTriangle size={16} />
            <div>
              {!env.colab_ok && <div>未找到 colab-cli：{env.colab_bin}</div>}
              {!env.rclone_ok && <div>未找到 rclone：{env.rclone_bin}</div>}
              {!env.rclone_conf_ok && <div>未检测到 rclone 授权文件：{env.rclone_conf}（接力训练需要）</div>}
            </div>
          </div>
        )}

        <div className="section-head">
          <h2 style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <ColabIcon size={22} />
            Colab 虚拟机 <span className="dim" style={{ fontWeight: 500, fontSize: 14 }}>· {vms.length}</span>
          </h2>
          <span>每 4 秒与 Colab 服务端计费列表同步 · 外部创建的实例也会自动纳入管控</span>
        </div>

        {vms.length === 0 ? (
          <div className="empty">
            <div className="empty-icon"><ColabIcon size={38} /></div>
            <h3>{ov ? "当前没有任何计费中的 Colab 虚拟机" : "正在连接 Google Colab…"}</h3>
            <p>
              服务端计费列表为空，当前消耗 {(ov?.ccu?.burn_rate_hourly ?? 0).toFixed(2)} CU/h。
              点击右上角「新增虚拟机」创建实例，并可一键开启 24h+ 全自动接力训练。
            </p>
            <button className="add-btn" onClick={() => setShowCreate(true)}>
              <span className="plus"><Plus size={16} strokeWidth={2.6} /></span>新增虚拟机
            </button>
          </div>
        ) : (
          <div className="vm-list">
            {vms.map((vm) => (
              <VmCard
                key={vm.name}
                vm={vm}
                pending={!!pending[vm.name]}
                onDelete={setDeleting}
                onProbe={onProbe}
                onStartRelay={onStartRelay}
                onStopRelay={setStopping}
              />
            ))}
          </div>
        )}
      </main>

      <CreateVmDialog open={showCreate} onClose={() => setShowCreate(false)} onCreate={onCreate} />
      <DeleteVmDialog vm={deleting} onClose={() => setDeleting(null)} onDelete={(vm) => api.deleteVm(vm.name, vm.endpoint)} />
      <StopRelayDialog vm={stopping} onClose={() => setStopping(null)} onStop={onStopRelay} />
      <ScriptDialog open={showScripts} onClose={() => setShowScripts(false)} />
    </>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Dashboard />
    </ToastProvider>
  );
}
