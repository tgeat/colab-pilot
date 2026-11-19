import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw, Save } from "lucide-react";
import { ColabIcon, Modal, useToast } from "./ui";
import { api, errText } from "../api";

export function ScriptDialog(props: {
  open: boolean;
  workspacePath?: string | null;
  vmName?: string | null;
  onClose: () => void;
}) {
  const toast = useToast();
  const [train, setTrain] = useState("");
  const [req, setReq] = useState("");
  const [origTrain, setOrigTrain] = useState("");
  const [origReq, setOrigReq] = useState("");
  const [ws, setWs] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadedOk, setLoadedOk] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [reloadNonce, setReloadNonce] = useState(0);

  const targetWs = props.workspacePath ?? null;
  const targetVm = props.vmName ?? null;

  useEffect(() => {
    if (!props.open) return;
    let cancelled = false;

    // #58：每次打开立即清空上一份残留内容并进入加载态，防止读取失败时误保存旧内容或空串
    setTrain("");
    setReq("");
    setOrigTrain("");
    setOrigReq("");
    setWs(targetWs ?? "");
    setLoadError(null);
    setLoadedOk(false);
    setConfirmDiscard(false);
    setLoading(true);

    api
      .readWorkspaceFiles(targetWs, targetVm)
      .then((f) => {
        if (cancelled) return;
        setTrain(f.train_py);
        setReq(f.requirements_txt);
        setOrigTrain(f.train_py);
        setOrigReq(f.requirements_txt);
        setWs(f.workspace || targetWs || "");
        setLoadedOk(true);
        setLoadError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        const msg = errText(e);
        setLoadError(msg);
        setLoadedOk(false);
        toast(`读取工作区脚本失败：${msg}`, "error");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [props.open, targetWs, targetVm, reloadNonce, toast]);

  const isDirty = loadedOk && !loadError && (train !== origTrain || req !== origReq);
  const canSave = !saving && !loading && loadedOk && !loadError;

  const requestClose = useCallback(() => {
    if (saving) return;
    if (isDirty) {
      setConfirmDiscard(true);
      return;
    }
    setConfirmDiscard(false);
    props.onClose();
  }, [saving, isDirty, props]);

  const forceDiscardAndClose = () => {
    if (saving) return;
    setConfirmDiscard(false);
    props.onClose();
  };

  const save = async () => {
    if (!canSave) return;
    setSaving(true);
    try {
      await api.saveWorkspaceFiles(train, req, targetWs, targetVm);
      setOrigTrain(train);
      setOrigReq(req);
      setConfirmDiscard(false);
      toast(
        ws
          ? `已保存至工作区 ${ws}，下一棒开始时自动上传最新脚本`
          : "已保存，下一棒开始时自动上传最新 train.py",
        "success",
      );
      props.onClose();
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setSaving(false);
    }
  };

  const displayWs = ws || targetWs || "";

  return (
    <Modal
      open={props.open}
      onClose={requestClose}
      closable={!saving}
      wide
      icon={<ColabIcon size={24} />}
      iconBg="#fff7e6"
      title={targetVm ? `接力训练脚本 · ${targetVm}` : "接力训练脚本"}
      sub={displayWs ? `工作目录：${displayWs}` : undefined}
      footer={
        <>
          {isDirty && (
            <span className="warn" style={{ marginRight: "auto", fontSize: 12, fontWeight: 600 }}>
              ● 有未保存的编辑内容
            </span>
          )}
          <button type="button" className="btn" onClick={requestClose} disabled={saving}>
            取消
          </button>
          <button type="button" className="btn btn-primary" onClick={save} disabled={!canSave}>
            {saving ? <Loader2 size={15} className="spin" /> : <Save size={15} />}
            保存
          </button>
        </>
      }
    >
      {loadError && (
        <div className="banner banner-danger" role="alert" style={{ margin: 0, alignItems: "center" }}>
          <AlertTriangle size={16} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700 }}>读取脚本失败（已禁用保存以防空白覆盖磁盘文件）</div>
            <div className="mono" style={{ fontSize: 12, marginTop: 2, wordBreak: "break-word" }}>
              {loadError}
            </div>
          </div>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setReloadNonce((n) => n + 1)}
            disabled={loading}
          >
            <RefreshCw size={13} className={loading ? "spin" : ""} />
            <span>重试读取</span>
          </button>
        </div>
      )}

      {confirmDiscard && (
        <div className="banner banner-warn" role="alert" style={{ margin: 0, alignItems: "center" }}>
          <AlertTriangle size={16} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <b>有未保存的修改，确定放弃编辑并关闭吗？</b>
          </div>
          <div style={{ display: "inline-flex", gap: 8 }}>
            <button type="button" className="btn btn-sm" onClick={() => setConfirmDiscard(false)}>
              继续编辑
            </button>
            <button type="button" className="btn btn-danger btn-sm" onClick={forceDiscardAndClose}>
              放弃修改并关闭
            </button>
          </div>
        </div>
      )}

      <div>
        <div className="field">
          <label>
            train.py —— 开机拉取断点 · 原子保存 · 定时 rclone 热备 · 到时安全交接（通过环境变量接收面板参数）
          </label>
        </div>
        <textarea
          className="editor"
          style={{ height: 360 }}
          spellCheck={false}
          disabled={loading || saving || !loadedOk || !!loadError}
          placeholder={loading ? "正在读取工作区 train.py…" : ""}
          value={train}
          onChange={(e) => {
            setTrain(e.target.value);
            if (confirmDiscard) setConfirmDiscard(false);
          }}
        />
      </div>
      <div>
        <div className="field">
          <label>requirements.txt —— 每棒开机通过 colab install -r 安装</label>
        </div>
        <textarea
          className="editor"
          style={{ height: 90 }}
          spellCheck={false}
          disabled={loading || saving || !loadedOk || !!loadError}
          placeholder={loading ? "正在读取工作区 requirements.txt…" : ""}
          value={req}
          onChange={(e) => {
            setReq(e.target.value);
            if (confirmDiscard) setConfirmDiscard(false);
          }}
        />
      </div>
    </Modal>
  );
}
