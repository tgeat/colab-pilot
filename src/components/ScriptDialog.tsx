import { type MutableRefObject, useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw, Save } from "lucide-react";
import { ColabIcon, Modal, useToast } from "./ui";
import { api, errText } from "../api";

export function ScriptDialog(props: {
  open: boolean;
  workspacePath?: string | null;
  vmName?: string | null;
  onDirtyChange?: (dirty: boolean) => void;
  saveNowRef?: MutableRefObject<(() => Promise<boolean>) | null>;
  discardNowRef?: MutableRefObject<(() => void) | null>;
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
  const onDirtyChange = props.onDirtyChange;

  const notifyDirty = useCallback(
    (dirty: boolean) => {
      onDirtyChange?.(dirty);
      api.setScriptDirty(dirty).catch(() => {});
    },
    [onDirtyChange],
  );

  useEffect(() => {
    if (!props.open) {
      notifyDirty(false);
      return;
    }
    let cancelled = false;

    // #58 & #83：每次打开立即清空上一份残留内容并重置 dirty 标记，防止读取失败时误保存旧内容或残留脏状态
    notifyDirty(false);
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
        notifyDirty(false);
      })
      .catch((e) => {
        if (cancelled) return;
        const msg = errText(e);
        setLoadError(msg);
        setLoadedOk(false);
        notifyDirty(false);
        toast(`读取工作区脚本失败：${msg}`, "error");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [props.open, targetWs, targetVm, reloadNonce, toast, notifyDirty]);

  useEffect(() => {
    return () => {
      api.setScriptDirty(false).catch(() => {});
    };
  }, []);

  const isDirty = loadedOk && !loadError && (train !== origTrain || req !== origReq);
  const canSave = !saving && !loading && loadedOk && !loadError;

  useEffect(() => {
    if (!props.open || !isDirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [props.open, isDirty]);

  const requestClose = useCallback(() => {
    if (saving) return;
    if (isDirty) {
      setConfirmDiscard(true);
      return;
    }
    setConfirmDiscard(false);
    notifyDirty(false);
    props.onClose();
  }, [saving, isDirty, notifyDirty, props]);

  const forceDiscardAndClose = useCallback(() => {
    if (saving) return;
    setConfirmDiscard(false);
    notifyDirty(false);
    props.onClose();
  }, [saving, notifyDirty, props]);

  const save = useCallback(async (): Promise<boolean> => {
    if (!canSave) return false;
    setSaving(true);
    try {
      await api.saveWorkspaceFiles(train, req, targetWs, targetVm);
      setOrigTrain(train);
      setOrigReq(req);
      setConfirmDiscard(false);
      notifyDirty(false);
      toast(
        ws
          ? `已保存至工作区 ${ws}，下一棒开始时自动上传最新脚本`
          : "已保存，下一棒开始时自动上传最新 train.py",
        "success",
      );
      props.onClose();
      return true;
    } catch (e) {
      toast(errText(e), "error");
      return false;
    } finally {
      setSaving(false);
    }
  }, [canSave, train, req, targetWs, targetVm, notifyDirty, toast, ws, props]);

  if (props.saveNowRef) {
    props.saveNowRef.current = props.open ? save : null;
  }
  if (props.discardNowRef) {
    props.discardNowRef.current = props.open ? forceDiscardAndClose : null;
  }

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
            const nextTrain = e.target.value;
            setTrain(nextTrain);
            notifyDirty(Boolean(loadedOk && !loadError && (nextTrain !== origTrain || req !== origReq)));
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
            const nextReq = e.target.value;
            setReq(nextReq);
            notifyDirty(Boolean(loadedOk && !loadError && (train !== origTrain || nextReq !== origReq)));
            if (confirmDiscard) setConfirmDiscard(false);
          }}
        />
      </div>
    </Modal>
  );
}
