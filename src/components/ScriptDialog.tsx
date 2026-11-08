import { useEffect, useState } from "react";
import { Loader2, Save } from "lucide-react";
import { ColabIcon, Modal, useToast } from "./ui";
import { api, errText } from "../api";

export function ScriptDialog(props: { open: boolean; onClose: () => void }) {
  const toast = useToast();
  const [train, setTrain] = useState("");
  const [req, setReq] = useState("");
  const [ws, setWs] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!props.open) return;
    setLoading(true);
    api
      .readWorkspaceFiles()
      .then((f) => {
        setTrain(f.train_py);
        setReq(f.requirements_txt);
        setWs(f.workspace);
      })
      .catch((e) => toast(errText(e), "error"))
      .finally(() => setLoading(false));
  }, [props.open]);

  const save = async () => {
    setSaving(true);
    try {
      await api.saveWorkspaceFiles(train, req);
      toast("已保存，下一棒开始时自动上传最新 train.py", "success");
      props.onClose();
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={props.open}
      onClose={props.onClose}
      wide
      icon={<ColabIcon size={24} />}
      iconBg="#fff7e6"
      title="接力训练脚本"
      sub={ws ? `工作目录：${ws}` : undefined}
      footer={
        <>
          <button className="btn" onClick={props.onClose}>关闭</button>
          <button className="btn btn-primary" onClick={save} disabled={saving || loading}>
            {saving ? <Loader2 size={15} className="spin" /> : <Save size={15} />}保存
          </button>
        </>
      }
    >
      <div>
        <div className="field">
          <label>train.py —— 开机拉取断点 · 原子保存 · 定时 rclone 热备 · 到时安全交接（通过环境变量接收面板参数）</label>
        </div>
        <textarea className="editor" style={{ height: 380 }} spellCheck={false} value={train} onChange={(e) => setTrain(e.target.value)} />
      </div>
      <div>
        <div className="field"><label>requirements.txt —— 每棒开机通过 colab install -r 安装</label></div>
        <textarea className="editor" style={{ height: 90 }} spellCheck={false} value={req} onChange={(e) => setReq(e.target.value)} />
      </div>
    </Modal>
  );
}
