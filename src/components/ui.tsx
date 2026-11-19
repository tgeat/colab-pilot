import {
  createContext,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { X } from "lucide-react";
import { pct } from "../format";
import { computeNextFocusTrapIndex } from "../chatUtils";

/* ───────── Google Colab 官方图标 ───────── */
export function ColabIcon({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label="Google Colab"
      style={{ flexShrink: 0, display: "block" }}
    >
      <path d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z" fill="#E8710A" />
      <path d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z" fill="#E8710A" />
      <path d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z" fill="#F9AB00" />
      <path d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09" fill="#F9AB00" />
      <path d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z" fill="#F9AB00" />
    </svg>
  );
}

/* ───────── Toast ───────── */
export type ToastKind = "info" | "success" | "error" | "warn";
interface ToastItem { id: number; kind: ToastKind; text: string }
const ToastCtx = createContext<(text: string, kind?: ToastKind) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

const toastColor: Record<ToastKind, string> = {
  info: "var(--primary)",
  success: "var(--success)",
  error: "var(--danger)",
  warn: "var(--warning)",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const push = useCallback((text: string, kind: ToastKind = "info") => {
    const id = Date.now() + Math.random();
    setItems((s) => [...s, { id, kind, text }]);
    setTimeout(
      () => setItems((s) => s.filter((t) => t.id !== id)),
      kind === "error" || kind === "warn" ? 7000 : 4200,
    );
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div className="toast" key={t.id}>
            <div className="bar" style={{ background: toastColor[t.kind] }} />
            <div style={{ flex: 1 }}>{t.text}</div>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* ───────── 无障碍弹窗焦点锁定 (Focus Trap #70) ───────── */
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not(:disabled)",
  'input:not(:disabled):not([type="hidden"])',
  "select:not(:disabled)",
  "textarea:not(:disabled)",
  "summary",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

function getVisibleFocusableElements(container: HTMLElement): HTMLElement[] {
  const nodes = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
  return nodes.filter((el) => {
    if (el.hasAttribute("disabled") || el.getAttribute("aria-hidden") === "true") {
      return false;
    }
    const closedDetails = el.closest("details:not([open])");
    if (closedDetails && container.contains(closedDetails)) {
      const firstSummary = closedDetails.querySelector(":scope > summary");
      if (el !== firstSummary) return false;
    }
    return el.offsetParent !== null || el.getClientRects().length > 0;
  });
}

function isTopmostModal(container: HTMLElement): boolean {
  const allModals = document.querySelectorAll<HTMLElement>(".overlay .modal");
  if (allModals.length === 0) return true;
  return allModals[allModals.length - 1] === container;
}

export function useModalFocusTrap(
  open: boolean,
  modalRef: RefObject<HTMLElement | null>,
  initialFocusRef?: RefObject<HTMLElement | null>,
) {
  const initialFocusRefHolder = useRef(initialFocusRef);
  initialFocusRefHolder.current = initialFocusRef;

  useEffect(() => {
    if (!open) return;
    const prevActive = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const focusInitial = () => {
      const modalEl = modalRef.current;
      if (!modalEl) return;
      if (modalEl.contains(document.activeElement)) return;

      const preferred = initialFocusRefHolder.current?.current;
      if (preferred && !preferred.hasAttribute("disabled")) {
        preferred.focus();
        return;
      }
      const bodyEl = modalEl.querySelector<HTMLElement>(".modal-body");
      const bodyFocusables = bodyEl ? getVisibleFocusableElements(bodyEl) : [];
      if (bodyFocusables.length > 0) {
        bodyFocusables[0].focus();
        return;
      }
      const allFocusables = getVisibleFocusableElements(modalEl);
      if (allFocusables.length > 0) {
        allFocusables[0].focus();
        return;
      }
      modalEl.focus();
    };

    const rafId = window.requestAnimationFrame(focusInitial);

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const modalEl = modalRef.current;
      if (!modalEl || !isTopmostModal(modalEl)) return;

      const focusables = getVisibleFocusableElements(modalEl);
      if (focusables.length === 0) {
        e.preventDefault();
        modalEl.focus();
        return;
      }

      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      const currentIdx = active && modalEl.contains(active) ? focusables.indexOf(active) : -1;
      if (
        currentIdx === -1 ||
        (!e.shiftKey && currentIdx === focusables.length - 1) ||
        (e.shiftKey && currentIdx === 0)
      ) {
        e.preventDefault();
        const nextIdx = computeNextFocusTrapIndex(focusables.length, currentIdx, e.shiftKey);
        if (nextIdx >= 0 && focusables[nextIdx]) {
          focusables[nextIdx].focus();
        }
      }
    };

    const onFocusIn = (e: FocusEvent) => {
      const modalEl = modalRef.current;
      if (!modalEl || !isTopmostModal(modalEl)) return;
      const target = e.target;
      if (target instanceof Node && !modalEl.contains(target)) {
        const focusables = getVisibleFocusableElements(modalEl);
        if (focusables.length > 0) {
          focusables[0].focus();
        } else {
          modalEl.focus();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("focusin", onFocusIn, true);

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("focusin", onFocusIn, true);
      if (prevActive && document.contains(prevActive) && typeof prevActive.focus === "function") {
        try {
          prevActive.focus();
        } catch {
          // ignore
        }
      }
    };
  }, [open, modalRef]);
}

/* ───────── Modal ───────── */
export function Modal(props: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  sub?: ReactNode;
  icon?: ReactNode;
  iconBg?: string;
  wide?: boolean;
  footer?: ReactNode;
  children: ReactNode;
  closable?: boolean;
}) {
  const closable = props.closable !== false;
  const modalRef = useRef<HTMLDivElement | null>(null);
  const titleId = useId();
  const onCloseRef = useRef(props.onClose);
  onCloseRef.current = props.onClose;

  useModalFocusTrap(props.open, modalRef);

  useEffect(() => {
    if (!props.open || !closable) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const modalEl = modalRef.current;
        if (modalEl && !isTopmostModal(modalEl)) return;
        e.preventDefault();
        onCloseRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.open, closable]);

  if (!props.open) return null;
  return (
    <div
      className="overlay"
      onMouseDown={(e) => e.target === e.currentTarget && closable && props.onClose()}
    >
      <div
        ref={modalRef}
        className={`modal ${props.wide ? "wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <div className="modal-head">
          {props.icon && (
            <div className="modal-icon" style={{ background: props.iconBg ?? "var(--primary-50)" }}>
              {props.icon}
            </div>
          )}
          <div style={{ flex: 1 }}>
            <h3 id={titleId}>{props.title}</h3>
            {props.sub && <div className="sub">{props.sub}</div>}
          </div>
          {closable && (
            <button type="button" className="icon-btn" onClick={props.onClose} aria-label="关闭">
              <X size={16} />
            </button>
          )}
        </div>
        <div className="modal-body">{props.children}</div>
        {props.footer && <div className="modal-foot">{props.footer}</div>}
      </div>
    </div>
  );
}

/* ───────── Meter ───────── */
export function Meter(props: { icon: ReactNode; name: string; value: number; display?: string; sub?: string }) {
  const v = pct(props.value);
  const cls = v >= 88 ? "fill high" : v >= 70 ? "fill mid" : "fill";
  return (
    <div>
      <div className="meter-head">
        <span className="meter-name">{props.icon}{props.name}</span>
        <span className="meter-num">{props.display ?? `${v.toFixed(0)}%`}</span>
      </div>
      <div className="track"><div className={cls} style={{ width: `${v}%` }} /></div>
      {props.sub && <div className="meter-sub" title={props.sub}>{props.sub}</div>}
    </div>
  );
}

export function KV(props: { label: string; value: ReactNode; mono?: boolean; wide?: boolean; title?: string; cls?: string }) {
  return (
    <div className={`kv-item ${props.wide ? "wide" : ""}`} title={props.title}>
      <div className="kv-label">{props.label}</div>
      <div className={`kv-value ${props.mono ? "mono" : ""} ${props.cls ?? ""}`}>{props.value}</div>
    </div>
  );
}
