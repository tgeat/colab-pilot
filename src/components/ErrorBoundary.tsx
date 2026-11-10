import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface ErrorBoundaryProps {
  children?: ReactNode;
  /** 区域名称，用于日志定位（可选） */
  label?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/** 渲染异常边界：防止单个组件（如遥测数据异常的卡片）抛错导致整个窗口白屏 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.label ? `:${this.props.label}` : ""}]`, error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <div className="banner banner-danger" role="alert" style={{ alignItems: "center" }}>
          <AlertTriangle size={16} />
          <span style={{ flex: 1, minWidth: 0, wordBreak: "break-word" }}>
            界面渲染异常：{error.message || String(error)}
          </span>
          <button className="btn btn-sm" onClick={this.reset} title="重新渲染该区域">
            <RotateCcw size={13} />
            <span>重试</span>
          </button>
        </div>
      );
    }
    return this.props.children ?? null;
  }
}
