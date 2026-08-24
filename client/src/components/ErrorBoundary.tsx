import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  // Called on catch so the app can reset any parent state (e.g. navigate
  // back to a known-good view) alongside the boundary's own reset.
  onReset?: () => void;
}

interface State {
  error: Error | null;
}

// Without this, any render-time throw (an unguarded .map/.length on a
// malformed API response, for instance) unmounts the whole tree — the nav
// disappears along with everything else, and the only way out is a manual
// page reload. One boundary around the routed view turns that into a
// recoverable, in-app error card.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.error) {
      return (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
          <div className="card" style={{ padding: 32, maxWidth: 440, textAlign: "center" }}>
            <div style={{ font: "700 17px var(--font-head)", marginBottom: 8 }}>Something went wrong</div>
            <div style={{ font: "400 13.5px/1.6 var(--font-body)", color: "var(--muted-2)", marginBottom: 20 }}>
              This screen hit an unexpected error. You can go back and try again.
            </div>
            <button className="btn btn-primary" onClick={this.reset}>Go back</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
