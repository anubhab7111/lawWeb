import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./styles/theme.css";

// Resolve the theme before first paint so there's no flash of the wrong
// ground. Stored choice wins; otherwise follow the OS preference.
const stored = localStorage.getItem("theme");
const initial = stored ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
document.documentElement.dataset.theme = initial;

createRoot(document.getElementById("root")!).render(<App />);
