import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Note: StrictMode removed to prevent double-mounting effects that cause
// duplicate process spawning. The refs-based locking doesn't work well
// with StrictMode's intentional double-invocation of effects and callbacks.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <App />
);
