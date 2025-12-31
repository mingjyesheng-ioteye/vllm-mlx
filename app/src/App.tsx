import { useState, useEffect, useRef } from "react";
import { Command } from "@tauri-apps/plugin-shell";
import { invoke } from "@tauri-apps/api/core";
import { fetch } from "@tauri-apps/plugin-http";
import "./App.css";

interface ServerStatus {
  running: boolean;
  ready: boolean;
  port: number;
  model: string | null;
  pid: number | null;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface CachedModel {
  name: string;
  org: string;
  full_name: string;
}

function App() {
  const [serverStatus, setServerStatus] = useState<ServerStatus>({
    running: false,
    ready: false,
    port: 8000,
    model: null,
    pid: null,
  });
  const [modelName, setModelName] = useState("mlx-community/Llama-3.2-3B-Instruct-4bit");
  const [port, setPort] = useState(8000);
  const [logs, setLogs] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [multiSession, setMultiSession] = useState(false);
  const childProcess = useRef<any>(null);
  const currentPid = useRef<number | null>(null);
  const isStarting = useRef<boolean>(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    const loadCachedModels = async () => {
      try {
        const models = await invoke<CachedModel[]>("get_cached_models");
        setCachedModels(models);
        // If we have cached models and current model is in cache, select it
        if (models.length > 0) {
          const currentInCache = models.find((m) => m.full_name === modelName);
          if (!currentInCache) {
            // Default to first cached model if current isn't in cache
            setModelName(models[0].full_name);
          }
        }
      } catch (error) {
        console.error("Failed to load cached models:", error);
      }
    };
    loadCachedModels();
  }, []);

  // Note: Server process cleanup on app exit is handled by the Rust backend
  // via on_window_event handler in lib.rs

  const startServer = async () => {
    addLog(`startServer called: isStarting=${isStarting.current}, childProcess=${!!childProcess.current}`);
    if (isStarting.current || childProcess.current) {
      addLog("Server start already in progress or running, skipping...");
      return;
    }
    isStarting.current = true;

    try {
      addLog(`Starting server with model: ${modelName}`);

      // Use Command.sidecar for bundled binary (no Python required)
      // The name must match externalBin in tauri.conf.json
      const args = [
        "serve",
        modelName,
        "--port",
        port.toString(),
      ];
      if (multiSession) {
        args.push("--continuous-batching");
        addLog("Multi-session mode enabled (continuous batching)");
      }
      const command = Command.sidecar("binaries/vllm-mlx-server", args);

      command.stdout.on("data", (line) => {
        addLog(`[stdout] ${line}`);
      });

      command.stderr.on("data", (line) => {
        addLog(`[stderr] ${line}`);
      });

      const child = await command.spawn();
      const spawnedPid = child.pid;

      // Set refs BEFORE releasing lock to prevent race conditions
      childProcess.current = child;
      currentPid.current = spawnedPid;

      command.on("close", (data) => {
        addLog(`Server process exited with code ${data.code}`);
        // Only update state if this is still the current process (not a stale close event)
        if (currentPid.current === spawnedPid) {
          setServerStatus((prev) => ({ ...prev, running: false, ready: false, pid: null }));
          childProcess.current = null;
          currentPid.current = null;
          isStarting.current = false;
        }
      });

      command.on("error", (error) => {
        addLog(`Error: ${error}`);
        isStarting.current = false;
      });

      setServerStatus({
        running: true,
        ready: false,
        port,
        model: modelName,
        pid: child.pid,
      });

      await invoke("set_server_status", {
        running: true,
        port,
        model: modelName,
        pid: child.pid,
      });

      addLog(`Server started with PID: ${child.pid}`);
      addLog("Waiting for model to load...");

      // Only release lock after everything is set up
      isStarting.current = false;

      // Poll until server is ready
      const checkReady = async () => {
        for (let i = 0; i < 120; i++) { // Wait up to 2 minutes
          // Stop polling if this process is no longer current
          if (currentPid.current !== spawnedPid) {
            addLog(`Polling cancelled for PID ${spawnedPid} (no longer current)`);
            return;
          }
          try {
            const resp = await fetch(`http://localhost:${port}/v1/models`);
            if (resp.ok) {
              // Double-check we're still the current process before updating state
              if (currentPid.current === spawnedPid) {
                addLog("Server is ready!");
                setServerStatus((prev) => ({ ...prev, ready: true }));
              }
              return;
            }
          } catch {
            // Server not ready yet
          }
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (currentPid.current === spawnedPid) {
          addLog("Server failed to become ready");
        }
      };
      checkReady();
    } catch (error) {
      addLog(`Failed to start server: ${error}`);
      isStarting.current = false;
    }
  };

  const stopServer = async (): Promise<boolean> => {
    addLog("stopServer called");
    isStarting.current = false;
    const processToKill = childProcess.current;
    if (processToKill) {
      try {
        addLog("Stopping server...");
        // Clear refs first to prevent race conditions with close event
        childProcess.current = null;
        currentPid.current = null;

        // Kill all vllm-mlx-server processes to ensure complete cleanup
        try {
          const killAll = Command.create("sh", ["-c",
            `pkill -9 -f "vllm-mlx-server" 2>/dev/null || true`
          ]);
          await killAll.execute();
          addLog("Killed all server processes");
        } catch {
          // Fall back to killing the main process directly
          await processToKill.kill();
        }

        setServerStatus((prev) => ({ ...prev, running: false, ready: false, pid: null }));
        await invoke("set_server_status", {
          running: false,
          port,
          model: null,
          pid: null,
        });
        addLog("Server stopped");
        return true;
      } catch (error) {
        addLog(`Failed to stop server: ${error}`);
        return false;
      }
    }
    return true;
  };

  const switchModel = async (newModel: string) => {
    addLog(`switchModel called: newModel=${newModel}, running=${serverStatus.running}, currentModel=${serverStatus.model}`);
    if (serverStatus.running && newModel !== serverStatus.model) {
      addLog(`Switching model from ${serverStatus.model} to ${newModel}...`);
      setModelName(newModel);
      // Stop the current server - user will manually start with new model
      await stopServer();
    } else {
      setModelName(newModel);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !serverStatus.ready) return;

    const userMessage: Message = { role: "user", content: inputMessage };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsGenerating(true);

    try {
      const response = await fetch(
        `http://localhost:${serverStatus.port}/v1/chat/completions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: serverStatus.model,
            messages: [...messages, userMessage].map((m) => ({
              role: m.role,
              content: m.content,
            })),
            max_tokens: 512,
            stream: false,
          }),
        }
      );

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.choices[0].message.content,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      addLog(`Chat error: ${error}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>vLLM-MLX</h1>
        <span className={`status ${serverStatus.ready ? "running" : serverStatus.running ? "loading" : "stopped"}`}>
          {serverStatus.ready ? "Ready" : serverStatus.running ? "Loading..." : "Stopped"}
        </span>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <div className="config-section">
            <h2>Server Configuration</h2>

            <label>
              Model:
              {cachedModels.length > 0 && !useCustomModel ? (
                <select
                  value={modelName}
                  onChange={(e) => switchModel(e.target.value)}
                >
                  {cachedModels.map((model) => (
                    <option key={model.full_name} value={model.full_name}>
                      {model.full_name}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  disabled={serverStatus.running}
                  placeholder="mlx-community/model-name"
                />
              )}
            </label>
            {cachedModels.length > 0 && (
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useCustomModel}
                  onChange={(e) => setUseCustomModel(e.target.checked)}
                  disabled={serverStatus.running}
                />
                Use custom model name
              </label>
            )}

            <label>
              Port:
              <input
                type="number"
                value={port}
                onChange={(e) => setPort(parseInt(e.target.value))}
                disabled={serverStatus.running}
              />
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={multiSession}
                onChange={(e) => setMultiSession(e.target.checked)}
                disabled={serverStatus.running}
              />
              Multi-session mode
              <span className="help-text">(for multiple concurrent users)</span>
            </label>

            <div className="button-group">
              {!serverStatus.running ? (
                <button className="btn-primary" onClick={(e) => { e.preventDefault(); e.stopPropagation(); startServer(); }}>
                  Start Server
                </button>
              ) : (
                <button className="btn-danger" onClick={(e) => { e.preventDefault(); e.stopPropagation(); stopServer(); }}>
                  Stop Server
                </button>
              )}
            </div>

            {serverStatus.running && (
              <div className="server-info">
                <p>
                  <strong>API:</strong> http://localhost:{serverStatus.port}/v1
                </p>
                <p>
                  <strong>Model:</strong> {serverStatus.model}
                </p>
                <p>
                  <strong>Mode:</strong> {multiSession ? "Multi-session" : "Single-session"}
                </p>
                <p>
                  <strong>PID:</strong> {serverStatus.pid}
                </p>
              </div>
            )}
          </div>

          <div className="logs-section">
            <h2>Logs</h2>
            <div className="logs">
              {logs.map((log, i) => (
                <div key={i} className="log-line">
                  {log}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </aside>

        <main className="chat-area">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <h3>Start a conversation</h3>
                <p>Send a message to chat with the model</p>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            {isGenerating && (
              <div className="message assistant">
                <div className="message-content typing">Thinking...</div>
              </div>
            )}
          </div>

          <div className="input-area">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && sendMessage()}
              placeholder={
                serverStatus.ready
                  ? "Type a message..."
                  : serverStatus.running
                  ? "Loading model..."
                  : "Start the server first..."
              }
              disabled={!serverStatus.ready || isGenerating}
            />
            <button
              onClick={sendMessage}
              disabled={!serverStatus.ready || isGenerating || !inputMessage.trim()}
            >
              Send
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
