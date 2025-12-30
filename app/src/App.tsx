import { useState, useEffect, useRef } from "react";
import { Command } from "@tauri-apps/plugin-shell";
import { invoke } from "@tauri-apps/api/core";
import { fetch } from "@tauri-apps/plugin-http";
import "./App.css";

interface ServerStatus {
  running: boolean;
  port: number;
  model: string | null;
  pid: number | null;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

function App() {
  const [serverStatus, setServerStatus] = useState<ServerStatus>({
    running: false,
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
  const childProcess = useRef<any>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const startServer = async () => {
    try {
      addLog(`Starting server with model: ${modelName}`);

      // Use Command.create to run Python with vllm_mlx module
      // The name must match the shell:allow-execute permission in capabilities
      const command = Command.create("python3", [
        "-m",
        "vllm_mlx.cli",
        "serve",
        modelName,
        "--port",
        port.toString(),
      ]);

      command.stdout.on("data", (line) => {
        addLog(`[stdout] ${line}`);
      });

      command.stderr.on("data", (line) => {
        addLog(`[stderr] ${line}`);
      });

      command.on("close", (data) => {
        addLog(`Server process exited with code ${data.code}`);
        setServerStatus((prev) => ({ ...prev, running: false, pid: null }));
        childProcess.current = null;
      });

      command.on("error", (error) => {
        addLog(`Error: ${error}`);
      });

      const child = await command.spawn();
      childProcess.current = child;

      setServerStatus({
        running: true,
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
    } catch (error) {
      addLog(`Failed to start server: ${error}`);
    }
  };

  const stopServer = async () => {
    if (childProcess.current) {
      try {
        addLog("Stopping server...");
        await childProcess.current.kill();
        childProcess.current = null;
        setServerStatus((prev) => ({ ...prev, running: false, pid: null }));
        await invoke("set_server_status", {
          running: false,
          port,
          model: null,
          pid: null,
        });
        addLog("Server stopped");
      } catch (error) {
        addLog(`Failed to stop server: ${error}`);
      }
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !serverStatus.running) return;

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
        <span className={`status ${serverStatus.running ? "running" : "stopped"}`}>
          {serverStatus.running ? "Running" : "Stopped"}
        </span>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <div className="config-section">
            <h2>Server Configuration</h2>

            <label>
              Model:
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={serverStatus.running}
                placeholder="mlx-community/model-name"
              />
            </label>

            <label>
              Port:
              <input
                type="number"
                value={port}
                onChange={(e) => setPort(parseInt(e.target.value))}
                disabled={serverStatus.running}
              />
            </label>

            <div className="button-group">
              {!serverStatus.running ? (
                <button className="btn-primary" onClick={startServer}>
                  Start Server
                </button>
              ) : (
                <button className="btn-danger" onClick={stopServer}>
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
                serverStatus.running
                  ? "Type a message..."
                  : "Start the server first..."
              }
              disabled={!serverStatus.running || isGenerating}
            />
            <button
              onClick={sendMessage}
              disabled={!serverStatus.running || isGenerating || !inputMessage.trim()}
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
