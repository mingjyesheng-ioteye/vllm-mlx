use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::Manager;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    pub port: u16,
    pub model: Option<String>,
    pub pid: Option<u32>,
}

#[derive(Default)]
pub struct AppState {
    pub server_status: Arc<Mutex<ServerStatus>>,
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self {
            running: false,
            port: 8000,
            model: None,
            pid: None,
        }
    }
}

#[tauri::command]
async fn get_server_status(state: tauri::State<'_, AppState>) -> Result<ServerStatus, String> {
    let status = state.server_status.lock().await;
    Ok(status.clone())
}

#[tauri::command]
async fn set_server_status(
    state: tauri::State<'_, AppState>,
    running: bool,
    port: u16,
    model: Option<String>,
    pid: Option<u32>,
) -> Result<(), String> {
    let mut status = state.server_status.lock().await;
    status.running = running;
    status.port = port;
    status.model = model;
    status.pid = pid;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![get_server_status, set_server_status])
        .setup(|app| {
            log::info!("vLLM-MLX Application starting...");

            let state = AppState::default();
            app.manage(state);

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
