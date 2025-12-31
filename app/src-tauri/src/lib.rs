use serde::{Deserialize, Serialize};
use std::fs;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub name: String,
    pub org: String,
    pub full_name: String,
}

#[tauri::command]
fn get_cached_models() -> Result<Vec<CachedModel>, String> {
    let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
    let cache_path = home_dir.join(".cache/huggingface/hub");

    if !cache_path.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    let entries = fs::read_dir(&cache_path).map_err(|e| e.to_string())?;

    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();

        // HuggingFace cache directories are named: models--{org}--{model}
        if dir_name.starts_with("models--") {
            let parts: Vec<&str> = dir_name.strip_prefix("models--").unwrap().splitn(2, "--").collect();
            if parts.len() == 2 {
                let org = parts[0].to_string();
                let model = parts[1].to_string();
                let full_name = format!("{}/{}", org, model);

                models.push(CachedModel {
                    name: model,
                    org,
                    full_name,
                });
            }
        }
    }

    // Sort by full name
    models.sort_by(|a, b| a.full_name.cmp(&b.full_name));

    Ok(models)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![get_server_status, set_server_status, get_cached_models])
        .setup(|app| {
            log::info!("vLLM-MLX Application starting...");

            let state = AppState::default();
            app.manage(state);

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
