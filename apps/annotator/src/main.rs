#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use gesture_model::{Session, data, train};
use std::{
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tauri::{Emitter, State};

#[derive(Default)]
struct Training {
    running: AtomicBool,
    stop: Arc<AtomicBool>,
}

#[tauri::command]
fn list_sessions(root: String) -> Result<Vec<Session>, String> {
    data::load_sessions(&PathBuf::from(root)).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn save_session(root: String, session: Session) -> Result<(), String> {
    data::save_session(&PathBuf::from(root), &session).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn delete_session(root: String, id: String) -> Result<(), String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("invalid session id".into());
    }
    std::fs::remove_file(PathBuf::from(root).join(format!("{id}.json"))).map_err(|e| e.to_string())
}

/// Train on every session in `root` and write the model to `root/model`. Progress is
/// emitted as `train-progress` events; only one training runs at a time.
#[tauri::command]
async fn train_model(
    app: tauri::AppHandle,
    training: State<'_, Training>,
    root: String,
    epochs: usize,
) -> Result<train::Report, String> {
    if training.running.swap(true, Ordering::SeqCst) {
        return Err("已有训练在进行".into());
    }
    training.stop.store(false, Ordering::SeqCst);
    let stop = training.stop.clone();
    let root = PathBuf::from(root);
    let result = tauri::async_runtime::spawn_blocking(move || {
        // Only human-reviewed sessions train the model; raw rule labels would teach the rules' mistakes.
        let sessions: Vec<Session> = data::load_sessions(&root)?
            .into_iter()
            .filter(|s| s.reviewed)
            .collect();
        // CUDA initialisation failures panic inside the backend; surface them as errors.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train::train(&sessions, &root.join("model"), epochs.clamp(1, 500), |p| {
                let _ = app.emit("train-progress", p);
                if stop.load(Ordering::SeqCst) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
        }))
        .map_err(|_| anyhow::anyhow!("CUDA 初始化或训练失败（需要 NVIDIA 显卡与 CUDA 运行时）"))?
    })
    .await;
    training.running.store(false, Ordering::SeqCst);
    result
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:#}"))
}

/// Ask a running training to stop after the current batch; the best epoch so far is kept.
#[tauri::command]
fn stop_training(training: State<'_, Training>) {
    training.stop.store(true, Ordering::SeqCst);
}

fn main() {
    tauri::Builder::default()
        .manage(Training::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            list_sessions,
            save_session,
            delete_session,
            train_model,
            stop_training
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Gesture Annotator");
}
