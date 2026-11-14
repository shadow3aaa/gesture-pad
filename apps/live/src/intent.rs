//! Optional GCN intent classifier. Without a loaded model the frontend falls back to rules.
use gesture_model::{Frame, Runtime};
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

#[derive(Default)]
pub struct Model(pub Mutex<Option<Runtime>>);

fn guarded<T>(work: impl FnOnce() -> anyhow::Result<T>) -> Result<T, String> {
    // CUDA failures surface as panics inside the backend; never let them take down the app.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .map_err(|_| "CUDA 推理失败（需要 NVIDIA 显卡与 CUDA 运行时）".to_string())?
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn load_model(path: String, model: tauri::State<'_, Model>) -> Result<(), String> {
    let runtime = tauri::async_runtime::spawn_blocking(move || {
        guarded(|| Runtime::load(&PathBuf::from(path)))
    })
    .await
    .map_err(|e| e.to_string())??;
    *model.0.lock().map_err(|e| e.to_string())? = Some(runtime);
    Ok(())
}

/// `data/model` in this repository, where the annotator saves when `data` is its dataset
/// directory; `None` until a model has been trained there.
#[tauri::command]
pub fn default_model_dir() -> Option<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2)?;
    let dir = root.join("data").join("model");
    dir.join("model.bpk")
        .is_file()
        .then(|| dir.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn unload_model(model: tauri::State<Model>) -> Result<(), String> {
    *model.0.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

/// Class probabilities for the newest frame, or `None` when no model is loaded.
#[tauri::command]
pub fn classify(frame: Frame, model: tauri::State<Model>) -> Result<Option<Vec<f32>>, String> {
    let mut slot = model.0.lock().map_err(|e| e.to_string())?;
    let Some(runtime) = slot.as_mut() else {
        return Ok(None);
    };
    match guarded(|| runtime.push(frame)) {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            *slot = None; // A failed device stays failed; fall back to rules visibly.
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn default_model_dir_is_data_model_in_the_workspace() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        assert!(root.join("Cargo.lock").is_file(), "{}", root.display());
        if let Some(dir) = super::default_model_dir() {
            assert_eq!(std::path::Path::new(&dir), root.join("data").join("model"));
        }
    }
}
