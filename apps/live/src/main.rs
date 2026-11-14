#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod control;
mod intent;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
fn main() {
    tauri::Builder::default()
        .manage(control::State::default())
        .manage(intent::Model::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _, event| {
                    if event.state() == ShortcutState::Pressed {
                        control::stop(app);
                    }
                })
                .build(),
        )
        .setup(|app| {
            app.global_shortcut().register("Ctrl+Shift+Space")?;
            // The hand is drawn in a transparent, click-through window above everything.
            let overlay = WebviewWindowBuilder::new(
                app,
                control::OVERLAY,
                WebviewUrl::App("index.html#overlay".into()),
            )
            .title("Gesture Pad Overlay")
            .transparent(true)
            .decorations(false)
            .shadow(false)
            .always_on_top(true)
            .skip_taskbar(true)
            .resizable(false)
            .focused(false)
            .visible(false)
            .build()?;
            if let Some(monitor) = overlay.primary_monitor()? {
                overlay.set_position(*monitor.position())?;
                overlay.set_size(*monitor.size())?;
            }
            overlay.set_ignore_cursor_events(true)?;
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let stale = handle
                        .state::<control::State>()
                        .0
                        .lock()
                        .is_ok_and(|p| p.armed && p.last_frame.elapsed().as_millis() > 500);
                    if stale {
                        control::stop(&handle);
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" && matches!(event, tauri::WindowEvent::Destroyed) {
                control::stop(window.app_handle());
                if let Some(overlay) = window.app_handle().get_webview_window(control::OVERLAY) {
                    let _ = overlay.close();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            control::set_armed,
            control::pointer_status,
            control::update_pointer,
            intent::load_model,
            intent::default_model_dir,
            intent::unload_model,
            intent::classify
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Gesture Pad");
}
