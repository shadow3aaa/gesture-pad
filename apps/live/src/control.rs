use enigo::{Axis, Button, Coordinate, Direction, Enigo, Mouse, Settings};
use serde::{Deserialize, Serialize};
use std::{sync::Mutex, time::Instant};
use tauri::{AppHandle, Manager};

pub const OVERLAY: &str = "overlay";

#[derive(Default)]
pub struct State(pub Mutex<Pointer>);

#[derive(Clone, Copy, Serialize)]
pub struct Status {
    armed: bool,
    epoch: u64,
}

/// One frame of hand control. `x`, `y` are fractions of the primary display.
/// While `scroll_id` is non-zero, `scroll_y` is cumulative vertical notches for that gesture.
#[derive(Deserialize)]
pub struct Sample {
    epoch: u64,
    sequence: u64,
    active: bool,
    down: bool,
    x: f64,
    y: f64,
    scroll_id: u64,
    scroll_y: f64,
}

trait Output {
    fn display(&self) -> Result<(i32, i32), String>;
    fn move_to(&mut self, x: i32, y: i32) -> Result<(), String>;
    fn button(&mut self, down: bool) -> Result<(), String>;
    fn scroll(&mut self, notches: i32) -> Result<(), String>;
}
impl Output for Enigo {
    fn display(&self) -> Result<(i32, i32), String> {
        self.main_display().map_err(|e| e.to_string())
    }
    fn move_to(&mut self, x: i32, y: i32) -> Result<(), String> {
        self.move_mouse(x, y, Coordinate::Abs)
            .map_err(|e| e.to_string())
    }
    fn button(&mut self, down: bool) -> Result<(), String> {
        let direction = if down {
            Direction::Press
        } else {
            Direction::Release
        };
        Mouse::button(self, Button::Left, direction).map_err(|e| e.to_string())
    }
    fn scroll(&mut self, notches: i32) -> Result<(), String> {
        Mouse::scroll(self, notches, Axis::Vertical).map_err(|e| e.to_string())
    }
}

#[derive(Default)]
struct Cursor {
    last: Option<(i32, i32)>,
    down: bool,
    scroll_id: u64,
    scrolled: i32,
}
impl Cursor {
    fn release(&mut self, output: &mut impl Output) -> Result<(), String> {
        if self.down {
            output.button(false)?;
        }
        *self = Self::default();
        Ok(())
    }
    fn update(&mut self, sample: &Sample, output: &mut impl Output) -> Result<(), String> {
        if !sample.active {
            return self.release(output);
        }
        if sample.scroll_id != 0 {
            if self.down {
                output.button(false)?;
                self.down = false;
            }
            if sample.scroll_id != self.scroll_id {
                self.scroll_id = sample.scroll_id;
                self.scrolled = 0;
            }
            // Content follows the fingers: moving down scrolls up.
            let target = (-sample.scroll_y) as i32;
            if target != self.scrolled {
                output.scroll(target - self.scrolled)?;
            }
            self.scrolled = target;
            return Ok(());
        }
        let (w, h) = output.display()?;
        if w <= 0 || h <= 0 {
            return Err("无法读取主屏幕尺寸".into());
        }
        let position = (
            (sample.x * (w - 1) as f64).round() as i32,
            (sample.y * (h - 1) as f64).round() as i32,
        );
        // Move before pressing so a click lands where the action point is.
        if self.last != Some(position) {
            output.move_to(position.0, position.1)?;
            self.last = Some(position);
        }
        if sample.down != self.down {
            output.button(sample.down)?;
            self.down = sample.down;
        }
        Ok(())
    }
}

pub struct Pointer {
    output: Option<Enigo>,
    cursor: Cursor,
    pub armed: bool,
    epoch: u64,
    sequence: u64,
    pub last_frame: Instant,
}
impl Default for Pointer {
    fn default() -> Self {
        Self {
            output: None,
            cursor: Cursor::default(),
            armed: false,
            epoch: 0,
            sequence: 0,
            last_frame: Instant::now(),
        }
    }
}
impl Pointer {
    fn status(&self) -> Status {
        Status {
            armed: self.armed,
            epoch: self.epoch,
        }
    }
    pub fn stop(&mut self) {
        if let Some(output) = &mut self.output {
            let _ = self.cursor.release(output);
        }
        // Dropping Enigo is an additional release safeguard on output errors.
        self.output = None;
        self.cursor = Cursor::default();
        self.armed = false;
        self.epoch += 1;
        self.sequence = 0;
    }
    fn update(&mut self, sample: &Sample) -> Result<Status, String> {
        // Queued packets cannot re-arm control after a global stop or a new session.
        if !self.armed || sample.epoch != self.epoch || sample.sequence <= self.sequence {
            return Ok(self.status());
        }
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        let notches = |v: f64| v.is_finite() && v.abs() <= 1000.0;
        if !unit(sample.x)
            || !unit(sample.y)
            || !notches(sample.scroll_y)
            || (!sample.active && sample.down)
            || (sample.scroll_id != 0 && sample.down)
        {
            self.stop();
            return Err("无效的控制数据，已停止控制".into());
        }
        self.sequence = sample.sequence;
        self.last_frame = Instant::now();
        if let Some(output) = &mut self.output
            && let Err(error) = self.cursor.update(sample, output)
        {
            self.stop();
            return Err(error);
        }
        Ok(self.status())
    }
}

/// Stop control and hide the hand overlay. Used by every stop path.
pub fn stop(app: &AppHandle) {
    if let Ok(mut pointer) = app.state::<State>().0.lock() {
        pointer.stop();
    }
    if let Some(overlay) = app.get_webview_window(OVERLAY) {
        let _ = overlay.hide();
    }
}

#[tauri::command]
pub fn set_armed(
    armed: bool,
    app: AppHandle,
    state: tauri::State<State>,
) -> Result<Status, String> {
    let status = {
        let mut pointer = state.0.lock().map_err(|e| e.to_string())?;
        pointer.stop();
        if armed {
            pointer.output = Some(Enigo::new(&Settings::default()).map_err(|e| e.to_string())?);
            pointer.last_frame = Instant::now();
            pointer.armed = true;
        }
        pointer.status()
    };
    if let Some(overlay) = app.get_webview_window(OVERLAY) {
        let shown = if armed {
            overlay.show()
        } else {
            overlay.hide()
        };
        shown.map_err(|e| e.to_string())?;
    }
    Ok(status)
}

#[tauri::command]
pub fn pointer_status(state: tauri::State<State>) -> Result<Status, String> {
    Ok(state.0.lock().map_err(|e| e.to_string())?.status())
}

#[tauri::command]
pub fn update_pointer(sample: Sample, state: tauri::State<State>) -> Result<Status, String> {
    state.0.lock().map_err(|e| e.to_string())?.update(&sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Fake {
        events: Vec<String>,
    }
    impl Output for Fake {
        fn display(&self) -> Result<(i32, i32), String> {
            Ok((1001, 601))
        }
        fn move_to(&mut self, x: i32, y: i32) -> Result<(), String> {
            self.events.push(format!("move {x},{y}"));
            Ok(())
        }
        fn button(&mut self, down: bool) -> Result<(), String> {
            self.events.push(if down { "down" } else { "up" }.into());
            Ok(())
        }
        fn scroll(&mut self, notches: i32) -> Result<(), String> {
            self.events.push(format!("vscroll {notches}"));
            Ok(())
        }
    }
    fn sample(x: f64, down: bool) -> Sample {
        Sample {
            epoch: 0,
            sequence: 1,
            active: true,
            down,
            x,
            y: 0.5,
            scroll_id: 0,
            scroll_y: 0.0,
        }
    }
    fn scroll(id: u64, y: f64) -> Sample {
        Sample {
            scroll_id: id,
            scroll_y: y,
            ..sample(0.5, false)
        }
    }
    #[test]
    fn absolute_position_then_press_drag_release() {
        let mut cursor = Cursor::default();
        let mut out = Fake::default();
        for s in [
            sample(0.25, false),
            sample(0.25, true),
            sample(0.5, true),
            sample(0.5, false),
        ] {
            cursor.update(&s, &mut out).unwrap();
        }
        assert_eq!(
            out.events,
            ["move 250,300", "down", "move 500,300", "up"].map(String::from)
        );
    }
    #[test]
    fn hand_loss_releases_a_held_button() {
        let mut cursor = Cursor::default();
        let mut out = Fake::default();
        cursor.update(&sample(0.1, true), &mut out).unwrap();
        let gone = Sample {
            active: false,
            ..sample(0.0, false)
        };
        cursor.update(&gone, &mut out).unwrap();
        cursor.update(&gone, &mut out).unwrap();
        assert_eq!(out.events, ["move 100,300", "down", "up"].map(String::from));
    }
    #[test]
    fn scroll_emits_whole_notches_once_per_gesture_and_never_moves() {
        let mut cursor = Cursor::default();
        let mut out = Fake::default();
        for s in [
            scroll(1, 0.4),
            scroll(1, 2.6),
            scroll(1, 2.9),
            scroll(2, 1.5),
        ] {
            cursor.update(&s, &mut out).unwrap();
        }
        assert_eq!(out.events, ["vscroll -2", "vscroll -1"].map(String::from));
    }
    #[test]
    fn stopped_and_stale_packets_are_ignored() {
        let mut pointer = Pointer::default();
        pointer.stop();
        assert!(!pointer.update(&sample(0.2, true)).unwrap().armed);
        pointer.armed = true;
        pointer.update(&sample(0.2, true)).unwrap();
        assert_eq!(pointer.sequence, 0);
    }
    #[test]
    fn invalid_data_disarms() {
        for bad in [
            sample(f64::NAN, false),
            sample(1.5, false),
            Sample {
                scroll_id: 1,
                ..sample(0.5, true)
            },
        ] {
            let mut pointer = Pointer {
                armed: true,
                ..Pointer::default()
            };
            assert!(pointer.update(&bad).is_err());
            assert!(!pointer.armed);
        }
    }
}
