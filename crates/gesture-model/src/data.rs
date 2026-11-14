use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Intent classes, in model output order. Swipe directions are finger motion as seen in
/// the mirrored preview (i.e. on screen); content follows the fingers.
/// Pressing is decided by rules (thumb–index gap), so a pinching hand is simply `point`:
/// the model only tells operating from idle and swipes from returns.
pub const LABELS: [&str; 4] = ["idle", "point", "swipe_up", "swipe_down"];

/// What a label from an earlier version becomes on load; `None` drops the segment.
/// Fast sideways swipes lost hand tracking, so they were removed (frames become unlabelled);
/// pinches merged into `point`, since the hand is still operating.
fn migrate(label: &str) -> Option<&str> {
    match label {
        "swipe_left" | "swipe_right" => None,
        "pinch" => Some("point"),
        other => Some(other),
    }
}
pub const JOINTS: usize = 21;

/// One camera frame. `image` is normalised (0–1) image coordinates; `world` is MediaPipe's
/// metric hand shape. Both are absent when no hand was detected.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub t_ms: f64,
    pub aspect: f32,
    pub image: Option<Vec<[f32; 2]>>,
    pub world: Option<Vec<[f32; 3]>>,
}
/// Image (normalised) and world (metric) landmarks of one hand.
pub type Hand<'a> = (&'a [[f32; 2]], &'a [[f32; 3]]);

impl Frame {
    pub fn hand(&self) -> Option<Hand<'_>> {
        match (&self.image, &self.world) {
            (Some(i), Some(w)) if i.len() == JOINTS && w.len() == JOINTS => Some((i, w)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: f64,
    pub end_ms: f64,
    pub label: String,
}

/// A recorded take: landmarks only, no video. Labels start as rule output and are corrected by hand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub id: String,
    pub prompt: String,
    pub reviewed: bool,
    pub frames: Vec<Frame>,
    pub segments: Vec<Segment>,
}
impl Session {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported session version {}",
            self.version
        );
        ensure!(
            !self.id.is_empty()
                && self
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "invalid session id"
        );
        ensure!(
            self.frames.windows(2).all(|w| w[0].t_ms < w[1].t_ms),
            "frame timestamps must increase"
        );
        for s in &self.segments {
            ensure!(s.start_ms < s.end_ms, "empty segment");
            ensure!(
                LABELS.contains(&s.label.as_str()),
                "unknown label {}",
                s.label
            );
        }
        ensure!(
            self.segments
                .windows(2)
                .all(|w| w[0].end_ms <= w[1].start_ms),
            "segments overlap or are unsorted"
        );
        Ok(())
    }
    /// Bring labels from earlier versions up to date; touching segments that end up with the
    /// same label become one.
    fn migrate(&mut self) {
        let mut out: Vec<Segment> = Vec::with_capacity(self.segments.len());
        for mut s in self.segments.drain(..) {
            let Some(label) = migrate(&s.label) else {
                continue;
            };
            s.label = label.to_string();
            match out.last_mut() {
                Some(prev) if prev.label == s.label && prev.end_ms == s.start_ms => {
                    prev.end_ms = s.end_ms
                }
                _ => out.push(s),
            }
        }
        self.segments = out;
    }
    /// Class index at a timestamp; frames outside every segment are unlabelled and skipped in training.
    pub fn label_at(&self, t: f64) -> Option<usize> {
        let s = self
            .segments
            .iter()
            .find(|s| s.start_ms <= t && t < s.end_ms)?;
        LABELS.iter().position(|l| *l == s.label)
    }
}

pub fn load_sessions(root: &Path) -> Result<Vec<Session>> {
    let mut sessions = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("cannot read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json")
            && path.file_stem().is_some_and(|s| s != "config")
        {
            let mut session: Session = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("invalid {}", path.display()))?;
            session.migrate();
            session
                .validate()
                .with_context(|| path.display().to_string())?;
            sessions.push(session);
        }
    }
    sessions.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(sessions)
}

pub fn save_session(root: &Path, session: &Session) -> Result<()> {
    session.validate()?;
    fs::create_dir_all(root)?;
    let path = root.join(format!("{}.json", session.id));
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec(session)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}
