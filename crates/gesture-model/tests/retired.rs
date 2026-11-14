//! Sessions recorded with earlier label sets still load: left/right swipes are dropped,
//! pinches become point.
use gesture_model::data::{load_sessions, save_session};
use gesture_model::{Frame, Segment, Session};

#[test]
fn old_labels_are_migrated_on_load() {
    let dir = std::env::temp_dir().join("gesture-model-retired-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let frames = vec![
        Frame {
            t_ms: 0.0,
            aspect: 1.0,
            image: None,
            world: None,
        },
        Frame {
            t_ms: 33.0,
            aspect: 1.0,
            image: None,
            world: None,
        },
    ];
    let seg = |a: f64, b: f64, l: &str| Segment {
        start_ms: a,
        end_ms: b,
        label: l.into(),
    };
    // Written as raw JSON: today's save_session would refuse the retired label.
    let old = Session {
        version: 1,
        id: "old-1".into(),
        prompt: "swipe_left".into(),
        reviewed: true,
        frames: frames.clone(),
        segments: vec![
            seg(0.0, 10.0, "point"),
            seg(10.0, 20.0, "pinch"),
            seg(20.0, 30.0, "swipe_left"),
            seg(30.0, 40.0, "idle"),
            seg(40.0, 50.0, "pinch"),
        ],
    };
    std::fs::write(dir.join("old-1.json"), serde_json::to_vec(&old).unwrap()).unwrap();
    let loaded = load_sessions(&dir).unwrap();
    assert_eq!(loaded.len(), 1);
    let segments: Vec<_> = loaded[0]
        .segments
        .iter()
        .map(|s| (s.start_ms, s.end_ms, s.label.as_str()))
        .collect();
    // The pinch joins the point before it; the swipe_left becomes a gap.
    assert_eq!(
        segments,
        [
            (0.0, 20.0, "point"),
            (30.0, 40.0, "idle"),
            (40.0, 50.0, "point")
        ]
    );
    // Saving the migrated session works.
    save_session(&dir, &loaded[0]).unwrap();
}
