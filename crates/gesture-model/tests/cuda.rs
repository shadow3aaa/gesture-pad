//! End-to-end on the GPU: synthetic sessions → train → save → load → online inference.
//! Needs an NVIDIA GPU: `cargo test -p gesture-model --test cuda -- --ignored`.
use gesture_model::train::Progress;
use gesture_model::{Frame, LABELS, Runtime, Segment, Session, train};
use std::ops::ControlFlow;

/// A schematic hand: `pinch` closes thumb onto index; the hand translates by `(dx, dy)` per frame.
fn hand(t: f64, x: f32, y: f32, pinch: bool) -> Frame {
    let mut world: Vec<[f32; 3]> = (0..21)
        .map(|j| {
            let finger = (j.max(1) - 1) / 4;
            let along = ((j.max(1) - 1) % 4 + 1) as f32;
            if j == 0 {
                [0.0, 0.0, 0.0]
            } else {
                [(finger as f32 - 2.0) * 0.02, -0.06 - along * 0.015, 0.0]
            }
        })
        .collect();
    if pinch {
        world[4] = [world[8][0] + 0.005, world[8][1], 0.0];
    } else {
        world[4] = [-0.07, -0.05, 0.0];
    }
    let image = world
        .iter()
        .map(|p| [x + p[0] * 2.0, y + p[1] * 2.0])
        .collect();
    Frame {
        t_ms: t,
        aspect: 1.0,
        image: Some(image),
        world: Some(world),
    }
}

/// One session per class. Swipes alternate a fast stroke (labelled) with a slow return (idle):
/// exactly the case rules cannot tell apart.
fn sessions(seed: u32) -> Vec<Session> {
    let mut out = Vec::new();
    for (k, label) in LABELS.iter().enumerate() {
        let (mut frames, mut segments) = (Vec::new(), Vec::new());
        let (mut x, mut y) = (0.5f32, 0.5f32);
        let dir = match *label {
            "swipe_up" => (0.0, -1.0),
            "swipe_down" => (0.0, 1.0),
            _ => (0.0, 0.0),
        };
        let mut t = 0.0f64;
        for cycle in 0..40 {
            let stroke = 10;
            for phase in 0..2 {
                let (n, speed, lab) = if phase == 0 {
                    (stroke, 0.012, *label)
                } else {
                    (stroke * 3, -0.004, "idle")
                };
                let start = t;
                for i in 0..n {
                    let wobble = ((cycle * 7 + i + seed as usize) % 5) as f32 * 0.0005;
                    match *label {
                        // Pointing wanders at under half the stroke speed (peak ≈ 0.005/frame),
                        // so the classes are separable by speed as well as by direction.
                        "point" => {
                            x = 0.5 + ((t / 400.0).sin() as f32) * 0.06;
                            y = 0.5 + ((t / 300.0).cos() as f32) * 0.045
                        }
                        "idle" => {
                            x = 0.5 + wobble;
                        }
                        _ => {
                            x += dir.0 * speed;
                            y += dir.1 * speed;
                        }
                    }
                    // Pointing includes pinches (presses are decided by rules, not the model).
                    let pinch = *label == "point" && cycle % 2 == 1;
                    frames.push(hand(t, x, y, pinch));
                    t += 33.0;
                }
                let lab = if dir == (0.0, 0.0) { *label } else { lab };
                segments.push(Segment {
                    start_ms: start,
                    end_ms: t,
                    label: lab.into(),
                });
            }
        }
        let _ = k;
        out.push(Session {
            version: 1,
            id: format!("s{seed}-{label}").replace('_', "-"),
            prompt: label.to_string(),
            reviewed: true,
            frames,
            segments,
        });
    }
    out
}

#[test]
#[ignore]
fn trains_on_cuda_and_runs_online() {
    let dir = std::env::temp_dir().join("gesture-model-cuda-test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut epochs_seen = 0;
    let report = train::train(&sessions(1), &dir, 12, |p| {
        if let Progress::Epoch(e) = &p {
            eprintln!(
                "epoch {}/{} loss {:.3}/{:.3} acc {:.3}/{:.3} recall {:?}{}",
                e.epoch,
                e.epochs,
                e.train_loss,
                e.val_loss,
                e.train_balanced,
                e.val_balanced,
                e.class_recall,
                if e.best { " *" } else { "" }
            );
            epochs_seen += 1;
        }
        ControlFlow::Continue(())
    })
    .unwrap();
    assert!(epochs_seen == 12 || report.early_stopped);
    assert!(!report.stopped);
    eprintln!("confusion: {:?}", report.confusion);
    assert!(
        report.val_balanced > 0.85,
        "val balanced accuracy {}",
        report.val_balanced
    );

    // Online: a fresh downward stroke is recognised as swipe_down, its slow return as not a swipe.
    let mut runtime = Runtime::load(&dir).unwrap();
    let (mut y, mut t) = (0.3f32, 0.0);
    let mut stroke = vec![];
    for _ in 0..12 {
        y += 0.012;
        stroke = runtime.push(hand(t, 0.5, y, false)).unwrap();
        t += 33.0;
    }
    let down = LABELS.iter().position(|l| *l == "swipe_down").unwrap();
    let argmax = |p: &[f32]| {
        p.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(&stroke), down, "stroke probs {stroke:?}");
    let mut back = vec![];
    for _ in 0..20 {
        y -= 0.004;
        back = runtime.push(hand(t, 0.5, y, false)).unwrap();
        t += 33.0;
    }
    assert!(
        !LABELS[argmax(&back)].starts_with("swipe"),
        "return probs {back:?}"
    );
}

#[test]
#[ignore]
fn stopping_keeps_the_best_epoch_so_far() {
    let dir = std::env::temp_dir().join("gesture-model-cuda-stop-test");
    let _ = std::fs::remove_dir_all(&dir);
    let report = train::train(&sessions(2), &dir, 50, |p| match p {
        Progress::Epoch(e) if e.epoch == 2 => ControlFlow::Break(()),
        _ => ControlFlow::Continue(()),
    })
    .unwrap();
    assert!(report.stopped);
    assert!(report.best_epoch <= 2);
    Runtime::load(&dir).unwrap();
    // Stopping before the first epoch finishes saves nothing and says so.
    let early = train::train(&sessions(2), &dir.join("early"), 5, |_| {
        ControlFlow::Break(())
    });
    assert!(early.is_err());
}
