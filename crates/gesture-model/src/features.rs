//! Per-joint features. Identical code runs for training and live inference, so the two
//! can never drift apart.
use crate::data::{Frame, JOINTS};

/// Channels per joint: shape relative to the wrist (3), shape velocity (3), image velocity (2).
pub const CHANNELS: usize = 8;
pub const FRAME_FEATURES: usize = JOINTS * CHANNELS;
const NOMINAL_DT_MS: f64 = 1000.0 / 30.0;

fn length3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Wrist-relative metric shape, in hand lengths (wrist → middle knuckle): position and size invariant.
fn shape(world: &[[f32; 3]]) -> Option<Vec<[f32; 3]>> {
    let l = length3(world[9], world[0]);
    (l > 1e-4).then(|| {
        world
            .iter()
            .map(|p| {
                [
                    (p[0] - world[0][0]) / l,
                    (p[1] - world[0][1]) / l,
                    (p[2] - world[0][2]) / l,
                ]
            })
            .collect()
    })
}

/// Features of `frame`, using `prev` for velocities. Velocities are scaled to a 30 fps step so
/// a different camera rate does not change their magnitude. Returns zeros without a hand.
pub fn frame_features(prev: Option<&Frame>, frame: &Frame) -> [f32; FRAME_FEATURES] {
    let mut out = [0.0; FRAME_FEATURES];
    let Some((image, world)) = frame.hand() else {
        return out;
    };
    let Some(rel) = shape(world) else { return out };
    let aspect = if frame.aspect.is_finite() && frame.aspect > 0.0 {
        frame.aspect
    } else {
        16.0 / 9.0
    };
    let px = |p: [f32; 2]| [p[0] * aspect, p[1]];
    let span = {
        let (a, b) = (px(image[9]), px(image[0]));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
            .sqrt()
            .max(1e-4)
    };
    let before = prev.and_then(|p| {
        let dt = frame.t_ms - p.t_ms;
        (dt > 0.0 && dt < 250.0).then_some(())?;
        let (pi, pw) = p.hand()?;
        Some((pi, shape(pw)?, (NOMINAL_DT_MS / dt) as f32))
    });
    for j in 0..JOINTS {
        let o = j * CHANNELS;
        out[o..o + 3].copy_from_slice(&rel[j]);
        if let Some((pimage, prel, scale)) = &before {
            for a in 0..3 {
                out[o + 3 + a] = (rel[j][a] - prel[j][a]) * scale;
            }
            let (now, then) = (px(image[j]), px(pimage[j]));
            out[o + 6] = (now[0] - then[0]) / span * scale;
            out[o + 7] = (now[1] - then[1]) / span * scale;
        }
    }
    out
}

/// Features for every frame of a sequence.
pub fn sequence_features(frames: &[Frame]) -> Vec<[f32; FRAME_FEATURES]> {
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| frame_features(i.checked_sub(1).map(|p| &frames[p]), f))
        .collect()
}

/// A model input window ending at `end` (inclusive), zero-padded before the start.
pub fn window(features: &[[f32; FRAME_FEATURES]], end: usize, length: usize, out: &mut Vec<f32>) {
    let start = (end + 1).saturating_sub(length);
    out.extend(std::iter::repeat_n(
        0.0,
        (length - (end + 1 - start)) * FRAME_FEATURES,
    ));
    for f in &features[start..=end] {
        out.extend_from_slice(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(t: f64, dx: f32) -> Frame {
        let world: Vec<[f32; 3]> = (0..JOINTS)
            .map(|j| [j as f32 * 0.01, -(j as f32) * 0.004, 0.0])
            .collect();
        let image = world
            .iter()
            .map(|p| [0.5 + p[0] + dx, 0.5 + p[1]])
            .collect();
        Frame {
            t_ms: t,
            aspect: 1.0,
            image: Some(image),
            world: Some(world),
        }
    }
    #[test]
    fn velocity_is_frame_rate_invariant_and_shape_is_translation_invariant() {
        let a = frame_features(Some(&frame(0.0, 0.0)), &frame(33.333, 0.01));
        let b = frame_features(Some(&frame(0.0, 0.0)), &frame(66.667, 0.02));
        for j in 0..JOINTS {
            assert!((a[j * CHANNELS + 6] - b[j * CHANNELS + 6]).abs() < 1e-3);
            assert!(a[j * CHANNELS + 6] > 0.0);
        }
        let c = frame_features(None, &frame(0.0, 0.3));
        let d = frame_features(None, &frame(0.0, 0.0));
        assert_eq!(c, d);
    }
    #[test]
    fn missing_hand_or_gap_gives_zero_velocity() {
        let none = Frame {
            t_ms: 0.0,
            aspect: 1.0,
            image: None,
            world: None,
        };
        assert!(frame_features(None, &none).iter().all(|v| *v == 0.0));
        let f = frame_features(Some(&frame(0.0, 0.0)), &frame(500.0, 0.2));
        assert!((0..JOINTS).all(|j| f[j * CHANNELS + 6] == 0.0));
    }
    #[test]
    fn window_pads_the_start() {
        let feats = vec![[1.0; FRAME_FEATURES]; 3];
        let mut out = Vec::new();
        window(&feats, 1, 4, &mut out);
        assert_eq!(out.len(), 4 * FRAME_FEATURES);
        assert!(out[..2 * FRAME_FEATURES].iter().all(|v| *v == 0.0));
        assert!(out[2 * FRAME_FEATURES..].iter().all(|v| *v == 1.0));
    }
}
