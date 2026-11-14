use crate::data::JOINTS;
use crate::data::{LABELS, Session};
use crate::features::CHANNELS;
use crate::features::{FRAME_FEATURES, sequence_features, window};
use crate::model::{Gcn, ModelConfig};
use anyhow::{Result, bail, ensure};
use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams},
    tensor::{Device, Int, Tensor, TensorData, activation::softmax},
};
use serde::Serialize;
use std::{
    fs,
    ops::ControlFlow,
    path::Path,
    time::{Duration, Instant},
};

/// Training events, in order: one `Start`, then `Batch` (throttled) and `Epoch` per epoch.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Progress {
    Start {
        labels: Vec<String>,
        epochs: usize,
        /// Samples per class, in `labels` order.
        train_counts: Vec<usize>,
        val_counts: Vec<usize>,
    },
    Batch {
        epoch: usize,
        batch: usize,
        batches: usize,
        loss: f32,
    },
    Epoch(EpochStats),
}

#[derive(Clone, Debug, Serialize)]
pub struct EpochStats {
    pub epoch: usize,
    pub epochs: usize,
    pub lr: f64,
    pub train_loss: f32,
    /// Balanced accuracy (see [`balanced`]) on augmented training batches, so it can trail validation.
    pub train_balanced: f32,
    pub val_loss: f32,
    /// Balanced accuracy on the validation set; the best epoch is chosen by it.
    pub val_balanced: f32,
    /// Validation recall per class; `None` when a class has no validation samples.
    pub class_recall: Vec<Option<f32>>,
    /// This epoch beat every earlier one and was saved.
    pub best: bool,
    /// Epochs since the best one; training stops early when it reaches `patience`.
    pub stale: usize,
    pub patience: usize,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub labels: Vec<String>,
    pub train_samples: usize,
    pub val_samples: usize,
    pub best_epoch: usize,
    pub val_balanced: f32,
    /// Rows: true class, columns: predicted class (validation set, best epoch).
    pub confusion: Vec<Vec<usize>>,
    /// Stopped by the user before the last epoch; the best epoch so far was kept.
    pub stopped: bool,
    /// Ended before the last epoch because validation stopped improving.
    pub early_stopped: bool,
}

const BATCH_EVENT_INTERVAL: Duration = Duration::from_millis(200);
/// Epochs without a better validation score before training ends on its own.
pub const PATIENCE: usize = 10;

/// Mean recall over the classes that have samples, plus each class's recall. Every class
/// counts the same however many frames it has, so plentiful idle frames cannot hide a
/// class the model gets wrong.
fn balanced(hits: &[usize], counts: &[usize]) -> (f32, Vec<Option<f32>>) {
    let recall: Vec<Option<f32>> = hits
        .iter()
        .zip(counts)
        .map(|(&h, &n)| (n > 0).then(|| h as f32 / n as f32))
        .collect();
    let present: Vec<f32> = recall.iter().flatten().copied().collect();
    let mean = present.iter().sum::<f32>() / present.len().max(1) as f32;
    (mean, recall)
}

struct Samples {
    x: Vec<f32>,
    y: Vec<i64>,
}
impl Samples {
    fn len(&self) -> usize {
        self.y.len()
    }
}

/// Every labelled frame with a hand becomes one sample: the window ending at it.
/// The last 20 % of each session (in time) is held out, so validation frames never share
/// windows with training frames.
fn build(sessions: &[Session], config: &ModelConfig) -> (Samples, Samples) {
    let (mut train, mut val) = (
        Samples {
            x: vec![],
            y: vec![],
        },
        Samples {
            x: vec![],
            y: vec![],
        },
    );
    for s in sessions {
        let features = sequence_features(&s.frames);
        let split = s.frames.len() * 4 / 5;
        for (i, f) in s.frames.iter().enumerate() {
            let (Some(label), Some(_)) = (s.label_at(f.t_ms), f.hand()) else {
                continue;
            };
            // A window straddling the split would leak training frames into validation.
            if i >= split && i < split + config.window {
                continue;
            }
            let set = if i < split { &mut train } else { &mut val };
            window(&features, i, config.window, &mut set.x);
            set.y.push(label as i64);
        }
    }
    (train, val)
}

/// Small deterministic generator for shuffling and augmentation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Small in-plane rotation and scale of every window: the same gesture at another wrist angle or speed.
fn augment(x: &mut [f32], rng: &mut Rng) {
    let angle = (rng.next() - 0.5) * 0.5;
    let (sin, cos) = angle.sin_cos();
    let speed = 0.8 + rng.next() * 0.4;
    for joint in x.as_chunks_mut::<CHANNELS>().0 {
        for (a, b) in [(0, 1), (3, 4), (6, 7)] {
            let (u, v) = (joint[a], joint[b]);
            joint[a] = cos * u - sin * v;
            joint[b] = sin * u + cos * v;
        }
        for v in &mut joint[3..] {
            *v *= speed;
        }
    }
}

fn tensor(x: &[f32], batch: usize, config: &ModelConfig, device: &Device) -> Tensor<4> {
    Tensor::from_data(
        TensorData::new(x.to_vec(), [batch, config.window, JOINTS, CHANNELS]),
        device,
    )
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Predicted classes and mean (unweighted) cross-entropy.
fn predict(
    model: &Gcn,
    samples: &Samples,
    config: &ModelConfig,
    device: &Device,
) -> Result<(Vec<usize>, f32)> {
    let per = config.window * FRAME_FEATURES;
    let (mut out, mut loss) = (Vec::with_capacity(samples.len()), 0.0f64);
    for (chunk, y) in samples.x.chunks(per * 512).zip(samples.y.chunks(512)) {
        let probs = softmax(model.forward(tensor(chunk, y.len(), config, device)), 1)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (p, &label) in probs.chunks(LABELS.len()).zip(y) {
            loss -= (p[label as usize].max(1e-7) as f64).ln();
            out.push(argmax(p));
        }
    }
    Ok((out, (loss / samples.len().max(1) as f64) as f32))
}

/// Train on CUDA and write `config.json` + `model.bpk` to `output`. Returning
/// `ControlFlow::Break` from `progress` stops training; the best epoch so far is kept.
pub fn train(
    sessions: &[Session],
    output: &Path,
    epochs: usize,
    mut progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Report> {
    let config = ModelConfig::current();
    config.validate()?;
    let (train, val) = build(sessions, &config);
    ensure!(
        train.len() >= 64,
        "训练样本太少（{}），请先采集并标注更多数据",
        train.len()
    );
    ensure!(val.len() > 0, "没有验证样本，请采集更长的会话");
    let tally = |ys: &[i64]| {
        let mut counts = vec![0usize; LABELS.len()];
        for &y in ys {
            counts[y as usize] += 1;
        }
        counts
    };
    let (counts, val_counts) = (tally(&train.y), tally(&val.y));
    let present: Vec<&str> = LABELS
        .iter()
        .zip(&counts)
        .filter(|(_, n)| **n > 0)
        .map(|(l, _)| *l)
        .collect();
    ensure!(present.len() >= 2, "至少需要两个类别的数据");
    // Inverse-frequency class weights; absent classes get weight 1 (never sampled anyway).
    let total = train.len() as f32;
    let weights = counts
        .iter()
        .map(|&n| {
            if n == 0 {
                1.0
            } else {
                total / (n as f32 * present.len() as f32)
            }
        })
        .collect();

    let device = Device::cuda(0);
    let autodiff = device.clone().autodiff();
    let mut model = config.init(&autodiff);
    let loss_fn = CrossEntropyLossConfig::new()
        .with_weights(Some(weights))
        .with_smoothing(Some(0.05))
        .init(&autodiff);
    // Decoupled weight decay: with this little data, small weights generalise better.
    let mut optim = AdamWConfig::new().with_weight_decay(1e-2).init();
    let per = config.window * FRAME_FEATURES;
    let batch = 128;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut order: Vec<usize> = (0..train.len()).collect();
    // (validation balanced accuracy, validation loss, epoch, predictions) of the saved epoch.
    let mut best: Option<(f32, f32, usize, Vec<usize>)> = None;
    let mut early_stopped = false;
    fs::create_dir_all(output)?;
    let labels: Vec<String> = LABELS.iter().map(|s| s.to_string()).collect();
    let started = Instant::now();
    let mut stopped = progress(Progress::Start {
        labels: labels.clone(),
        epochs,
        train_counts: counts.clone(),
        val_counts: val_counts.clone(),
    })
    .is_break();
    let batches = order.len().div_ceil(batch);

    for epoch in 1..=epochs {
        if stopped {
            break;
        }
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next() * (i + 1) as f32) as usize % (i + 1));
        }
        let lr =
            2e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (epoch - 1) as f64 / epochs as f64).cos());
        let mut loss_sum = 0.0;
        let mut train_hits = vec![0usize; LABELS.len()];
        let mut last_event = Instant::now();
        for (b, idx) in order.chunks(batch).enumerate() {
            let mut x = Vec::with_capacity(idx.len() * per);
            for &i in idx {
                let start = x.len();
                x.extend_from_slice(&train.x[i * per..(i + 1) * per]);
                augment(&mut x[start..], &mut rng);
            }
            let y: Vec<i64> = idx.iter().map(|&i| train.y[i]).collect();
            let logits = model.forward(tensor(&x, idx.len(), &config, &autodiff));
            let predicted = logits
                .clone()
                .detach()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (p, &y) in predicted.chunks(LABELS.len()).zip(&y) {
                if argmax(p) as i64 == y {
                    train_hits[y as usize] += 1;
                }
            }
            let loss = loss_fn.forward(
                logits,
                Tensor::<1, Int>::from_data(TensorData::new(y, [idx.len()]), &autodiff),
            );
            loss_sum += loss.clone().into_scalar::<f32>();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            if last_event.elapsed() >= BATCH_EVENT_INTERVAL {
                last_event = Instant::now();
                let event = Progress::Batch {
                    epoch,
                    batch: b + 1,
                    batches,
                    loss: loss_sum / (b + 1) as f32,
                };
                if progress(event).is_break() {
                    stopped = true;
                    break;
                }
            }
        }
        // An interrupted epoch is discarded: its weights saw only part of the data.
        if stopped {
            break;
        }
        let eval = model.valid();
        let (predicted, val_loss) = predict(&eval, &val, &config, &autodiff)?;
        let mut hits = vec![0usize; LABELS.len()];
        for (p, y) in predicted.iter().zip(&val.y) {
            if *p as i64 == *y {
                hits[*p] += 1;
            }
        }
        let (score, class_recall) = balanced(&hits, &val_counts);
        let train_loss = loss_sum / batches as f32;
        if !score.is_finite() || !train_loss.is_finite() || !val_loss.is_finite() {
            bail!("训练发散（出现非有限值）");
        }
        // Higher balanced accuracy wins; on a tie, lower validation loss (more confident).
        let improved = best.as_ref().is_none_or(|(s, l, _, _)| {
            score > s + 1e-6 || ((score - s).abs() <= 1e-6 && val_loss < *l)
        });
        if improved {
            eval.clone()
                .into_record()
                .save(output.join("model.bpk"))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            best = Some((score, val_loss, epoch, predicted));
        }
        let stale = epoch - best.as_ref().map_or(epoch, |b| b.2);
        stopped = progress(Progress::Epoch(EpochStats {
            epoch,
            epochs,
            lr,
            train_loss,
            train_balanced: balanced(&train_hits, &counts).0,
            val_loss,
            val_balanced: score,
            class_recall,
            best: improved,
            stale,
            patience: PATIENCE,
            elapsed_ms: started.elapsed().as_millis() as u64,
        }))
        .is_break();
        if !stopped && stale >= PATIENCE && epoch < epochs {
            early_stopped = true;
            break;
        }
    }
    let Some((val_balanced, _, best_epoch, predicted)) = best else {
        bail!("训练已停止，第一轮未完成，没有保存模型");
    };
    fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let mut confusion = vec![vec![0; LABELS.len()]; LABELS.len()];
    for (p, y) in predicted.iter().zip(&val.y) {
        confusion[*y as usize][*p] += 1;
    }
    let _ = device;
    Ok(Report {
        labels,
        train_samples: train.len(),
        val_samples: val.len(),
        best_epoch,
        val_balanced,
        confusion,
        stopped,
        early_stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Frame, Segment};
    fn session(frames: usize) -> Session {
        let frames = (0..frames)
            .map(|i| Frame {
                t_ms: i as f64 * 33.0,
                aspect: 1.0,
                image: Some(vec![[0.5, 0.5]; 21]),
                world: Some((0..21).map(|j| [j as f32 * 0.01, 0.0, 0.0]).collect()),
            })
            .collect();
        Session {
            version: 1,
            id: "a".into(),
            prompt: String::new(),
            reviewed: true,
            frames,
            segments: vec![Segment {
                start_ms: 0.0,
                end_ms: 1e9,
                label: "point".into(),
            }],
        }
    }
    #[test]
    fn split_has_no_overlapping_windows() {
        let config = ModelConfig::current();
        let (train, val) = build(&[session(100)], &config);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 100 - 80 - config.window);
        assert_eq!(train.x.len(), train.len() * config.window * FRAME_FEATURES);
    }
    #[test]
    fn augmentation_preserves_shape_length() {
        let mut x = vec![1.0; CHANNELS * 4];
        augment(&mut x, &mut Rng(1));
        let n0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        assert!((n0 - 2f32.sqrt()).abs() < 1e-5);
    }
    #[test]
    fn balanced_accuracy_weighs_classes_equally_and_skips_empty_ones() {
        // 90/100 idle right, 1/10 point right, no swipes: frame accuracy would be 83 %.
        let (mean, recall) = balanced(&[90, 1, 0, 0], &[100, 10, 0, 0]);
        assert!((mean - 0.5).abs() < 1e-6);
        assert_eq!(recall, [Some(0.9), Some(0.1), None, None]);
    }
}
