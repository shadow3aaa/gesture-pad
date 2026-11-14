//! Spatio-temporal graph convolution over the 21-joint hand skeleton (ST-GCN style),
//! with causal temporal convolution so it can run online with no look-ahead.
use crate::data::{JOINTS, LABELS};
use crate::features::CHANNELS;
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Device, Tensor, activation::relu},
};

/// MediaPipe hand skeleton.
pub const EDGES: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Frames per input window (≈ 0.5 s at 30 fps).
    #[config(default = 16)]
    pub window: usize,
    /// Causal temporal kernel length.
    #[config(default = 5)]
    pub kernel: usize,
    #[config(default = 48)]
    pub width: usize,
    #[config(default = 3)]
    pub layers: usize,
    /// Dropout after each block and before the classifier, during training only.
    #[config(default = 0.2)]
    pub dropout: f64,
    /// Output classes the model was trained with; must match today's `LABELS`.
    pub classes: usize,
}

const RETRAIN: &str = "模型是旧版本训练的（类别或结构与当前不同），请在标注 app 中重新训练";

impl ModelConfig {
    /// Config for a new model with today's classes.
    pub fn current() -> Self {
        Self::new(LABELS.len())
    }
    /// Read a saved `config.json`. Configs from earlier versions (missing fields, other
    /// classes) are refused with a hint to retrain.
    pub fn from_json(bytes: &[u8]) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_slice(bytes).map_err(|_| anyhow::anyhow!(RETRAIN))?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.classes == LABELS.len(), RETRAIN);
        anyhow::ensure!(
            self.window >= 2
                && self.kernel >= 1
                && self.kernel <= self.window
                && self.width > 0
                && self.layers > 0
                && (0.0..1.0).contains(&self.dropout),
            "invalid model config"
        );
        Ok(())
    }
    pub fn init(&self, device: &Device) -> Gcn {
        let blocks = (0..self.layers)
            .map(|i| {
                let input = if i == 0 { CHANNELS } else { self.width };
                Block {
                    spatial: LinearConfig::new(input, self.width).init(device),
                    temporal: LinearConfig::new(self.width * self.kernel, self.width).init(device),
                    residual: (input != self.width)
                        .then(|| LinearConfig::new(input, self.width).init(device)),
                    norm: LayerNormConfig::new(self.width).init(device),
                    kernel: self.kernel,
                }
            })
            .collect();
        Gcn {
            blocks,
            dropout: DropoutConfig::new(self.dropout).init(),
            head: LinearConfig::new(self.width, LABELS.len()).init(device),
        }
    }
}

/// Symmetrically normalised adjacency with self loops, D^-½ (A + I) D^-½.
fn adjacency() -> [f32; JOINTS * JOINTS] {
    let mut a = [0.0; JOINTS * JOINTS];
    for j in 0..JOINTS {
        a[j * JOINTS + j] = 1.0;
    }
    for (i, j) in EDGES {
        a[i * JOINTS + j] = 1.0;
        a[j * JOINTS + i] = 1.0;
    }
    let degree: Vec<f32> = (0..JOINTS)
        .map(|i| a[i * JOINTS..(i + 1) * JOINTS].iter().sum())
        .collect();
    for i in 0..JOINTS {
        for j in 0..JOINTS {
            a[i * JOINTS + j] /= (degree[i] * degree[j]).sqrt();
        }
    }
    a
}

#[derive(Module, Debug)]
pub struct Block {
    spatial: Linear,
    temporal: Linear,
    residual: Option<Linear>,
    norm: LayerNorm,
    kernel: usize,
}

impl Block {
    /// `x`: [batch, time, joints, channels].
    fn forward(&self, x: Tensor<4>, graph: &Tensor<2>) -> Tensor<4> {
        let [b, t, j, c] = x.dims();
        // Spatial: mix each joint with its skeleton neighbours, then project channels.
        let mixed = x
            .clone()
            .permute([0, 1, 3, 2])
            .reshape([b * t * c, j])
            .matmul(graph.clone())
            .reshape([b, t, c, j])
            .permute([0, 1, 3, 2]);
        let h = relu(self.spatial.forward(mixed));
        let w = h.dims()[3];
        // Temporal: causal convolution, as a projection of the last `kernel` steps stacked on channels.
        let padded = Tensor::cat(
            vec![Tensor::zeros([b, self.kernel - 1, j, w], &h.device()), h],
            1,
        );
        let stacked = Tensor::cat(
            (0..self.kernel)
                .map(|k| padded.clone().narrow(1, k, t))
                .collect(),
            3,
        );
        let h = self.temporal.forward(stacked);
        let r = match &self.residual {
            Some(l) => l.forward(x),
            None => x,
        };
        relu(self.norm.forward(h + r))
    }
}

#[derive(Module, Debug)]
pub struct Gcn {
    blocks: Vec<Block>,
    dropout: Dropout,
    head: Linear,
}

impl Gcn {
    /// `x`: [batch, window, joints, channels] → logits [batch, classes] for the last frame.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let device = x.device();
        let graph =
            Tensor::<1>::from_floats(adjacency().as_slice(), &device).reshape([JOINTS, JOINTS]);
        let mut h = x;
        for block in &self.blocks {
            h = self.dropout.forward(block.forward(h, &graph));
        }
        let [b, t, j, w] = h.dims();
        let last = h
            .narrow(1, t - 1, 1)
            .reshape([b, j, w])
            .mean_dim(1)
            .reshape([b, w]);
        self.head.forward(last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adjacency_is_symmetric_and_normalised() {
        let a = adjacency();
        for i in 0..JOINTS {
            for j in 0..JOINTS {
                assert!((a[i * JOINTS + j] - a[j * JOINTS + i]).abs() < 1e-6);
            }
        }
        // Wrist (degree 3 + self) to index knuckle (degree 3 + self).
        assert!((a[5] - 1.0 / 4.0).abs() < 1e-6);
    }
    #[test]
    fn configs_from_other_label_sets_are_refused() {
        let retrain = |json: &str| {
            ModelConfig::from_json(json.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("重新训练")
        };
        // Written before dropout and the class count were saved.
        assert!(retrain(r#"{"window":16,"kernel":5,"width":48,"layers":3}"#));
        let other = serde_json::to_string(&ModelConfig::new(LABELS.len() + 1)).unwrap();
        assert!(retrain(&other));
        let saved = serde_json::to_vec(&ModelConfig::current()).unwrap();
        ModelConfig::from_json(&saved).unwrap();
    }
}
