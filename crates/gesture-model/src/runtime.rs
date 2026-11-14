use crate::data::JOINTS;
use crate::data::{Frame, LABELS};
use crate::features::CHANNELS;
use crate::features::{FRAME_FEATURES, frame_features};
use crate::model::{Gcn, ModelConfig};
use anyhow::{Context, Result, anyhow, ensure};
use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Device, Tensor, TensorData, activation::softmax},
};
use std::{collections::VecDeque, fs, path::Path};

/// Online intent classifier: feed frames one by one, get class probabilities for the latest.
pub struct Runtime {
    model: Gcn,
    config: ModelConfig,
    device: Device,
    history: VecDeque<[f32; FRAME_FEATURES]>,
    last: Option<Frame>,
}

impl Runtime {
    pub fn load(dir: &Path) -> Result<Self> {
        let config = ModelConfig::from_json(
            &fs::read(dir.join("config.json")).context("缺少 config.json")?,
        )?;
        let device = Device::cuda(0);
        let record = ModuleRecord::load(dir.join("model.bpk"))
            .map_err(|e| anyhow!("无法读取 model.bpk：{e:?}"))?;
        let model = config.init(&device).try_load_record(record).map_err(|e| {
            anyhow!("模型与当前版本不匹配（类别或结构已变，请在标注 app 中重新训练）：{e:?}")
        })?;
        let runtime = Self {
            model,
            config,
            device,
            history: VecDeque::new(),
            last: None,
        };
        // Warm up kernels now, not on the first live frame, and fail early on a broken device.
        let probs = runtime.infer()?;
        ensure!(probs.iter().all(|p| p.is_finite()), "模型预热输出非有限值");
        Ok(runtime)
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.last = None;
    }

    pub fn push(&mut self, frame: Frame) -> Result<Vec<f32>> {
        if self
            .last
            .as_ref()
            .is_some_and(|l| frame.t_ms <= l.t_ms || frame.t_ms - l.t_ms > 250.0)
        {
            self.reset();
        }
        let features = frame_features(self.last.as_ref(), &frame);
        self.last = Some(frame);
        self.history.push_back(features);
        while self.history.len() > self.config.window {
            self.history.pop_front();
        }
        self.infer()
    }

    fn infer(&self) -> Result<Vec<f32>> {
        let mut x = vec![0.0; (self.config.window - self.history.len()) * FRAME_FEATURES];
        for f in &self.history {
            x.extend_from_slice(f);
        }
        let input = Tensor::from_data(
            TensorData::new(x, [1, self.config.window, JOINTS, CHANNELS]),
            &self.device,
        );
        let probs = softmax(self.model.forward(input), 1)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow!("{e:?}"))?;
        ensure!(
            probs.len() == LABELS.len() && probs.iter().all(|p| p.is_finite()),
            "模型输出无效"
        );
        Ok(probs)
    }
}
