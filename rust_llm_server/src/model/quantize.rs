use serde::{Deserialize, Serialize};

use super::parallel::ShardStrategy;
use super::tensor::DType;

/// Quantization scheme for a weight tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantScheme {
    /// No quantization — original precision (FP16/BF16).
    None,

    /// Per-tensor symmetric quantization.
    /// `weight_int = round(weight_fp / scale)`
    PerTensor {
        dtype: DType,
        scale: f32,
    },

    /// Per-channel (per-output-channel) quantization.
    /// Each output channel has its own scale factor.
    PerChannel {
        dtype: DType,
        num_channels: usize,
    },

    /// Group-wise quantization (GPTQ / AWQ style).
    /// Weights are split into groups along the input dimension,
    /// each group has its own scale and zero-point.
    GroupWise {
        dtype: DType,
        group_size: usize,
    },
}

impl QuantScheme {
    /// Whether this is a quantized scheme (not None).
    pub fn is_quantized(&self) -> bool {
        !matches!(self, QuantScheme::None)
    }

    /// The storage dtype.
    pub fn storage_dtype(&self) -> DType {
        match self {
            QuantScheme::None => DType::Float16,
            QuantScheme::PerTensor { dtype, .. } => *dtype,
            QuantScheme::PerChannel { dtype, .. } => *dtype,
            QuantScheme::GroupWise { dtype, .. } => *dtype,
        }
    }
}

/// Per-model quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Default scheme applied to all linear weights.
    pub default_scheme: QuantScheme,

    /// Override: scheme for attention projections (Q/K/V/O).
    pub attention_scheme: Option<QuantScheme>,

    /// Override: scheme for MLP projections (gate/up/down).
    pub mlp_scheme: Option<QuantScheme>,

    /// Override: scheme for the LM head.
    pub lm_head_scheme: Option<QuantScheme>,
}

impl QuantConfig {
    /// No quantization — everything stays FP16.
    pub fn none() -> Self {
        Self {
            default_scheme: QuantScheme::None,
            attention_scheme: None,
            mlp_scheme: None,
            lm_head_scheme: None,
        }
    }

    /// INT8 per-tensor quantization for all linear layers.
    pub fn int8_per_tensor() -> Self {
        Self {
            default_scheme: QuantScheme::PerTensor {
                dtype: DType::Int8,
                scale: 1.0, // placeholder, actual scale from calibration
            },
            attention_scheme: None,
            mlp_scheme: None,
            lm_head_scheme: Some(QuantScheme::None), // keep LM head in FP16
        }
    }

    /// AWQ-style INT4 group-wise quantization.
    pub fn awq_int4(group_size: usize) -> Self {
        Self {
            default_scheme: QuantScheme::GroupWise {
                dtype: DType::Int4,
                group_size,
            },
            attention_scheme: None,
            mlp_scheme: None,
            lm_head_scheme: Some(QuantScheme::None),
        }
    }

    /// Get the quantization scheme for a given weight name.
    pub fn scheme_for(&self, weight_name: &str) -> &QuantScheme {
        if weight_name.contains("lm_head") {
            return self.lm_head_scheme.as_ref().unwrap_or(&self.default_scheme);
        }
        if weight_name.contains("self_attn") {
            return self.attention_scheme.as_ref().unwrap_or(&self.default_scheme);
        }
        if weight_name.contains("mlp") {
            return self.mlp_scheme.as_ref().unwrap_or(&self.default_scheme);
        }
        &self.default_scheme
    }
}

/// A physical weight — logical weight + shard + quantization metadata.
///
/// This is what the execution plan operates on. Created at plan compilation
/// time by combining the logical model, parallel config, and quant config.
#[derive(Debug, Clone)]
pub struct PhysicalWeight {
    /// Original weight name (e.g., "model.layers.0.self_attn.q_proj.weight").
    pub name: String,

    /// Logical shape (before sharding/quantization).
    pub logical_shape: Vec<usize>,

    /// Physical shape (after sharding, before quantization packing).
    pub physical_shape: Vec<usize>,

    /// How this weight is sharded across TP devices.
    pub shard: ShardStrategy,

    /// Quantization scheme.
    pub quant: QuantScheme,

    /// Physical storage dtype.
    pub physical_dtype: DType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_none() {
        let qc = QuantConfig::none();
        assert!(!qc.scheme_for("model.layers.0.self_attn.q_proj.weight").is_quantized());
        assert!(!qc.scheme_for("lm_head.weight").is_quantized());
    }

    #[test]
    fn test_quant_config_int8() {
        let qc = QuantConfig::int8_per_tensor();
        assert!(qc.scheme_for("model.layers.0.self_attn.q_proj.weight").is_quantized());
        assert!(qc.scheme_for("model.layers.0.mlp.gate_proj.weight").is_quantized());
        // LM head stays FP16
        assert!(!qc.scheme_for("lm_head.weight").is_quantized());
    }

    #[test]
    fn test_quant_config_awq() {
        let qc = QuantConfig::awq_int4(128);
        match qc.scheme_for("model.layers.0.mlp.gate_proj.weight") {
            QuantScheme::GroupWise { group_size, .. } => assert_eq!(*group_size, 128),
            other => panic!("Expected GroupWise, got {:?}", other),
        }
    }
}
