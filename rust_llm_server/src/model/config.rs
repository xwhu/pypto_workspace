use serde::{Deserialize, Serialize};

/// Qwen3 model configuration, matching HuggingFace `config.json` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Hidden dimension of the transformer (e.g., 4096 for Qwen3-8B).
    pub hidden_size: usize,

    /// FFN intermediate dimension (e.g., 12288 for Qwen3-8B).
    pub intermediate_size: usize,

    /// Number of transformer layers (e.g., 36 for Qwen3-8B).
    pub num_hidden_layers: usize,

    /// Number of attention heads for queries (e.g., 32 for Qwen3-8B).
    pub num_attention_heads: usize,

    /// Number of key-value heads for GQA (e.g., 8 for Qwen3-8B).
    pub num_key_value_heads: usize,

    /// Dimension per attention head (e.g., 128).
    pub head_dim: usize,

    /// Vocabulary size (e.g., 151936).
    pub vocab_size: usize,

    /// Maximum sequence length (e.g., 40960).
    pub max_position_embeddings: usize,

    /// RMSNorm epsilon (e.g., 1e-6).
    pub rms_norm_eps: f64,

    /// RoPE theta for rotary embeddings (e.g., 1_000_000.0).
    pub rope_theta: f64,

    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
}

fn default_model_type() -> String {
    "qwen3".to_string()
}

impl Qwen3Config {
    /// Default Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            model_type: "qwen3".to_string(),
        }
    }

    /// Default Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            model_type: "qwen3".to_string(),
        }
    }

    /// Default Qwen3-8B configuration.
    pub fn qwen3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            model_type: "qwen3".to_string(),
        }
    }

    /// Load configuration from a JSON file.
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }

    /// Number of query heads per KV head group (for GQA).
    pub fn num_queries_per_kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total KV size per token per layer in bytes (FP16).
    pub fn kv_size_per_token_per_layer(&self) -> usize {
        // K + V, each: num_kv_heads * head_dim * 2 bytes (FP16)
        2 * self.num_key_value_heads * self.head_dim * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_8b_config() {
        let cfg = Qwen3Config::qwen3_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.num_queries_per_kv_group(), 4);
    }

    #[test]
    fn test_qwen3_config_serde_roundtrip() {
        let cfg = Qwen3Config::qwen3_8b();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: Qwen3Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.hidden_size, cfg2.hidden_size);
        assert_eq!(cfg.num_hidden_layers, cfg2.num_hidden_layers);
    }

    #[test]
    fn test_kv_size_per_token() {
        let cfg = Qwen3Config::qwen3_8b();
        // 2 (K+V) * 8 kv_heads * 128 head_dim * 2 bytes = 4096 bytes
        assert_eq!(cfg.kv_size_per_token_per_layer(), 4096);
    }
}
