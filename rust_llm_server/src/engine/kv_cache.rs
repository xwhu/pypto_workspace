use crate::model::config::Qwen3Config;
use crate::model::tensor::{DType, Tensor};

/// Per-layer KV cache entry for one sequence.
#[derive(Debug)]
pub struct LayerKVCache {
    /// Key cache: [kv_len, num_kv_heads, head_dim]
    pub k: Tensor,
    /// Value cache: [kv_len, num_kv_heads, head_dim]
    pub v: Tensor,
    /// Number of tokens currently cached.
    pub len: usize,
}

/// KV cache for one sequence across all layers.
#[derive(Debug)]
pub struct SequenceKVCache {
    /// Per-layer caches.
    pub layers: Vec<LayerKVCache>,
    /// Maximum capacity in tokens.
    pub max_len: usize,
}

impl SequenceKVCache {
    /// Create a new KV cache for a sequence.
    ///
    /// In stub mode, this only creates shape descriptors — no actual
    /// memory is allocated.
    pub fn new(config: &Qwen3Config, max_len: usize) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|i| LayerKVCache {
                k: Tensor::new(
                    vec![max_len, config.num_key_value_heads, config.head_dim],
                    DType::Float16,
                    format!("kv_cache.layer.{i}.k"),
                ),
                v: Tensor::new(
                    vec![max_len, config.num_key_value_heads, config.head_dim],
                    DType::Float16,
                    format!("kv_cache.layer.{i}.v"),
                ),
                len: 0,
            })
            .collect();

        Self { layers, max_len }
    }

    /// Append new K/V entries after a forward pass step.
    pub fn append(&mut self, num_new_tokens: usize) {
        for layer in &mut self.layers {
            layer.len += num_new_tokens;
        }
    }

    /// Current cached length (same for all layers).
    pub fn current_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len)
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.max_len.saturating_sub(self.current_len())
    }
}

/// Manager for all active sequence KV caches.
pub struct KVCacheManager {
    config: Qwen3Config,
    max_seq_len: usize,
}

impl KVCacheManager {
    pub fn new(config: Qwen3Config, max_seq_len: usize) -> Self {
        Self {
            config,
            max_seq_len,
        }
    }

    /// Allocate a new KV cache for a sequence.
    pub fn allocate(&self) -> SequenceKVCache {
        SequenceKVCache::new(&self.config, self.max_seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let config = Qwen3Config::qwen3_8b();
        let cache = SequenceKVCache::new(&config, 2048);

        assert_eq!(cache.layers.len(), 36);
        assert_eq!(cache.current_len(), 0);
        assert_eq!(cache.remaining(), 2048);
    }

    #[test]
    fn test_kv_cache_append() {
        let config = Qwen3Config::qwen3_8b();
        let mut cache = SequenceKVCache::new(&config, 2048);

        cache.append(10);
        assert_eq!(cache.current_len(), 10);
        assert_eq!(cache.remaining(), 2038);

        cache.append(5);
        assert_eq!(cache.current_len(), 15);
    }
}
