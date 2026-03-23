use serde::{Deserialize, Serialize};

/// Parallel execution configuration.
///
/// Describes how the model is distributed across multiple devices.
/// Used at plan compilation time to determine weight sharding and
/// communication insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Tensor parallelism degree (how many devices share one layer).
    pub tp_size: usize,
    /// Pipeline parallelism degree (how many stages the layers are split into).
    pub pp_size: usize,
    /// This device's tensor parallel rank (0..tp_size-1).
    pub tp_rank: usize,
    /// This device's pipeline parallel rank (0..pp_size-1).
    pub pp_rank: usize,
}

impl ParallelConfig {
    /// Single-device config (no parallelism).
    pub fn single_device() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            tp_rank: 0,
            pp_rank: 0,
        }
    }

    /// Tensor-parallel only.
    pub fn tensor_parallel(tp_size: usize, tp_rank: usize) -> Self {
        Self {
            tp_size,
            pp_size: 1,
            tp_rank,
            pp_rank: 0,
        }
    }

    /// Pipeline-parallel only.
    pub fn pipeline_parallel(pp_size: usize, pp_rank: usize) -> Self {
        Self {
            tp_size: 1,
            pp_size,
            tp_rank: 0,
            pp_rank,
        }
    }

    /// Whether tensor parallelism is active.
    pub fn is_tp(&self) -> bool {
        self.tp_size > 1
    }

    /// Whether pipeline parallelism is active.
    pub fn is_pp(&self) -> bool {
        self.pp_size > 1
    }

    /// Whether running on a single device (no parallelism).
    pub fn is_single(&self) -> bool {
        !self.is_tp() && !self.is_pp()
    }

    /// Compute which layers are assigned to this PP stage.
    /// Returns (start_layer, end_layer) exclusive.
    pub fn pp_layer_range(&self, total_layers: usize) -> (usize, usize) {
        if !self.is_pp() {
            return (0, total_layers);
        }
        let layers_per_stage = total_layers / self.pp_size;
        let remainder = total_layers % self.pp_size;
        // Distribute remainder layers to earlier stages.
        let start = if self.pp_rank < remainder {
            self.pp_rank * (layers_per_stage + 1)
        } else {
            remainder * (layers_per_stage + 1) + (self.pp_rank - remainder) * layers_per_stage
        };
        let count = if self.pp_rank < remainder {
            layers_per_stage + 1
        } else {
            layers_per_stage
        };
        (start, start + count)
    }
}

/// How a weight tensor is sharded across TP devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Full copy on every device.
    Replicate,
    /// Split along columns (output dimension). Used for Q/K/V projections,
    /// gate_proj, up_proj. Each device gets `cols / tp_size` columns.
    /// Requires AllReduce after the corresponding output projection.
    ShardColumns,
    /// Split along rows (input dimension). Used for O projection, down_proj.
    /// Each device gets `rows / tp_size` rows.
    /// The input must be gathered or the output must be reduced.
    ShardRows,
}

/// Describes the TP shard strategy for each weight in a transformer layer.
///
/// Follows the Megatron-LM column/row parallel convention:
/// - QKV projections: shard columns (each device has a subset of heads)
/// - O projection: shard rows (each device has partial output, then AllReduce)
/// - gate_proj, up_proj: shard columns
/// - down_proj: shard rows
/// - LayerNorm / RMSNorm weights: replicate
pub struct Qwen3ShardMap;

impl Qwen3ShardMap {
    pub fn q_proj() -> ShardStrategy { ShardStrategy::ShardColumns }
    pub fn k_proj() -> ShardStrategy { ShardStrategy::ShardColumns }
    pub fn v_proj() -> ShardStrategy { ShardStrategy::ShardColumns }
    pub fn o_proj() -> ShardStrategy { ShardStrategy::ShardRows }
    pub fn gate_proj() -> ShardStrategy { ShardStrategy::ShardColumns }
    pub fn up_proj() -> ShardStrategy { ShardStrategy::ShardColumns }
    pub fn down_proj() -> ShardStrategy { ShardStrategy::ShardRows }
    pub fn norm_weight() -> ShardStrategy { ShardStrategy::Replicate }
    pub fn embed_tokens() -> ShardStrategy { ShardStrategy::Replicate }
    pub fn lm_head() -> ShardStrategy { ShardStrategy::ShardColumns }
}

/// Compute the physical shape of a weight after TP sharding.
pub fn shard_weight_shape(
    logical_shape: &[usize],
    strategy: ShardStrategy,
    tp_size: usize,
) -> Vec<usize> {
    if tp_size <= 1 {
        return logical_shape.to_vec();
    }
    match strategy {
        ShardStrategy::Replicate => logical_shape.to_vec(),
        ShardStrategy::ShardColumns => {
            // Split last dimension
            let mut shape = logical_shape.to_vec();
            let last = shape.last_mut().expect("empty shape");
            assert!(*last % tp_size == 0, "cols {} not divisible by tp_size {}", *last, tp_size);
            *last /= tp_size;
            shape
        }
        ShardStrategy::ShardRows => {
            // Split first dimension
            let mut shape = logical_shape.to_vec();
            let first = shape.first_mut().expect("empty shape");
            assert!(*first % tp_size == 0, "rows {} not divisible by tp_size {}", *first, tp_size);
            *first /= tp_size;
            shape
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_device() {
        let cfg = ParallelConfig::single_device();
        assert!(cfg.is_single());
        assert_eq!(cfg.pp_layer_range(36), (0, 36));
    }

    #[test]
    fn test_pp_layer_range() {
        let cfg = ParallelConfig::pipeline_parallel(4, 0);
        assert_eq!(cfg.pp_layer_range(36), (0, 9));

        let cfg = ParallelConfig::pipeline_parallel(4, 3);
        assert_eq!(cfg.pp_layer_range(36), (27, 36));
    }

    #[test]
    fn test_pp_layer_range_uneven() {
        // 36 layers / 5 stages = 7 + 7 + 7 + 7 + 8? No: 7*5=35, remainder=1
        // Stage 0 gets 8, stages 1-4 get 7
        let cfg = ParallelConfig::pipeline_parallel(5, 0);
        let (s, e) = cfg.pp_layer_range(36);
        assert_eq!((s, e), (0, 8)); // 7+1 remainder

        let cfg = ParallelConfig::pipeline_parallel(5, 4);
        let (s, e) = cfg.pp_layer_range(36);
        assert_eq!(e - s, 7);
    }

    #[test]
    fn test_shard_weight_shape() {
        // q_proj [4096, 4096] with TP=4 → [4096, 1024]
        let shape = shard_weight_shape(&[4096, 4096], ShardStrategy::ShardColumns, 4);
        assert_eq!(shape, vec![4096, 1024]);

        // o_proj [4096, 4096] with TP=4 → [1024, 4096]
        let shape = shard_weight_shape(&[4096, 4096], ShardStrategy::ShardRows, 4);
        assert_eq!(shape, vec![1024, 4096]);

        // norm [4096] replicate
        let shape = shard_weight_shape(&[4096], ShardStrategy::Replicate, 4);
        assert_eq!(shape, vec![4096]);
    }
}
