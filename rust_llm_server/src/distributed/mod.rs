//! Distributed execution support for multi-device inference.
//!
//! Provides rank mapping, process group management, and HCCL communicator
//! initialization for Tensor Parallelism (TP), Pipeline Parallelism (PP),
//! and Data Parallelism (DP).
//!
//! # Process Launch Convention
//!
//! Each NPU device runs as a separate OS process. Processes are coordinated
//! via environment variables (matching the PyTorch distributed convention):
//!
//! ```bash
//! WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 ./rust-llm-server --tp 4 --pp 2 ...
//! ```
//!
//! The binary reads these env vars to determine its role in the distributed
//! topology.

#[cfg(all(feature = "ascend", feature = "hccl"))]
pub mod process_group;

use crate::model::parallel::ParallelConfig;

/// Configuration for a distributed inference process.
///
/// Maps a world rank to its (tp_rank, pp_rank, dp_rank) position
/// in the 3D parallelism grid.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of processes.
    pub world_size: usize,
    /// This process's global rank (0..world_size-1).
    pub world_rank: usize,
    /// This process's local rank on the node (= device ID).
    pub local_rank: usize,
    /// Tensor parallelism degree.
    pub tp_size: usize,
    /// Pipeline parallelism degree.
    pub pp_size: usize,
    /// Data parallelism degree.
    pub dp_size: usize,
    /// This process's TP rank.
    pub tp_rank: usize,
    /// This process's PP rank.
    pub pp_rank: usize,
    /// This process's DP rank.
    pub dp_rank: usize,
}

impl DistributedConfig {
    /// Create a distributed config from parallelism degrees and world rank.
    ///
    /// Rank mapping (innermost = TP, then PP, outermost = DP):
    /// ```text
    /// world_rank = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
    /// ```
    pub fn new(
        tp_size: usize,
        pp_size: usize,
        dp_size: usize,
        world_rank: usize,
    ) -> Self {
        let world_size = tp_size * pp_size * dp_size;
        assert!(
            world_rank < world_size,
            "world_rank {} >= world_size {} (tp={} * pp={} * dp={})",
            world_rank,
            world_size,
            tp_size,
            pp_size,
            dp_size
        );

        let tp_pp_size = tp_size * pp_size;
        let dp_rank = world_rank / tp_pp_size;
        let remainder = world_rank % tp_pp_size;
        let pp_rank = remainder / tp_size;
        let tp_rank = remainder % tp_size;

        Self {
            world_size,
            world_rank,
            local_rank: world_rank, // assumes 1 node; override with LOCAL_RANK env
            tp_size,
            pp_size,
            dp_size,
            tp_rank,
            pp_rank,
            dp_rank,
        }
    }

    /// Create from environment variables.
    ///
    /// Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from the environment.
    /// `tp_size`, `pp_size`, `dp_size` come from CLI args.
    pub fn from_env(
        tp_size: usize,
        pp_size: usize,
        dp_size: usize,
    ) -> Result<Self, String> {
        let world_rank: usize = std::env::var("RANK")
            .map_err(|_| "RANK env var not set")?
            .parse()
            .map_err(|_| "RANK env var is not a valid integer")?;

        let world_size_env: usize = std::env::var("WORLD_SIZE")
            .map_err(|_| "WORLD_SIZE env var not set")?
            .parse()
            .map_err(|_| "WORLD_SIZE env var is not a valid integer")?;

        let expected_world_size = tp_size * pp_size * dp_size;
        if world_size_env != expected_world_size {
            return Err(format!(
                "WORLD_SIZE={} but tp*pp*dp = {}*{}*{} = {}",
                world_size_env, tp_size, pp_size, dp_size, expected_world_size
            ));
        }

        let local_rank: usize = std::env::var("LOCAL_RANK")
            .unwrap_or_else(|_| world_rank.to_string())
            .parse()
            .map_err(|_| "LOCAL_RANK env var is not a valid integer")?;

        let mut config = Self::new(tp_size, pp_size, dp_size, world_rank);
        config.local_rank = local_rank;
        Ok(config)
    }

    /// Convert to a `ParallelConfig` for the model/engine.
    pub fn to_parallel_config(&self) -> ParallelConfig {
        ParallelConfig {
            tp_size: self.tp_size,
            pp_size: self.pp_size,
            tp_rank: self.tp_rank,
            pp_rank: self.pp_rank,
            dp_size: self.dp_size,
            dp_rank: self.dp_rank,
        }
    }

    /// Get the Ascend device ID for this process.
    pub fn device_id(&self) -> i32 {
        self.local_rank as i32
    }

    /// Compute the world ranks that form this process's TP group.
    ///
    /// TP group: ranks with the same (pp_rank, dp_rank), varying tp_rank.
    #[allow(dead_code)]
    pub fn tp_group_ranks(&self) -> Vec<usize> {
        let base = self.dp_rank * (self.tp_size * self.pp_size) + self.pp_rank * self.tp_size;
        (0..self.tp_size).map(|tp| base + tp).collect()
    }

    /// Compute the world ranks that form this process's PP group.
    ///
    /// PP group: ranks with the same (tp_rank, dp_rank), varying pp_rank.
    #[allow(dead_code)]
    pub fn pp_group_ranks(&self) -> Vec<usize> {
        let base = self.dp_rank * (self.tp_size * self.pp_size);
        (0..self.pp_size)
            .map(|pp| base + pp * self.tp_size + self.tp_rank)
            .collect()
    }

    /// This process's rank within its TP group (0..tp_size-1).
    #[allow(dead_code)]
    pub fn tp_group_rank(&self) -> usize {
        self.tp_rank
    }

    /// This process's rank within its PP group (0..pp_size-1).
    #[allow(dead_code)]
    pub fn pp_group_rank(&self) -> usize {
        self.pp_rank
    }

    /// Whether this process is a single-device run (no distribution).
    #[allow(dead_code)]
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_mapping_tp4_pp2() {
        // 8 processes: TP=4, PP=2, DP=1
        // world_rank = pp_rank * tp_size + tp_rank
        let cfg = DistributedConfig::new(4, 2, 1, 0);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (0, 0, 0));

        let cfg = DistributedConfig::new(4, 2, 1, 3);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (3, 0, 0));

        let cfg = DistributedConfig::new(4, 2, 1, 4);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (0, 1, 0));

        let cfg = DistributedConfig::new(4, 2, 1, 7);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (3, 1, 0));
    }

    #[test]
    fn test_rank_mapping_tp2_pp2_dp2() {
        // 8 processes: TP=2, PP=2, DP=2
        let cfg = DistributedConfig::new(2, 2, 2, 0);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (0, 0, 0));

        let cfg = DistributedConfig::new(2, 2, 2, 1);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (1, 0, 0));

        let cfg = DistributedConfig::new(2, 2, 2, 2);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (0, 1, 0));

        let cfg = DistributedConfig::new(2, 2, 2, 4);
        assert_eq!((cfg.tp_rank, cfg.pp_rank, cfg.dp_rank), (0, 0, 1));
    }

    #[test]
    fn test_tp_group_ranks() {
        // TP=4, PP=2, DP=1: rank 5 is (tp=1, pp=1, dp=0)
        let cfg = DistributedConfig::new(4, 2, 1, 5);
        assert_eq!(cfg.tp_group_ranks(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_pp_group_ranks() {
        // TP=4, PP=2, DP=1: rank 1 is (tp=1, pp=0, dp=0)
        let cfg = DistributedConfig::new(4, 2, 1, 1);
        assert_eq!(cfg.pp_group_ranks(), vec![1, 5]);
    }

    #[test]
    fn test_to_parallel_config() {
        let cfg = DistributedConfig::new(4, 2, 1, 5);
        let parallel = cfg.to_parallel_config();
        assert_eq!(parallel.tp_size, 4);
        assert_eq!(parallel.pp_size, 2);
        assert_eq!(parallel.tp_rank, 1);
        assert_eq!(parallel.pp_rank, 1);
    }

    #[test]
    fn test_single_device() {
        let cfg = DistributedConfig::new(1, 1, 1, 0);
        assert!(cfg.is_single());
        assert_eq!(cfg.tp_group_ranks(), vec![0]);
        assert_eq!(cfg.pp_group_ranks(), vec![0]);
    }
}
