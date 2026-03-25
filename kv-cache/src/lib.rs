//! KV cache block management for LLM inference serving.
//!
//! This crate provides the **logical layer** of KV cache management:
//! block allocation, per-sequence tracking, and block-table generation
//! for paged attention kernels. It is independent of any specific GPU
//! runtime — no CUDA, CANN, or device memory is touched here.
//!
//! # Architecture
//!
//! ```text
//!  Scheduler
//!    │
//!    ├── can_allocate(n)?
//!    ├── allocate_for_seq(seq, prompt_len)
//!    ├── append_token(seq)          (per decode step)
//!    ├── build_block_table([seqs])  → Vec<Vec<u32>> for kernel
//!    └── release_seq(seq)
//!    │
//!    ▼
//!  SimpleKvManager
//!    └── BlockPool (free-list of block IDs)
//! ```
//!
//! # Example
//!
//! ```rust
//! use kv_cache::types::{KvCacheConfig, SeqId};
//! use kv_cache::kv_manager::SimpleKvManager;
//!
//! let config = KvCacheConfig {
//!     block_size: 16,
//!     num_layers: 32,
//!     num_kv_heads: 8,
//!     head_dim: 128,
//!     dtype_size: 2,
//! };
//!
//! let mut mgr = SimpleKvManager::new(config, 100);
//!
//! // Prefill
//! let seq = SeqId(42);
//! mgr.allocate_for_seq(seq, 50); // 50 tokens → 4 blocks (block_size=16)
//!
//! // Decode
//! mgr.append_token(seq);
//!
//! // Build block table for attention kernel
//! let table = mgr.build_block_table(&[seq]);
//!
//! // Release when done
//! mgr.release_seq(seq);
//! ```

pub mod block_hash;
pub mod block_pool;
pub mod kv_manager;
pub mod radix_tree;
pub mod types;
