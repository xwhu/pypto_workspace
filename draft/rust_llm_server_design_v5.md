# Rust LLM Inference Server — Design & Implementation Plan (v5)

> **Target:** 1–N machines, 8–16 GPU/NPU cards per machine, high-performance inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI.
> **New in v5:** KVCache pool management with multi-tier storage, KV connector interface, cross-instance cache sharing, and cache-aware routing. Inspired by Mooncake and LMCache, kept minimal.

---

## Table of Contents

1. [Critique of v4 and What v5 Fixes](#1-critique-of-v4-and-what-v5-fixes)
2. [KVCache Pool Architecture Overview](#2-kvcache-pool-architecture-overview)
3. [KV Connector Interface](#3-kv-connector-interface)
4. [Multi-Tier Storage Backends](#4-multi-tier-storage-backends)
5. [Transfer Engine](#5-transfer-engine)
6. [KVCache Pool Manager](#6-kvcache-pool-manager)
7. [Cache-Aware Routing](#7-cache-aware-routing)
8. [Integration with Existing Components](#8-integration-with-existing-components)
9. [Configuration](#9-configuration)
10. [Updated Directory Layout](#10-updated-directory-layout)
11. [Implementation Phase: KVCache Pool (Phase 10)](#11-implementation-phase-kvcache-pool-phase-10)
12. [Performance Targets (Updated)](#12-performance-targets-updated)

---

## 1. Critique of v4 and What v5 Fixes

### Critique

v4 is a solid single-instance design. Its `RadixKvManager` handles prefix caching within one server process well. However, it has blind spots for production deployments with multiple instances:

| # | v4 Problem | Why It Matters |
|---|---|---|
| 1 | **KV cache is GPU-only with simple CPU swap.** The only overflow path is CPU DRAM swap during preemption. There is no persistent storage tier (SSD, remote). | Long-context agent workloads generate GBs of KV per session. When GPU blocks are evicted, the KV is lost — the next turn must recompute from scratch. A persistent tier would save that recomputation. |
| 2 | **No cross-instance cache sharing.** Each server instance maintains its own radix tree in isolation. | In a multi-instance deployment, requests from the same user/session may hit different instances. Without shared cache, prefix hit rates drop from >90% to near zero. |
| 3 | **Disaggregated serving KV transfer is point-to-point.** v4 §14 describes IPC/RDMA between a prefill node and a decode node, but there is no shared pool. | If the decode node chosen differs from the one the prefill node expected, the KV must be re-transferred. A shared store decouples producer and consumer. |
| 4 | **No cache-aware routing.** v4 has no mechanism to steer requests toward the instance that already has their prefix cached. | Cache-blind load balancing (round-robin, least-connections) scatters related requests, destroying locality. This is the #1 cause of low prefix cache hit rates at cluster scale. |
| 5 | **No standardized connector API.** The `RadixKvManager` is tightly coupled to the scheduler. Adding a new storage backend (SSD, Redis, remote RDMA pool) requires modifying core scheduler code. | A plugin interface lets backends evolve independently. This is the pattern proven by vLLM's KV Connector API and LMCache. |

### What v5 Adds

v5 retains all of v4 and adds:

- **§3**: `KvConnector` trait — standardized interface for cache backends
- **§4**: Multi-tier storage: GPU HBM → CPU DRAM → Local SSD → Remote RDMA pool
- **§5**: Transfer engine for high-throughput KV data movement
- **§6**: `KvCachePoolManager` — orchestrates tiers, eviction, and cross-instance sharing
- **§7**: Cache-aware routing for multi-instance deployments
- **§8**: Integration points with v4's scheduler, radix tree, and disagg serving

### What v5 Does NOT Change

Everything in v4 §2–§22 remains. v5 is purely additive. The existing `RadixKvManager` becomes the L1 (GPU-local) tier within the new multi-tier hierarchy.

---

## 2. KVCache Pool Architecture Overview

```
                     ┌─────────────────────────────────────┐
                     │        Cache-Aware Router            │
                     │  Prefix hash → best instance         │
                     └──────────────┬──────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │  Instance 0   │     │  Instance 1   │     │  Instance 2   │
    │               │     │               │     │               │
    │ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
    │ │ L1: GPU   │ │     │ │ L1: GPU   │ │     │ │ L1: GPU   │ │
    │ │ RadixTree  │ │     │ │ RadixTree  │ │     │ │ RadixTree  │ │
    │ └─────┬─────┘ │     │ └─────┬─────┘ │     │ └─────┬─────┘ │
    │       │ miss   │     │       │ miss   │     │       │ miss   │
    │ ┌─────▼─────┐ │     │ ┌─────▼─────┐ │     │ ┌─────▼─────┐ │
    │ │ L2: CPU   │ │     │ │ L2: CPU   │ │     │ │ L2: CPU   │ │
    │ │ DRAM Pool │ │     │ │ DRAM Pool │ │     │ │ DRAM Pool │ │
    │ └─────┬─────┘ │     │ └─────┬─────┘ │     │ └─────┬─────┘ │
    └───────┼───────┘     └───────┼───────┘     └───────┼───────┘
            │ miss                │ miss                │ miss
            └─────────────────────┼─────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │    L3: Shared Store       │
                    │  (Remote DRAM / SSD pool) │
                    │  Addressed by block hash  │
                    │  Transfer via RDMA / TCP   │
                    └──────────────────────────┘
```

### Key Principle: Hash-Based Block Addressing

Every KV block is identified by the SHA-256 hash of its token prefix (same hash v4's radix tree uses). This hash is the universal address across all tiers and instances. When instance 0 computes a block, it can publish the block to the shared store under this hash. When instance 1 receives a request with the same prefix, it looks up the hash in the store and fetches the pre-computed block instead of recomputing.

---

## 3. KV Connector Interface

The `KvConnector` trait is the boundary between the inference engine and cache storage. It is intentionally minimal — backends implement only what they support.

```rust
/// Universal block identifier: SHA-256 hash of the token prefix up to and including this block.
pub type BlockHash = [u8; 32];

/// Metadata for a single KV block.
#[derive(Clone, Debug)]
pub struct KvBlockMeta {
    pub hash: BlockHash,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,  // tokens per block
    pub dtype: KvDtype,
}

/// A handle to KV block data in some storage tier.
/// The data layout is: [2, num_layers, block_size, num_kv_heads, head_dim]
/// (K and V interleaved, matching the GPU block shape from v4 §5).
pub struct KvBlockData {
    pub meta: KvBlockMeta,
    pub data: Vec<u8>,  // owned byte buffer (for transfer)
}

/// The connector interface. Each storage backend implements this.
pub trait KvConnector: Send + Sync {
    /// Check if blocks exist in this tier. Returns which hashes are present.
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool>;

    /// Fetch blocks from this tier into CPU memory.
    /// Returns None for blocks not found.
    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>>;

    /// Store blocks into this tier.
    fn store(&self, blocks: &[KvBlockData]);

    /// Remove blocks from this tier (eviction).
    fn remove(&self, hashes: &[BlockHash]);

    /// Tier name for logging/metrics.
    fn tier_name(&self) -> &str;

    /// Approximate capacity in blocks.
    fn capacity(&self) -> usize;

    /// Current usage in blocks.
    fn usage(&self) -> usize;
}
```

### Why This Interface

- **`contains` before `fetch`**: Avoids transferring data when we only need to know if a block exists (used by the router).
- **`BlockHash` as the key**: Same hash the radix tree computes. No translation layer needed.
- **`KvBlockData` owns its bytes**: The data must cross thread/process/network boundaries. Owned buffers avoid lifetime issues.
- **No async**: Storage operations are called from a background thread, not the scheduler hot path. The scheduler queries L1 (GPU radix tree) synchronously; L2+ lookups happen asynchronously via a prefetch pipeline.

---

## 4. Multi-Tier Storage Backends

### 4.1 L1: GPU (Existing RadixKvManager)

v4's `RadixKvManager` + `BlockPool` is the L1 tier. No change to its internal design. It gains one new capability: on eviction, instead of just freeing the block, it can optionally push the block down to L2.

```rust
/// Extension to RadixKvManager for tier demotion.
impl RadixKvManager {
    /// Evict blocks, but write them to the next tier before freeing GPU memory.
    pub fn evict_to_tier(&mut self, target: usize, next_tier: &dyn KvConnector) -> usize {
        let mut freed = 0;
        while freed < target {
            if let Some(leaf) = self.find_oldest_unreferenced_leaf() {
                // Demote to next tier before freeing
                let block_data = self.gpu_pool.read_block(leaf.block_id);
                let kv_data = KvBlockData {
                    meta: self.block_meta(leaf),
                    data: block_data,
                };
                next_tier.store(&[kv_data]);

                self.gpu_pool.free(leaf.block_id);
                self.remove_leaf(leaf);
                freed += 1;
            } else {
                break;
            }
        }
        freed
    }
}
```

### 4.2 L2: CPU DRAM Connector

A simple in-process hash map storing KV blocks in pinned host memory. This is the first fallback and has the lowest latency after GPU.

```rust
pub struct CpuDramConnector {
    /// Hash → pinned host buffer
    blocks: RwLock<HashMap<BlockHash, KvBlockData>>,
    /// Max blocks to store
    capacity: usize,
    /// LRU tracking
    lru: Mutex<LruCache<BlockHash, ()>>,
}

impl CpuDramConnector {
    pub fn new(capacity_blocks: usize) -> Self {
        Self {
            blocks: RwLock::new(HashMap::with_capacity(capacity_blocks)),
            capacity: capacity_blocks,
            lru: Mutex::new(LruCache::new(capacity_blocks)),
        }
    }
}

impl KvConnector for CpuDramConnector {
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        let map = self.blocks.read();
        hashes.iter().map(|h| map.contains_key(h)).collect()
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        let map = self.blocks.read();
        let mut lru = self.lru.lock();
        hashes.iter().map(|h| {
            map.get(h).map(|b| {
                lru.promote(h);
                b.clone()
            })
        }).collect()
    }

    fn store(&self, blocks: &[KvBlockData]) {
        let mut map = self.blocks.write();
        let mut lru = self.lru.lock();
        for block in blocks {
            // Evict if full
            while map.len() >= self.capacity {
                if let Some(evicted) = lru.pop_lru() {
                    map.remove(&evicted);
                }
            }
            lru.insert(block.meta.hash, ());
            map.insert(block.meta.hash, block.clone());
        }
    }

    fn remove(&self, hashes: &[BlockHash]) {
        let mut map = self.blocks.write();
        let mut lru = self.lru.lock();
        for h in hashes {
            map.remove(h);
            lru.remove(h);
        }
    }

    fn tier_name(&self) -> &str { "cpu_dram" }
    fn capacity(&self) -> usize { self.capacity }
    fn usage(&self) -> usize { self.blocks.read().len() }
}
```

### 4.3 L3: Remote Store Connector

Connects to a shared KV store over the network. This is the cross-instance tier. In production, this would be backed by a distributed DRAM pool (Mooncake Store) or Redis. We define the connector in terms of the transfer engine (§5).

```rust
pub struct RemoteStoreConnector {
    /// Transfer engine for data movement
    transfer: Arc<TransferEngine>,
    /// Address of the store service
    store_addr: SocketAddr,
    /// Instance ID (for publishing block locations)
    instance_id: u32,
}

impl KvConnector for RemoteStoreConnector {
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        // RPC: send hash list, receive bitmap
        self.transfer.rpc_contains(self.store_addr, hashes)
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        // Batch fetch via transfer engine (RDMA if available, TCP fallback)
        self.transfer.fetch_blocks(self.store_addr, hashes)
    }

    fn store(&self, blocks: &[KvBlockData]) {
        // Batch store: coalesce small blocks into a single transfer
        self.transfer.store_blocks(self.store_addr, self.instance_id, blocks)
    }

    fn remove(&self, hashes: &[BlockHash]) {
        self.transfer.rpc_remove(self.store_addr, hashes)
    }

    fn tier_name(&self) -> &str { "remote_store" }
    fn capacity(&self) -> usize { self.transfer.rpc_capacity(self.store_addr) }
    fn usage(&self) -> usize { self.transfer.rpc_usage(self.store_addr) }
}
```

### 4.4 Optional: Local SSD Connector

For single-node deployments with very long contexts, an NVMe SSD tier sits between CPU DRAM and remote. Uses `io_uring` for async IO.

```rust
pub struct SsdConnector {
    /// Base directory for block files
    base_dir: PathBuf,
    /// Index: hash → file offset (in-memory for fast lookup)
    index: RwLock<HashMap<BlockHash, (u64, u32)>>,  // (offset, length)
    /// File handle for the block store (single large file, append-only)
    file: Mutex<File>,
    capacity: usize,
    lru: Mutex<LruCache<BlockHash, ()>>,
}

impl KvConnector for SsdConnector {
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        let idx = self.index.read();
        hashes.iter().map(|h| idx.contains_key(h)).collect()
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        let idx = self.index.read();
        // Batch read: sort by offset for sequential access, use io_uring
        let mut reads: Vec<_> = hashes.iter().enumerate()
            .filter_map(|(i, h)| idx.get(h).map(|&(off, len)| (i, off, len)))
            .collect();
        reads.sort_by_key(|&(_, off, _)| off);

        let mut results = vec![None; hashes.len()];
        for (i, offset, len) in reads {
            let data = self.read_at(offset, len as usize);
            results[i] = Some(KvBlockData {
                meta: self.decode_meta(&data),
                data,
            });
        }
        results
    }

    fn store(&self, blocks: &[KvBlockData]) {
        let mut file = self.file.lock();
        let mut idx = self.index.write();
        let mut lru = self.lru.lock();

        for block in blocks {
            let offset = file.seek(SeekFrom::End(0)).unwrap();
            let serialized = self.serialize_block(block);
            file.write_all(&serialized).unwrap();

            while idx.len() >= self.capacity {
                if let Some(evicted) = lru.pop_lru() {
                    idx.remove(&evicted);
                    // Note: space is not reclaimed immediately (compaction is separate)
                }
            }

            idx.insert(block.meta.hash, (offset, serialized.len() as u32));
            lru.insert(block.meta.hash, ());
        }
    }

    fn remove(&self, hashes: &[BlockHash]) {
        let mut idx = self.index.write();
        let mut lru = self.lru.lock();
        for h in hashes {
            idx.remove(h);
            lru.remove(h);
        }
    }

    fn tier_name(&self) -> &str { "local_ssd" }
    fn capacity(&self) -> usize { self.capacity }
    fn usage(&self) -> usize { self.index.read().len() }
}
```

---

## 5. Transfer Engine

The transfer engine handles KV block data movement between instances. It abstracts over transport mechanisms, selecting the best available.

### 5.1 Transport Selection

```rust
pub enum Transport {
    /// Intra-node: CUDA/Ascend IPC (zero-copy, ~300 GB/s)
    DeviceIpc,
    /// Intra-node: shared memory (for CPU-to-CPU, ~50 GB/s)
    Shm,
    /// Inter-node: RDMA (InfiniBand / RoCEv2, ~25 GB/s per port)
    Rdma,
    /// Fallback: TCP (~3 GB/s)
    Tcp,
}

pub struct TransferEngine {
    /// Available transports, ordered by preference
    transports: Vec<Transport>,
    /// Peer registry: instance_id → (address, available transports)
    peers: RwLock<HashMap<u32, PeerInfo>>,
    /// Listener for incoming block transfers
    listener: TransferListener,
}

struct PeerInfo {
    addr: SocketAddr,
    transports: Vec<Transport>,
    /// Is this peer on the same node?
    same_node: bool,
}
```

### 5.2 Transport Auto-Detection

```rust
impl TransferEngine {
    pub fn new(config: &TransferConfig) -> Self {
        let mut transports = Vec::new();

        // Check for RDMA capability
        if config.enable_rdma {
            if Self::detect_rdma_devices().is_some() {
                transports.push(Transport::Rdma);
            }
        }

        // IPC is always available for same-node peers
        transports.push(Transport::DeviceIpc);
        transports.push(Transport::Shm);

        // TCP is always the fallback
        transports.push(Transport::Tcp);

        Self {
            transports,
            peers: RwLock::new(HashMap::new()),
            listener: TransferListener::bind(config.listen_addr),
        }
    }

    /// Select the best transport for a given peer.
    fn select_transport(&self, peer: &PeerInfo) -> Transport {
        for t in &self.transports {
            match t {
                Transport::DeviceIpc | Transport::Shm if peer.same_node => return *t,
                Transport::Rdma if peer.transports.contains(&Transport::Rdma) => return *t,
                Transport::Tcp => return *t,
                _ => continue,
            }
        }
        Transport::Tcp
    }
}
```

### 5.3 Batch Transfer (Coalescing)

Small KV blocks transferred individually waste network bandwidth on per-message overhead. The transfer engine coalesces multiple blocks into a single contiguous buffer before sending.

```rust
impl TransferEngine {
    /// Send multiple KV blocks to a peer in a single transfer.
    pub fn send_blocks(&self, peer_id: u32, blocks: &[KvBlockData]) -> Result<(), TransferError> {
        let peer = self.peers.read().get(&peer_id).cloned()
            .ok_or(TransferError::PeerNotFound(peer_id))?;
        let transport = self.select_transport(&peer);

        // Coalesce: [header][block0_data][block1_data]...[blockN_data]
        let total_bytes: usize = blocks.iter().map(|b| b.data.len()).sum();
        let header_bytes = blocks.len() * std::mem::size_of::<BlockTransferHeader>();
        let mut buf = Vec::with_capacity(header_bytes + total_bytes);

        // Write headers
        for block in blocks {
            let header = BlockTransferHeader {
                hash: block.meta.hash,
                data_len: block.data.len() as u32,
            };
            buf.extend_from_slice(bytemuck::bytes_of(&header));
        }

        // Write data
        for block in blocks {
            buf.extend_from_slice(&block.data);
        }

        // Send in one shot
        match transport {
            Transport::Rdma => self.rdma_send(&peer.addr, &buf),
            Transport::Tcp => self.tcp_send(&peer.addr, &buf),
            Transport::DeviceIpc => self.ipc_send(&peer.addr, &buf),
            Transport::Shm => self.shm_send(&peer.addr, &buf),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockTransferHeader {
    hash: BlockHash,
    data_len: u32,
}
```

---

## 6. KVCache Pool Manager

The `KvCachePoolManager` sits between the scheduler and the storage tiers. It coordinates lookups, prefetches, and promotions/demotions across the tier hierarchy.

### 6.1 Structure

```rust
pub struct KvCachePoolManager {
    /// L1: GPU radix tree (v4's RadixKvManager, unchanged)
    l1_gpu: RadixKvManager,
    /// L2: CPU DRAM
    l2_cpu: Option<Box<dyn KvConnector>>,
    /// L3: Local SSD (optional)
    l3_ssd: Option<Box<dyn KvConnector>>,
    /// L4: Remote shared store (optional)
    l4_remote: Option<Box<dyn KvConnector>>,
    /// Transfer engine for cross-instance data movement
    transfer: Option<Arc<TransferEngine>>,
    /// Background prefetch channel
    prefetch_tx: mpsc::Sender<PrefetchRequest>,
    /// Metrics
    metrics: Arc<KvPoolMetrics>,
}

struct PrefetchRequest {
    hashes: Vec<BlockHash>,
    /// Where the blocks need to end up
    target_tier: Tier,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier { Gpu, Cpu, Ssd, Remote }
```

### 6.2 Tiered Lookup

When the scheduler needs to know how many tokens are cached for a sequence, the pool manager checks tiers in order:

```rust
impl KvCachePoolManager {
    /// Find the longest prefix cached across all tiers.
    /// L1 (GPU) is checked synchronously. Lower tiers are checked only
    /// for the portion not found in L1.
    pub fn find_prefix_all_tiers(&mut self, tokens: &[u32]) -> PrefixLookupResult {
        // L1: fast, synchronous
        let (l1_cached, l1_block_ids) = self.l1_gpu.find_prefix(tokens);

        if l1_cached == tokens.len() / BLOCK_SIZE * BLOCK_SIZE {
            // Fully cached in GPU — best case
            return PrefixLookupResult {
                gpu_cached_tokens: l1_cached,
                gpu_block_ids: l1_block_ids,
                lower_tier_cached_tokens: 0,
                lower_tier_hashes: vec![],
            };
        }

        // Check lower tiers for the remaining blocks
        let remaining_tokens = &tokens[l1_cached..];
        let remaining_hashes: Vec<BlockHash> = remaining_tokens
            .chunks(BLOCK_SIZE)
            .filter(|c| c.len() == BLOCK_SIZE)
            .map(|chunk| compute_block_hash(&tokens[..l1_cached + chunk.len()]))
            .collect();

        let lower_cached = self.check_lower_tiers(&remaining_hashes);

        // Trigger background prefetch: promote lower-tier blocks to GPU
        if lower_cached > 0 {
            let hashes_to_prefetch: Vec<_> = remaining_hashes[..lower_cached].to_vec();
            let _ = self.prefetch_tx.try_send(PrefetchRequest {
                hashes: hashes_to_prefetch,
                target_tier: Tier::Gpu,
            });
        }

        PrefixLookupResult {
            gpu_cached_tokens: l1_cached,
            gpu_block_ids: l1_block_ids,
            lower_tier_cached_tokens: lower_cached * BLOCK_SIZE,
            lower_tier_hashes: remaining_hashes[..lower_cached].to_vec(),
        }
    }

    /// Check L2 → L3 → L4 in order. Returns how many contiguous blocks are found.
    fn check_lower_tiers(&self, hashes: &[BlockHash]) -> usize {
        let tiers: Vec<&dyn KvConnector> = [
            self.l2_cpu.as_deref(),
            self.l3_ssd.as_deref(),
            self.l4_remote.as_deref(),
        ].into_iter().flatten().collect();

        let mut found = vec![false; hashes.len()];

        for tier in &tiers {
            let present = tier.contains(hashes);
            for (i, &p) in present.iter().enumerate() {
                if p { found[i] = true; }
            }
            // Early exit if all found
            if found.iter().all(|&f| f) { break; }
        }

        // Return count of contiguous found blocks from the start
        found.iter().take_while(|&&f| f).count()
    }
}
```

### 6.3 Block Promotion (Lower Tier → GPU)

When a prefetch request arrives, the background worker promotes blocks up the hierarchy:

```rust
impl KvCachePoolManager {
    /// Background worker: promotes blocks from lower tiers to GPU.
    async fn prefetch_worker(
        mut rx: mpsc::Receiver<PrefetchRequest>,
        l1: Arc<Mutex<RadixKvManager>>,
        tiers: Vec<Arc<dyn KvConnector>>,
    ) {
        while let Some(req) = rx.recv().await {
            // Fetch from the lowest tier that has the blocks
            let mut blocks = vec![None; req.hashes.len()];

            for tier in &tiers {
                let fetched = tier.fetch(&req.hashes);
                for (i, block) in fetched.into_iter().enumerate() {
                    if blocks[i].is_none() {
                        blocks[i] = block;
                    }
                }
                if blocks.iter().all(|b| b.is_some()) { break; }
            }

            // Upload to GPU
            let mut l1 = l1.lock();
            for block in blocks.into_iter().flatten() {
                // Allocate a GPU block and copy data
                if let Some(gpu_block_id) = l1.try_allocate_one() {
                    l1.upload_block(gpu_block_id, &block.data);
                    l1.insert_by_hash(block.meta.hash, gpu_block_id);
                }
                // If GPU is full, the block stays in the lower tier
                // and will be fetched on-demand during prefill
            }
        }
    }
}
```

### 6.4 Block Demotion (GPU → Lower Tier)

On GPU eviction, blocks cascade down:

```rust
impl KvCachePoolManager {
    /// Called by the scheduler when GPU blocks must be freed.
    /// Demotes evicted blocks to the next available tier instead of discarding.
    pub fn evict_gpu_blocks(&mut self, target: usize) -> usize {
        if let Some(ref l2) = self.l2_cpu {
            self.l1_gpu.evict_to_tier(target, l2.as_ref())
        } else {
            // No lower tier — just evict (v4 behavior)
            self.l1_gpu.evict(target)
        }
    }
}
```

### 6.5 Publishing Blocks to Shared Store

After a prefill computes new KV blocks, they are published to the shared store so other instances can reuse them:

```rust
impl KvCachePoolManager {
    /// Called after prefill: publish newly computed blocks to the shared store.
    pub fn publish_new_blocks(&self, tokens: &[u32], block_ids: &[u32]) {
        let l4 = match &self.l4_remote {
            Some(l4) => l4,
            None => return,  // no shared store configured
        };

        // Read GPU blocks and publish
        let mut blocks = Vec::with_capacity(block_ids.len());
        for (i, &bid) in block_ids.iter().enumerate() {
            let prefix_end = (i + 1) * BLOCK_SIZE;
            let hash = compute_block_hash(&tokens[..prefix_end]);
            let data = self.l1_gpu.gpu_pool.read_block(bid);
            blocks.push(KvBlockData {
                meta: KvBlockMeta {
                    hash,
                    num_layers: self.l1_gpu.layers_per_rank(),
                    num_kv_heads: self.l1_gpu.kv_heads_per_rank(),
                    head_dim: self.l1_gpu.head_dim(),
                    block_size: BLOCK_SIZE,
                    dtype: self.l1_gpu.kv_dtype(),
                },
                data,
            });
        }

        // Async publish — don't block the scheduler
        let l4 = Arc::clone(l4);
        tokio::spawn(async move {
            l4.store(&blocks);
        });
    }
}
```

---

## 7. Cache-Aware Routing

### 7.1 The Problem

With N instances, a naive load balancer sends each request to any instance. If instance 0 has the prefix cached but the request goes to instance 2, the cache is wasted.

### 7.2 Design: Block Hash Index

A lightweight router maintains a global index: which instance has which block hashes cached in L1 (GPU).

```rust
pub struct CacheAwareRouter {
    /// Global index: block_hash → set of instance IDs that have it in GPU
    index: RwLock<HashMap<BlockHash, SmallVec<[u32; 4]>>>,
    /// Instance health + load info
    instances: RwLock<Vec<InstanceInfo>>,
}

struct InstanceInfo {
    id: u32,
    addr: SocketAddr,
    running_requests: u32,
    gpu_cache_usage: f32,
    healthy: bool,
}
```

### 7.3 Routing Algorithm

```rust
impl CacheAwareRouter {
    /// Route a request to the best instance.
    pub fn route(&self, prompt_tokens: &[u32]) -> u32 {
        let index = self.index.read();
        let instances = self.instances.read();

        // Score each instance by prefix match length
        let mut scores: Vec<(u32, usize)> = instances.iter()
            .filter(|inst| inst.healthy)
            .map(|inst| (inst.id, 0usize))
            .collect();

        // Walk prefix blocks, count matches per instance
        let mut prefix_so_far = Vec::new();
        for chunk in prompt_tokens.chunks(BLOCK_SIZE) {
            if chunk.len() < BLOCK_SIZE { break; }
            prefix_so_far.extend_from_slice(chunk);
            let hash = compute_block_hash(&prefix_so_far);

            if let Some(holders) = index.get(&hash) {
                for &inst_id in holders.iter() {
                    if let Some(score) = scores.iter_mut().find(|(id, _)| *id == inst_id) {
                        score.1 += 1;
                    }
                }
            }
        }

        // Pick instance with highest prefix match.
        // Tiebreak: least running requests (load balancing).
        scores.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                let load_a = instances.iter().find(|i| i.id == a.0).map(|i| i.running_requests).unwrap_or(u32::MAX);
                let load_b = instances.iter().find(|i| i.id == b.0).map(|i| i.running_requests).unwrap_or(u32::MAX);
                load_a.cmp(&load_b)
            })
        });

        scores[0].0
    }
}
```

### 7.4 Index Updates

Each instance publishes cache events to the router. This runs as a lightweight background task:

```rust
/// Events published by each instance to the router.
#[derive(Serialize, Deserialize)]
pub enum CacheEvent {
    /// Block was added to GPU cache (after prefill or promotion).
    BlockStored { hash: BlockHash, instance_id: u32 },
    /// Block was evicted from GPU cache.
    BlockRemoved { hash: BlockHash, instance_id: u32 },
}

impl CacheAwareRouter {
    /// Process a batch of events from instances.
    pub fn apply_events(&self, events: &[CacheEvent]) {
        let mut index = self.index.write();
        for event in events {
            match event {
                CacheEvent::BlockStored { hash, instance_id } => {
                    index.entry(*hash)
                        .or_insert_with(SmallVec::new)
                        .push(*instance_id);
                }
                CacheEvent::BlockRemoved { hash, instance_id } => {
                    if let Some(holders) = index.get_mut(hash) {
                        holders.retain(|&id| id != *instance_id);
                        if holders.is_empty() {
                            index.remove(hash);
                        }
                    }
                }
            }
        }
    }
}
```

### 7.5 Deployment: Embedded vs. Standalone

- **Single-instance:** No router needed. L1 + L2 only.
- **Multi-instance, single-node:** Router embedded in the HTTP gateway process. Events via shared memory or Unix socket.
- **Multi-instance, multi-node:** Standalone router process. Events via TCP/ZMQ. This is the pattern llm-d uses.

---

## 8. Integration with Existing Components

### 8.1 Scheduler Changes

The scheduler's `schedule()` function in v4 §8.2 uses `self.kv_manager.get_token_layout(seq)` to determine how many tokens are cached. In v5, this call goes through the pool manager instead:

```rust
// v4:
let layout = self.kv_manager.get_token_layout(seq);

// v5:
let lookup = self.pool_manager.find_prefix_all_tiers(seq.all_tokens());
let layout = TokenLayout {
    computed_tokens: lookup.gpu_cached_tokens + lookup.lower_tier_cached_tokens,
    total_tokens: seq.prompt_len(),
    remaining: seq.prompt_len() - lookup.gpu_cached_tokens - lookup.lower_tier_cached_tokens,
};
```

The scheduler still only allocates GPU blocks for the remaining (uncached) tokens. Blocks found in lower tiers are promoted to GPU asynchronously before the prefill step executes.

### 8.2 Prefill Integration

When a prefill step is about to execute, blocks that were found in lower tiers need to be in GPU memory. The executor waits for pending promotions:

```rust
impl Executor {
    async fn forward_prefill(&self, input: &ForwardInput) -> ForwardOutput {
        // Wait for any pending prefetch (promotion from lower tiers)
        for seq in &input.prefill_seqs {
            if seq.has_pending_promotions() {
                seq.wait_promotions_complete().await;
            }
        }

        // Proceed with normal prefill — only computes the cache-miss portion
        self.forward_prefill_inner(input).await
    }
}
```

### 8.3 Disaggregated Serving Integration

v4's disaggregated serving (§14) transfers KV directly from prefill node to decode node. v5 adds an alternative path via the shared store:

```
v4 path: Prefill → (IPC/RDMA) → Decode     [point-to-point]
v5 path: Prefill → Shared Store → Decode    [decoupled]
```

The v5 path is preferred when:
- Prefill and decode nodes are on different machines (RDMA not available or high latency)
- Multiple decode nodes may serve the same user's follow-up requests
- The shared store already has the blocks from a previous request

```rust
impl DisaggPrefillEngine {
    async fn after_prefill(&self, seq: &Sequence) {
        // v4: direct transfer
        if self.direct_transfer_available() {
            self.transfer_kv_to_decode(seq).await;
        }

        // v5: also publish to shared store for future reuse
        self.pool_manager.publish_new_blocks(seq.all_tokens(), seq.block_ids());
    }
}

impl DisaggDecodeEngine {
    async fn before_decode(&self, seq: &Sequence) {
        // Try local GPU first (v4 behavior)
        if self.has_kv_locally(seq) { return; }

        // v5: fetch from shared store
        let lookup = self.pool_manager.find_prefix_all_tiers(seq.all_tokens());
        if lookup.lower_tier_cached_tokens > 0 {
            // Promote to GPU and wait
            self.pool_manager.promote_and_wait(&lookup.lower_tier_hashes).await;
        }
    }
}
```

### 8.4 Process Results Integration

After each forward step, newly computed blocks are published to the shared store:

```rust
// v4 §8: process_results
async fn process_results(&mut self, output: &ForwardOutput) {
    for (seq_id, token_id) in &output.sampled_tokens {
        let seq = self.find_running_mut(*seq_id);
        seq.append_token(*token_id);

        // v4: insert into local radix tree
        self.pool_manager.l1_gpu.insert_prefix(seq.all_tokens(), seq.block_ids());

        // v5: publish to shared store (async, non-blocking)
        self.pool_manager.publish_new_blocks(seq.all_tokens(), seq.block_ids());

        // ... rest unchanged (finish check, metrics, etc.)
    }
}
```

### 8.5 Block Hash Computation

The hash used across all tiers must be identical to v4's radix tree hash:

```rust
/// Compute the block hash for a prefix ending at the given tokens.
/// This is the same hash used by the radix tree for prefix matching.
pub fn compute_block_hash(prefix_tokens: &[u32]) -> BlockHash {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    // Hash the raw token bytes
    let bytes: &[u8] = bytemuck::cast_slice(prefix_tokens);
    hasher.update(bytes);
    hasher.finalize().into()
}
```

---

## 9. Configuration

New configuration fields added to the existing v4 config:

```yaml
kv_cache_pool:
  # Enable multi-tier caching (default: false = v4 behavior)
  enabled: false

  # L2: CPU DRAM tier
  cpu_dram:
    enabled: true
    capacity_gb: 32            # pinned host memory for KV blocks
    # 0 = auto (use remaining system RAM after model weights)

  # L3: Local SSD tier (optional)
  local_ssd:
    enabled: false
    path: "/tmp/kv_cache"
    capacity_gb: 200

  # L4: Remote shared store (optional, for multi-instance)
  remote_store:
    enabled: false
    addr: "kv-store:50051"     # address of the shared store service

  # Transfer engine
  transfer:
    listen_addr: "0.0.0.0:50052"
    enable_rdma: true          # auto-detect RDMA devices

  # Cache-aware routing (only for the router process)
  routing:
    enabled: false
    listen_addr: "0.0.0.0:8090"  # router HTTP endpoint
    event_transport: "tcp"        # "tcp" or "shm"

  # Background prefetch
  prefetch:
    enabled: true
    max_inflight: 16           # max concurrent prefetch operations
    promote_on_miss: true      # auto-promote lower-tier blocks to GPU

  # Demotion policy: what to do when GPU blocks are evicted
  eviction_policy:
    demote_to_cpu: true        # push evicted GPU blocks to L2
    publish_to_remote: false   # also publish to L4 on eviction
```

### Minimal Config: Single Instance (v4-compatible)

```yaml
kv_cache_pool:
  enabled: false
# Everything else from v4 config.yaml
```

This disables all v5 features. The server behaves exactly like v4.

### Single Instance with CPU Offload

```yaml
kv_cache_pool:
  enabled: true
  cpu_dram:
    enabled: true
    capacity_gb: 64
  eviction_policy:
    demote_to_cpu: true
```

GPU evictions push blocks to CPU DRAM. Next time the same prefix appears, blocks are promoted back to GPU instead of being recomputed.

### Multi-Instance with Shared Store

```yaml
kv_cache_pool:
  enabled: true
  cpu_dram:
    enabled: true
    capacity_gb: 32
  remote_store:
    enabled: true
    addr: "kv-store.svc:50051"
  transfer:
    listen_addr: "0.0.0.0:50052"
    enable_rdma: true
  eviction_policy:
    demote_to_cpu: true
    publish_to_remote: true
```

---

## 10. Updated Directory Layout

New files added to v4's directory structure:

```
llm-server/src/
├── kv_cache/
│   ├── radix_tree.rs         ← (v4, unchanged)
│   ├── block_pool.rs         ← (v4, unchanged)
│   ├── swap.rs               ← (v4, unchanged)
│   ├── pool_manager.rs       ← NEW: KvCachePoolManager (tiered orchestration)
│   ├── connector.rs          ← NEW: KvConnector trait + BlockHash + KvBlockData
│   ├── cpu_dram.rs            ← NEW: CpuDramConnector
│   ├── ssd.rs                 ← NEW: SsdConnector (optional)
│   ├── remote_store.rs        ← NEW: RemoteStoreConnector
│   └── hash.rs                ← NEW: compute_block_hash
├── transfer/
│   ├── engine.rs              ← NEW: TransferEngine
│   ├── rdma.rs                ← NEW: RDMA transport
│   ├── tcp.rs                 ← NEW: TCP transport
│   ├── ipc.rs                 ← NEW: Device IPC transport
│   └── coalesce.rs            ← NEW: Block coalescing for batch transfer
├── routing/
│   ├── router.rs              ← NEW: CacheAwareRouter
│   └── events.rs              ← NEW: CacheEvent + event publishing
```

### New crate dependency

| Crate | Purpose |
|---|---|
| `sha2` | Block hash computation |
| `lru` | LRU eviction for CPU/SSD tiers |
| `bytemuck` | Zero-copy casting for token bytes |
| `io-uring` (optional) | Async SSD I/O on Linux |

---

## 11. Implementation Phase: KVCache Pool (Phase 10)

This phase follows v4's Phase 9 (disaggregated serving) and builds on its foundation.

### Phase 10a — CPU DRAM Tier (1 week)

#### Deliverables

- [ ] `KvConnector` trait defined
- [ ] `compute_block_hash` matching radix tree hash
- [ ] `CpuDramConnector` with LRU eviction
- [ ] `KvCachePoolManager` with L1 (GPU) + L2 (CPU) only
- [ ] `RadixKvManager::evict_to_tier` — GPU → CPU demotion
- [ ] Prefetch worker: CPU → GPU promotion

#### Test Plan

**T10a.1 — CPU demotion round-trip:**
```rust
#[test]
fn test_gpu_to_cpu_round_trip() {
    let mut pool = KvCachePoolManager::new_test(gpu_blocks: 10, cpu_blocks: 100);
    let tokens = vec![1u32; 16];  // 1 block

    // Insert into GPU
    pool.l1_gpu.insert_prefix(&tokens, &[pool.l1_gpu.allocate_one()]);
    assert_eq!(pool.l1_gpu.find_prefix(&tokens).0, 16);

    // Force GPU eviction → should land in CPU
    pool.evict_gpu_blocks(1);
    assert_eq!(pool.l1_gpu.find_prefix(&tokens).0, 0);  // gone from GPU

    // Check CPU has it
    let hash = compute_block_hash(&tokens);
    assert!(pool.l2_cpu.as_ref().unwrap().contains(&[hash])[0]);

    // Promote back to GPU
    pool.promote_blocks(&[hash]);
    assert_eq!(pool.l1_gpu.find_prefix(&tokens).0, 16);  // back in GPU
}
```

**T10a.2 — Multi-turn agent workload with CPU offload:**
```bash
python tools/agent_workload_test.py \
  --server http://localhost:8080 \
  --num-turns 20 \
  --system-prompt-tokens 4096 \
  --tool-history-tokens 2048

# Compare: kv_cache_pool.enabled=false vs true
# With CPU offload, TTFT on later turns should be lower
# because evicted prefix blocks are fetched from CPU instead of recomputed.

# Pass criteria:
#   - p50 TTFT with CPU offload < p50 TTFT without (for turns > 5)
#   - cpu_dram tier hit rate > 0 (from metrics)
#   - No correctness errors (output matches reference)
```

**T10a.3 — CPU tier capacity enforcement:**
```rust
#[test]
fn test_cpu_tier_lru_eviction() {
    let cpu = CpuDramConnector::new(2);  // only 2 blocks
    let b1 = make_test_block([1u8; 32]);
    let b2 = make_test_block([2u8; 32]);
    let b3 = make_test_block([3u8; 32]);

    cpu.store(&[b1.clone(), b2.clone()]);
    assert_eq!(cpu.usage(), 2);

    // Storing b3 should evict b1 (oldest)
    cpu.store(&[b3.clone()]);
    assert_eq!(cpu.usage(), 2);
    assert!(!cpu.contains(&[b1.meta.hash])[0]);  // evicted
    assert!(cpu.contains(&[b2.meta.hash])[0]);    // still there
    assert!(cpu.contains(&[b3.meta.hash])[0]);    // new
}
```

**Pass criteria:** GPU→CPU→GPU round-trip preserves block data. CPU LRU eviction works correctly. Agent workload shows measurable TTFT improvement on later turns.

---

### Phase 10b — Shared Store + Cache-Aware Routing (2 weeks)

#### Deliverables

- [ ] `RemoteStoreConnector` with TCP transport
- [ ] `TransferEngine` with TCP transport (RDMA as stretch goal)
- [ ] Block coalescing for batch transfers
- [ ] `CacheAwareRouter` with prefix hash scoring
- [ ] Cache event publishing (BlockStored / BlockRemoved)
- [ ] Multi-instance integration test

#### Test Plan

**T10b.1 — Cross-instance cache sharing:**
```bash
# Start 2 instances + shared store + router
./kv-store --listen 0.0.0.0:50051 &
./llm-server --instance-id 0 --kv-store kv-store:50051 &
./llm-server --instance-id 1 --kv-store kv-store:50051 &
./cache-router --instances localhost:8080,localhost:8081 --listen 0.0.0.0:8090 &

# Send request to instance 0 (through router)
curl -X POST http://localhost:8090/v1/chat/completions \
  -d '{"messages": [{"role": "system", "content": "You are helpful..."}, {"role": "user", "content": "Hello"}], "max_tokens": 32}'

# Send same-prefix request again (router should send to instance 0)
curl -X POST http://localhost:8090/v1/chat/completions \
  -d '{"messages": [{"role": "system", "content": "You are helpful..."}, {"role": "user", "content": "World"}], "max_tokens": 32}'

# Verify:
#   - Second request routed to instance 0 (check router logs)
#   - prefix_cache_hit_rate on instance 0 > 0
#   - Both responses are correct
```

**T10b.2 — Router load balancing under uniform cache:**
```bash
# Send 100 requests with different prefixes (no cache hits possible)
python tools/router_load_test.py \
  --router http://localhost:8090 \
  --num-requests 100 \
  --unique-prefixes

# Verify: requests distributed roughly evenly across instances
# (within 60/40 split or better)
```

**T10b.3 — Shared store fetch latency:**
```bash
# Measure: time to fetch a 16-token KV block from the shared store
python tools/transfer_benchmark.py \
  --store-addr kv-store:50051 \
  --block-count 100 \
  --transport tcp

# Pass criteria:
#   - p50 fetch latency < 1ms (same node, TCP)
#   - p50 fetch latency < 5ms (cross node, TCP)
#   - p50 fetch latency < 0.5ms (same node, RDMA, if available)
```

**T10b.4 — Block coalescing efficiency:**
```rust
#[test]
fn test_block_coalescing() {
    let blocks: Vec<KvBlockData> = (0..64).map(|i| make_test_block_with_size(i, 32768)).collect();
    // 64 blocks × 32KB = 2MB total

    let coalesced = coalesce_blocks(&blocks);
    // Should produce a single buffer (below 4MB threshold)
    assert_eq!(coalesced.len(), 1);
    // Verify round-trip: decoalesce and compare
    let recovered = decoalesce_blocks(&coalesced[0]);
    assert_eq!(recovered.len(), 64);
    for (orig, recv) in blocks.iter().zip(recovered.iter()) {
        assert_eq!(orig.meta.hash, recv.meta.hash);
        assert_eq!(orig.data, recv.data);
    }
}
```

**Pass criteria:** Cross-instance cache sharing works. Router steers prefix-matched requests correctly. Transfer latency meets targets.

---

### Phase 10c — SSD Tier + RDMA Transport (1 week, optional)

#### Deliverables

- [ ] `SsdConnector` with `io_uring`
- [ ] RDMA transport in `TransferEngine`
- [ ] SSD compaction (background defragmentation)

#### Test Plan

**T10c.1 — SSD round-trip:**
```rust
#[test]
fn test_ssd_store_fetch() {
    let ssd = SsdConnector::new("/tmp/test_kv_cache", 1000);
    let block = make_test_block([42u8; 32]);
    ssd.store(&[block.clone()]);

    let fetched = ssd.fetch(&[block.meta.hash]);
    assert!(fetched[0].is_some());
    assert_eq!(fetched[0].as_ref().unwrap().data, block.data);
}
```

**T10c.2 — RDMA transfer throughput:**
```bash
python tools/transfer_benchmark.py \
  --store-addr remote-host:50051 \
  --block-count 1000 \
  --transport rdma

# Pass criteria:
#   - Sustained throughput > 20 GB/s (InfiniBand HDR)
#   - p99 latency < 100μs per block
```

**Pass criteria:** SSD tier works as L3 fallback. RDMA achieves near-line-rate throughput.

---

## 12. Performance Targets (Updated)

### Single Instance: Agent Workload with CPU Offload

| Metric | Without Pool | With CPU Pool | How Measured |
|---|---|---|---|
| TTFT (turn 10, shared prefix) | ~80ms (full recompute) | < 20ms (cache hit + GPU promote) | Agent workload test, p50 |
| GPU KV utilization | 100% (evict = lose) | 100% GPU + 80% CPU overflow | Prometheus metrics |
| Prefix cache effective hit rate | 60% (GPU only) | > 90% (GPU + CPU) | Prometheus metric |

### Multi-Instance: 4× Instances with Cache-Aware Routing

| Metric | Round-Robin | Cache-Aware | How Measured |
|---|---|---|---|
| Prefix cache hit rate (cluster) | ~25% (random scatter) | > 80% (affinity) | Router + instance metrics |
| p50 TTFT (repeated prefix) | ~80ms | < 20ms | Load test through router |
| Request distribution balance | Perfect 25/25/25/25 | Skewed by cache (~40/30/20/10) | Router logs |

### Disaggregated Serving: KV Transfer via Shared Store

| Metric | v4 (Point-to-Point) | v5 (Shared Store) | How Measured |
|---|---|---|---|
| Decode cold-start (no local KV) | KV transfer from prefill node | Fetch from store (any instance's blocks) | Disagg benchmark |
| Multi-turn reuse | Recompute if different decode node | Hit store from any previous turn | Agent workload test |

---

## Key Design Decisions — Rationale

**`KvConnector` as a sync trait, not async:** Lower tiers (CPU, SSD, remote) are accessed from a background prefetch thread, not the scheduler hot path. The scheduler only queries L1 (GPU) synchronously. Making the trait sync simplifies implementations and avoids async infection in the storage layer.

**Hash-based block addressing across all tiers:** Using the same SHA-256 prefix hash as the radix tree means no translation between local cache keys and global store keys. A block's identity is its content hash, universally.

**Coalesced transfers:** Mooncake's key insight: naïve per-block transfers produce thousands of tiny messages that saturate message-processing overhead, not bandwidth. Coalescing into large buffers amortizes this, achieving near-line-rate on both RDMA and TCP.

**Router as a scoring function, not a hard assignment:** The router scores instances by prefix match length and tiebreaks by load. This gracefully degrades: even with zero cache hits (cold start), it falls back to least-loaded balancing.

**CPU DRAM as the default L2:** CPU DRAM is always available (unlike RDMA or SSD), low-latency (~2μs for a memcpy of a 32KB block), and high-capacity (256–512 GB on typical servers). It provides the biggest win-per-complexity of any tier.

**Optional SSD and remote tiers:** Not every deployment needs SSD or cluster-wide sharing. The pool manager accepts `None` for tiers it doesn't need. Single-instance users get the CPU offload benefit without any network complexity.

---

*v5 adds a minimal but complete KVCache pool management layer on top of v4. The single-instance path (GPU + CPU offload) requires no infrastructure changes. The multi-instance path (shared store + routing) adds two new processes (store service + router) but reuses all existing per-instance logic unchanged.*
