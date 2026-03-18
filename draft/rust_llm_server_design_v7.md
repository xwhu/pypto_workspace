# Rust LLM Inference Server — Design & Implementation Plan (v7)

> **Target:** 1–N machines with CXL-interconnected memory pools, 8–16 GPU/NPU cards per machine, hardware-limit inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI. Exploit CXL/PGAS to eliminate data copies. **Never trust shared memory for control structures.**
> **New in v7:** Corruption-resilient CXL architecture. Separates control plane (local per-host) from data plane (CXL shared). Block-level checksums, centralized allocator with failover, background scrubbing, gossip-based index replication. Designed to survive bit-level CXL memory corruption without cluster-wide failure.

---

## Table of Contents

1. [Critique of v6 and What v7 Fixes](#1-critique-of-v6-and-what-v7-fixes)
2. [CXL Reliability Model](#2-cxl-reliability-model)
3. [Architecture: Control Plane / Data Plane Split](#3-architecture-control-plane--data-plane-split)
4. [Data Plane: Checksummed Blocks in CXL Shared Memory](#4-data-plane-checksummed-blocks-in-cxl-shared-memory)
5. [Control Plane: Local Index with Gossip Replication](#5-control-plane-local-index-with-gossip-replication)
6. [Control Plane: Centralized Allocator with Failover](#6-control-plane-centralized-allocator-with-failover)
7. [Corruption Detection and Recovery](#7-corruption-detection-and-recovery)
8. [Background Scrubber](#8-background-scrubber)
9. [Heartbeat and Liveness (Side-Channel)](#9-heartbeat-and-liveness-side-channel)
10. [Revised PGAS Model: Safe Wrappers](#10-revised-pgas-model-safe-wrappers)
11. [Revised CXL KV Connector](#11-revised-cxl-kv-connector)
12. [Weight Pool Integrity](#12-weight-pool-integrity)
13. [Integration with v6 Components](#13-integration-with-v6-components)
14. [Configuration](#14-configuration)
15. [Updated Directory Layout](#15-updated-directory-layout)
16. [Implementation Phases](#16-implementation-phases)
17. [Performance Targets](#17-performance-targets)
18. [Mapping to Linqu Hierarchy](#18-mapping-to-linqu-hierarchy)

---

## 1. Critique of v6 and What v7 Fixes

### Critique

v6 introduces CXL/PGAS for zero-copy KV cache sharing — a major performance win. But it puts **critical control structures** (the block index, free list, refcounts) directly in CXL shared memory as lock-free atomic data structures. This is dangerous because CXL shared memory is less reliable than local DRAM:

| # | v6 Problem | Why It's Dangerous |
|---|---|---|
| 1 | **Lock-free free list (`AtomicFreeList`) in shared memory.** The Treiber stack head and next-pointers are `AtomicU64` values in the CXL region. | A single bit flip in the head pointer corrupts the free list for ALL hosts. Double-allocation → two hosts write to the same block → data corruption or segfault. A flipped next-pointer creates a cycle → `pop()` infinite-loops → the allocator hangs cluster-wide. |
| 2 | **Hash index (`AtomicBlockIndexEntry`) in shared memory.** The open-addressing hash table with CAS-based insert/lookup lives in the CXL region. | A flipped bit in an `offset` field → `block_ptr()` returns a garbage pointer → segfault or silent read of wrong data. A flipped `hash` field → false positive or false negative lookups. A flipped `refcount` → memory leak or use-after-free. Every host trusts this single shared index. |
| 3 | **Heartbeat in shared memory.** v6 stores heartbeat timestamps as `AtomicU64` in CXL memory. | A bit flip in a heartbeat slot → false liveness detection (dead host appears alive) or false failure detection (live host appears dead → unnecessary failover → cluster instability). |
| 4 | **No data integrity verification.** v6 reads KV block data from CXL via raw pointer dereference. No checksum. | A bit flip in KV data → the model silently produces wrong output. There is no way to detect this. This is the worst kind of failure: silent corruption of inference results. |
| 5 | **No recovery model for bit-level corruption.** v6's §9 only handles host-level failures (crash, switch down). It has no answer for "what if a single cache line in CXL memory has a flipped bit." | Host failures are fail-stop and detectable. Bit-level corruption is **silent** — the system continues operating with bad data. v6 has no defense against this. |

### The Failure Surface

Unlike local DRAM where ECC protects data end-to-end from DIMM to CPU cache, CXL data traverses a longer path:

```
Local DRAM:     DIMM (ECC) ──→ Memory Controller ──→ CPU L3 Cache
                1 hop, ECC covers it

CXL memory:     DIMM (ECC) ──→ CXL Device Controller ──→ CXL Link (PCIe PHY)
                ──→ CXL Retimer(s) ──→ CXL Switch ──→ CXL Link (PCIe PHY)
                ──→ Host CXL Controller ──→ Memory Controller ──→ CPU L3 Cache
                6+ hops, each with its own failure mode
```

CXL 3.0 has link-level CRC (32-bit LCRC per 256-byte flit) and optional IDE (Integrity and Data Encryption). But:
- CRC **detects**, it doesn't **correct**. Detected errors cause link retries or poison. Poisoned cache lines trigger MCE (machine check exception) → process death.
- A 32-bit CRC has ~2⁻³² undetected error probability per flit. At billions of flits/second across a busy fabric, undetected errors are rare but non-zero.
- CXL device controller firmware bugs can cause silent corruption that no link-level CRC catches.
- Standard DRAM ECC (SECDED) corrects 1-bit, detects 2-bit. A 3-bit error in the same ECC word passes undetected.

### What v7 Adds

v7 retains all of v6's performance architecture (CXL shared memory for KV blocks, weight sharing, distance-aware routing) but makes it **corruption-resilient**:

- **§3**: Architectural split: control plane (index, allocator, refcounts, heartbeat) in LOCAL per-host memory; data plane (KV blocks, weights) in CXL shared memory
- **§4**: Every block in CXL memory gets a checksum header; every read verifies it
- **§5**: Block index replicated across hosts via gossip, not shared via CXL atomics
- **§6**: Centralized block allocator (bitmap in local memory) with standby failover, replacing lock-free atomics in shared memory
- **§7**: Corruption detection and recovery: checksum failure → quarantine + recompute
- **§8**: Background scrubber periodically verifies all CXL blocks
- **§9**: Heartbeat over UDP side-channel, not in CXL memory
- **§10**: Safe PGAS wrappers that always verify before trusting

### What v7 Does NOT Change

- CXL topology discovery (v6 §2) — unchanged
- CXL-distance-aware routing (v6 §7) — unchanged, uses local index instead of shared index
- PD disaggregation over CXL (v6 §8) — unchanged, adds checksum on write/verify on read
- Shared weight pool (v6 §6) — unchanged, adds periodic integrity verification
- All v4/v5 components — unchanged
- Graceful degradation to v5 on non-CXL hardware — unchanged

### Key Principle

**KV cache is a cache, not a database.** Every block is deterministically recomputable from its prompt tokens. Corruption is a **performance event** (cache miss → recompute), never a **correctness event** (wrong model output) — provided we detect it.

---

## 2. CXL Reliability Model

### 2.1 Error Categories

| Category | Mechanism | Detection | Probability | v6 Handling | v7 Handling |
|---|---|---|---|---|---|
| **DRAM single-bit** | Particle strike, voltage drift | ECC corrects silently | ~1 per GB per year (DDR5) | Hidden by ECC | Hidden by ECC |
| **DRAM multi-bit (2)** | Clustered strike, row hammer | ECC detects, uncorrectable → **poison** | ~1 per TB per year | MCE kills process | MCE kills process (same) |
| **DRAM multi-bit (3+)** | Rare | ECC misses → **silent corruption** | ~1 per PB per year | **Undetected** | **Block checksum catches it** |
| **CXL link bit flip** | PHY noise, retimer glitch | LCRC detects → retry | ~10⁻¹² BER after retry | Retry (transparent) | Retry (transparent) |
| **CXL link undetected** | Multi-bit pattern evading 32-bit CRC | Undetected by CRC | ~2⁻³² per flit (~10⁻¹⁰ effective) | **Undetected** | **Block checksum catches it** |
| **CXL controller bug** | Firmware defect | None (silent) | Depends on maturity | **Undetected** | **Block checksum catches it** |
| **CXL switch corruption** | SRAM upset in switch fabric | Switch internal ECC | Rare but impactful | Not handled | **Block checksum catches it** |

### 2.2 Blast Radius Analysis

| What Gets Corrupted | v6 Blast Radius | v7 Blast Radius |
|---|---|---|
| KV block data (32 KB payload) | **Silent wrong output** (no detection) | 1 block → checksum fail → recompute (~3ms) |
| Free list head (8 bytes) | **Cluster-wide allocator corruption** | Impossible (allocator in local memory) |
| Index entry offset (8 bytes) | **Cluster-wide: all hosts read wrong pointer** | 1 host (local index); gossip corrects within 10ms |
| Index entry refcount (4 bytes) | **Cluster-wide leak or use-after-free** | 1 host (local refcount); no global impact |
| Heartbeat slot (8 bytes) | **False failover → cluster instability** | Impossible (heartbeat over UDP) |
| Weight data (read-only) | **Silent wrong output until restart** | Periodic checksum verification → reload |

---

## 3. Architecture: Control Plane / Data Plane Split

### 3.1 The Principle

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE — LOCAL memory, per-host, replicated via gossip       │
│                                                                      │
│  • Block index: hash → GlobalBlockAddr         (HashMap, local DRAM) │
│  • Block allocator: bitmap of free/used blocks  (Vec<u64>, local)    │
│  • Refcount tracking: per-block local refcounts (HashMap, local)     │
│  • Heartbeat: UDP packets between hosts         (network, not memory)│
│                                                                      │
│  Corruption impact: kills ONE host. Others unaffected.               │
│  Recovery: restart host, rebuild index from CXL block headers.       │
├──────────────────────────────────────────────────────────────────────┤
│  DATA PLANE — CXL shared memory, globally accessible                 │
│                                                                      │
│  • KV block data: [CxlBlockHeader][payload]     (checksummed)        │
│  • Model weights: [WeightHeader][tensor data]   (checksummed)        │
│                                                                      │
│  Corruption impact: ONE block has bad data → detected by checksum.   │
│  Recovery: quarantine block, recompute (KV) or reload (weights).     │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 What Moved Out of CXL Shared Memory (vs. v6)

| Structure | v6 Location | v7 Location | Why |
|---|---|---|---|
| `AtomicFreeList` (head + next ptrs) | CXL shared memory | Local memory on allocator host | Bit flip in head → cluster-wide hang. Local bitmap is protected by local ECC and isolated to one host. |
| `AtomicBlockIndexEntry` (hash table) | CXL shared memory | Local memory on each host (gossip-replicated) | Bit flip in shared index → all hosts affected. Local indexes fail independently. |
| Refcounts (`AtomicU32`) | CXL shared memory (per-entry) | Local memory per host | Bit flip in shared refcount → use-after-free across all hosts. Local refcounts only affect one host. |
| Heartbeat timestamps | CXL shared memory (`AtomicU64` slots) | UDP side-channel | Bit flip → false failure detection → spurious failover. |

### 3.3 What Stays in CXL Shared Memory

| Structure | Why It Stays | How It's Protected |
|---|---|---|
| KV block payload (bulk tensor data) | This IS the data we want to share. Moving it out defeats the purpose of CXL. | Per-block checksum header (§4). Corruption detected on every read. |
| `CxlBlockHeader` (written once per block) | Must be co-located with data for self-describing blocks. Needed for allocator rebuild (§6.4). | Magic number + internal consistency checks. Written once, never mutated. |
| Model weight tensors | Sharing weights saves N×140 GB. | Per-tensor checksum + periodic verification (§12). |

### 3.4 Trade-Off: Latency Cost

| Operation | v6 (Shared Index) | v7 (Local Index + Gossip) | Delta |
|---|---|---|---|
| Index lookup (same host) | ~300ns (CXL load) | ~80ns (local HashMap) | **Faster** |
| Index lookup (cross host, routing) | ~300ns (CXL load) | ~2–5μs (RPC to owner, or local replica if gossip is fresh) | **Slower for cold queries** |
| Index consistency lag | 0 (single copy) | ~1–10ms (gossip propagation) | **Trade-off** |
| Allocation throughput | Lock-free, any host, ~10M ops/s | Centralized, ~100K allocs/s via RPC | **Lower peak, but sufficient** (KV blocks allocated at ~1K/s per host peak) |

The cross-host index lookup is slower, but this only affects the **routing decision** (once per request). Once routed, the actual KV data read from CXL is still at full CXL speed. The 2–5μs routing overhead is negligible compared to the ~10ms+ prefill latency of a typical request.

---

## 4. Data Plane: Checksummed Blocks in CXL Shared Memory

### 4.1 Block Layout

Every KV block in CXL shared memory is self-describing and checksummed:

```
┌─────────────────────────── CXL Shared Memory Block ──────────────────────┐
│ Offset 0:   CxlBlockHeader (64 bytes, cache-line aligned)                │
│   ├── magic: u32          = 0x4B56_4358 ("KVCX")                        │
│   ├── hash: BlockHash     = SHA-256 of token prefix                      │
│   ├── data_checksum: u64  = xxHash64 of payload                          │
│   ├── data_size: u32      = payload size in bytes                        │
│   ├── writer_host: u32    = host ID that wrote this block                │
│   ├── write_epoch: u64    = monotonic epoch (for staleness detection)    │
│   └── _pad: [u8; 4]       (alignment)                                   │
│                                                                          │
│ Offset 64:  KV payload (block_size_bytes, e.g., 32768 bytes)             │
│   [2 × num_layers × block_size × num_kv_heads × head_dim × dtype_size]  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Block Header

```rust
/// Header prepended to every KV block in CXL shared memory.
/// Written once by the producing host. Verified on every read by any host.
#[repr(C, align(64))]
pub struct CxlBlockHeader {
    /// Magic number: detects uninitialized memory or gross corruption.
    pub magic: u32,
    /// Writer host ID (for debugging, GC, and allocator rebuild).
    pub writer_host: u32,
    /// SHA-256 prefix hash — the block's identity.
    pub hash: BlockHash,
    /// xxHash64 of the KV data payload. Fast, non-cryptographic.
    pub data_checksum: u64,
    /// Payload size in bytes.
    pub data_size: u32,
    /// Monotonic write epoch (prevents ABA from stale reads after reuse).
    pub write_epoch: u32,
    /// Reserved for future use.
    pub _reserved: [u8; 8],
}

const CXL_BLOCK_MAGIC: u32 = 0x4B56_4358; // "KVCX"

impl CxlBlockHeader {
    /// Quick sanity check (before reading the full payload).
    pub fn is_valid_header(&self) -> bool {
        self.magic == CXL_BLOCK_MAGIC
            && self.data_size > 0
            && self.data_size <= MAX_BLOCK_DATA_SIZE as u32
    }
}
```

### 4.3 Write Path: Checksum on Write

```rust
/// Write a KV block to CXL shared memory with integrity header.
///
/// Write order matters: data first, then header.
/// The header write (specifically, setting magic to CXL_BLOCK_MAGIC)
/// is the "publish" barrier — readers check magic before trusting data.
pub unsafe fn cxl_write_block(
    dest: *mut u8,
    hash: BlockHash,
    kv_data: &[u8],
    writer_host: u32,
    write_epoch: u32,
) {
    let header_size = std::mem::size_of::<CxlBlockHeader>();

    // Step 1: Write payload FIRST (after header space).
    std::ptr::copy_nonoverlapping(
        kv_data.as_ptr(),
        dest.add(header_size),
        kv_data.len(),
    );

    // Step 2: Store fence — ensure payload is fully written before header.
    std::sync::atomic::fence(Ordering::Release);

    // Step 3: Write header (including checksum computed over the payload).
    let header = CxlBlockHeader {
        magic: CXL_BLOCK_MAGIC,
        writer_host,
        hash,
        data_checksum: xxhash64(kv_data),
        data_size: kv_data.len() as u32,
        write_epoch,
        _reserved: [0; 8],
    };
    std::ptr::copy_nonoverlapping(
        &header as *const _ as *const u8,
        dest,
        header_size,
    );

    // Step 4: Final fence — header is now visible to other hosts.
    std::sync::atomic::fence(Ordering::Release);
}
```

### 4.4 Read Path: Verify on Every Read

```rust
/// Read a KV block from CXL shared memory with full integrity verification.
///
/// Returns Ok(data) if all checks pass, Err with specific failure mode otherwise.
/// This is called on EVERY block read — no unverified reads.
pub unsafe fn cxl_read_block(
    src: *const u8,
    expected_hash: &BlockHash,
) -> Result<Vec<u8>, CxlBlockError> {
    let header_size = std::mem::size_of::<CxlBlockHeader>();

    // Load fence — ensure we see the most recent write.
    std::sync::atomic::fence(Ordering::Acquire);

    // Step 1: Read header.
    let header = &*(src as *const CxlBlockHeader);

    // Check 1: Magic number (detects uninitialized / grossly corrupted memory).
    if !header.is_valid_header() {
        return Err(CxlBlockError::InvalidMagic {
            got: header.magic,
        });
    }

    // Check 2: Hash matches expectation (detects index corruption or wrong block).
    if header.hash != *expected_hash {
        return Err(CxlBlockError::HashMismatch {
            expected: *expected_hash,
            got: header.hash,
        });
    }

    // Step 2: Read payload.
    let data_size = header.data_size as usize;
    let mut data = vec![0u8; data_size];
    std::ptr::copy_nonoverlapping(
        src.add(header_size),
        data.as_mut_ptr(),
        data_size,
    );

    // Check 3: Data checksum (detects bit-level corruption in the payload).
    let actual_checksum = xxhash64(&data);
    if actual_checksum != header.data_checksum {
        return Err(CxlBlockError::ChecksumMismatch {
            hash: *expected_hash,
            expected: header.data_checksum,
            actual: actual_checksum,
        });
    }

    Ok(data)
}

#[derive(Debug)]
pub enum CxlBlockError {
    /// Magic number wrong — memory uninitialized or gross corruption.
    InvalidMagic { got: u32 },
    /// Block hash mismatch — index pointed to wrong block.
    HashMismatch { expected: BlockHash, got: BlockHash },
    /// Payload checksum failed — bit-level data corruption.
    ChecksumMismatch { hash: BlockHash, expected: u64, actual: u64 },
}
```

### 4.5 Checksum Cost

xxHash64 throughput on modern CPUs: ~30 GB/s single-core. For a 32 KB KV block:
- Checksum computation: ~1μs
- Added to write path: ~1μs (overlapped with D2H DMA — effectively free)
- Added to read path: ~1μs (before H2D DMA — adds ~1μs to promotion)

| Path | v6 Latency | v7 Latency | Delta |
|---|---|---|---|
| Write block (GPU → CXL) | ~3μs (DMA only) | ~3μs (DMA) + ~1μs (checksum, overlapped) | **~0μs effective** |
| Read block (CXL → GPU) | ~3μs (DMA only) | ~1μs (checksum) + ~3μs (DMA) | **+1μs** (+25%) |
| Promote 100 blocks | ~330μs | ~430μs | **+100μs** (+30%) |

The +1μs per block is the cost of detecting corruption before it reaches the GPU. This is a good trade: 100μs of extra latency for 100 blocks prevents silent wrong output.

---

## 5. Control Plane: Local Index with Gossip Replication

### 5.1 Per-Host Local Index

Each host maintains its own block index in **local DRAM** (protected by local ECC, isolated from CXL failures):

```rust
/// Per-host block index. Replaces v6's shared AtomicBlockIndexEntry table.
/// Lives in LOCAL DRAM — a CXL bit flip cannot corrupt this.
pub struct LocalBlockIndex {
    /// hash → entry. Standard HashMap in local memory.
    map: RwLock<HashMap<BlockHash, IndexEntry>>,
    /// This host's ID.
    my_host: u32,
}

#[derive(Clone, Debug)]
struct IndexEntry {
    /// Where the block data lives in CXL shared memory.
    addr: GlobalBlockAddr,
    /// Which host wrote this block.
    writer_host: u32,
    /// How many local sequences (on THIS host) are actively using this block.
    local_refcount: u32,
    /// Write epoch from the CxlBlockHeader (for staleness detection).
    write_epoch: u32,
    /// When we learned about this entry (for GC of stale gossip).
    learned_at: Instant,
}

impl LocalBlockIndex {
    /// Look up a block by hash. Fast local HashMap lookup (~80ns).
    pub fn lookup(&self, hash: &BlockHash) -> Option<GlobalBlockAddr> {
        self.map.read().get(hash).map(|e| e.addr)
    }

    /// Insert a new block (we just wrote it to CXL).
    pub fn insert_local(&self, hash: BlockHash, addr: GlobalBlockAddr, epoch: u32) {
        self.map.write().insert(hash, IndexEntry {
            addr,
            writer_host: self.my_host,
            local_refcount: 0,
            write_epoch: epoch,
            learned_at: Instant::now(),
        });
    }

    /// Insert a block learned via gossip from another host.
    pub fn insert_gossip(&self, hash: BlockHash, addr: GlobalBlockAddr, writer: u32, epoch: u32) {
        let mut map = self.map.write();
        match map.entry(hash) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // Keep the newer entry (higher epoch).
                if epoch > e.get().write_epoch {
                    e.get_mut().addr = addr;
                    e.get_mut().writer_host = writer;
                    e.get_mut().write_epoch = epoch;
                    e.get_mut().learned_at = Instant::now();
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(IndexEntry {
                    addr, writer_host: writer, local_refcount: 0,
                    write_epoch: epoch, learned_at: Instant::now(),
                });
            }
        }
    }

    /// Increment local refcount (a sequence on this host is using the block).
    pub fn ref_inc(&self, hash: &BlockHash) {
        if let Some(entry) = self.map.write().get_mut(hash) {
            entry.local_refcount += 1;
        }
    }

    /// Decrement local refcount.
    pub fn ref_dec(&self, hash: &BlockHash) {
        if let Some(entry) = self.map.write().get_mut(hash) {
            entry.local_refcount = entry.local_refcount.saturating_sub(1);
        }
    }

    /// Get all entries (for gossip broadcast).
    pub fn snapshot_for_gossip(&self) -> Vec<(BlockHash, GlobalBlockAddr, u32, u32)> {
        self.map.read().iter()
            .map(|(h, e)| (*h, e.addr, e.writer_host, e.write_epoch))
            .collect()
    }

    /// Remove entries for blocks that have been freed by the allocator.
    pub fn gc_freed_blocks(&self, freed_addrs: &HashSet<GlobalBlockAddr>) {
        self.map.write().retain(|_, e| !freed_addrs.contains(&e.addr));
    }
}
```

### 5.2 Gossip Protocol

Hosts periodically broadcast their index changes to all peers. This replaces v6's "single shared index visible to everyone instantly."

```rust
/// Lightweight gossip protocol for block index replication.
/// Each host periodically sends its recent index updates to all peers.
pub struct IndexGossip {
    local_index: Arc<LocalBlockIndex>,
    peers: Vec<SocketAddr>,
    socket: UdpSocket,
    /// Track what we've already sent (to send only deltas).
    last_broadcast_epoch: AtomicU32,
}

/// A single gossip message: a batch of index updates.
#[derive(Serialize, Deserialize)]
struct GossipMessage {
    sender_host: u32,
    entries: Vec<GossipEntry>,
}

#[derive(Serialize, Deserialize)]
struct GossipEntry {
    hash: BlockHash,
    addr: GlobalBlockAddr,
    writer_host: u32,
    write_epoch: u32,
    /// true = block exists; false = block was freed (tombstone).
    alive: bool,
}

impl IndexGossip {
    /// Broadcast recent index changes to all peers.
    /// Called periodically (e.g., every 1–5ms).
    pub fn broadcast(&self) {
        let snapshot = self.local_index.snapshot_for_gossip();
        let current_epoch = self.last_broadcast_epoch.load(Ordering::Relaxed);

        // Only send entries newer than last broadcast.
        let delta: Vec<GossipEntry> = snapshot.into_iter()
            .filter(|(_, _, _, epoch)| *epoch > current_epoch)
            .map(|(hash, addr, writer, epoch)| GossipEntry {
                hash, addr, writer_host: writer, write_epoch: epoch, alive: true,
            })
            .collect();

        if delta.is_empty() { return; }

        let msg = GossipMessage {
            sender_host: self.local_index.my_host,
            entries: delta,
        };
        let bytes = bincode::serialize(&msg).unwrap();

        for peer in &self.peers {
            let _ = self.socket.send_to(&bytes, peer);
        }

        self.last_broadcast_epoch.store(
            snapshot.iter().map(|(_, _, _, e)| *e).max().unwrap_or(current_epoch),
            Ordering::Relaxed,
        );
    }

    /// Receive gossip from peers and merge into local index.
    pub fn receive_loop(&self) {
        let mut buf = vec![0u8; 65536]; // max UDP payload
        loop {
            if let Ok((n, _)) = self.socket.recv_from(&mut buf) {
                if let Ok(msg) = bincode::deserialize::<GossipMessage>(&buf[..n]) {
                    for entry in &msg.entries {
                        if entry.alive {
                            self.local_index.insert_gossip(
                                entry.hash, entry.addr,
                                entry.writer_host, entry.write_epoch,
                            );
                        } else {
                            // Tombstone: block was freed.
                            self.local_index.map.write().remove(&entry.hash);
                        }
                    }
                }
            }
        }
    }
}
```

### 5.3 Consistency Model

The gossip-replicated index is **eventually consistent** with a propagation delay of ~1–10ms (one gossip interval). This means:

- **Hot path (same-host lookup):** A block written by this host is immediately visible in the local index. Zero delay.
- **Cross-host discovery:** A block written by Host A becomes visible to Host B after one gossip round (~1–10ms).
- **Routing decisions:** The router can use a slightly stale index. A stale miss means a request goes to a less-optimal instance — it doesn't cause a wrong answer, just a cache miss and recompute. The performance impact of a 10ms gossip lag is negligible for LLM workloads.

---

## 6. Control Plane: Centralized Allocator with Failover

### 6.1 Why Centralize?

v6 used a lock-free `AtomicFreeList` in CXL shared memory so any host could allocate. But that put the most critical data structure (the allocator) in the least reliable memory. v7 centralizes allocation on a single **memory manager host** with a standby:

```
v6:  Any Host ──CAS──→ [Shared Free List in CXL] ←──CAS── Any Host
     (lock-free, but a bit flip in the head kills everyone)

v7:  Any Host ──RPC──→ [Memory Manager Host]
                       bitmap in LOCAL DRAM        ←──RPC── Any Host
                       (local ECC, isolated)
     Standby Host monitors via heartbeat; takes over if primary dies.
```

### 6.2 Memory Manager

```rust
/// Centralized CXL block allocator. Runs on ONE host.
/// Uses a simple bitmap in LOCAL DRAM — no atomics in shared memory.
pub struct CxlMemoryManager {
    /// 1 bit per block: 1 = allocated, 0 = free.
    /// In LOCAL DRAM, protected by local ECC.
    bitmap: Vec<u64>,
    /// Total blocks in the CXL region.
    total_blocks: u32,
    /// Free block count (for fast capacity queries).
    free_count: u32,
    /// The CXL region this manager covers.
    region_id: u64,
    /// Block stride (header + payload size).
    block_stride: usize,
    /// Base address of the CXL region (for address computation).
    region_base_offset: u64,
    /// Monotonic epoch counter (for write_epoch in block headers).
    epoch: u32,
}

impl CxlMemoryManager {
    /// Allocate N blocks. Returns global addresses.
    /// Called via RPC from any host.
    pub fn allocate(&mut self, count: u32) -> Vec<(GlobalBlockAddr, u32)> {
        let mut results = Vec::with_capacity(count as usize);
        for word_idx in 0..self.bitmap.len() {
            if results.len() >= count as usize { break; }
            let word = &mut self.bitmap[word_idx];
            while *word != u64::MAX && results.len() < count as usize {
                let bit = (!*word).trailing_zeros();
                *word |= 1u64 << bit;
                let block_idx = word_idx as u32 * 64 + bit;
                self.epoch += 1;
                results.push((
                    GlobalBlockAddr {
                        region_id: self.region_id,
                        offset: self.block_offset(block_idx),
                    },
                    self.epoch,
                ));
                self.free_count -= 1;
            }
        }
        results
    }

    /// Free blocks. Called via RPC from any host.
    pub fn free(&mut self, addrs: &[GlobalBlockAddr]) {
        for addr in addrs {
            let block_idx = self.offset_to_index(addr.offset);
            let word_idx = block_idx as usize / 64;
            let bit = block_idx % 64;
            self.bitmap[word_idx] &= !(1u64 << bit);
            self.free_count += 1;
        }
    }

    /// How many blocks are free?
    pub fn free_blocks(&self) -> u32 {
        self.free_count
    }

    fn block_offset(&self, block_idx: u32) -> u64 {
        self.region_base_offset + (block_idx as u64 * self.block_stride as u64)
    }

    fn offset_to_index(&self, offset: u64) -> u32 {
        ((offset - self.region_base_offset) / self.block_stride as u64) as u32
    }
}
```

### 6.3 RPC Interface

```rust
/// Lightweight RPC for memory manager. TCP with length-prefixed messages.
/// The allocator is not on the hot path (allocations happen at block granularity,
/// ~1K/s per host peak), so TCP overhead is fine.
#[derive(Serialize, Deserialize)]
pub enum AllocatorRequest {
    Allocate { count: u32 },
    Free { addrs: Vec<GlobalBlockAddr> },
    Status,
}

#[derive(Serialize, Deserialize)]
pub enum AllocatorResponse {
    Allocated { addrs: Vec<(GlobalBlockAddr, u32)> },  // (addr, epoch)
    Freed,
    Status { total: u32, free: u32 },
}
```

### 6.4 Standby Failover

A standby host monitors the primary allocator via heartbeat. On primary failure, it rebuilds the bitmap by scanning CXL block headers:

```rust
/// Standby allocator: monitors primary, takes over on failure.
pub struct AllocatorStandby {
    primary_host: u32,
    heartbeat: Arc<UdpHeartbeat>,
    /// CXL region info for bitmap rebuild.
    region_id: u64,
    region_base: *const u8,
    total_blocks: u32,
    block_stride: usize,
}

impl AllocatorStandby {
    /// Monitor loop: detect primary failure and take over.
    pub async fn monitor(&self) -> CxlMemoryManager {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if !self.heartbeat.is_alive(self.primary_host) {
                log::warn!(
                    "Memory manager primary (host {}) down. Rebuilding allocator.",
                    self.primary_host
                );
                return self.rebuild();
            }
        }
    }

    /// Rebuild bitmap by scanning CXL block headers.
    /// A block is allocated if its header has magic == CXL_BLOCK_MAGIC.
    /// A block is free if its header has any other magic value.
    fn rebuild(&self) -> CxlMemoryManager {
        let num_words = (self.total_blocks as usize + 63) / 64;
        let mut bitmap = vec![0u64; num_words];
        let mut allocated = 0u32;

        for block_idx in 0..self.total_blocks {
            let offset = block_idx as usize * self.block_stride;
            let ptr = unsafe { self.region_base.add(offset) };
            let header = unsafe { &*(ptr as *const CxlBlockHeader) };

            if header.is_valid_header() {
                let word_idx = block_idx as usize / 64;
                let bit = block_idx % 64;
                bitmap[word_idx] |= 1u64 << bit;
                allocated += 1;
            }
        }

        log::info!(
            "Allocator rebuilt: {}/{} blocks allocated, {} free",
            allocated, self.total_blocks, self.total_blocks - allocated
        );

        CxlMemoryManager {
            bitmap,
            total_blocks: self.total_blocks,
            free_count: self.total_blocks - allocated,
            region_id: self.region_id,
            block_stride: self.block_stride,
            region_base_offset: 0,
            epoch: allocated, // start epoch after existing blocks
        }
    }
}
```

### 6.5 Allocation Throughput

The centralized allocator is NOT on the hot path:
- KV blocks are allocated at block granularity (16 tokens per block).
- A busy host processing 1000 tokens/s allocates ~62 blocks/s.
- 4 hosts: ~250 blocks/s total.
- The allocator can handle ~100K alloc/s over TCP — 400× headroom.
- Batch allocation (allocate N blocks per RPC) further amortizes overhead.

---

## 7. Corruption Detection and Recovery

### 7.1 Detection Points

Every read from CXL shared memory passes through `cxl_read_block` (§4.4), which performs three checks:

1. **Magic number** → detects uninitialized memory or gross corruption
2. **Hash match** → detects index corruption (local index pointed to wrong block)
3. **Data checksum** → detects bit-level corruption in the KV payload

### 7.2 Recovery: Recompute, Don't Propagate

```rust
impl CxlAwarePoolManager {
    /// Promote a block from CXL to GPU, with integrity verification.
    /// On corruption: quarantine and return cache-miss (scheduler will recompute).
    pub fn promote_from_cxl_verified(
        &mut self,
        addr: GlobalBlockAddr,
        expected_hash: &BlockHash,
    ) -> PromoteResult {
        let pgas = self.pgas_pool.as_ref().expect("CXL not available");

        unsafe {
            let src = pgas.block_ptr(addr);
            match cxl_read_block(src, expected_hash) {
                Ok(data) => {
                    // Data verified. Upload to GPU.
                    if let Some(gpu_id) = self.l1_gpu.try_allocate_one() {
                        self.l1_gpu.gpu_pool.upload_block(gpu_id, &data);
                        self.metrics.cxl_promote_ok.inc();
                        PromoteResult::Ok(gpu_id)
                    } else {
                        PromoteResult::GpuFull
                    }
                }
                Err(e) => {
                    log::warn!("CXL block corruption: {:?}. Recomputing.", e);
                    self.metrics.cxl_corruption_detected.inc();

                    // Remove from local index.
                    self.local_index.map.write().remove(expected_hash);

                    // Notify allocator to quarantine.
                    self.allocator_client.quarantine(addr);

                    // Broadcast tombstone via gossip so peers also remove it.
                    self.gossip.broadcast_tombstone(*expected_hash);

                    // Scheduler will see this as a cache miss and recompute.
                    PromoteResult::CorruptionDetected
                }
            }
        }
    }
}

pub enum PromoteResult {
    /// Successfully promoted. Contains GPU block ID.
    Ok(u32),
    /// GPU is full; need to evict first.
    GpuFull,
    /// Block was corrupted in CXL memory. Treat as cache miss.
    CorruptionDetected,
}
```

### 7.3 Quarantine

Corrupted blocks are quarantined (not freed for reuse) until the scrubber determines the corruption pattern. If many blocks in the same CXL region are corrupted, it may indicate a failing CXL device:

```rust
impl CxlMemoryManager {
    /// Quarantine a block: mark as allocated but unusable.
    /// Quarantined blocks are not freed back to the pool.
    pub fn quarantine(&mut self, addr: GlobalBlockAddr) {
        // Block stays allocated in the bitmap (not freed for reuse).
        // Track separately for monitoring.
        self.quarantined.insert(addr);

        // Check: too many quarantined blocks from the same device?
        let device_quarantine_count = self.quarantined.iter()
            .filter(|a| self.same_cxl_device(a, &addr))
            .count();

        if device_quarantine_count > QUARANTINE_THRESHOLD {
            log::error!(
                "CXL device containing region {} has {} quarantined blocks — possible hardware failure",
                addr.region_id, device_quarantine_count
            );
            // TODO: alert monitoring, consider draining this device
        }
    }
}
```

---

## 8. Background Scrubber

### 8.1 Purpose

The scrubber catches **latent corruption** — blocks that were corrupted after being written but before being read. Without scrubbing, a corrupted block sits undetected until a request needs it, causing an unexpected recompute at read time.

### 8.2 Implementation

```rust
/// Background scrubber: periodically verifies all allocated blocks.
pub struct CxlScrubber {
    /// CXL region base pointer.
    region_base: *const u8,
    /// Block stride (header + max payload).
    block_stride: usize,
    /// Total blocks in the region.
    total_blocks: u32,
    /// Blocks confirmed corrupted.
    quarantined: Arc<Mutex<HashSet<u32>>>,
    /// Scrub interval (how long to complete one full pass).
    full_pass_duration: Duration,
    /// Metrics.
    metrics: Arc<ScrubberMetrics>,
}

struct ScrubberMetrics {
    blocks_scrubbed: AtomicU64,
    corruptions_found: AtomicU64,
    last_full_pass_ms: AtomicU64,
}

impl CxlScrubber {
    /// Scrub all blocks in the region. Returns count of newly corrupted blocks.
    pub fn scrub_pass(&self) -> u32 {
        let mut new_corruptions = 0;
        let quarantined = self.quarantined.lock();
        let start = Instant::now();

        for block_idx in 0..self.total_blocks {
            if quarantined.contains(&block_idx) { continue; }

            let offset = block_idx as usize * self.block_stride;
            let ptr = unsafe { self.region_base.add(offset) };
            let header = unsafe { &*(ptr as *const CxlBlockHeader) };

            // Skip unallocated blocks.
            if !header.is_valid_header() { continue; }

            // Verify data checksum.
            let data_ptr = unsafe { ptr.add(std::mem::size_of::<CxlBlockHeader>()) };
            let data = unsafe {
                std::slice::from_raw_parts(data_ptr, header.data_size as usize)
            };
            let actual = xxhash64(data);

            if actual != header.data_checksum {
                log::error!(
                    "Scrubber: block {} corrupted (hash={:x?}, expected={:#x}, actual={:#x})",
                    block_idx, &header.hash[..8], header.data_checksum, actual
                );
                drop(quarantined);
                self.quarantined.lock().insert(block_idx);
                new_corruptions += 1;
                self.metrics.corruptions_found.fetch_add(1, Ordering::Relaxed);
            }

            self.metrics.blocks_scrubbed.fetch_add(1, Ordering::Relaxed);

            // Pace: spread the scrub across full_pass_duration to avoid
            // saturating CXL bandwidth with scrub reads.
            if block_idx % 1000 == 0 {
                let elapsed = start.elapsed();
                let expected = self.full_pass_duration * block_idx / self.total_blocks;
                if elapsed < expected {
                    std::thread::sleep(expected - elapsed);
                }
            }
        }

        self.metrics.last_full_pass_ms.store(
            start.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        new_corruptions
    }

    /// Run scrubber loop.
    pub async fn run(&self) {
        loop {
            let found = tokio::task::spawn_blocking({
                let this = self.clone();
                move || this.scrub_pass()
            }).await.unwrap();

            if found > 0 {
                log::warn!("Scrubber found {} corrupted blocks in this pass", found);
            }
        }
    }
}
```

### 8.3 Scrub Pacing

The scrubber must not saturate CXL bandwidth. For a 1 TB CXL region with 32 KB blocks (~32M blocks):
- Full pass read volume: ~1 TB
- At 10 GB/s scrub rate (small fraction of CXL bandwidth): ~100 seconds per pass
- Scrub pacing: read ~10 GB/s, yielding between batches

This means every block is verified at least every ~100 seconds. A block corrupted in between scrub passes will be caught either on the next scrub or on the next read (whichever comes first).

---

## 9. Heartbeat and Liveness (Side-Channel)

### 9.1 Why Not in CXL Memory

v6 stored heartbeats as `AtomicU64` in CXL shared memory. A bit flip in a heartbeat slot causes false liveness or false failure detection — both dangerous. v7 uses a **UDP side-channel** that doesn't touch CXL memory at all.

### 9.2 Implementation

```rust
/// Host heartbeat over UDP. Does NOT use CXL shared memory.
pub struct UdpHeartbeat {
    my_host: u32,
    peers: Vec<SocketAddr>,
    socket: UdpSocket,
    /// Last heartbeat received from each peer.
    last_seen: Mutex<HashMap<u32, Instant>>,
    interval: Duration,
    timeout: Duration,
}

#[derive(Serialize, Deserialize)]
struct HeartbeatMsg {
    host_id: u32,
    timestamp_ms: u64,
    /// Include allocator role info for failover coordination.
    is_allocator_primary: bool,
}

impl UdpHeartbeat {
    pub fn new(
        my_host: u32,
        peers: Vec<SocketAddr>,
        listen_addr: SocketAddr,
        interval: Duration,
        timeout: Duration,
    ) -> Self {
        let socket = UdpSocket::bind(listen_addr).expect("heartbeat bind failed");
        socket.set_nonblocking(true).unwrap();
        Self {
            my_host, peers, socket,
            last_seen: Mutex::new(HashMap::new()),
            interval, timeout,
        }
    }

    /// Send heartbeat to all peers.
    pub fn beat(&self, is_allocator_primary: bool) {
        let msg = HeartbeatMsg {
            host_id: self.my_host,
            timestamp_ms: unix_timestamp_ms(),
            is_allocator_primary,
        };
        let bytes = bincode::serialize(&msg).unwrap();
        for peer in &self.peers {
            let _ = self.socket.send_to(&bytes, peer);
        }
    }

    /// Check if a peer is alive.
    pub fn is_alive(&self, host_id: u32) -> bool {
        self.last_seen.lock()
            .get(&host_id)
            .map(|t| t.elapsed() < self.timeout)
            .unwrap_or(false)
    }

    /// Receive loop (runs in background).
    pub fn receive_loop(&self) {
        let mut buf = [0u8; 128];
        loop {
            match self.socket.recv_from(&mut buf) {
                Ok((n, _)) => {
                    if let Ok(msg) = bincode::deserialize::<HeartbeatMsg>(&buf[..n]) {
                        self.last_seen.lock().insert(msg.host_id, Instant::now());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {}
            }
        }
    }
}

fn unix_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
```

---

## 10. Revised PGAS Model: Safe Wrappers

### 10.1 Changes from v6

v6's `SymmetricHeap` exposed raw `shmem_put/get/ptr` that bypassed integrity checks. v7 wraps these with mandatory checksum verification:

```rust
/// REVISED SymmetricHeap: all data-plane operations go through checksummed paths.
/// Raw pointer access is private — external callers must use safe wrappers.
pub struct SymmetricHeap {
    /// CXL region base address.
    region_base: *mut u8,
    region_id: u64,
    /// Block stride: header + max payload.
    block_stride: usize,
    /// This host's ID.
    my_host: u32,
    /// CXL topology.
    topology: Arc<CxlTopology>,
}

impl SymmetricHeap {
    /// Write a KV block to a global address WITH checksum.
    /// Replaces v6's unsafe `shmem_put_block`.
    pub fn write_block(
        &self,
        addr: GlobalBlockAddr,
        hash: BlockHash,
        kv_data: &[u8],
        epoch: u32,
    ) {
        let ptr = self.addr_to_ptr(addr);
        unsafe {
            cxl_write_block(ptr, hash, kv_data, self.my_host, epoch);
        }
    }

    /// Read a KV block from a global address WITH checksum verification.
    /// Replaces v6's unsafe `shmem_get_block`.
    /// Returns Err on corruption (caller should recompute).
    pub fn read_block(
        &self,
        addr: GlobalBlockAddr,
        expected_hash: &BlockHash,
    ) -> Result<Vec<u8>, CxlBlockError> {
        let ptr = self.addr_to_ptr(addr);
        unsafe { cxl_read_block(ptr as *const u8, expected_hash) }
    }

    /// Read the header only (for scrubbing, allocator rebuild).
    pub fn read_header(&self, addr: GlobalBlockAddr) -> Option<CxlBlockHeader> {
        let ptr = self.addr_to_ptr(addr) as *const CxlBlockHeader;
        let header = unsafe { &*ptr };
        if header.is_valid_header() {
            Some(unsafe { std::ptr::read(ptr) })
        } else {
            None
        }
    }

    /// Compute CXL distance from this host to the NUMA node hosting an address.
    pub fn distance_to(&self, addr: GlobalBlockAddr) -> u32 {
        self.topology.shared_regions.iter()
            .find(|r| r.region_id == addr.region_id)
            .and_then(|r| r.distance_ns_per_host.iter()
                .find(|(h, _)| *h == self.my_host)
                .map(|(_, d)| *d))
            .unwrap_or(u32::MAX)
    }

    /// Raw pointer (PRIVATE — only for DMA source/dest within this crate).
    fn addr_to_ptr(&self, addr: GlobalBlockAddr) -> *mut u8 {
        assert_eq!(addr.region_id, self.region_id, "wrong region");
        unsafe { self.region_base.add(addr.offset as usize) }
    }
}
```

### 10.2 DMA Path: GPU ↔ CXL with Verification

For GPU promotion, we still need a pointer for DMA. But we verify AFTER the DMA:

```rust
impl CxlAwarePoolManager {
    /// Promote: CXL → GPU with post-DMA verification.
    pub fn promote_from_cxl(
        &mut self,
        addr: GlobalBlockAddr,
        expected_hash: &BlockHash,
    ) -> PromoteResult {
        let heap = self.pgas_heap.as_ref().unwrap();

        // Step 1: Read block with checksum verification.
        let data = match heap.read_block(addr, expected_hash) {
            Ok(d) => d,
            Err(e) => {
                self.handle_corruption(addr, expected_hash, e);
                return PromoteResult::CorruptionDetected;
            }
        };

        // Step 2: Upload verified data to GPU.
        if let Some(gpu_id) = self.l1_gpu.try_allocate_one() {
            self.l1_gpu.gpu_pool.upload_block(gpu_id, &data);
            self.metrics.cxl_promote_ok.inc();
            PromoteResult::Ok(gpu_id)
        } else {
            PromoteResult::GpuFull
        }
    }

    /// Demote: GPU → CXL (write with checksum).
    pub fn evict_gpu_to_cxl(&mut self, target: usize) -> usize {
        let heap = match &self.pgas_heap {
            Some(h) => h,
            None => return self.l1_gpu.evict(target),
        };

        let mut freed = 0;
        while freed < target {
            if let Some(leaf) = self.l1_gpu.find_oldest_unreferenced_leaf() {
                // Request allocation from centralized allocator.
                let allocs = self.allocator_client.allocate(1);
                if let Some((global_addr, epoch)) = allocs.first() {
                    // Read from GPU.
                    let data = self.l1_gpu.gpu_pool.read_block(leaf.block_id);
                    let hash = self.l1_gpu.block_hash(leaf);

                    // Write to CXL WITH checksum.
                    heap.write_block(*global_addr, hash, &data, *epoch);

                    // Update local index.
                    self.local_index.insert_local(hash, *global_addr, *epoch);

                    // Free GPU block.
                    self.l1_gpu.gpu_pool.free(leaf.block_id);
                    self.l1_gpu.remove_leaf(leaf);
                    freed += 1;
                } else {
                    // CXL pool full — just discard.
                    self.l1_gpu.gpu_pool.free(leaf.block_id);
                    self.l1_gpu.remove_leaf(leaf);
                    freed += 1;
                }
            } else {
                break;
            }
        }
        freed
    }

    fn handle_corruption(
        &self,
        addr: GlobalBlockAddr,
        hash: &BlockHash,
        error: CxlBlockError,
    ) {
        log::warn!("CXL corruption at {:?}: {:?}", addr, error);
        self.metrics.cxl_corruption_detected.inc();
        self.local_index.map.write().remove(hash);
        self.allocator_client.quarantine(addr);
        self.gossip.broadcast_tombstone(*hash);
    }
}
```

---

## 11. Revised CXL KV Connector

v6's `KvConnectorZeroCopy` returned raw pointers. v7 replaces it with a safe interface:

```rust
/// v7 CXL connector: always returns verified data, never raw pointers.
pub struct CxlKvConnector {
    heap: Arc<SymmetricHeap>,
    local_index: Arc<LocalBlockIndex>,
    allocator_client: Arc<AllocatorClient>,
    gossip: Arc<IndexGossip>,
    metrics: Arc<CxlConnectorMetrics>,
}

struct CxlConnectorMetrics {
    lookups: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    corruption_detected: AtomicU64,
}

/// v7 still implements the v5 KvConnector trait for compatibility,
/// but backed by CXL with checksums.
impl KvConnector for CxlKvConnector {
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        self.metrics.lookups.fetch_add(hashes.len() as u64, Ordering::Relaxed);
        hashes.iter().map(|h| {
            let found = self.local_index.lookup(h).is_some();
            if found {
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            }
            found
        }).collect()
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        hashes.iter().map(|h| {
            let addr = self.local_index.lookup(h)?;

            // Read with checksum verification.
            match self.heap.read_block(addr, h) {
                Ok(data) => {
                    self.local_index.ref_inc(h);
                    Some(KvBlockData {
                        meta: KvBlockMeta { hash: *h, ..Default::default() },
                        data,
                    })
                }
                Err(e) => {
                    // Corruption detected. Quarantine and return miss.
                    log::warn!("CXL fetch corruption for {:x?}: {:?}", &h[..8], e);
                    self.metrics.corruption_detected.fetch_add(1, Ordering::Relaxed);
                    self.local_index.map.write().remove(h);
                    self.allocator_client.quarantine(addr);
                    self.gossip.broadcast_tombstone(*h);
                    None
                }
            }
        }).collect()
    }

    fn store(&self, blocks: &[KvBlockData]) {
        // Batch-allocate from centralized allocator.
        let allocs = self.allocator_client.allocate(blocks.len() as u32);

        for (block, (addr, epoch)) in blocks.iter().zip(allocs.iter()) {
            // Write with checksum.
            self.heap.write_block(*addr, block.meta.hash, &block.data, *epoch);
            // Update local index.
            self.local_index.insert_local(block.meta.hash, *addr, *epoch);
        }
        // Gossip will broadcast these entries to peers.
    }

    fn remove(&self, hashes: &[BlockHash]) {
        let mut addrs_to_free = Vec::new();
        for h in hashes {
            if let Some(addr) = self.local_index.lookup(h) {
                addrs_to_free.push(addr);
                self.local_index.map.write().remove(h);
            }
        }
        if !addrs_to_free.is_empty() {
            self.allocator_client.free(&addrs_to_free);
        }
    }

    fn tier_name(&self) -> &str { "cxl_shared" }
    fn capacity(&self) -> usize { self.allocator_client.status().total as usize }
    fn usage(&self) -> usize {
        self.allocator_client.status().total as usize - self.allocator_client.status().free as usize
    }
}
```

---

## 12. Weight Pool Integrity

Model weights are read-only after loading, so corruption detection is simpler but the stakes are higher (weight corruption → every output is wrong, not just one block).

```rust
impl SharedWeightPool {
    /// Leader: load weights AND compute per-tensor checksums.
    pub fn load_leader(
        region: &SharedRegionInfo,
        model_path: &Path,
        tp_rank: u32,
        tp_size: u32,
    ) -> Self {
        let base_addr = region.base_addr;
        let mut layout = HashMap::new();
        let mut checksums = HashMap::new();
        let mut offset = 0u64;

        for shard in load_safetensors_index(model_path) {
            let (name, tensor) = load_tensor(&shard);
            let sharded = apply_tp_sharding(&name, &tensor, tp_rank, tp_size);
            let size = sharded.data.len() as u64;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    sharded.data.as_ptr(),
                    base_addr.add(offset as usize) as *mut u8,
                    size as usize,
                );
            }

            checksums.insert(name.clone(), xxhash64(&sharded.data));
            layout.insert(name, WeightSlice { offset, size, dtype: sharded.dtype, shape: sharded.shape });
            offset += size;
        }

        std::sync::atomic::fence(Ordering::SeqCst);

        Self { region_id: region.region_id, base_addr: base_addr as *const u8, layout, checksums, total_bytes: offset }
    }

    /// Verify all weight tensor checksums. Called at startup (follower) and periodically.
    pub fn verify_integrity(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for (name, slice) in &self.layout {
            let expected = self.checksums.get(name).copied().unwrap_or(0);
            let actual = unsafe {
                let data = std::slice::from_raw_parts(
                    self.base_addr.add(slice.offset as usize),
                    slice.size as usize,
                );
                xxhash64(data)
            };
            if expected != actual {
                errors.push(format!(
                    "Weight '{}' corrupted: expected checksum {:#x}, got {:#x}",
                    name, expected, actual
                ));
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Periodic weight verification (background task).
    pub async fn periodic_verify(&self, interval: Duration) {
        loop {
            tokio::time::sleep(interval).await;
            if let Err(errors) = self.verify_integrity() {
                for e in &errors {
                    log::error!("{}", e);
                }
                log::error!("Weight corruption detected — initiating reload from disk");
                // Weight corruption is catastrophic: model outputs are wrong.
                // Must reload from disk and re-push to all GPUs.
                self.initiate_weight_reload().await;
            }
        }
    }
}
```

---

## 13. Integration with v6 Components

### 13.1 What Changed

| v6 Component | v7 Change | Reason |
|---|---|---|
| `PgasBlockPool` (§3.3) | **Replaced** by `CxlMemoryManager` (§6) + `LocalBlockIndex` (§5) + `SymmetricHeap` (§10) | Split control from data plane |
| `AtomicFreeList` (§3.4) | **Removed**. Replaced by bitmap in `CxlMemoryManager`. | No atomics in shared memory |
| `AtomicBlockIndexEntry` (§3.3) | **Removed**. Replaced by `LocalBlockIndex` + `IndexGossip`. | No shared hash table |
| `SymmetricHeap::shmem_put_block` | **Replaced** by `SymmetricHeap::write_block` (with checksum) | Mandatory integrity |
| `SymmetricHeap::shmem_get_block` | **Replaced** by `SymmetricHeap::read_block` (with verification) | Mandatory integrity |
| `SymmetricHeap::shmem_ptr` | **Removed** (private only) | No unverified raw pointers |
| `CxlKvConnector` (zero-copy trait) | **Replaced** by v5-compatible `KvConnector` backed by CXL | Corruption-safe interface |
| `CxlHeartbeat` (in CXL memory) | **Replaced** by `UdpHeartbeat` (§9) | Side-channel, no shared memory |
| `CxlAwarePoolManager::promote_from_cxl` | **Updated**: now calls `promote_from_cxl_verified` with checksum | Verify before GPU upload |
| `CxlAwarePoolManager::evict_gpu_to_cxl` | **Updated**: allocates via RPC, writes with checksum | Centralized allocation |
| `CxlAwareRouter` (v6 §7) | **Updated**: queries `LocalBlockIndex` instead of shared `PgasBlockPool` | Local index |

### 13.2 Unchanged from v6

- CXL topology discovery (v6 §2)
- Distance-aware routing scoring logic (v6 §7) — only the index source changed
- PD disaggregation flow (v6 §8) — adds checksum on write/read
- Graceful degradation to v5
- All v4/v5 components

### 13.3 Unchanged from v4/v5

- `RadixKvManager`, `BlockPool`, forward pass, detokenizer, CUDA graphs, metrics, HTTP API

---

## 14. Configuration

New/changed configuration fields vs. v6:

```yaml
cxl:
  enabled: false

  topology:
    auto_discover: true

  kv_pool:
    enabled: true
    region_id: 0
    capacity_gb: 0

  weight_pool:
    enabled: true
    region_id: 0
    is_leader: false
    # NEW: periodic weight verification interval.
    verify_interval_secs: 300

  # CHANGED: heartbeat is now UDP, not CXL memory.
  heartbeat:
    listen_addr: "0.0.0.0:9100"   # NEW: UDP listen address
    interval_ms: 100
    timeout_ms: 500

  # NEW: centralized allocator.
  allocator:
    # Is this host the primary allocator?
    is_primary: false
    # Is this host the standby allocator?
    is_standby: false
    # Primary allocator address (for RPC from non-primary hosts).
    primary_addr: "host0:9200"
    # Standby address (for failover coordination).
    standby_addr: "host1:9200"

  # NEW: gossip protocol for index replication.
  gossip:
    listen_addr: "0.0.0.0:9300"
    # Broadcast interval (trade-off: lower = faster consistency, higher = less bandwidth).
    interval_ms: 5
    # Peer addresses.
    peers: []

  # NEW: background scrubber.
  scrubber:
    enabled: true
    # Target duration for one full scrub pass.
    full_pass_secs: 120

  # NEW: integrity settings.
  integrity:
    # Checksum algorithm for block data. "xxhash64" (fast) or "crc32c" (HW-accelerated).
    checksum_algo: "xxhash64"
    # Quarantine threshold: alert if this many blocks from one device are corrupted.
    quarantine_alert_threshold: 10

  routing:
    gpu_hit_weight: 10.0
    cxl_hit_base_weight: 1200.0
    min_distance_ns: 150

# Legacy v5 settings unchanged.
kv_cache_pool:
  enabled: true
  cpu_dram:
    enabled: true
    capacity_gb: 32
```

### Minimal Config (2 hosts)

```yaml
# Host 0 (allocator primary + weight leader):
cxl:
  enabled: true
  topology: { auto_discover: true }
  kv_pool: { enabled: true }
  weight_pool: { enabled: true, is_leader: true }
  allocator: { is_primary: true, primary_addr: "host0:9200" }
  gossip: { peers: ["host1:9300"] }

# Host 1 (allocator standby):
cxl:
  enabled: true
  topology: { auto_discover: true }
  kv_pool: { enabled: true }
  weight_pool: { enabled: true, is_leader: false }
  allocator: { is_standby: true, primary_addr: "host0:9200", standby_addr: "host1:9200" }
  gossip: { peers: ["host0:9300"] }
```

---

## 15. Updated Directory Layout

```
llm-server/src/
├── cxl/
│   ├── topology.rs           ← (v6, unchanged)
│   ├── symmetric_heap.rs     ← MODIFIED: safe wrappers with mandatory checksum
│   ├── block_header.rs       ← NEW: CxlBlockHeader, cxl_write_block, cxl_read_block
│   ├── local_index.rs        ← NEW: LocalBlockIndex (replaces shared index)
│   ├── gossip.rs             ← NEW: IndexGossip protocol
│   ├── allocator.rs          ← NEW: CxlMemoryManager (centralized, local bitmap)
│   ├── allocator_client.rs   ← NEW: RPC client for allocator
│   ├── allocator_standby.rs  ← NEW: AllocatorStandby (failover + bitmap rebuild)
│   ├── scrubber.rs           ← NEW: CxlScrubber (background integrity checker)
│   ├── heartbeat.rs          ← REWRITTEN: UdpHeartbeat (replaces CXL-memory heartbeat)
│   ├── connector.rs          ← REWRITTEN: CxlKvConnector (checksummed, v5-compatible)
│   ├── weight_pool.rs        ← MODIFIED: added checksums + periodic verify
│   └── quarantine.rs         ← NEW: block quarantine tracking
├── kv_cache/
│   ├── pool_manager.rs       ← MODIFIED: uses local index, centralized allocator, checksum
│   └── ...                   ← unchanged from v5
├── routing/
│   ├── router.rs             ← MODIFIED: queries local index instead of shared
│   └── ...                   ← unchanged from v5
```

### New crate dependencies

| Crate | Purpose |
|---|---|
| `xxhash-rust` | Fast non-cryptographic checksum (xxHash64) |
| `bincode` | Serialization for gossip + allocator RPC |
| `memmap2` | (v6, unchanged) Memory-map CXL device files |

---

## 16. Implementation Phases

### Phase 11a — CXL Topology + Checksummed Block I/O (1 week)

#### Deliverables

- [ ] `CxlTopology::discover()` — (from v6, unchanged)
- [ ] `CxlBlockHeader` + `cxl_write_block` + `cxl_read_block` — checksummed block I/O
- [ ] `SymmetricHeap` with safe `write_block` / `read_block` wrappers
- [ ] Unit tests: write/read round-trip, corruption injection

#### Test Plan

**T11a.1 — Checksum round-trip:**
```rust
#[test]
fn test_cxl_block_write_read_roundtrip() {
    let region = mmap_test_region(1 << 20);  // 1 MB
    let heap = SymmetricHeap::new(region, 0, 32768 + 64, 0);
    let addr = GlobalBlockAddr { region_id: 0, offset: 0 };
    let hash = [0xABu8; 32];
    let data = vec![0x42u8; 32768];

    heap.write_block(addr, hash, &data, 1);
    let read_back = heap.read_block(addr, &hash).unwrap();
    assert_eq!(read_back, data);
}
```

**T11a.2 — Corruption detection (inject bit flip):**
```rust
#[test]
fn test_cxl_block_corruption_detected() {
    let region = mmap_test_region(1 << 20);
    let heap = SymmetricHeap::new(region, 0, 32768 + 64, 0);
    let addr = GlobalBlockAddr { region_id: 0, offset: 0 };
    let hash = [0xABu8; 32];
    let data = vec![0x42u8; 32768];

    heap.write_block(addr, hash, &data, 1);

    // Inject corruption: flip one bit in the payload.
    unsafe {
        let ptr = region.as_mut_ptr().add(64 + 100);  // byte 100 of payload
        *ptr ^= 0x01;  // flip bit 0
    }

    // Read should detect checksum mismatch.
    let result = heap.read_block(addr, &hash);
    assert!(matches!(result, Err(CxlBlockError::ChecksumMismatch { .. })));
}
```

**T11a.3 — Magic number detection (uninitialized read):**
```rust
#[test]
fn test_cxl_uninitialized_block_detected() {
    let region = mmap_test_region(1 << 20);
    let heap = SymmetricHeap::new(region, 0, 32768 + 64, 0);
    let addr = GlobalBlockAddr { region_id: 0, offset: 0 };
    let hash = [0xABu8; 32];

    // Region is zero-initialized. Reading without writing should fail.
    let result = heap.read_block(addr, &hash);
    assert!(matches!(result, Err(CxlBlockError::InvalidMagic { .. })));
}
```

---

### Phase 11b — Local Index + Gossip + Centralized Allocator (2 weeks)

#### Deliverables

- [ ] `LocalBlockIndex` — per-host HashMap index
- [ ] `IndexGossip` — UDP gossip protocol for index replication
- [ ] `CxlMemoryManager` — centralized bitmap allocator
- [ ] `AllocatorClient` — RPC client
- [ ] `AllocatorStandby` — failover with bitmap rebuild from headers
- [ ] `UdpHeartbeat` — side-channel liveness

#### Test Plan

**T11b.1 — Gossip convergence:**
```rust
#[test]
fn test_gossip_convergence() {
    // 3 hosts, each with a local index.
    let indexes: Vec<Arc<LocalBlockIndex>> = (0..3).map(|i| Arc::new(LocalBlockIndex::new(i))).collect();

    // Host 0 inserts 100 blocks.
    for i in 0..100u32 {
        let hash = test_hash(i);
        let addr = GlobalBlockAddr { region_id: 0, offset: i as u64 * 32768 };
        indexes[0].insert_local(hash, addr, i);
    }

    // Simulate one gossip round: host 0 → hosts 1,2.
    let snapshot = indexes[0].snapshot_for_gossip();
    for (hash, addr, writer, epoch) in &snapshot {
        indexes[1].insert_gossip(*hash, *addr, *writer, *epoch);
        indexes[2].insert_gossip(*hash, *addr, *writer, *epoch);
    }

    // All 3 hosts should have the same 100 entries.
    for idx in &indexes {
        assert_eq!(idx.map.read().len(), 100);
    }
}
```

**T11b.2 — Allocator failover and rebuild:**
```rust
#[test]
fn test_allocator_rebuild_from_headers() {
    let region = mmap_test_region(100 * (32768 + 64));
    let heap = SymmetricHeap::new(region, 0, 32768 + 64, 0);

    // Write 50 blocks with valid headers.
    for i in 0..50u32 {
        let addr = GlobalBlockAddr { region_id: 0, offset: i as u64 * (32768 + 64) };
        let hash = test_hash(i);
        heap.write_block(addr, hash, &vec![i as u8; 32768], i);
    }

    // Simulate standby rebuild.
    let standby = AllocatorStandby::new(0, region.as_ptr(), 100, 32768 + 64);
    let manager = standby.rebuild();

    // 50 allocated, 50 free.
    assert_eq!(manager.free_blocks(), 50);
}
```

**T11b.3 — Allocator throughput (ensure it's not a bottleneck):**
```rust
#[bench]
fn bench_allocator_throughput(b: &mut Bencher) {
    let mut mgr = CxlMemoryManager::new(0, 0, 32768 + 64, 1_000_000);
    b.iter(|| {
        let addrs = mgr.allocate(10);
        mgr.free(&addrs.iter().map(|(a, _)| *a).collect::<Vec<_>>());
    });
    // Expect > 100K alloc+free pairs/sec.
}
```

---

### Phase 11c — CXL Pool Manager + Connector + Scrubber (1.5 weeks)

#### Deliverables

- [ ] `CxlKvConnector` implementing v5's `KvConnector` trait (checksummed)
- [ ] `CxlAwarePoolManager` updated: uses local index, centralized allocator, checksum on all paths
- [ ] `CxlScrubber` — background integrity checker
- [ ] Quarantine tracking + alerting

#### Test Plan

**T11c.1 — End-to-end: promote with corruption recovery:**
```rust
#[test]
fn test_promote_corrupt_block_triggers_recompute() {
    let mut pool = CxlAwarePoolManager::new_test_v7(gpu_blocks: 10, cxl_blocks: 1000);
    let tokens = vec![1u32; 16];

    // Write a block to GPU, evict to CXL.
    pool.l1_gpu.insert_prefix(&tokens, &[pool.l1_gpu.allocate_one()]);
    pool.l1_gpu.gpu_pool.write_test_data(0, &[0xAB; BLOCK_SIZE_BYTES]);
    pool.evict_gpu_to_cxl(1);

    // Inject corruption in CXL.
    let hash = compute_block_hash(&tokens);
    let addr = pool.local_index.lookup(&hash).unwrap();
    unsafe { *pool.pgas_heap.as_ref().unwrap().addr_to_ptr(addr).add(64 + 100) ^= 0x01; }

    // Promote should detect corruption and return CorruptionDetected.
    let result = pool.promote_from_cxl(addr, &hash);
    assert!(matches!(result, PromoteResult::CorruptionDetected));

    // Block should be removed from local index.
    assert!(pool.local_index.lookup(&hash).is_none());
}
```

**T11c.2 — Scrubber finds latent corruption:**
```rust
#[test]
fn test_scrubber_detects_latent_corruption() {
    let region = mmap_test_region(100 * BLOCK_STRIDE);
    let heap = SymmetricHeap::new(region, 0, BLOCK_STRIDE, 0);

    // Write 50 valid blocks.
    for i in 0..50u32 {
        let addr = GlobalBlockAddr { region_id: 0, offset: i as u64 * BLOCK_STRIDE as u64 };
        heap.write_block(addr, test_hash(i), &vec![i as u8; 32768], i);
    }

    // Corrupt block 25.
    unsafe { *region.as_mut_ptr().add(25 * BLOCK_STRIDE + 64 + 500) ^= 0xFF; }

    let scrubber = CxlScrubber::new(region.as_ptr(), BLOCK_STRIDE, 100);
    let found = scrubber.scrub_pass();
    assert_eq!(found, 1);
    assert!(scrubber.quarantined.lock().contains(&25));
}
```

---

### Phase 11d — Shared Weight Pool Integrity + Routing (1 week)

Same as v6 Phase 11c–11d, plus:
- [ ] Weight checksum computation at load time
- [ ] `periodic_verify` background task
- [ ] `CxlAwareRouter` updated to use `LocalBlockIndex`

---

## 17. Performance Targets

### v7 vs. v6: Integrity Cost

| Metric | v6 (No Checksum) | v7 (Checksummed) | Delta |
|---|---|---|---|
| Block write (GPU → CXL) | ~3μs | ~3μs (checksum overlapped with DMA) | **~0** |
| Block read (CXL → GPU) | ~3μs | ~4μs (1μs checksum + 3μs DMA) | **+1μs** (+25%) |
| 100-block promotion | ~330μs | ~430μs | **+100μs** (+30%) |
| Scrubber bandwidth | 0 | ~10 GB/s (paced, <10% of CXL link) | Background |
| Gossip bandwidth | 0 | ~1 MB/s per host (UDP, negligible) | Background |
| Allocator RPC latency | 0 (lock-free) | ~5–10μs per alloc RPC | **+5μs per allocation** |

### v7 vs. v5: Net Performance Gain (What Matters)

| Metric | v5 (Copy-Based) | v7 (CXL + Checksums) | Net Gain |
|---|---|---|---|
| Block promotion latency | ~15μs | ~4μs | **3.75× faster** (was 5× in v6) |
| Cross-instance KV fetch | ~50μs | ~4μs + ~5μs gossip lag = ~9μs worst case | **5.5× faster** |
| 131K-token PD transfer (8192 blocks) | ~140–510ms | ~57ms (8192 × 7μs) | **2.5–9× faster** |
| Silent corruption probability | Same as v7 (no detection) | **~0** (checksum + scrubber) | **Safety** |

### Corruption Blast Radius Comparison

| Scenario | v6 Blast Radius | v7 Blast Radius |
|---|---|---|
| 1 bit flip in KV block data | **Silent wrong output** | 1 block checksum fail → recompute (3ms) |
| 1 bit flip in free list head | **Cluster allocator corrupted** | Impossible (local bitmap) |
| 1 bit flip in index entry | **All hosts read wrong pointer** | 1 host, corrected by gossip in ~10ms |
| 1 bit flip in heartbeat | **False failover** | Impossible (UDP side-channel) |

---

## 18. Mapping to Linqu Hierarchy

Same as v6 §15, with one refinement:

| Linqu Level | Control Plane | Data Plane |
|---|---|---|
| **Level 2** (Chip) | GPU radix tree (local) | GPU HBM (KV blocks) |
| **Level 3** (Host) | `LocalBlockIndex` + `CxlMemoryManager` (local DRAM) | CXL-attached DRAM (checksummed blocks) |
| **Level 4** (Cluster-level-0) | `IndexGossip` (UDP, ~5ms propagation) | CXL-pooled memory (checksummed blocks) |
| **Level 5** (Cluster-level-1) | `IndexGossip` (UDP, ~10ms propagation) | CXL fabric memory (checksummed blocks) |
| **Level 6** (Cluster-level-2) | v5 `RemoteStoreConnector` (RPC) | RDMA/TCP (v5 fallback) |

The control/data separation maps naturally to the Linqu hierarchy: **control structures propagate via the network at each level's latency** (gossip within a switch domain, RPC across fabrics), while **data stays in place and is accessed via CXL load/store** with per-access integrity verification.

---

## Key Design Decisions — Rationale

**Local control plane, shared data plane:** The fundamental insight is that control structures (index, allocator, refcounts) are small but their corruption cascades cluster-wide, while data (KV blocks) is large but its corruption is isolated to one block. Putting control in local ECC-protected memory and data in checksummed CXL memory minimizes blast radius per bit of corruption.

**xxHash64 for block checksums (not CRC32, not SHA-256):** xxHash64 at ~30 GB/s is fast enough to checksum every read without measurable latency impact. CRC32C is HW-accelerated but only 32 bits (2⁻³² collision). SHA-256 is 64 bits but ~3 GB/s (10× slower). xxHash64's 2⁻⁶⁴ collision probability is sufficient for corruption detection (not security).

**Centralized allocator over lock-free:** The allocation rate (~1K blocks/s per host) is 100× below what a centralized allocator can handle. Lock-free atomics in shared memory optimize for a bottleneck that doesn't exist, at the cost of catastrophic failure modes. The simplest correct solution wins.

**Gossip over consensus:** We don't need strong consistency for the block index. A stale miss just means a cache miss (recompute), not a wrong answer. Gossip is simpler, has no leader election, and tolerates partial failures gracefully.

**Quarantine over immediate free:** A corrupted block may indicate a hardware problem. Freeing it immediately would allow re-allocation at the same address, which might corrupt the new block too. Quarantining preserves the evidence and prevents repeated failures at the same location.

**Scrubber as defense-in-depth:** The per-read checksum catches corruption at read time. The scrubber catches it proactively, before a request needs the block. This converts surprise recomputation (user-visible latency spike) into background cleanup.

---

## Defense in Depth Summary

```
Layer 1: Hardware ECC (DRAM)       → corrects most single-bit errors
Layer 2: CXL link CRC (LCRC)      → detects most link errors, retries transparently
Layer 3: CXL IDE (optional)        → end-to-end encryption + integrity
Layer 4: Block magic number        → detects uninitialized / gross corruption (cost: ~0)
Layer 5: Block data checksum       → detects bit-level payload corruption (cost: ~1μs/block)
Layer 6: Hash match verification   → detects index corruption (wrong block) (cost: ~0)
Layer 7: Local control plane       → isolates index/allocator faults to one host
Layer 8: Background scrubber       → catches latent corruption before read
Layer 9: Quarantine + alerting     → prevents re-use of failing memory regions
Layer 10: Recomputation            → KV cache is deterministic — corruption = cache miss, not data loss
```

**The cost of all 10 layers: ~1μs per block read + ~10 GB/s background scrub bandwidth.
The benefit: a single bit flip never causes a wrong model output or a cluster-wide failure.**

---

*v7 inherits v6's CXL performance architecture and adds corruption resilience. The key architectural change — local control plane, checksummed shared data plane — trades ~25% of v6's block-promotion speedup for the guarantee that bit-level CXL failures are detected and contained. On reliable CXL hardware, v7 performs within 1μs/block of v6. On unreliable hardware, v7 is the difference between a cache miss and a wrong answer.*
