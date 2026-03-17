# Rust LLM Inference Server — Design & Implementation Plan (v6)

> **Target:** 1–N machines with CXL-interconnected memory pools, 8–16 GPU/NPU cards per machine, hardware-limit inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI. **Exploit CXL/PGAS hardware to eliminate data copies and unify the memory hierarchy.**
> **New in v6:** CXL memory pooling, PGAS (OpenSHMEM-style) global address space for KV blocks, zero-copy cross-instance cache sharing, NUMA-aware CXL tiering, shared-memory weight deduplication, and CXL-distance-aware routing. Designed to push CXL 3.0 hardware to its limits.

---

## Table of Contents

1. [Critique of v5 and What v6 Fixes](#1-critique-of-v5-and-what-v6-fixes)
2. [CXL Memory Architecture Overview](#2-cxl-memory-architecture-overview)
3. [PGAS Memory Model for KV Blocks](#3-pgas-memory-model-for-kv-blocks)
4. [Revised Memory Hierarchy: CXL-Aware Tiering](#4-revised-memory-hierarchy-cxl-aware-tiering)
5. [CXL KV Connector: Zero-Copy Interface](#5-cxl-kv-connector-zero-copy-interface)
6. [Shared Weight Pool](#6-shared-weight-pool)
7. [CXL-Distance-Aware Routing](#7-cxl-distance-aware-routing)
8. [PD Disaggregation over CXL Shared Memory](#8-pd-disaggregation-over-cxl-shared-memory)
9. [Fault Tolerance in Shared Memory](#9-fault-tolerance-in-shared-memory)
10. [Integration with v4/v5 Components](#10-integration-with-v4v5-components)
11. [Configuration](#11-configuration)
12. [Updated Directory Layout](#12-updated-directory-layout)
13. [Implementation Phases](#13-implementation-phases)
14. [Performance Targets](#14-performance-targets)
15. [Mapping to Linqu Hierarchy](#15-mapping-to-linqu-hierarchy)

---

## 1. Critique of v5 and What v6 Fixes

### Critique

v5 adds a solid multi-tier KV cache pool with cross-instance sharing via a remote store. But it was designed for commodity hardware where memory is **node-local** and cross-node data movement is **explicit and expensive**. On CXL-equipped hardware, v5 leaves enormous performance on the table:

| # | v5 Problem | Why It Matters on CXL Hardware |
|---|---|---|
| 1 | **Every cross-tier transfer is a copy.** `KvConnector::fetch()` returns `Vec<u8>` — owned, copied data. Even `RemoteStoreConnector` copies bytes through an RPC. | CXL 3.0 memory pooling enables **load/store access** to remote memory at ~150–300ns. Copying a 32KB KV block through RPC takes ~5–50μs. The copy overhead is 10–100× the hardware capability. |
| 2 | **L2 (local CPU DRAM) and L4 (remote store) are separate tiers.** The pool manager treats them as fundamentally different: L2 is a local `HashMap`, L4 is a network service. | With CXL fabric memory, there is no meaningful distinction between "my DRAM" and "that node's DRAM" — it's all accessible via the same address space, just at different latencies (local CXL ~150ns, fabric CXL ~300ns, switched CXL ~500ns). The artificial tier boundary forces unnecessary data movement. |
| 3 | **Transfer engine is message-based.** Even RDMA is a two-sided or one-sided verb with registration, pinning, and completion notification overhead. | CXL gives you **cache-coherent load/store** — the CPU's memory controller handles everything. No registration, no pinning, no completion events. The transfer engine is solving a problem that CXL hardware eliminates. |
| 4 | **Model weights are replicated per instance.** Each instance loads and stores its own copy of model weights in local GPU/CPU memory. | With CXL memory pooling, a single copy of model weights in shared memory can serve all instances on the CXL fabric. For a 70B model at FP16 (~140GB), this saves 140GB × (N-1) across N instances. |
| 5 | **Cache-aware routing optimizes for GPU L1 hits only.** The router routes requests to the instance whose GPU has the prefix cached. If no GPU has it, the request goes to the least-loaded instance and recomputes from scratch (or fetches from the remote store). | With CXL shared memory, every block stored by any instance is accessible by every other instance at memory-read latency. Routing should optimize for **CXL topology distance** (which NUMA node/CXL switch is the block closest to?), not binary "have it / don't have it." |
| 6 | **No exploitation of one-sided operations.** v5's `RemoteStoreConnector` requires the remote store service to be running and responsive — it's a request/response protocol. | OpenSHMEM-style one-sided operations (`shmem_get`, `shmem_put`) let any PE read/write any other PE's memory without the remote CPU being involved. This eliminates the store service as a bottleneck and enables true peer-to-peer KV sharing. |
| 7 | **Prefetch is reactive.** The background worker promotes blocks only after the scheduler discovers a lower-tier hit. The promotion must complete before prefill can start. | With CXL, "promotion" from shared memory to local memory is just a cache-line fetch by the memory controller. Hardware prefetching can speculatively pull data ahead of demand. The software prefetch pipeline is solving for 5–50μs transfers, not 300ns CXL loads. |

### What v6 Adds

v6 retains all of v4 and v5 and adds:

- **§2–§3**: CXL memory topology discovery and a PGAS (Partitioned Global Address Space) model for KV blocks
- **§4**: Revised memory hierarchy that replaces v5's discrete tiers with a **continuous latency-distance model** over CXL fabric
- **§5**: New `CxlKvConnector` that exposes **global pointers** instead of owned byte buffers — zero-copy block access
- **§6**: Shared weight pool — a single copy of model weights in CXL shared memory, accessible by all instances
- **§7**: CXL-distance-aware routing that accounts for memory topology, not just binary cache presence
- **§8**: PD disaggregation over CXL shared memory — prefill writes KV directly to a global region; decode reads it with no transfer step
- **§9**: Fault tolerance model for shared memory (what happens when a node fails that owns CXL-pooled memory)

### What v6 Does NOT Change

Everything in v4 §2–§22 and v5 §2–§9 remains as fallback. v6 is a hardware-accelerated fast path layered on top. When CXL is not available, the system degrades to v5 behavior (copy-based tiers, message-based transfers).

---

## 2. CXL Memory Architecture Overview

### 2.1 CXL Protocol Layers Relevant to Us

| CXL Sub-Protocol | What It Does | How We Use It |
|---|---|---|
| **CXL.mem** | Host accesses CXL-attached memory via load/store. Memory controller handles coherence. | Attach large CXL DRAM expanders as additional NUMA nodes. KV blocks live here. |
| **CXL.cache** | Device caches host memory with hardware coherence. | GPU/NPU with CXL.cache can directly access KV blocks in CXL-attached memory — potential future zero-copy path from shared pool to device. |
| **CXL 3.0 Fabric** | Multi-host memory pooling via CXL switches. Any host on the fabric can map any memory region. | The KV cache pool becomes a **fabric-wide shared region**. Any instance on any host reads/writes KV blocks via its local memory controller. |

### 2.2 CXL Memory Topology

A CXL-equipped cluster has a richer topology than the flat "local vs. remote" model in v5:

```
                         ┌──────────────────────┐
                         │   CXL Fabric Switch   │
                         └──────┬───────┬───────┘
                                │       │
                    ┌───────────┘       └───────────┐
                    │                               │
            ┌───────┴──────┐                ┌───────┴──────┐
            │  CXL Switch 0 │                │  CXL Switch 1 │
            └──┬─────┬─────┘                └──┬─────┬─────┘
               │     │                         │     │
         ┌─────┘     └─────┐             ┌─────┘     └─────┐
         │                 │             │                 │
    ┌────┴────┐      ┌────┴────┐   ┌────┴────┐      ┌────┴────┐
    │ Host 0  │      │ Host 1  │   │ Host 2  │      │ Host 3  │
    │ CPU+GPU │      │ CPU+GPU │   │ CPU+GPU │      │ CPU+GPU │
    │ Local   │      │ Local   │   │ Local   │      │ Local   │
    │ DRAM    │      │ DRAM    │   │ DRAM    │      │ DRAM    │
    └────┬────┘      └────┬────┘   └────┬────┘      └────┬────┘
         │                 │             │                 │
    ┌────┴────┐      ┌────┴────┐   ┌────┴────┐      ┌────┴────┐
    │CXL DRAM │      │CXL DRAM │   │CXL DRAM │      │CXL DRAM │
    │Expander │      │Expander │   │Expander │      │Expander │
    └─────────┘      └─────────┘   └─────────┘      └─────────┘
         │                 │             │                 │
         └────────┬────────┘             └────────┬────────┘
              ┌───┴────┐                     ┌───┴────┐
              │ Pooled  │                     │ Pooled  │
              │Memory 0 │                     │Memory 1 │
              │(Shared) │                     │(Shared) │
              └─────────┘                     └─────────┘
```

### 2.3 Latency Tiers (Continuous, Not Discrete)

| Access Pattern | Approximate Latency | v5 Tier Equivalent |
|---|---|---|
| GPU HBM (local) | ~1–2ns per element (bandwidth-limited) | L1 |
| CPU ↔ Local DRAM | ~80ns | L2 (CpuDramConnector) |
| CPU ↔ CXL-attached DRAM (same host) | ~150–200ns | Not modeled in v5 |
| CPU ↔ CXL-pooled DRAM (same switch) | ~250–350ns | Between L2 and L4 |
| CPU ↔ CXL-pooled DRAM (cross switch) | ~400–600ns | Comparable to L4 fetch |
| CPU ↔ Remote DRAM (RDMA) | ~1–5μs | L4 (RemoteStoreConnector) |
| CPU ↔ NVMe SSD | ~10–100μs | L3 (SsdConnector) |

**Key insight:** CXL inserts a new latency tier (~150–600ns) between local DRAM (~80ns) and RDMA (~1μs+). This tier is large (terabytes across a fabric), directly addressable (load/store), and cache-coherent. It is the ideal home for the KV cache pool.

### 2.4 Topology Discovery

At startup, the server must discover the CXL topology to make routing and placement decisions:

```rust
/// Represents the CXL memory topology visible to this host.
pub struct CxlTopology {
    /// This host's ID in the CXL fabric.
    pub local_host_id: u32,
    /// NUMA nodes on this host, with their type and capacity.
    pub numa_nodes: Vec<NumaNodeInfo>,
    /// Distance matrix: numa_nodes[i] to numa_nodes[j] in nanoseconds.
    /// Includes remote NUMA nodes exposed via CXL fabric.
    pub distance_ns: Vec<Vec<u32>>,
    /// Shared memory regions (CXL 3.0 pooled memory).
    pub shared_regions: Vec<SharedRegionInfo>,
}

#[derive(Clone, Debug)]
pub struct NumaNodeInfo {
    pub id: u32,
    pub kind: NumaKind,
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    /// Which host owns this NUMA node (local_host_id for local nodes).
    pub owner_host_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumaKind {
    /// Standard DDR on the local CPU.
    LocalDram,
    /// CXL-attached memory expander on the local host.
    CxlLocalExpander,
    /// CXL-pooled memory accessible via fabric (may be on another host).
    CxlPooled,
    /// Remote host's memory, exposed via CXL 3.0 multi-host sharing.
    CxlRemote,
}

#[derive(Clone, Debug)]
pub struct SharedRegionInfo {
    /// Fabric-wide region ID.
    pub region_id: u64,
    /// Base virtual address (after mmap).
    pub base_addr: *mut u8,
    /// Size in bytes.
    pub size: u64,
    /// Which hosts can access this region.
    pub accessible_by: Vec<u32>,
    /// NUMA distance from each accessible host.
    pub distance_ns_per_host: Vec<(u32, u32)>,
}

impl CxlTopology {
    /// Discover CXL topology from Linux sysfs and ACPI CEDT/CFMWS tables.
    pub fn discover() -> Result<Self, CxlError> {
        // 1. Read /sys/bus/cxl/devices/ for CXL devices
        // 2. Parse ACPI CEDT (CXL Early Discovery Table) for fabric topology
        // 3. Read /sys/devices/system/node/ for NUMA distance matrix
        // 4. Identify CXL-backed NUMA nodes via HMAT (Heterogeneous Memory Attribute Table)
        // 5. Map shared regions via CXL fabric manager API
        todo!()
    }

    /// Get the latency (in ns) from a local NUMA node to a target NUMA node.
    pub fn distance_ns(&self, from_numa: u32, to_numa: u32) -> u32 {
        self.distance_ns[from_numa as usize][to_numa as usize]
    }

    /// Find the best (lowest-latency) shared region for the local host.
    pub fn best_shared_region(&self) -> Option<&SharedRegionInfo> {
        self.shared_regions.iter().min_by_key(|r| {
            r.distance_ns_per_host.iter()
                .find(|(h, _)| *h == self.local_host_id)
                .map(|(_, d)| *d)
                .unwrap_or(u32::MAX)
        })
    }
}
```

---

## 3. PGAS Memory Model for KV Blocks

### 3.1 The Concept

Instead of v5's copy-based `KvConnector` where `fetch()` returns an owned `Vec<u8>`, v6 introduces a **Partitioned Global Address Space** (PGAS) model inspired by OpenSHMEM. Every KV block in the system has a **global address** — a (region_id, offset) pair that any node on the CXL fabric can use to access the block's data directly via load/store.

```
v5 model (copy-based):
  Instance 0 computes KV block → serialize → RPC to store → deserialize → store
  Instance 1 needs block → RPC to store → serialize → transfer → deserialize → memcpy to GPU

v6 model (PGAS):
  Instance 0 computes KV block → DMA from GPU to shared CXL region at global_addr
  Instance 1 needs block → DMA from global_addr to GPU
  (No serialization, no RPC, no store service)
```

### 3.2 Global Block Address

```rust
/// A globally-unique address for a KV block in the PGAS.
/// Any node on the CXL fabric can access this block via its global address.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct GlobalBlockAddr {
    /// Which shared memory region contains this block.
    pub region_id: u64,
    /// Byte offset within the region.
    pub offset: u64,
}

impl GlobalBlockAddr {
    /// Convert to a raw pointer (valid on any host that has mapped this region).
    pub unsafe fn as_ptr(&self, topology: &CxlTopology) -> *const u8 {
        let region = topology.shared_regions.iter()
            .find(|r| r.region_id == self.region_id)
            .expect("region not mapped");
        region.base_addr.add(self.offset as usize)
    }

    /// Convert to a mutable raw pointer.
    pub unsafe fn as_mut_ptr(&self, topology: &CxlTopology) -> *mut u8 {
        let region = topology.shared_regions.iter()
            .find(|r| r.region_id == self.region_id)
            .expect("region not mapped");
        region.base_addr.add(self.offset as usize) as *mut u8
    }
}
```

### 3.3 PGAS Block Allocator

The shared CXL memory region is managed as a block pool, similar to v4's GPU `BlockPool` but operating on global memory:

```rust
/// Manages allocation of KV blocks in a CXL shared memory region.
/// This is the PGAS equivalent of v4's BlockPool, but for shared memory.
pub struct PgasBlockPool {
    /// The shared region this pool manages.
    region_id: u64,
    /// Base address in this process's virtual address space.
    base_addr: *mut u8,
    /// Size of each KV block in bytes.
    block_size_bytes: usize,
    /// Total number of blocks in the pool.
    total_blocks: u32,
    /// Free list: lock-free stack of available block indices.
    /// Uses atomic operations so multiple hosts can allocate concurrently.
    free_list: AtomicFreeList,
    /// Block index: hash → GlobalBlockAddr.
    /// Stored IN the shared region itself (at a reserved prefix) so all
    /// hosts see the same index without synchronization.
    index_base: *mut AtomicBlockIndexEntry,
    index_capacity: usize,
}

/// A single entry in the shared block index (stored in CXL shared memory).
/// Uses atomic operations for lock-free concurrent access from multiple hosts.
#[repr(C, align(64))]  // cache-line aligned to avoid false sharing
pub struct AtomicBlockIndexEntry {
    /// Block hash (written once, then immutable).
    pub hash: BlockHash,
    /// Global block address (0 = empty slot).
    pub offset: AtomicU64,
    /// Reference count (how many hosts are actively using this block).
    pub refcount: AtomicU32,
    /// Owning host ID (which host computed this block).
    pub owner_host: AtomicU32,
    /// Epoch: incremented on each reuse, for ABA detection.
    pub epoch: AtomicU64,
}

impl PgasBlockPool {
    /// Create a new PGAS block pool over a CXL shared memory region.
    /// The region must already be mapped (via mmap on the CXL device file).
    pub unsafe fn new(
        region_id: u64,
        base_addr: *mut u8,
        region_size: u64,
        block_size_bytes: usize,
    ) -> Self {
        // Reserve the first portion of the region for the block index.
        // The rest is the data area.
        let index_capacity = (region_size as usize / block_size_bytes).min(1 << 20);
        let index_bytes = index_capacity * std::mem::size_of::<AtomicBlockIndexEntry>();
        let index_base = base_addr as *mut AtomicBlockIndexEntry;

        let data_base = base_addr.add(index_bytes);
        let data_size = region_size as usize - index_bytes;
        let total_blocks = (data_size / block_size_bytes) as u32;

        // Initialize free list with all block indices
        let free_list = AtomicFreeList::new(total_blocks);

        Self {
            region_id,
            base_addr,
            block_size_bytes,
            total_blocks,
            free_list,
            index_base,
            index_capacity,
        }
    }

    /// Allocate a block and return its global address.
    pub fn allocate(&self) -> Option<GlobalBlockAddr> {
        let block_idx = self.free_list.pop()?;
        let offset = self.data_offset(block_idx);
        Some(GlobalBlockAddr {
            region_id: self.region_id,
            offset,
        })
    }

    /// Free a block back to the pool.
    pub fn free(&self, addr: GlobalBlockAddr) {
        let block_idx = self.addr_to_index(addr);
        self.free_list.push(block_idx);
    }

    /// Get a raw pointer to block data (valid on any host with the region mapped).
    pub unsafe fn block_ptr(&self, addr: GlobalBlockAddr) -> *mut u8 {
        self.base_addr.add(addr.offset as usize)
    }

    /// Look up a block by hash in the shared index.
    /// Returns the global address if found.
    pub fn lookup(&self, hash: &BlockHash) -> Option<GlobalBlockAddr> {
        // Hash-based probe into the shared index (open addressing).
        let mut slot = self.hash_slot(hash);
        for _ in 0..self.index_capacity {
            let entry = unsafe { &*self.index_base.add(slot) };
            let offset = entry.offset.load(Ordering::Acquire);
            if offset == 0 {
                return None; // empty slot — not found
            }
            if entry.hash == *hash {
                // Found. Increment refcount to prevent eviction.
                entry.refcount.fetch_add(1, Ordering::AcqRel);
                return Some(GlobalBlockAddr {
                    region_id: self.region_id,
                    offset,
                });
            }
            slot = (slot + 1) % self.index_capacity;
        }
        None
    }

    /// Insert a block into the shared index.
    /// Called after computing KV data and writing it to the block address.
    pub fn insert(&self, hash: BlockHash, addr: GlobalBlockAddr, host_id: u32) {
        let mut slot = self.hash_slot(&hash);
        for _ in 0..self.index_capacity {
            let entry = unsafe { &*self.index_base.add(slot) };
            // Try to claim an empty slot.
            if entry.offset.compare_exchange(
                0, addr.offset, Ordering::AcqRel, Ordering::Relaxed
            ).is_ok() {
                // We claimed the slot. Write metadata.
                // Safety: hash is a plain [u8; 32] — no atomics needed
                // because offset=0→non-zero is the publishing barrier.
                unsafe {
                    let entry_mut = &mut *(self.index_base.add(slot) as *mut AtomicBlockIndexEntry);
                    std::ptr::write(&mut entry_mut.hash, hash);
                }
                entry.owner_host.store(host_id, Ordering::Release);
                entry.refcount.store(1, Ordering::Release);
                entry.epoch.fetch_add(1, Ordering::Release);
                return;
            }
            // Slot taken — check if it's the same hash (duplicate insert).
            if entry.hash == hash {
                return; // already inserted by another host
            }
            slot = (slot + 1) % self.index_capacity;
        }
        // Index full — need to evict. See §4.3.
    }

    fn hash_slot(&self, hash: &BlockHash) -> usize {
        let h = u64::from_le_bytes(hash[..8].try_into().unwrap());
        (h as usize) % self.index_capacity
    }

    fn data_offset(&self, block_idx: u32) -> u64 {
        let index_bytes = self.index_capacity * std::mem::size_of::<AtomicBlockIndexEntry>();
        index_bytes as u64 + (block_idx as u64 * self.block_size_bytes as u64)
    }

    fn addr_to_index(&self, addr: GlobalBlockAddr) -> u32 {
        let index_bytes = self.index_capacity * std::mem::size_of::<AtomicBlockIndexEntry>();
        ((addr.offset - index_bytes as u64) / self.block_size_bytes as u64) as u32
    }
}
```

### 3.4 Lock-Free Free List (for Multi-Host Concurrent Allocation)

The free list must be safe for concurrent access from multiple hosts over CXL. We use a Treiber stack with CAS on a shared atomic:

```rust
/// Lock-free free list stored in CXL shared memory.
/// Uses a Treiber stack with atomic CAS.
pub struct AtomicFreeList {
    /// Head of the free list (index into a next-pointer array).
    head: *mut AtomicU64,  // stored in shared memory
    /// Next pointers: next[i] = next free block after block i.
    /// Stored in shared memory. Uses packed (index, tag) for ABA prevention.
    next: *mut AtomicU64,
    capacity: u32,
}

impl AtomicFreeList {
    /// Pack index + tag into a single u64 for ABA prevention.
    fn pack(index: u32, tag: u32) -> u64 {
        ((tag as u64) << 32) | (index as u64)
    }

    fn unpack(val: u64) -> (u32, u32) {
        ((val & 0xFFFF_FFFF) as u32, (val >> 32) as u32)
    }

    pub fn pop(&self) -> Option<u32> {
        loop {
            let head_val = unsafe { (*self.head).load(Ordering::Acquire) };
            if head_val == u64::MAX {
                return None; // empty
            }
            let (index, tag) = Self::unpack(head_val);
            let next_val = unsafe { (*self.next.add(index as usize)).load(Ordering::Acquire) };
            let new_head = if Self::unpack(next_val).0 == u32::MAX {
                u64::MAX
            } else {
                Self::pack(Self::unpack(next_val).0, tag.wrapping_add(1))
            };
            if unsafe { (*self.head).compare_exchange_weak(
                head_val, new_head, Ordering::AcqRel, Ordering::Relaxed
            ) }.is_ok() {
                return Some(index);
            }
            // CAS failed — another host took the head. Retry.
            std::hint::spin_loop();
        }
    }

    pub fn push(&self, index: u32) {
        loop {
            let head_val = unsafe { (*self.head).load(Ordering::Acquire) };
            let (_, tag) = if head_val == u64::MAX {
                (u32::MAX, 0u32)
            } else {
                Self::unpack(head_val)
            };
            unsafe { (*self.next.add(index as usize)).store(head_val, Ordering::Release) };
            let new_head = Self::pack(index, tag.wrapping_add(1));
            if unsafe { (*self.head).compare_exchange_weak(
                head_val, new_head, Ordering::AcqRel, Ordering::Relaxed
            ) }.is_ok() {
                return;
            }
            std::hint::spin_loop();
        }
    }
}
```

### 3.5 OpenSHMEM-Style Symmetric Operations

For interoperability with OpenSHMEM runtimes (SOS, OSHMPI) and to provide a familiar PGAS API, v6 wraps the CXL shared memory operations in OpenSHMEM-style primitives:

```rust
/// OpenSHMEM-inspired symmetric operations over CXL shared memory.
/// "Symmetric" means the same virtual address range is mapped on all PEs (hosts).
pub struct SymmetricHeap {
    /// The CXL shared region backing this heap.
    pool: Arc<PgasBlockPool>,
    /// This PE's (Processing Element) ID = host ID.
    my_pe: u32,
    /// Total PEs in the fabric.
    num_pes: u32,
    /// CXL topology for distance-aware operations.
    topology: Arc<CxlTopology>,
}

impl SymmetricHeap {
    /// shmem_malloc: Allocate a KV block from the shared pool.
    /// Returns a global address visible to all PEs.
    pub fn shmem_malloc_block(&self) -> Option<GlobalBlockAddr> {
        self.pool.allocate()
    }

    /// shmem_free: Return a KV block to the shared pool.
    pub fn shmem_free_block(&self, addr: GlobalBlockAddr) {
        self.pool.free(addr);
    }

    /// shmem_put: Write KV block data from local buffer to a global address.
    /// One-sided: the remote PE does not need to participate.
    /// On CXL, this is a memcpy to a CXL-mapped address — the hardware handles it.
    pub unsafe fn shmem_put_block(&self, dest: GlobalBlockAddr, src: &[u8]) {
        let dest_ptr = self.pool.block_ptr(dest);
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest_ptr, src.len());
        // CXL.mem ensures the write is visible to all hosts after the store completes.
        // A fence ensures ordering on weakly-ordered architectures.
        std::sync::atomic::fence(Ordering::Release);
    }

    /// shmem_get: Read KV block data from a global address into a local buffer.
    /// One-sided: the remote PE does not need to participate.
    pub unsafe fn shmem_get_block(&self, src: GlobalBlockAddr, dest: &mut [u8]) {
        let src_ptr = self.pool.block_ptr(src);
        std::sync::atomic::fence(Ordering::Acquire);
        std::ptr::copy_nonoverlapping(src_ptr, dest.as_mut_ptr(), dest.len());
    }

    /// shmem_ptr: Get a raw pointer to a global block.
    /// On CXL, this is always possible (all shared memory is mapped).
    /// On non-CXL fallback, this returns None and the caller must use get/put.
    pub unsafe fn shmem_ptr(&self, addr: GlobalBlockAddr) -> Option<*const u8> {
        Some(self.pool.block_ptr(addr) as *const u8)
    }

    /// shmem_quiet: Ensure all outstanding puts from this PE are visible to all PEs.
    pub fn shmem_quiet(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        // On CXL, SeqCst fence ensures all CXL stores are globally visible.
    }

    /// shmem_barrier_all: Barrier across all PEs.
    /// Uses a shared atomic counter in CXL memory.
    pub fn shmem_barrier_all(&self) {
        // Implementation: shared atomic counter + spin-wait.
        // Details in §9 (fault tolerance).
        todo!()
    }

    /// Compute the CXL distance (in ns) from this PE to the NUMA node
    /// hosting a given block address.
    pub fn distance_to(&self, addr: GlobalBlockAddr) -> u32 {
        let region = self.topology.shared_regions.iter()
            .find(|r| r.region_id == addr.region_id)
            .expect("region not mapped");
        region.distance_ns_per_host.iter()
            .find(|(h, _)| *h == self.my_pe)
            .map(|(_, d)| *d)
            .unwrap_or(u32::MAX)
    }
}
```

---

## 4. Revised Memory Hierarchy: CXL-Aware Tiering

### 4.1 From Discrete Tiers to Distance-Based Placement

v5's tier model (L1=GPU, L2=CPU, L3=SSD, L4=Remote) becomes a **continuous latency space**:

```
v5 tiers:           L1 ──── L2 ──── L3 ──── L4
                   (GPU)  (CPU)   (SSD)  (Remote)
                   [gap]  [gap]   [gap]   [gap]

v6 latency space:  ────────────────────────────────────────────→ latency
                   GPU    Local   CXL     CXL       CXL         RDMA    SSD
                   HBM    DRAM    Local   Same-Sw   Cross-Sw    Remote
                   1ns    80ns    150ns   300ns     500ns       2μs     50μs
                   ▲      ▲       ▲       ▲         ▲           ▲       ▲
                   always  always  CXL     CXL 3.0  CXL 3.0    legacy  optional
                   present present expander pooled   fabric     fallback
```

The pool manager no longer hard-codes tier boundaries. Instead, it uses the CXL distance matrix to make placement decisions:

```rust
/// Replaces v5's tiered pool manager with a distance-aware model.
pub struct CxlAwarePoolManager {
    /// L1: GPU radix tree (unchanged from v4).
    l1_gpu: RadixKvManager,

    /// CXL PGAS pool (replaces v5's L2 CPU + L4 Remote).
    /// All KV blocks not in GPU live here, globally accessible.
    pgas_pool: Option<Arc<SymmetricHeap>>,

    /// Legacy v5 tiers (used when CXL is not available).
    legacy_l2_cpu: Option<Box<dyn KvConnector>>,
    legacy_l4_remote: Option<Box<dyn KvConnector>>,

    /// Local SSD (still useful as a cold tier, even with CXL).
    ssd_tier: Option<Box<dyn KvConnector>>,

    /// CXL topology for distance-aware decisions.
    topology: Option<Arc<CxlTopology>>,

    /// Background prefetch (only for non-CXL tiers; CXL access is fast enough inline).
    prefetch_tx: mpsc::Sender<PrefetchRequest>,

    metrics: Arc<KvPoolMetrics>,
}
```

### 4.2 CXL-Path Lookup

When the scheduler queries for cached prefix blocks, the CXL path is fundamentally different from v5's tiered lookup:

```rust
impl CxlAwarePoolManager {
    /// Find prefix blocks across GPU and CXL shared memory.
    pub fn find_prefix_all_tiers(&mut self, tokens: &[u32]) -> PrefixLookupResult {
        // L1: GPU (fast, synchronous — same as v5)
        let (l1_cached, l1_block_ids) = self.l1_gpu.find_prefix(tokens);

        if l1_cached == (tokens.len() / BLOCK_SIZE) * BLOCK_SIZE {
            return PrefixLookupResult {
                gpu_cached_tokens: l1_cached,
                gpu_block_ids: l1_block_ids,
                shared_mem_tokens: 0,
                shared_mem_addrs: vec![],
                lower_tier_cached_tokens: 0,
                lower_tier_hashes: vec![],
            };
        }

        // CXL path: look up remaining blocks in the PGAS shared index.
        // This is a LOCAL memory read (the index is in CXL shared memory,
        // mapped into our address space). No RPC, no network.
        let remaining_tokens = &tokens[l1_cached..];
        let mut shared_addrs = Vec::new();

        if let Some(ref pgas) = self.pgas_pool {
            for chunk in remaining_tokens.chunks(BLOCK_SIZE) {
                if chunk.len() < BLOCK_SIZE { break; }
                let hash = compute_block_hash(&tokens[..l1_cached + shared_addrs.len() * BLOCK_SIZE + BLOCK_SIZE]);
                if let Some(addr) = pgas.pool.lookup(&hash) {
                    shared_addrs.push(addr);
                } else {
                    break; // contiguous prefix ends here
                }
            }
        }

        let shared_tokens = shared_addrs.len() * BLOCK_SIZE;

        // If CXL found everything, no need to check legacy tiers.
        if l1_cached + shared_tokens >= (tokens.len() / BLOCK_SIZE) * BLOCK_SIZE {
            return PrefixLookupResult {
                gpu_cached_tokens: l1_cached,
                gpu_block_ids: l1_block_ids,
                shared_mem_tokens: shared_tokens,
                shared_mem_addrs: shared_addrs,
                lower_tier_cached_tokens: 0,
                lower_tier_hashes: vec![],
            };
        }

        // Legacy fallback: check v5-style tiers for remaining blocks.
        let further_remaining = &tokens[l1_cached + shared_tokens..];
        let legacy_hashes: Vec<BlockHash> = further_remaining
            .chunks(BLOCK_SIZE)
            .filter(|c| c.len() == BLOCK_SIZE)
            .map(|chunk| compute_block_hash(&tokens[..l1_cached + shared_tokens + chunk.len()]))
            .collect();
        let legacy_cached = self.check_legacy_tiers(&legacy_hashes);

        PrefixLookupResult {
            gpu_cached_tokens: l1_cached,
            gpu_block_ids: l1_block_ids,
            shared_mem_tokens: shared_tokens,
            shared_mem_addrs: shared_addrs,
            lower_tier_cached_tokens: legacy_cached * BLOCK_SIZE,
            lower_tier_hashes: legacy_hashes[..legacy_cached].to_vec(),
        }
    }
}

pub struct PrefixLookupResult {
    pub gpu_cached_tokens: usize,
    pub gpu_block_ids: Vec<u32>,
    /// NEW: tokens found in CXL shared memory (zero-copy accessible).
    pub shared_mem_tokens: usize,
    pub shared_mem_addrs: Vec<GlobalBlockAddr>,
    /// Legacy: tokens found in v5-style tiers (need copy-based promotion).
    pub lower_tier_cached_tokens: usize,
    pub lower_tier_hashes: Vec<BlockHash>,
}
```

### 4.3 CXL Promotion: DMA Instead of Copy Pipeline

v5 promotes blocks through a background worker (fetch from tier → memcpy to pinned host memory → H2D copy to GPU). With CXL, the shared memory IS host memory from the CPU's perspective, so promotion is a single DMA:

```rust
impl CxlAwarePoolManager {
    /// Promote KV blocks from CXL shared memory to GPU.
    /// This replaces v5's multi-hop background prefetch pipeline.
    pub fn promote_from_cxl(
        &mut self,
        addrs: &[GlobalBlockAddr],
    ) -> Vec<u32> {
        let pgas = self.pgas_pool.as_ref().expect("CXL not available");
        let mut gpu_block_ids = Vec::with_capacity(addrs.len());

        for addr in addrs {
            if let Some(gpu_block_id) = self.l1_gpu.try_allocate_one() {
                // Direct DMA: CXL shared memory → GPU HBM.
                // The CXL address is in the CPU's virtual address space,
                // so we can use it as a source for a device H2D transfer.
                unsafe {
                    let src_ptr = pgas.pool.block_ptr(*addr);
                    self.l1_gpu.gpu_pool.upload_from_host_ptr(
                        gpu_block_id,
                        src_ptr,
                        self.block_size_bytes(),
                    );
                }
                gpu_block_ids.push(gpu_block_id);
            } else {
                // GPU full — need to evict first. Same as v5.
                self.evict_gpu_to_cxl(1);
                if let Some(gpu_block_id) = self.l1_gpu.try_allocate_one() {
                    unsafe {
                        let src_ptr = pgas.pool.block_ptr(*addr);
                        self.l1_gpu.gpu_pool.upload_from_host_ptr(
                            gpu_block_id,
                            src_ptr,
                            self.block_size_bytes(),
                        );
                    }
                    gpu_block_ids.push(gpu_block_id);
                }
            }
        }

        gpu_block_ids
    }

    /// Demote GPU blocks to CXL shared memory (instead of v5's CPU DRAM).
    /// The demoted blocks are immediately accessible by all other instances.
    pub fn evict_gpu_to_cxl(&mut self, target: usize) -> usize {
        let pgas = match &self.pgas_pool {
            Some(p) => p,
            None => return self.l1_gpu.evict(target), // no CXL — v4 discard
        };

        let mut freed = 0;
        while freed < target {
            if let Some(leaf) = self.l1_gpu.find_oldest_unreferenced_leaf() {
                // Allocate in CXL shared pool
                if let Some(global_addr) = pgas.pool.allocate() {
                    // D2H: GPU → CXL shared memory (single DMA)
                    unsafe {
                        let dest_ptr = pgas.pool.block_ptr(global_addr);
                        self.l1_gpu.gpu_pool.download_to_host_ptr(
                            leaf.block_id,
                            dest_ptr,
                            self.block_size_bytes(),
                        );
                    }
                    // Publish to shared index
                    let hash = self.l1_gpu.block_hash(leaf);
                    pgas.pool.insert(hash, global_addr, self.host_id());

                    self.l1_gpu.gpu_pool.free(leaf.block_id);
                    self.l1_gpu.remove_leaf(leaf);
                    freed += 1;
                } else {
                    // CXL pool full — evict from CXL (LRU) then retry,
                    // or just discard the GPU block.
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
}
```

### 4.4 Why CXL Promotion Is Faster

| Step | v5 (Copy Pipeline) | v6 (CXL DMA) |
|---|---|---|
| Locate block | RPC to remote store (~5μs) | Load from CXL-mapped index (~300ns) |
| Read block data | Fetch over RDMA/TCP (~5–50μs for 32KB) | Already in addressable memory (0μs) |
| Copy to pinned host memory | memcpy (~2μs for 32KB) | Skipped — CXL memory IS host memory |
| Upload to GPU | H2D DMA (~3μs for 32KB on PCIe 5.0) | H2D DMA (~3μs for 32KB) — same |
| **Total** | **~15–60μs** | **~3.3μs** |

The CXL path is **5–18× faster** per block promotion.

---

## 5. CXL KV Connector: Zero-Copy Interface

### 5.1 Zero-Copy Connector

v5's `KvConnector` trait copies data (`fetch` returns `Vec<u8>`). v6 adds a parallel trait for zero-copy access via global pointers:

```rust
/// Zero-copy KV connector for CXL-backed storage.
/// Instead of copying block data, returns a global pointer.
pub trait KvConnectorZeroCopy: Send + Sync {
    /// Look up blocks by hash. Returns global addresses (not data).
    fn lookup(&self, hashes: &[BlockHash]) -> Vec<Option<GlobalBlockAddr>>;

    /// Publish a block: register (hash → global_addr) in the shared index.
    /// Caller has already written data to global_addr via shmem_put.
    fn publish(&self, hash: BlockHash, addr: GlobalBlockAddr);

    /// Release a block reference. When refcount reaches 0, the block
    /// is eligible for eviction/reuse.
    fn release(&self, addrs: &[GlobalBlockAddr]);

    /// Get a raw pointer to block data (zero-copy).
    /// Safety: caller must ensure the block is not freed while the pointer is in use.
    unsafe fn block_ptr(&self, addr: GlobalBlockAddr) -> *const u8;

    /// Get a mutable pointer (for writing new block data).
    unsafe fn block_ptr_mut(&self, addr: GlobalBlockAddr) -> *mut u8;
}

/// CXL implementation of the zero-copy connector.
pub struct CxlKvConnector {
    heap: Arc<SymmetricHeap>,
    host_id: u32,
}

impl KvConnectorZeroCopy for CxlKvConnector {
    fn lookup(&self, hashes: &[BlockHash]) -> Vec<Option<GlobalBlockAddr>> {
        hashes.iter().map(|h| self.heap.pool.lookup(h)).collect()
    }

    fn publish(&self, hash: BlockHash, addr: GlobalBlockAddr) {
        self.heap.pool.insert(hash, addr, self.host_id);
    }

    fn release(&self, addrs: &[GlobalBlockAddr]) {
        for addr in addrs {
            // Decrement refcount in the shared index.
            if let Some(entry) = self.heap.pool.find_entry_by_addr(*addr) {
                entry.refcount.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    unsafe fn block_ptr(&self, addr: GlobalBlockAddr) -> *const u8 {
        self.heap.pool.block_ptr(addr) as *const u8
    }

    unsafe fn block_ptr_mut(&self, addr: GlobalBlockAddr) -> *mut u8 {
        self.heap.pool.block_ptr(addr)
    }
}
```

### 5.2 Prefill Direct-to-Shared-Memory

Instead of v5's "compute on GPU → read back to CPU → push to remote store," v6 writes prefill results directly to CXL shared memory:

```rust
impl CxlAwarePoolManager {
    /// After prefill: write new KV blocks directly to CXL shared memory.
    /// No intermediate CPU buffer, no store service RPC.
    pub fn publish_new_blocks_cxl(&self, tokens: &[u32], gpu_block_ids: &[u32]) {
        let pgas = match &self.pgas_pool {
            Some(p) => p,
            None => {
                // Fallback to v5 publish path.
                self.publish_new_blocks_legacy(tokens, gpu_block_ids);
                return;
            }
        };

        for (i, &gpu_bid) in gpu_block_ids.iter().enumerate() {
            let prefix_end = (i + 1) * BLOCK_SIZE;
            let hash = compute_block_hash(&tokens[..prefix_end]);

            // Check if already in shared pool (another instance may have computed it).
            if pgas.pool.lookup(&hash).is_some() {
                continue; // already shared — skip
            }

            // Allocate in shared pool and D2H directly to CXL memory.
            if let Some(global_addr) = pgas.pool.allocate() {
                unsafe {
                    let dest_ptr = pgas.pool.block_ptr(global_addr);
                    self.l1_gpu.gpu_pool.download_to_host_ptr(
                        gpu_bid,
                        dest_ptr,
                        self.block_size_bytes(),
                    );
                }
                // Publish: register in shared index (visible to all hosts).
                pgas.pool.insert(hash, global_addr, self.host_id());
            }
        }
    }
}
```

---

## 6. Shared Weight Pool

### 6.1 The Opportunity

A Llama-3.1-70B model at FP16 occupies ~140 GB. In v5, each of N instances loads its own copy. With 4 instances, that's 560 GB of CPU/GPU memory for weights alone — a massive waste when CXL can provide shared access.

### 6.2 Design: Read-Only Weights in CXL Shared Memory

Model weights are loaded once into CXL shared memory by a leader instance. All other instances map the same region and read weights directly.

```rust
/// Shared model weight pool in CXL memory.
pub struct SharedWeightPool {
    /// The CXL region holding the weights.
    region_id: u64,
    base_addr: *const u8,
    /// Weight layout: name → (offset, size, dtype, shape)
    layout: HashMap<String, WeightSlice>,
    /// Total size in bytes.
    total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct WeightSlice {
    pub offset: u64,
    pub size: u64,
    pub dtype: WeightDtype,
    pub shape: Vec<usize>,
}

impl SharedWeightPool {
    /// Leader: load weights from disk into CXL shared memory.
    /// Called once by the first instance to start.
    pub fn load_leader(
        region: &SharedRegionInfo,
        model_path: &Path,
        tp_rank: u32,
        tp_size: u32,
    ) -> Self {
        let base_addr = region.base_addr;
        let mut layout = HashMap::new();
        let mut offset = 0u64;

        for shard in load_safetensors_index(model_path) {
            let (name, tensor) = load_tensor(&shard);
            // Apply TP sharding (same logic as v4 §6).
            let sharded = apply_tp_sharding(&name, &tensor, tp_rank, tp_size);

            let size = sharded.data.len() as u64;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sharded.data.as_ptr(),
                    base_addr.add(offset as usize) as *mut u8,
                    size as usize,
                );
            }
            layout.insert(name, WeightSlice {
                offset, size,
                dtype: sharded.dtype,
                shape: sharded.shape,
            });
            offset += size;
        }

        // Fence: ensure all writes are visible before followers read.
        std::sync::atomic::fence(Ordering::SeqCst);

        Self {
            region_id: region.region_id,
            base_addr: base_addr as *const u8,
            layout,
            total_bytes: offset,
        }
    }

    /// Follower: map the same region and use the leader's layout.
    /// Layout is communicated out-of-band (e.g., written to a small
    /// metadata region in the same CXL memory, or via a control RPC).
    pub fn attach_follower(
        region: &SharedRegionInfo,
        layout: HashMap<String, WeightSlice>,
    ) -> Self {
        Self {
            region_id: region.region_id,
            base_addr: region.base_addr as *const u8,
            layout,
            total_bytes: layout.values().map(|w| w.offset + w.size).max().unwrap_or(0),
        }
    }

    /// Get a raw pointer to a weight tensor. Used as source for H2D copy to GPU.
    pub unsafe fn weight_ptr(&self, name: &str) -> Option<(*const u8, &WeightSlice)> {
        self.layout.get(name).map(|w| {
            (self.base_addr.add(w.offset as usize), w)
        })
    }
}
```

### 6.3 Impact

| Scenario | v5 Memory Usage | v6 Memory Usage | Savings |
|---|---|---|---|
| 4 instances, Llama-70B (FP16) | 4 × 140 GB = 560 GB | 140 GB shared + 4 × 0 GB = 140 GB | **420 GB freed** |
| 4 instances, Llama-405B (FP8) | 4 × 200 GB = 800 GB | 200 GB shared = 200 GB | **600 GB freed** |

The freed memory can be used for KV cache, dramatically increasing effective context length and concurrency.

### 6.4 GPU Weight Loading from Shared Pool

Each GPU loads its TP shard from the shared pool (CXL → GPU HBM), rather than from disk:

```rust
impl ModelLoader {
    pub fn load_weights_from_shared_pool(
        &self,
        pool: &SharedWeightPool,
        device: &dyn HalDevice,
    ) -> ModelWeights {
        let mut weights = ModelWeights::new();

        for (name, slice) in &pool.layout {
            unsafe {
                let (src_ptr, _) = pool.weight_ptr(name).unwrap();
                // H2D: CXL memory → GPU. The CXL address is accessible
                // to the CPU, so standard H2D DMA works.
                let gpu_buf = device.alloc(slice.size as usize);
                device.h2d_async(gpu_buf, src_ptr, slice.size as usize);
                weights.insert(name.clone(), gpu_buf);
            }
        }

        device.synchronize();
        weights
    }
}
```

**Benefit beyond memory savings:** Weight loading from CXL DRAM (~80 GB/s per CXL x16 link) is much faster than from NVMe SSD (~7 GB/s) or network storage. For a 140 GB model: CXL loads in ~1.75s vs NVMe in ~20s.

---

## 7. CXL-Distance-Aware Routing

### 7.1 The Problem with v5 Routing on CXL Hardware

v5's router scores instances by "how many prefix blocks does this instance's GPU have cached?" On CXL hardware, this is the wrong question. Every block in the shared pool is accessible by every instance — the question is **how close** (in CXL latency) is the block to the instance.

### 7.2 Distance-Weighted Scoring

```rust
pub struct CxlAwareRouter {
    /// Global index: block_hash → GlobalBlockAddr (in CXL shared memory).
    /// Replaces v5's hash → instance_id mapping.
    shared_index: Arc<PgasBlockPool>,

    /// Per-instance GPU cache index (v5's index, retained for GPU L1 hits).
    gpu_index: RwLock<HashMap<BlockHash, SmallVec<[u32; 4]>>>,

    /// Instance info with CXL topology.
    instances: RwLock<Vec<CxlInstanceInfo>>,

    /// CXL topology.
    topology: Arc<CxlTopology>,
}

struct CxlInstanceInfo {
    id: u32,
    addr: SocketAddr,
    running_requests: u32,
    gpu_cache_usage: f32,
    healthy: bool,
    /// Which NUMA node this instance's GPU is closest to.
    gpu_numa_node: u32,
    /// Which CXL switch domain this instance belongs to.
    cxl_switch_id: u32,
}

impl CxlAwareRouter {
    /// Route a request, considering both GPU cache hits and CXL distance.
    pub fn route(&self, prompt_tokens: &[u32]) -> u32 {
        let gpu_index = self.gpu_index.read();
        let instances = self.instances.read();

        // Score each instance.
        let mut scores: Vec<(u32, f64)> = instances.iter()
            .filter(|i| i.healthy)
            .map(|i| (i.id, 0.0))
            .collect();

        let mut prefix_so_far = Vec::new();
        for chunk in prompt_tokens.chunks(BLOCK_SIZE) {
            if chunk.len() < BLOCK_SIZE { break; }
            prefix_so_far.extend_from_slice(chunk);
            let hash = compute_block_hash(&prefix_so_far);

            // Score 1: GPU L1 hit (highest value — zero latency).
            if let Some(holders) = gpu_index.get(&hash) {
                for &inst_id in holders.iter() {
                    if let Some(s) = scores.iter_mut().find(|(id, _)| *id == inst_id) {
                        s.1 += 10.0;  // GPU hit: maximum score per block
                    }
                }
            }

            // Score 2: CXL shared memory hit (weighted by distance).
            if let Some(global_addr) = self.shared_index.lookup(&hash) {
                for (inst_id, score) in scores.iter_mut() {
                    let inst = instances.iter().find(|i| i.id == *inst_id).unwrap();
                    let distance_ns = self.topology.distance_ns(
                        inst.gpu_numa_node,
                        self.addr_to_numa(global_addr),
                    );
                    // Inverse distance scoring: closer = higher score.
                    // Normalize: 150ns (same-host CXL) → 8.0, 500ns (cross-switch) → 2.0
                    let cxl_score = 1200.0 / (distance_ns as f64).max(150.0);
                    *score += cxl_score;
                }
                // Release the reference we acquired during lookup.
                self.shared_index.release_lookup(&hash);
            }
        }

        // Tiebreak: least running requests.
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let load_a = instances.iter().find(|i| i.id == a.0)
                        .map(|i| i.running_requests).unwrap_or(u32::MAX);
                    let load_b = instances.iter().find(|i| i.id == b.0)
                        .map(|i| i.running_requests).unwrap_or(u32::MAX);
                    load_a.cmp(&load_b)
                })
        });

        scores[0].0
    }

    fn addr_to_numa(&self, addr: GlobalBlockAddr) -> u32 {
        // Map a global address to its NUMA node in the CXL topology.
        // This uses the region → NUMA mapping from CxlTopology.
        self.topology.shared_regions.iter()
            .find(|r| r.region_id == addr.region_id)
            .map(|r| {
                // Find the NUMA node that owns this region.
                // For CXL pooled memory, this is the CXL memory controller's NUMA node.
                r.accessible_by[0]  // simplified — use owning host's NUMA
            })
            .unwrap_or(0)
    }
}
```

### 7.3 Routing Example

Consider 4 instances on 2 CXL switches (Instance 0,1 on Switch 0; Instance 2,3 on Switch 1). A request's prefix has 100 blocks:

| Block Location | Instance 0 Score | Instance 2 Score |
|---|---|---|
| GPU of Instance 0 | +10.0 each | +0.0 |
| CXL shared memory, Switch 0 | +8.0 each (150ns) | +2.4 each (500ns) |
| CXL shared memory, Switch 1 | +2.4 each (500ns) | +8.0 each (150ns) |

If 60 blocks are in Switch 0's pool and 40 are in Instance 0's GPU:
- Instance 0 score: 40×10.0 + 60×8.0 = 880
- Instance 2 score: 0 + 60×2.4 + 40×2.4 = 240

Instance 0 wins decisively. But if Instance 0 is overloaded (5× the requests of Instance 2), the load tiebreaker may favor Instance 2 — it can still access all blocks, just at higher CXL latency.

---

## 8. PD Disaggregation over CXL Shared Memory

### 8.1 v5's KV Transfer Bottleneck — Eliminated

v5's PD disaggregation requires an explicit KV transfer step after prefill completes. With CXL shared memory, this step disappears:

```
v5: Prefill GPU → D2H → CPU → RPC/RDMA → CPU → H2D → Decode GPU
    (prefill)   (copy)       (transfer)       (copy)  (decode)
    ~3μs        ~3μs         ~5-50μs          ~3μs    ~3μs
    Total: ~17-62μs per block

v6: Prefill GPU → D2H → CXL shared memory ← H2D ← Decode GPU
    (prefill)   (DMA to shared)            (DMA from shared) (decode)
    ~3μs        ~3μs                       ~3μs              ~3μs
    Total: ~6μs per block (both DMAs can overlap with compute)
```

### 8.2 Implementation

```rust
impl CxlDisaggPrefillEngine {
    /// After prefill: KV blocks are already in CXL shared memory
    /// (written by publish_new_blocks_cxl). No transfer step needed.
    async fn after_prefill(&self, seq: &Sequence) {
        // Publish KV blocks to CXL shared pool.
        // This is a D2H DMA from GPU to CXL shared memory —
        // it happens AS PART of the normal publish path.
        self.pool_manager.publish_new_blocks_cxl(
            seq.all_tokens(),
            seq.block_ids(),
        );

        // Notify the decode engine that KV is ready.
        // The notification carries GlobalBlockAddrs, not the data itself.
        self.notify_decode_ready(seq.id(), seq.block_addrs());
    }
}

impl CxlDisaggDecodeEngine {
    /// Before decode: promote KV blocks from CXL shared memory to GPU.
    /// This is a single H2D DMA per block — no intermediate copies.
    async fn before_decode(&self, seq: &Sequence) {
        // Check if KV is already in local GPU (fast path).
        if self.has_kv_locally(seq) { return; }

        // Promote from CXL shared memory → GPU.
        let gpu_block_ids = self.pool_manager.promote_from_cxl(
            &seq.shared_mem_addrs(),
        );

        // Update sequence with new GPU block IDs.
        seq.set_block_ids(gpu_block_ids);
    }
}
```

### 8.3 Overlapping DMA with Compute

On hardware that supports concurrent DMA and compute (most modern GPUs/NPUs), the prefill engine can overlap KV writes to CXL memory with computing the next batch:

```rust
impl CxlDisaggPrefillEngine {
    async fn prefill_loop(&mut self) {
        loop {
            let batch = self.scheduler.get_next_prefill_batch();

            // Start prefill computation.
            let output = self.executor.forward_prefill(&batch).await;

            // Overlap: while the next batch is being prepared,
            // DMA the completed KV blocks to CXL shared memory.
            let publish_handle = tokio::spawn({
                let pool = self.pool_manager.clone();
                let tokens = batch.all_tokens();
                let block_ids = output.new_block_ids.clone();
                async move {
                    pool.publish_new_blocks_cxl(&tokens, &block_ids);
                }
            });

            // Process results (update sequences, notify decode, etc.)
            self.process_prefill_results(&output).await;

            // Ensure publish completed before reusing GPU blocks.
            publish_handle.await.unwrap();
        }
    }
}
```

---

## 9. Fault Tolerance in Shared Memory

### 9.1 The Problem

CXL shared memory introduces a new failure mode: if a host fails, the CXL memory it owns may become inaccessible to other hosts (depending on the CXL fabric manager and memory pooling implementation).

### 9.2 Failure Modes and Recovery

| Failure | Impact | Recovery |
|---|---|---|
| Host crash, CXL memory survives (pooled) | KV blocks in pooled memory are still accessible. The shared index entries with `owner_host == crashed_host` are orphaned but valid. | Other hosts continue accessing blocks. The crashed host's index entries are garbage-collected when refcount reaches 0. |
| Host crash, CXL memory lost (attached) | Blocks in that host's CXL expander are lost. The shared index has dangling entries. | Detect stale entries (heartbeat-based). Mark blocks as lost. Requests that need these blocks trigger recomputation. |
| CXL switch failure | All memory behind that switch is inaccessible. | Fall back to v5 behavior: use RDMA/TCP to fetch blocks from hosts on other switches, or recompute. |
| CXL fabric manager failure | New region allocations fail, but existing mappings continue. | Degrade gracefully: stop allocating new shared blocks, use local memory only. |

### 9.3 Heartbeat and Liveness

```rust
/// Each host periodically writes a heartbeat to a well-known slot
/// in CXL shared memory. Other hosts read it to detect failures.
pub struct CxlHeartbeat {
    /// Heartbeat array in shared memory: one slot per host.
    slots: *mut AtomicU64,
    my_pe: u32,
    num_pes: u32,
    interval: Duration,
}

impl CxlHeartbeat {
    /// Write heartbeat (called periodically by each host).
    pub fn beat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        unsafe {
            (*self.slots.add(self.my_pe as usize)).store(now, Ordering::Release);
        }
    }

    /// Check if a host is alive (called by any host).
    pub fn is_alive(&self, host_id: u32, timeout: Duration) -> bool {
        let last_beat = unsafe {
            (*self.slots.add(host_id as usize)).load(Ordering::Acquire)
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        now - last_beat < timeout.as_millis() as u64
    }
}
```

### 9.4 Block Ownership and GC

When a host is detected as dead, its blocks are either:
1. **Adopted** by a surviving host (change `owner_host` in the index entry).
2. **Freed** if no other host holds a reference (refcount == 0).

```rust
impl PgasBlockPool {
    /// Garbage-collect blocks owned by a dead host.
    pub fn gc_dead_host(&self, dead_host_id: u32) -> usize {
        let mut freed = 0;
        for slot in 0..self.index_capacity {
            let entry = unsafe { &*self.index_base.add(slot) };
            let offset = entry.offset.load(Ordering::Acquire);
            if offset == 0 { continue; }

            if entry.owner_host.load(Ordering::Acquire) == dead_host_id {
                let refcount = entry.refcount.load(Ordering::Acquire);
                if refcount == 0 {
                    // No one is using this block. Free it.
                    if entry.offset.compare_exchange(
                        offset, 0, Ordering::AcqRel, Ordering::Relaxed
                    ).is_ok() {
                        let addr = GlobalBlockAddr { region_id: self.region_id, offset };
                        self.free(addr);
                        freed += 1;
                    }
                } else {
                    // Block is in use. Transfer ownership to the first live user.
                    // (Simplified — in practice, use a lease-based protocol.)
                    entry.owner_host.store(0, Ordering::Release); // host 0 adopts
                }
            }
        }
        freed
    }
}
```

---

## 10. Integration with v4/v5 Components

### 10.1 Graceful Degradation

v6 is designed as a fast path layered on top of v5. Every CXL-specific path has a fallback:

| v6 Path | Condition | Fallback |
|---|---|---|
| `CxlAwarePoolManager::find_prefix_all_tiers` → PGAS lookup | `pgas_pool.is_some()` | v5: `check_lower_tiers` (RPC-based) |
| `promote_from_cxl` (DMA from shared memory) | `pgas_pool.is_some()` | v5: `prefetch_worker` (background copy) |
| `evict_gpu_to_cxl` (DMA to shared memory) | `pgas_pool.is_some()` | v5: `evict_to_tier` (CPU DRAM copy) |
| `SharedWeightPool` (shared model weights) | CXL region available + leader loaded | v4: per-instance weight loading |
| `CxlAwareRouter` (distance-aware routing) | CXL topology discovered | v5: `CacheAwareRouter` (binary scoring) |

### 10.2 Scheduler Changes

The scheduler's only change from v5 is handling the new `shared_mem_addrs` field:

```rust
// v5:
let layout = TokenLayout {
    computed_tokens: lookup.gpu_cached_tokens + lookup.lower_tier_cached_tokens,
    ...
};

// v6: CXL shared memory tokens count as "cached" with lower promotion cost.
let layout = TokenLayout {
    computed_tokens: lookup.gpu_cached_tokens
        + lookup.shared_mem_tokens       // NEW: CXL hits
        + lookup.lower_tier_cached_tokens,
    total_tokens: seq.prompt_len(),
    remaining: seq.prompt_len()
        - lookup.gpu_cached_tokens
        - lookup.shared_mem_tokens
        - lookup.lower_tier_cached_tokens,
    // NEW: scheduler prioritizes sequences with CXL hits over legacy-tier hits,
    // because CXL promotion is 5-18x faster.
    promotion_cost: if lookup.shared_mem_tokens > 0 {
        PromotionCost::CxlFast  // ~3μs/block
    } else if lookup.lower_tier_cached_tokens > 0 {
        PromotionCost::LegacySlow  // ~15-60μs/block
    } else {
        PromotionCost::None
    },
};
```

### 10.3 No Changes Required

These v4/v5 components are completely unchanged:
- `RadixKvManager` (L1 GPU cache)
- `BlockPool` (GPU block allocation)
- Forward pass execution (prefill and decode kernels)
- Incremental detokenization
- CUDA/ACL graph capture
- Metrics schema (new metrics are additive)
- HTTP API and SSE streaming

---

## 11. Configuration

New configuration fields for CXL/PGAS:

```yaml
cxl:
  # Enable CXL memory features (default: false = v5 behavior).
  enabled: false

  # CXL topology discovery.
  topology:
    # Auto-discover from sysfs/ACPI (recommended).
    auto_discover: true
    # Manual override (for testing or non-standard setups).
    # shared_regions:
    #   - region_id: 0
    #     device: "/dev/cxl/mem0"
    #     size_gb: 512

  # PGAS KV block pool in CXL shared memory.
  kv_pool:
    enabled: true
    # Region to use for KV blocks. 0 = auto (best region for this host).
    region_id: 0
    # Maximum bytes to use from the CXL region for KV blocks.
    # 0 = use all available space in the region.
    capacity_gb: 0

  # Shared weight pool.
  weight_pool:
    enabled: true
    # Region to use for model weights (can be same as kv_pool region).
    region_id: 0
    # This instance is the leader (loads weights from disk → CXL).
    # Other instances set this to false and wait for the leader.
    is_leader: false

  # Heartbeat for fault detection.
  heartbeat:
    interval_ms: 100
    timeout_ms: 500

  # Distance-aware routing.
  routing:
    # Weight for GPU L1 cache hits in routing score.
    gpu_hit_weight: 10.0
    # Weight for CXL shared memory hits (scaled by inverse distance).
    cxl_hit_base_weight: 1200.0
    # Minimum CXL distance (ns) for normalization.
    min_distance_ns: 150

# Legacy v5 settings remain for non-CXL fallback.
kv_cache_pool:
  enabled: true   # still used for CPU DRAM and SSD tiers
  cpu_dram:
    enabled: true
    capacity_gb: 32
  # ... rest of v5 config unchanged
```

### Minimal CXL Config (2 instances, same CXL switch)

```yaml
cxl:
  enabled: true
  topology:
    auto_discover: true
  kv_pool:
    enabled: true
  weight_pool:
    enabled: true
    is_leader: true  # set false on the other instance
```

---

## 12. Updated Directory Layout

New files added to v5's directory structure:

```
llm-server/src/
├── cxl/
│   ├── topology.rs          ← NEW: CxlTopology, NumaNodeInfo, discovery
│   ├── pgas_pool.rs         ← NEW: PgasBlockPool, AtomicBlockIndexEntry
│   ├── free_list.rs         ← NEW: AtomicFreeList (lock-free, CXL-safe)
│   ├── symmetric_heap.rs    ← NEW: SymmetricHeap (OpenSHMEM-style API)
│   ├── connector.rs         ← NEW: CxlKvConnector (zero-copy)
│   ├── weight_pool.rs       ← NEW: SharedWeightPool
│   ├── heartbeat.rs         ← NEW: CxlHeartbeat (fault detection)
│   └── gc.rs                ← NEW: Dead-host block garbage collection
├── kv_cache/
│   ├── pool_manager.rs      ← MODIFIED: CxlAwarePoolManager (extends v5)
│   └── ...                  ← unchanged from v5
├── routing/
│   ├── router.rs            ← MODIFIED: CxlAwareRouter (extends v5)
│   └── ...                  ← unchanged from v5
```

### New crate dependencies

| Crate | Purpose |
|---|---|
| `memmap2` | Memory-map CXL device files |
| `libc` (extended) | `mbind`, `set_mempolicy` for NUMA-aware CXL allocation |

---

## 13. Implementation Phases

### Phase 11a — CXL Topology Discovery + PGAS Pool (1.5 weeks)

#### Deliverables

- [ ] `CxlTopology::discover()` — read sysfs, ACPI HMAT/CEDT
- [ ] `PgasBlockPool` — allocator with lock-free free list over CXL shared memory
- [ ] `AtomicBlockIndexEntry` — shared hash index with CAS-based insert/lookup
- [ ] `SymmetricHeap` — OpenSHMEM-style wrapper

#### Test Plan

**T11a.1 — CXL topology discovery (requires CXL hardware or QEMU emulation):**
```rust
#[test]
fn test_cxl_topology_discovery() {
    let topo = CxlTopology::discover().unwrap();
    assert!(topo.numa_nodes.len() > 0);
    // At least one CXL node should be present
    assert!(topo.numa_nodes.iter().any(|n|
        n.kind == NumaKind::CxlLocalExpander || n.kind == NumaKind::CxlPooled
    ));
    // Distance matrix should be symmetric
    for i in 0..topo.numa_nodes.len() {
        for j in 0..topo.numa_nodes.len() {
            assert_eq!(topo.distance_ns[i][j], topo.distance_ns[j][i]);
        }
    }
}
```

**T11a.2 — PGAS block pool concurrent allocation (multi-process):**
```rust
#[test]
fn test_pgas_concurrent_alloc() {
    // Simulate 4 hosts allocating from the same pool.
    // Use threads as a proxy for separate processes (shared mmap).
    let pool = Arc::new(unsafe {
        PgasBlockPool::new(0, mmap_shared_region(), REGION_SIZE, BLOCK_SIZE_BYTES)
    });
    let handles: Vec<_> = (0..4).map(|_| {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let mut addrs = Vec::new();
            for _ in 0..1000 {
                if let Some(addr) = pool.allocate() {
                    addrs.push(addr);
                }
            }
            addrs
        })
    }).collect();

    let all_addrs: Vec<_> = handles.into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();
    // No duplicates
    let unique: HashSet<_> = all_addrs.iter().map(|a| a.offset).collect();
    assert_eq!(unique.len(), all_addrs.len());
}
```

**T11a.3 — Shared index insert/lookup consistency:**
```rust
#[test]
fn test_shared_index_concurrent_insert_lookup() {
    let pool = make_test_pgas_pool(10000);
    let hashes: Vec<BlockHash> = (0..1000u32).map(|i| {
        let mut h = [0u8; 32];
        h[..4].copy_from_slice(&i.to_le_bytes());
        h
    }).collect();

    // Thread 0: insert all
    let pool_w = pool.clone();
    let hashes_w = hashes.clone();
    let writer = std::thread::spawn(move || {
        for hash in &hashes_w {
            let addr = pool_w.allocate().unwrap();
            pool_w.insert(*hash, addr, 0);
        }
    });

    // Thread 1: lookup in parallel
    let pool_r = pool.clone();
    let hashes_r = hashes.clone();
    let reader = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(1));
        let mut found = 0;
        for hash in &hashes_r {
            if pool_r.lookup(hash).is_some() {
                found += 1;
            }
        }
        found
    });

    writer.join().unwrap();
    let found = reader.join().unwrap();
    // Reader should find most/all blocks (depending on timing)
    assert!(found > 900);
}
```

---

### Phase 11b — CXL Integration with Pool Manager + Disagg (1.5 weeks)

#### Deliverables

- [ ] `CxlAwarePoolManager` — extends v5 pool manager with CXL fast path
- [ ] `promote_from_cxl` — single-DMA promotion from CXL to GPU
- [ ] `evict_gpu_to_cxl` — single-DMA demotion from GPU to CXL shared memory
- [ ] `publish_new_blocks_cxl` — direct prefill → CXL write path
- [ ] PD disaggregation over CXL shared memory

#### Test Plan

**T11b.1 — CXL promotion round-trip (GPU → CXL → GPU):**
```rust
#[test]
fn test_cxl_promotion_round_trip() {
    let mut pool = CxlAwarePoolManager::new_test_cxl(gpu_blocks: 10, cxl_blocks: 1000);
    let tokens = vec![1u32; 16];

    // Insert into GPU
    pool.l1_gpu.insert_prefix(&tokens, &[pool.l1_gpu.allocate_one()]);
    // Write known data to the GPU block
    pool.l1_gpu.gpu_pool.write_test_data(0, &[0xABu8; BLOCK_SIZE_BYTES]);

    // Evict to CXL
    pool.evict_gpu_to_cxl(1);
    assert_eq!(pool.l1_gpu.find_prefix(&tokens).0, 0);  // gone from GPU

    // Promote back from CXL
    let hash = compute_block_hash(&tokens);
    let addr = pool.pgas_pool.as_ref().unwrap().pool.lookup(&hash).unwrap();
    let gpu_ids = pool.promote_from_cxl(&[addr]);
    assert_eq!(gpu_ids.len(), 1);

    // Verify data integrity
    let data = pool.l1_gpu.gpu_pool.read_test_data(gpu_ids[0]);
    assert_eq!(data, vec![0xABu8; BLOCK_SIZE_BYTES]);
}
```

**T11b.2 — CXL disagg latency benchmark:**
```bash
# Compare PD disagg latency: v5 (RDMA) vs v6 (CXL shared memory)
python tools/disagg_benchmark.py \
  --mode v5_rdma --blocks 1000
# Expected: ~17-62μs per block

python tools/disagg_benchmark.py \
  --mode v6_cxl --blocks 1000
# Expected: ~6μs per block

# Pass criteria: v6 CXL is at least 3× faster than v5 RDMA
```

**T11b.3 — Multi-instance KV sharing over CXL:**
```bash
# Start 2 instances sharing the same CXL region.
./llm-server --instance-id 0 --cxl-enabled --cxl-leader &
./llm-server --instance-id 1 --cxl-enabled &

# Send a request to instance 0.
curl http://localhost:8080/v1/completions -d '{"prompt":"The capital of France is"}'

# Send the SAME prefix to instance 1.
curl http://localhost:8081/v1/completions -d '{"prompt":"The capital of France is"}'

# Verify: instance 1's metrics show CXL shared memory hits (not recompute).
# cxl_shared_mem_hits > 0
```

---

### Phase 11c — Shared Weight Pool (1 week)

#### Deliverables

- [ ] `SharedWeightPool::load_leader` — load weights into CXL shared memory
- [ ] `SharedWeightPool::attach_follower` — map shared weights read-only
- [ ] Integration with v4's weight loading path

#### Test Plan

**T11c.1 — Shared weight loading correctness:**
```rust
#[test]
fn test_shared_weight_correctness() {
    // Leader loads weights into CXL shared memory.
    let leader = SharedWeightPool::load_leader(&region, model_path, tp_rank=0, tp_size=1);
    // Follower attaches.
    let follower = SharedWeightPool::attach_follower(&region, leader.layout.clone());
    // Compare: every weight tensor should be byte-identical.
    for (name, slice) in &leader.layout {
        unsafe {
            let leader_ptr = leader.weight_ptr(name).unwrap().0;
            let follower_ptr = follower.weight_ptr(name).unwrap().0;
            let leader_data = std::slice::from_raw_parts(leader_ptr, slice.size as usize);
            let follower_data = std::slice::from_raw_parts(follower_ptr, slice.size as usize);
            assert_eq!(leader_data, follower_data);
        }
    }
}
```

**T11c.2 — Weight loading speedup:**
```bash
# Benchmark: disk vs CXL weight loading for Llama-3.1-70B
python tools/weight_load_benchmark.py --source disk --model meta-llama/Llama-3.1-70B
# Expected: ~20s (NVMe)

python tools/weight_load_benchmark.py --source cxl --model meta-llama/Llama-3.1-70B
# Expected: ~1.75s (CXL x16)

# Pass criteria: CXL loading is at least 5× faster
```

---

### Phase 11d — CXL-Distance-Aware Routing (1 week)

#### Deliverables

- [ ] `CxlAwareRouter` with distance-weighted scoring
- [ ] Integration with v5's cache event system
- [ ] NUMA-aware block placement (prefer allocating in local CXL expander)

#### Test Plan

**T11d.1 — Distance-aware routing preference:**
```rust
#[test]
fn test_cxl_routing_prefers_proximity() {
    let mut router = CxlAwareRouter::new_test(
        // 4 instances: 0,1 on switch 0; 2,3 on switch 1
        instances: vec![
            CxlInstanceInfo { id: 0, cxl_switch_id: 0, .. },
            CxlInstanceInfo { id: 1, cxl_switch_id: 0, .. },
            CxlInstanceInfo { id: 2, cxl_switch_id: 1, .. },
            CxlInstanceInfo { id: 3, cxl_switch_id: 1, .. },
        ],
    );

    // Insert blocks into CXL region on switch 0.
    let tokens = vec![1u32; 160];  // 10 blocks
    for i in 0..10 {
        let hash = compute_block_hash(&tokens[..(i+1)*16]);
        router.shared_index.insert(hash, GlobalBlockAddr { region_id: 0, offset: i as u64 * 32768 }, 0);
    }

    // Route should prefer instance 0 or 1 (same switch as blocks).
    let chosen = router.route(&tokens);
    assert!(chosen == 0 || chosen == 1);
}
```

---

## 14. Performance Targets

### Single Instance: CXL vs. v5 CPU Offload

| Metric | v5 (CPU DRAM) | v6 (CXL Shared) | How Measured |
|---|---|---|---|
| Block promotion latency | ~15μs (fetch + memcpy + H2D) | ~3.3μs (direct H2D from CXL) | Micro-benchmark |
| TTFT (turn 10, 100 evicted blocks) | ~20ms (100 × 15μs promote + overlap) | ~8ms (100 × 3.3μs + overlap) | Agent workload |
| Effective cache capacity | GPU + CPU DRAM (~100 + 512 GB) | GPU + CXL pool (100 + 2048 GB+) | Prometheus |

### Multi-Instance: 4 Instances, CXL Fabric

| Metric | v5 (Remote Store) | v6 (CXL PGAS) | How Measured |
|---|---|---|---|
| Cross-instance KV fetch | ~50μs (RPC + RDMA transfer) | ~300ns (CXL load) + ~3μs (H2D) | Micro-benchmark |
| Prefix cache effective hit rate | > 80% (with routing) | > 95% (all blocks globally visible) | Router metrics |
| Weight memory per instance | 140 GB (70B model, FP16) | 0 GB (shared pool) | Memory accounting |
| Weight loading time | ~20s from NVMe | ~1.75s from CXL (leader) / ~0s (followers) | Startup benchmark |

### PD Disaggregation: CXL vs. RDMA

| Metric | v5 (RDMA) | v6 (CXL Shared) | How Measured |
|---|---|---|---|
| KV transfer per block | ~17–62μs | ~6μs | Disagg benchmark |
| 131K-token prefill KV transfer | ~140–510ms (8192 blocks) | ~49ms (8192 blocks) | Disagg benchmark |
| Transfer overhead as % of prefill | ~5–20% | ~2% | End-to-end profile |

### Hardware Utilization

| Resource | v5 | v6 | Delta |
|---|---|---|---|
| CXL bandwidth utilized | 0% (not used) | ~60–80% of link BW | **New** |
| GPU HBM for KV cache | Limited by local GPU | Same GPU + overflow to CXL | +20× capacity |
| CPU DRAM for weights | 140 GB × N instances | 0 (shared in CXL) | **Freed for KV cache** |
| Total effective KV capacity (4-node) | ~600 GB (GPU) + 2 TB (CPU) | ~600 GB (GPU) + 8 TB (CXL fabric) | **+4×** |

---

## 15. Mapping to Linqu Hierarchy

v6's CXL memory model maps naturally onto the Linqu hierarchical machine (see `machine_hierarchy_and_function_hierarchy.md`):

| Linqu Level | Memory Domain | v6 Component |
|---|---|---|
| **Level 0** (Core) | Core-local scratchpad / registers | Attention kernel local state |
| **Level 2** (Chip) | GPU/NPU HBM | L1: `RadixKvManager` + `BlockPool` |
| **Level 3** (Host) | Local DRAM + CXL-attached expander | CXL local tier (~150ns) |
| **Level 4** (Cluster-level-0) | CXL-pooled memory within a switch domain | `PgasBlockPool` same-switch region |
| **Level 5** (Cluster-level-1) | CXL-pooled memory across switches (same fabric) | `PgasBlockPool` cross-switch region |
| **Level 6** (Cluster-level-2) | RDMA across CXL fabrics or data centers | v5's `RemoteStoreConnector` (fallback) |
| **Level 7** (Global Coordinator) | Global routing and scheduling | `CxlAwareRouter` |

This mapping means the compiler's **hierarchy labels** (§2.1 of the hierarchy doc) can directly inform memory placement decisions:

- A KV block tagged at **Level 4** should be placed in the CXL-pooled region for same-switch sharing.
- A KV block tagged at **Level 2** stays in GPU HBM.
- Model weights tagged at **Level 5** should go in the shared weight pool for fabric-wide access.

The `SymmetricHeap` API (shmem_put/get) naturally maps to data movement operations across Linqu levels, providing a uniform programming model regardless of which level boundary the data crosses.

---

## Key Design Decisions — Rationale

**CXL as the primary cross-instance tier, not a new tier stacked on v5:** CXL shared memory has lower latency than v5's RDMA-based remote store (~300ns vs ~5μs) and higher capacity (terabytes on a fabric). It replaces the L2+L4 tiers rather than adding a new one, simplifying the hierarchy.

**Lock-free data structures in shared memory:** Multiple hosts concurrently access the block allocator and hash index. We use CAS-based lock-free structures (Treiber stack, open-addressing hash table) to avoid cross-host lock contention, which would be catastrophic over CXL latencies.

**OpenSHMEM-style API:** Wrapping CXL operations in `shmem_put/get/ptr` provides a well-understood PGAS programming model. This enables future integration with actual OpenSHMEM runtimes (SOS, OSHMPI) for portability across CXL and non-CXL PGAS fabrics.

**Shared weight pool as an optimization, not a requirement:** Weight sharing is pure win (saves N×140GB), but it introduces a startup dependency (leader must load first). The follower waits only for weight loading, not for any ongoing coordination — after startup, weight access is a read-only mmap.

**Distance-weighted routing instead of binary routing:** On CXL fabric, every block is accessible from every instance — the question is not "who has it?" but "who is closest to it?" Distance-weighted scoring captures the continuous nature of CXL latency tiers.

**Fault tolerance via heartbeat + GC, not replication:** v5's remote store can replicate blocks for durability. In CXL shared memory, replication is expensive (doubles memory usage) and often unnecessary — KV blocks are recomputable. Instead, we detect failures via heartbeat and let affected requests recompute their KV. This trades rare recomputation for 2× memory efficiency.

---

*v6 takes the v5 multi-tier cache pool and replaces its copy-based, message-passing data movement with CXL load/store semantics. On CXL hardware, this eliminates the transfer engine bottleneck, unifies local and remote memory into a single PGAS, enables zero-copy weight sharing, and pushes the system to the hardware limits of CXL bandwidth and latency. On non-CXL hardware, every v6 path degrades gracefully to v5 behavior.*
