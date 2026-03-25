# Ascend Paged KV Cache Integration (V2) - Design Spec

## 重新设计目标
我们需要基于 `vllm` 与 `vllm-ascend` 的最新源码架构，在 Rust 中从头实现一套高性能、适配 NPU 的 Paged KV Cache 与 Radix Tree / Prefix Cache 管理器。

通过阅读 `vllm` 及 `vllm-ascend` 的核心代码，我们获取了以下关键设计灵感：
1. **Prefix Caching 的数据结构 (类似 vLLM v1 core)**: 
   - 传统 Radix Tree 需要复杂的节点分裂与合并逻辑。vLLM 最新版（V1）使用了基于密码学哈希的 **Hash Chain Prefix Cache**（即缓存块的哈希值 = `Hash(Parent Hash + Token IDs)`）。
   - 通过维持一个 `HashMap<BlockHash, KVCacheBlock>` 和自定义的 **双向链表 LRU Queue** (`FreeKVCacheBlockQueue`)，即可实现 **O(1)** 复杂度的完美前缀匹配和淘汰。
2. **底层 Paged Attention 的调用机制**: 
   - `vllm-ascend` 底层直接透传调用 `torch_npu._npu_paged_attention`。这对应着我们 CANN FFI 层的 `aclnnIncreFlashAttentionV4`（Decode）以及全量 Attention 算子。
   - 由于我们已知 CANN 算子异步调度存在严格的生命周期要求（之前的 Bug），结合 NPU 对显存连续性的极度敏感，我们将按 Layer-first 分配全局固定缓存，通过 Block Table 的映射机制调度物理 Block。

## User Review Required
> [!IMPORTANT]
> 针对新的 `radix_tree.rs` (或者 `prefix_cache.rs`) 我们有两个架构路线，请您定夺：
> 
> **方案A（vLLM V0 路线：严格基数树结构）**: 
> 严格实现 `RadixNode` 对象，维护 parent 和 dict 形式的 children。带有显式的节点分裂（Split）和合并逻辑。管理比较直观，但实现复杂度略高，且多级遍历有一定的开销。
> 
> **方案B (vLLM V1 路线：哈希链前缀缓存 O(1))【推荐】**: 
> 参考 `vllm/v1/core/kv_cache_utils.py`。不维护庞大的物理树，只为每个满载 Block 算一个全局级联 Hash。基于此我们可以直接 O(1) 用 Hash 在散列表中快速查到前缀匹配。淘汰时使用简单的双向链表 LRU。Rust 实现起来非常安全高效！

## Proposed Architecture (Rust)

我们将创建一个独立的子包/模块 `kv-cache`，包含以下核心文件：

### 1. `kv-cache/src/radix_tree.rs` (或者 `prefix_cache.rs`)
- [NEW] 实现前缀匹配逻辑。如果选择方案B，我们将实现 `BlockHash` 计算和 `BlockHash -> BlockId` 的高速映射。
- [NEW] 实现 `FreeBlockQueue`（基于双向链表数组或 Rust `VecDeque` 的 O(1) LRU 淘汰机制）。

### 2. `kv-cache/src/block_manager.rs`
- [NEW] `BlockManager`: 管理 Token 到虚拟 Block 的切分（比如每个 block_size=16）。
- 结合 `PrefixCache` 执行 `allocate` (分配新块) 和 `free` (释放/驱逐无用块) 逻辑。

### 3. `kv-cache/src/npu_memory.rs` (与 CANN 的抽象对接)
- 在框架初始化时，通过 CANN 一次性预分配全局巨大的 `DeviceTensor` (按 Layer 大小切分保证单层内的 contiguous 内存排布，这也是 CANN IncreFlashAttention 的必须要求)。
- 提供按照 `block_id` 在预分配的大 tensor 内进行内存写偏移（Offset）的计算方法。

## Verification Plan
1. **单元测试 (`cargo test -p kv-cache`)**: 独立跑通 Radix Cache / Prefix Cache 的分配、淘汰、前缀命中的边界逻辑验证。
2. **算子联调**: 注入 Dummy NPU 数据，跑通 `aclnnIncreFlashAttentionV4`。用 `stream.synchronize()` 强封锁规避 CANN 底层的异步 UAF (Use-After-Free) 崩溃问题。
