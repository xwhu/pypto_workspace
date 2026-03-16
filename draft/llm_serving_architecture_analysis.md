# LLM Serving Project Architecture Analysis

## Executive Summary

Based on the analysis of 5 reference implementations (nano-vllm, nano-vllm-v1, mini-sglang, mistral.rs, and lmdeploy), this report identifies the architectural patterns, best practices, and design principles that make an excellent LLM serving system.

---

## 1. Nano-vLLM (Minimal vLLM Implementation)

### Overall Architecture
Nano-vLLM is a lightweight (~1,200 lines) educational implementation of vLLM that demonstrates core serving concepts without the complexity of the full vLLM codebase.

**Main Components:**
- **`LLMEngine`** (`ref/nano-vllm/nanovllm/engine/llm_engine.py`): Central orchestrator that coordinates all components
- **`Scheduler`** (`ref/nano-vllm/nanovllm/engine/scheduler.py`): Handles request batching and scheduling decisions
- **`ModelRunner`** (`ref/nano-vllm/nanovllm/engine/model_runner.py`): Manages model execution, tensor parallelism, and CUDA graphs
- **`BlockManager`** (`ref/nano-vllm/nanovllm/engine/block_manager.py`): Implements PagedAttention-style KV cache management with prefix caching
- **`Sequence`** (`ref/nano-vllm/nanovllm/engine/sequence.py`): Represents individual inference requests with state management

### Request Routing/Scheduling
- **Two-phase scheduling**: Prefill phase (processes new prompts) followed by decode phase (generates tokens)
- **Simple FCFS (First-Come-First-Served)** with token budget constraints
- **Preemption support**: Running sequences can be preempted and returned to waiting queue when memory is constrained
- **Separate queues**: `waiting` queue for new requests, `running` queue for active sequences

```python
# From scheduler.py - two-phase scheduling logic
def schedule(self) -> tuple[list[Sequence], bool]:
    # prefill
    while self.waiting and num_seqs < self.max_num_seqs:
        # ... allocate blocks and schedule
    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill=True

    # decode
    while self.running and num_seqs < self.max_num_seqs:
        # ... handle decode scheduling with preemption
```

### Model Loading and Management
- Uses PyTorch distributed for tensor parallelism (NCCL backend)
- Model weights loaded via HuggingFace transformers
- KV cache dynamically allocated based on available GPU memory
- CUDA graph capture for decode phase optimization

### Inference Execution Flow
1. **Request Ingestion**: Tokenize prompts and create Sequence objects
2. **Scheduling**: Determine batch composition (prefill vs decode)
3. **Block Allocation**: Allocate KV cache blocks via BlockManager
4. **Model Execution**: Run forward pass with appropriate context (cu_seqlens, slot_mapping, etc.)
5. **Sampling**: Generate next tokens using temperature-based sampling
6. **Post-processing**: Update sequence states, check completion criteria

### Unique Architectural Patterns
- **SharedMemory IPC**: Uses multiprocessing SharedMemory for cross-process communication in tensor parallelism
- **Context pattern**: Uses `set_context()`/`get_context()`/`reset_context()` for managing attention metadata
- **Block-level prefix caching**: Hash-based block reuse for common prefixes

---

## 2. Nano-vLLM-v1 (vLLM v1 Scheduler Reproduction)

### Overall Architecture
An evolution of nano-vLLM that reproduces the vLLM v1 scheduler architecture with chunked prefill support.

**Key Differences from nano-vLLM:**
- Implements **vLLM v1 scheduler** with unified prefill/decode scheduling
- **Chunked prefill**: Long prompts can be split across multiple forward passes
- More sophisticated token budget management

### Request Routing/Scheduling
- **Unified scheduling**: Running and waiting sequences compete for the same token budget
- **Chunked prefill support**: Sequences can be partially processed and resumed
- **Priority-based scheduling**: Running sequences are scheduled first, then new requests

```python
# From scheduler.py - unified scheduling with chunked prefill
def schedule(self) -> tuple[list[Sequence], bool]:
    token_budget = self.max_num_batched_tokens
    
    # schedule from the running queue first
    while req_index < len(self.running) and token_budget > 0:
        num_new_tokens = min(num_new_tokens, token_budget)  # chunked prefill
        
    # schedule from the waiting queue
    while self.waiting and token_budget > 0:
        # ... allocate with prefix caching
```

### Unique Architectural Patterns
- **Token-level scheduling**: Tracks `num_new_tokens` per sequence for chunked processing
- **Prefix caching integration**: Block manager computes cache hits before allocation
- **Online serving support**: Includes `serving_bench.py` for request rate testing

---

## 3. Mini-SGLang (Minimal SGLang Implementation)

### Overall Architecture
A ~5,000 line implementation of SGLang with advanced optimizations like RadixCache and overlap scheduling.

**Main Components:**
- **`Engine`** (`ref/mini-sglang/python/minisgl/engine/engine.py`): Core inference engine
- **`Scheduler`** (`ref/mini-sglang/python/minisgl/scheduler/scheduler.py`): Advanced scheduler with overlap support
- **`PrefillManager`/`DecodeManager`**: Separate managers for prefill and decode phases
- **`CacheManager`**: Implements RadixCache for prefix caching
- **`APIServer`** (`ref/mini-sglang/python/minisgl/server/api_server.py`): FastAPI-based HTTP server

### Request Routing/Scheduling
- **Overlap scheduling**: CPU scheduling overhead is hidden behind GPU computation
- **Two-stream architecture**: Separate CUDA streams for scheduling and execution
- **RadixCache**: Tree-based prefix caching for efficient KV cache reuse

```python
# From scheduler.py - overlap scheduling
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    # Process messages (non-blocking)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)
    
    # Schedule next batch
    forward_input = self._schedule_next_batch()
    
    # Run batch in engine stream while processing continues in scheduler stream
    with self.engine_stream_ctx:
        ongoing_data = (forward_input, self._forward(forward_input))
    
    # Process last batch's results
    self._process_last_data(last_data)
```

### Model Loading and Management
- **Meta-tensor initialization**: Models initialized on meta device before weight loading
- **Dynamic KV cache allocation**: Based on available memory after model loading
- **Multiple attention backends**: FlashAttention, FlashInfer, TRT-LLM support
- **CUDA graph capture**: Automatic graph capture for common batch sizes

### Inference Execution Flow
1. **Request Ingestion**: ZMQ-based message passing from API server to scheduler
2. **Tokenization**: Separate tokenizer process for parallelization
3. **Prefix Matching**: RadixCache lookup for prefix reuse
4. **Batch Preparation**: Prepare input tensors, positions, attention metadata
5. **Model Execution**: Forward pass with selected attention backend
6. **Sampling**: Generate next tokens
7. **Response Streaming**: Async streaming back to client

### Unique Architectural Patterns
- **ZMQ-based IPC**: Decouples API server, tokenizer, and scheduler
- **Context manager pattern**: `forward_batch()` context for global state management
- **Modular backend system**: Pluggable attention and MoE backends
- **Lazy free region**: Deferred deallocation for cache management

---

## 4. Mistral.rs (Rust-based LLM Inference Engine)

### Overall Architecture
A production-grade Rust implementation with comprehensive model support (text, vision, speech, embeddings) and advanced features.

**Main Components:**
- **`Engine`** (`ref/mistral.rs/mistralrs-core/src/engine/mod.rs`): Core async engine with request handling
- **`Scheduler`** (`ref/mistral.rs/mistralrs-core/src/scheduler/`): Multiple scheduler implementations
- **`Pipeline`**: Abstraction for different model types
- **`Model`**: High-level API wrapper (`ref/mistral.rs/mistralrs/src/model.rs`)

### Request Routing/Scheduling
- **Multiple scheduler types**: DefaultScheduler and PagedAttentionScheduler
- **Bucketing**: Groups sequences by length for efficient batching
- **Priority and urgency**: Sequences can have urgency levels affecting scheduling

```rust
// From default_scheduler.rs - bucketing strategy
fn bucket_and_waitlist_seqs(&mut self, running: Vec<Sequence>) -> Vec<Sequence> {
    // Move sequences into buckets by (length, has_images, token_offset)
    // Allow shortest sequences to run, waitlist others
}
```

### Model Loading and Management
- **Builder pattern**: Multiple model builders (TextModelBuilder, VisionModelBuilder, etc.)
- **ISQ (In-Situ Quantization)**: Dynamic quantization during model loading
- **Device mapping**: Automatic multi-GPU placement
- **Adapter support**: LoRA, X-LoRA, AnyMoE

### Inference Execution Flow
1. **Request Submission**: Async request submission via channels
2. **Request Handling**: `handle_request()` processes different request types
3. **Scheduling**: Scheduler selects sequences for next step
4. **Pipeline Step**: Model forward pass with cache management
5. **Response Generation**: Stream or batch responses back

### Unique Architectural Patterns
- **Tokio-based async**: Full async/await throughout
- **Arc<Mutex<>> pattern**: Shared state management with interior mutability
- **Pipeline abstraction**: Unified interface across model types
- **Prefix caching v2**: Sophisticated prefix caching with recurrent state support
- **Tool calling**: Built-in support for function calling and MCP

---

## 5. LMDeploy (Deployment Toolkit for LLMs)

### Overall Architecture
A comprehensive deployment toolkit with both PyTorch and TurboMind backends, supporting production features like disaggregated serving.

**Main Components:**
- **`Engine`** (`ref/lmdeploy/lmdeploy/pytorch/engine/engine.py`): Main engine with async loop
- **`EngineLoop`** (`ref/lmdeploy/lmdeploy/pytorch/engine/engine_loop.py`): Async event loop management
- **`Scheduler`** (`ref/lmdeploy/lmdeploy/pytorch/paging/scheduler.py`): PagedAttention-based scheduler
- **`BlockManager`**: Memory management with block trie for prefix caching

### Request Routing/Scheduling
- **Session-based**: Requests organized into sessions with multiple sequences
- **Multi-phase scheduling**: Separate prefill and decode scheduling
- **Migration support**: For disaggregated serving (Prefill/Decode separation)
- **Eviction policies**: Configurable eviction strategies for memory pressure

```python
# From scheduler.py - session-based scheduling
def _schedule_prefill(self, prealloc_size: int = 0):
    max_batches = self.scheduler_config.max_batches - self.num_ready() - self.num_running()
    # ... schedule with prefix matching via block_trie
```

### Model Loading and Management
- **Strategy pattern**: Different strategies for different model types
- **Executor abstraction**: Pluggable execution backends
- **Adapter management**: Built-in LoRA/adapter support
- **Quantization**: AWQ, GPTQ, and other quantization methods

### Inference Execution Flow
1. **Request Management**: Request manager handles incoming requests
2. **Session Management**: Sessions maintain conversation state
3. **Scheduling**: Scheduler selects sequences based on priority and constraints
4. **Input Preparation**: InputsMaker prepares model inputs
5. **Execution**: Executor runs forward pass
6. **Response Handling**: Async response streaming

### Unique Architectural Patterns
- **Async/await throughout**: Fully async architecture
- **Strategy factory**: Pluggable strategies for different behaviors
- **Disaggregated serving**: Support for separate prefill and decode engines
- **Block trie**: Efficient prefix matching data structure
- **Event-driven**: Uses asyncio events for coordination

---

## Summary: What Makes a Good Serving Project Architecture

### Common Patterns Across All Projects

1. **Layered Architecture**
   - API Layer (HTTP server, protocol handling)
   - Scheduling Layer (batching, prioritization)
   - Execution Layer (model inference, kernel optimization)
   - Memory Management Layer (KV cache, block allocation)

2. **Async/Concurrent Design**
   - All projects use async patterns (asyncio in Python, Tokio in Rust)
   - Separation of I/O-bound and compute-bound work
   - Non-blocking request handling

3. **Modular Scheduling**
   - Clear separation between scheduler and executor
   - Pluggable scheduling policies
   - Support for both prefill and decode phases

4. **Memory Efficiency**
   - PagedAttention-style KV cache management
   - Prefix caching for common prompts
   - Dynamic memory allocation based on workload

5. **Request Lifecycle Management**
   - Explicit state machines for sequences (WAITING -> RUNNING -> FINISHED)
   - Proper cleanup and resource deallocation
   - Support for request cancellation

### Best Practices Observed

1. **Separation of Concerns**
   - Nano-vLLM: Clean separation between scheduler, model runner, and block manager
   - Mini-SGLang: Distinct managers for prefill, decode, and cache
   - Mistral.rs: Pipeline abstraction separates model-specific logic

2. **Performance Optimizations**
   - CUDA graphs for decode (nano-vLLM, mini-SGLang)
   - Overlap scheduling (mini-SGLang)
   - Bucketing for efficient batching (mistral.rs)

3. **Scalability Features**
   - Tensor parallelism support (all projects)
   - Multi-GPU support with proper synchronization
   - Distributed serving capabilities (LMDeploy)

4. **Production Readiness**
   - Comprehensive error handling (mistral.rs)
   - Metrics and logging (all projects)
   - Graceful degradation under load

### Architectural Principles

1. **Composability**: Components should be easily replaceable (schedulers, attention backends)
2. **Observability**: Built-in metrics, logging, and tracing
3. **Flexibility**: Support for different model types, quantization, and optimizations
4. **Efficiency**: Minimize memory overhead, maximize GPU utilization
5. **Reliability**: Proper error handling, request isolation, resource cleanup

### Key File References

| Project | Core Engine | Scheduler | Model Runner | Cache Management |
|---------|-------------|-----------|--------------|------------------|
| nano-vllm | `engine/llm_engine.py` | `engine/scheduler.py` | `engine/model_runner.py` | `engine/block_manager.py` |
| nano-vllm-v1 | `engine/llm_engine.py` | `engine/scheduler.py` | `engine/model_runner.py` | `engine/block_manager.py` |
| mini-sglang | `engine/engine.py` | `scheduler/scheduler.py` | `engine/engine.py` | `scheduler/cache.py` |
| mistral.rs | `engine/mod.rs` | `scheduler/` | `pipeline/` | `paged_attention/` |
| lmdeploy | `engine/engine.py` | `paging/scheduler.py` | `engine/executor.py` | `paging/block_manager.py` |

---

## Recommendations for New Serving Projects

Based on this analysis, a new LLM serving project should:

1. **Start with a clean layered architecture** separating API, scheduling, execution, and memory layers
2. **Implement async/await patterns** from the beginning for scalability
3. **Use PagedAttention-style memory management** with prefix caching
4. **Design for composability** with pluggable schedulers and attention backends
5. **Include production features** like metrics, logging, and error handling from day one
6. **Support both prefill and decode phases** with appropriate scheduling strategies
7. **Consider overlap scheduling** to hide CPU overhead behind GPU computation
8. **Implement proper request lifecycle management** with state machines and cleanup
