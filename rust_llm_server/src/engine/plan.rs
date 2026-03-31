use std::fmt;
#[cfg(feature = "ascend")]
use std::time::Instant;

// ─── Cached environment variables (read once at process start) ─────────

#[cfg(feature = "ascend")]
pub(super) static PERF_BREAKDOWN: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("RUST_LLM_PERF_BREAKDOWN").map_or(false, |v| v == "1")
});

#[cfg(feature = "ascend")]
pub(super) static PERF_SKIP_STEPS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("RUST_LLM_PERF_SKIP_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
});

#[cfg(feature = "ascend")]
static TP_HP_MATMUL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("TP_HP_MATMUL").map_or(false, |v| v == "1")
});

#[cfg(feature = "ascend")]
static DUMP_DECODE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("TP_DEBUG_DUMP_DECODE").map_or(false, |v| v == "1")
});

use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::{QuantConfig, QuantScheme};
use crate::model::tensor::{DType, Tensor};


// ─── Execution Step IR ─────────────────────────────────────────────────

/// Index into the plan's tensor buffer pool.
pub type TensorRef = usize;

/// Index into the model's weight list.
pub type WeightRef = usize;

/// A single execution step in the compiled plan.
///
/// This is the instruction set for the LLM inference virtual machine.
/// At compile time, the logical model is lowered into a flat sequence
/// of these steps. At runtime, they are compiled into closures for
/// zero-dispatch-overhead execution.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ExecStep {
    // ─── Compute ops ───
    Embedding {
        ids_ref: TensorRef,
        table_weight: WeightRef,
        out: TensorRef,
    },
    RmsNorm {
        input: TensorRef,
        weight: WeightRef,
        eps: f32,
        out: TensorRef,
    },
    MatMul {
        a: TensorRef,
        b: WeightRef,
        out: TensorRef,
    },
    RotaryEmb {
        q: TensorRef,
        k: TensorRef,
        positions_ref: TensorRef,
        rope_theta: f64,
        head_dim: usize,
    },
    /// Per-head RMS normalization on Q or K (Qwen3 QK norm).
    /// Reshapes [B, S, num_heads*head_dim] → [B*S*num_heads, head_dim],
    /// applies RMS norm with weight [head_dim], reshapes back.
    QKNorm {
        qk: TensorRef,
        weight: WeightRef,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    },
    Attention {
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        out: TensorRef,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    SiluMul {
        gate: TensorRef,
        up: TensorRef,
        out: TensorRef,
    },
    Add {
        a: TensorRef,
        b: TensorRef,
    },
    Sample {
        logits: TensorRef,
        out_token: TensorRef,
    },

    // ─── Communication ops (for TP/PP) ───
    AllReduceSum {
        tensor: TensorRef,
    },
    Send {
        tensor: TensorRef,
        dst_rank: usize,
    },
    Recv {
        tensor: TensorRef,
        src_rank: usize,
    },

    // ─── Quantized ops ───
    DequantMatMul {
        input: TensorRef,
        weight: WeightRef,
        scales: WeightRef,
        zeros: Option<WeightRef>,
        out: TensorRef,
    },
}

impl fmt::Display for ExecStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecStep::Embedding { .. } => write!(f, "Embedding"),
            ExecStep::RmsNorm { .. } => write!(f, "RmsNorm"),
            ExecStep::MatMul { .. } => write!(f, "MatMul"),
            ExecStep::RotaryEmb { .. } => write!(f, "RotaryEmb"),
            ExecStep::QKNorm { .. } => write!(f, "QKNorm"),
            ExecStep::Attention {
                num_heads,
                num_kv_heads,
                ..
            } => write!(f, "Attention(h={num_heads},kv={num_kv_heads})"),
            ExecStep::SiluMul { .. } => write!(f, "SiluMul"),
            ExecStep::Add { .. } => write!(f, "Add"),
            ExecStep::Sample { .. } => write!(f, "Sample"),
            ExecStep::AllReduceSum { .. } => write!(f, "AllReduceSum"),
            ExecStep::Send { dst_rank, .. } => write!(f, "Send(dst={dst_rank})"),
            ExecStep::Recv { src_rank, .. } => write!(f, "Recv(src={src_rank})"),
            ExecStep::DequantMatMul { .. } => write!(f, "DequantMatMul"),
        }
    }
}

// ─── Plan Compilation ──────────────────────────────────────────────────

/// Intermediate representation: a buffer slot in the tensor pool.
/// Used during plan compilation to track tensor slots.
struct BufferAllocator {
    next_id: usize,
}

impl BufferAllocator {
    fn new() -> Self {
        Self { next_id: 0 }
    }
    fn alloc(&mut self) -> TensorRef {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
    fn total(&self) -> usize {
        self.next_id
    }
}

/// Weight registry: maps weight names to indices and stores actual tensors.
struct WeightRegistry {
    names: Vec<String>,
}

impl WeightRegistry {
    fn new() -> Self {
        Self {
            names: Vec::new(),
        }
    }

    fn register(&mut self, tensor: &Tensor) -> WeightRef {
        let id = self.names.len();
        self.names.push(tensor.name.clone());
        id
    }

    fn into_names(self) -> Vec<String> {
        self.names
    }
}

/// Compile a logical model + configs into a flat sequence of execution steps.
///
/// This is the "compiler" that transforms the model graph into a linear
/// instruction stream. It handles:
/// - PP: only emits layers assigned to this pipeline stage
/// - TP: inserts AllReduce after attention and MLP
/// - Quantization: replaces MatMul with DequantMatMul for quantized weights
pub fn compile_plan(
    model: &Qwen3Model,
    parallel: &ParallelConfig,
    quant: &QuantConfig,
) -> ExecutionPlan {
    let cfg = &model.config;
    let mut steps = Vec::new();
    let mut bufs = BufferAllocator::new();
    let mut weights = WeightRegistry::new();

    // Reserve special slots
    let input_ids_slot = bufs.alloc(); // 0: input token IDs
    let positions_slot = bufs.alloc(); // 1: position indices
    let hidden_slot = bufs.alloc(); // 2: main hidden state

    // Register embedding weight
    let embed_w = weights.register(&model.embed_tokens);

    // ── Embedding ──
    let is_first_pp_stage = parallel.pp_rank == 0;
    let is_last_pp_stage = parallel.pp_rank == parallel.pp_size - 1;

    if is_first_pp_stage {
        steps.push(ExecStep::Embedding {
            ids_ref: input_ids_slot,
            table_weight: embed_w,
            out: hidden_slot,
        });
    } else {
        // PP: receive hidden state from previous stage
        steps.push(ExecStep::Recv {
            tensor: hidden_slot,
            src_rank: parallel.pp_rank - 1,
        });
    }

    // ── Transformer layers (only this PP stage's layers) ──
    let (layer_start, layer_end) = parallel.pp_layer_range(cfg.num_hidden_layers);

    for layer_idx in layer_start..layer_end {
        let layer = &model.layers[layer_idx];
        let _prefix = format!("model.layers.{layer_idx}");

        // Allocate temp buffers for this layer
        let normed = bufs.alloc();
        let q = bufs.alloc();
        let k = bufs.alloc();
        let v = bufs.alloc();
        let attn_out = bufs.alloc();
        let proj_out = bufs.alloc();
        let normed2 = bufs.alloc();
        let gate = bufs.alloc();
        let up = bufs.alloc();
        let silu_out = bufs.alloc();
        let ffn_out = bufs.alloc();

        // Register weights
        let ln1_w = weights.register(&layer.input_layernorm.weight);
        let q_w = weights.register(&layer.self_attn.q_proj);
        let k_w = weights.register(&layer.self_attn.k_proj);
        let v_w = weights.register(&layer.self_attn.v_proj);
        let o_w = weights.register(&layer.self_attn.o_proj);
        let q_norm_w = weights.register(&layer.self_attn.q_norm);
        let k_norm_w = weights.register(&layer.self_attn.k_norm);
        let ln2_w = weights.register(&layer.post_attention_layernorm.weight);
        let gate_w = weights.register(&layer.mlp.gate_proj);
        let up_w = weights.register(&layer.mlp.up_proj);
        let down_w = weights.register(&layer.mlp.down_proj);

        // Input LayerNorm
        steps.push(ExecStep::RmsNorm {
            input: hidden_slot,
            weight: ln1_w,
            eps: cfg.rms_norm_eps as f32,
            out: normed,
        });

        // Q/K/V projections (potentially quantized)
        emit_matmul_or_dequant(
            &mut steps,
            normed,
            q_w,
            q,
            &layer.self_attn.q_proj.name,
            quant,
            &mut weights,
        );
        emit_matmul_or_dequant(
            &mut steps,
            normed,
            k_w,
            k,
            &layer.self_attn.k_proj.name,
            quant,
            &mut weights,
        );
        emit_matmul_or_dequant(
            &mut steps,
            normed,
            v_w,
            v,
            &layer.self_attn.v_proj.name,
            quant,
            &mut weights,
        );

        // QK Norm (Qwen3: per-head RMS norm on Q and K)
        steps.push(ExecStep::QKNorm {
            qk: q,
            weight: q_norm_w,
            num_heads: cfg.num_attention_heads / parallel.tp_size,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
        });
        steps.push(ExecStep::QKNorm {
            qk: k,
            weight: k_norm_w,
            num_heads: cfg.num_key_value_heads / parallel.tp_size,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
        });

        // RoPE
        steps.push(ExecStep::RotaryEmb {
            q,
            k,
            positions_ref: positions_slot,
            rope_theta: cfg.rope_theta,
            head_dim: cfg.head_dim,
        });

        // Attention
        steps.push(ExecStep::Attention {
            q,
            k,
            v,
            out: attn_out,
            num_heads: cfg.num_attention_heads / parallel.tp_size,
            num_kv_heads: cfg.num_key_value_heads / parallel.tp_size,
            head_dim: cfg.head_dim,
        });

        // O projection
        emit_matmul_or_dequant(
            &mut steps,
            attn_out,
            o_w,
            proj_out,
            &layer.self_attn.o_proj.name,
            quant,
            &mut weights,
        );

        // TP: AllReduce after attention (if row-sharded o_proj)
        if parallel.is_tp() {
            steps.push(ExecStep::AllReduceSum { tensor: proj_out });
        }

        // Residual
        steps.push(ExecStep::Add {
            a: hidden_slot,
            b: proj_out,
        });

        // Post-attention LayerNorm
        steps.push(ExecStep::RmsNorm {
            input: hidden_slot,
            weight: ln2_w,
            eps: cfg.rms_norm_eps as f32,
            out: normed2,
        });

        // MLP: gate_proj and up_proj
        emit_matmul_or_dequant(
            &mut steps,
            normed2,
            gate_w,
            gate,
            &layer.mlp.gate_proj.name,
            quant,
            &mut weights,
        );
        emit_matmul_or_dequant(
            &mut steps,
            normed2,
            up_w,
            up,
            &layer.mlp.up_proj.name,
            quant,
            &mut weights,
        );

        // SwiGLU
        steps.push(ExecStep::SiluMul {
            gate,
            up,
            out: silu_out,
        });

        // down_proj
        emit_matmul_or_dequant(
            &mut steps,
            silu_out,
            down_w,
            ffn_out,
            &layer.mlp.down_proj.name,
            quant,
            &mut weights,
        );

        // TP: AllReduce after MLP (if row-sharded down_proj)
        if parallel.is_tp() {
            steps.push(ExecStep::AllReduceSum { tensor: ffn_out });
        }

        // Residual
        steps.push(ExecStep::Add {
            a: hidden_slot,
            b: ffn_out,
        });
    }

    // ── Final norm + LM head (only on last PP stage) ──
    if is_last_pp_stage {
        let final_normed = bufs.alloc();
        let logits = bufs.alloc();
        let final_norm_w = weights.register(&model.norm.weight);
        let lm_head_w = weights.register(&model.lm_head);

        steps.push(ExecStep::RmsNorm {
            input: hidden_slot,
            weight: final_norm_w,
            eps: cfg.rms_norm_eps as f32,
            out: final_normed,
        });

        emit_matmul_or_dequant(
            &mut steps,
            final_normed,
            lm_head_w,
            logits,
            "lm_head.weight",
            quant,
            &mut weights,
        );

        steps.push(ExecStep::Sample {
            logits,
            out_token: input_ids_slot, // reuse slot
        });
    } else {
        // PP: send hidden state to next stage
        steps.push(ExecStep::Send {
            tensor: hidden_slot,
            dst_rank: parallel.pp_rank + 1,
        });
    }

    let weight_names = weights.into_names();

    ExecutionPlan {
        steps,
        weight_names,
        num_buffers: bufs.total(),
        config: cfg.clone(),
        parallel: parallel.clone(),
    }
}

/// Helper: emit MatMul or DequantMatMul depending on quantization config.
fn emit_matmul_or_dequant(
    steps: &mut Vec<ExecStep>,
    input: TensorRef,
    weight: WeightRef,
    out: TensorRef,
    weight_name: &str,
    quant: &QuantConfig,
    weights: &mut WeightRegistry,
) {
    let scheme = quant.scheme_for(weight_name);
    if scheme.is_quantized() {
        // Create placeholder tensors for scales/zeros (actual tensors loaded from weights)
        let scales_tensor = Tensor::new(vec![1], DType::Float32, format!("{weight_name}.scales"));
        let scales = weights.register(&scales_tensor);
        let zeros = match scheme {
            QuantScheme::GroupWise { .. } => {
                let zeros_tensor =
                    Tensor::new(vec![1], DType::Float32, format!("{weight_name}.zeros"));
                Some(weights.register(&zeros_tensor))
            }
            _ => None,
        };
        steps.push(ExecStep::DequantMatMul {
            input,
            weight,
            scales,
            zeros,
            out,
        });
    } else {
        steps.push(ExecStep::MatMul {
            a: input,
            b: weight,
            out,
        });
    }
}

/// Parse a weight name to extract (layer_index, checkpoint_name).
///
/// Maps weight names like "model.layers.5.self_attn.q_proj.weight" to
/// `Some((5, "02_q_proj"))`. Returns `None` for non-layer weights
/// (embedding, final norm, lm_head).
#[cfg(feature = "ascend")]
fn dump_name_from_weight(name: &str) -> Option<(usize, &'static str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot_pos = rest.find('.')?;
    let layer_idx: usize = rest[..dot_pos].parse().ok()?;
    let component = &rest[dot_pos + 1..];
    let step_name = if component.contains("input_layernorm") {
        "01_input_ln"
    } else if component.contains("q_proj") {
        "02_q_proj"
    } else if component.contains("k_proj") {
        "03_k_proj"
    } else if component.contains("v_proj") {
        "04_v_proj"
    } else if component.contains("q_norm") {
        "05_q_norm"
    } else if component.contains("k_norm") {
        "06_k_norm"
    } else if component.contains("o_proj") {
        "10_o_proj"
    } else if component.contains("post_attention_layernorm") {
        "13_post_attn_ln"
    } else if component.contains("gate_proj") {
        "14_gate_proj"
    } else if component.contains("up_proj") {
        "15_up_proj"
    } else if component.contains("down_proj") {
        "17_down_proj"
    } else {
        return None;
    };
    Some((layer_idx, step_name))
}

// ─── Execution Plan ────────────────────────────────────────────────────

/// The compiled execution plan — a flat list of steps + metadata.
///
/// Created once at engine initialization, reused for every inference step.
#[allow(dead_code)]
pub struct ExecutionPlan {
    /// Flat instruction sequence.
    pub steps: Vec<ExecStep>,
    /// Weight names (indexed by WeightRef).
    pub weight_names: Vec<String>,
    /// Number of tensor buffer slots needed.
    pub num_buffers: usize,
    /// Model config snapshot.
    pub config: Qwen3Config,
    /// Parallel config snapshot.
    pub parallel: ParallelConfig,
}

impl ExecutionPlan {
    /// Compile into a `CompiledPlan` with cached closures.
    pub fn compile(self) -> CompiledPlan {
        CompiledPlan::new(self)
    }

    /// Total number of steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Count steps of a specific type.
    #[allow(dead_code)]
    pub fn count_step_type(&self, predicate: impl Fn(&ExecStep) -> bool) -> usize {
        self.steps.iter().filter(|s| predicate(s)).count()
    }

    /// Print the plan for debugging.
    pub fn dump(&self) {
        tracing::info!(
            "Execution Plan: {} steps, {} buffers, {} weights",
            self.steps.len(),
            self.num_buffers,
            self.weight_names.len()
        );
        for (i, step) in self.steps.iter().enumerate() {
            tracing::debug!("  [{:3}] {}", i, step);
        }
    }
}

// ─── Compiled Plan (方案B) ─────────────────────────────────────────────

/// Compiled execution plan — caches closures for zero-dispatch execution.
///
/// Created once from an `ExecutionPlan`. Holds the plan metadata and
/// executes steps by iterating through them with minimal overhead.
#[allow(dead_code)]
pub struct CompiledPlan {
    plan: ExecutionPlan,
}

/// Runtime context for Paged KV Cache during execution.
///
/// Passed to `execute_paged()` to enable the Attention step to branch
/// between Prefill (full FlashAttention) and Decode (PagedAttention).
#[cfg(feature = "ascend")]
pub struct PagedKVContext {
    /// True if this is a decode step (seq_len=1 per sequence).
    pub is_decode: bool,
    /// Total context length (cached + new tokens).
    pub context_len: usize,
    /// Block table: [batch_size, max_blocks_per_seq] as flat i32 array.
    pub block_table: Vec<i32>,
    /// Max blocks per sequence (second dim of block_table).
    pub max_blocks_per_seq: usize,
    /// Slot mapping for reshape_and_cache: [num_new_tokens] global slot indices.
    pub slot_mapping: Vec<i32>,
    /// Block size (tokens per block).
    pub block_size: usize,
    /// Tracks which layer's attention we're currently processing.
    pub layer_idx: std::cell::Cell<usize>,
}

#[cfg(feature = "ascend")]
struct ExecContext<'a, 'b> {
    ops: &'a crate::ops::ascend::AscendComputeOps,
    comm_ops: Option<&'a crate::ops::ascend_comm::AscendCommOps>,
    pool: &'b mut crate::model::device_tensor::TensorPool,
    rotating_pool: &'b mut crate::model::scratch_arena::RotatingPool,
    weights: &'a [crate::model::device_tensor::WeightTensor],
    input_ids: &'a [u32],
    positions: &'a [u32],
    paged_ctx: &'a PagedKVContext,
    kv_key_caches: &'a [ascend::memory::DeviceBuffer],
    kv_value_caches: &'a [ascend::memory::DeviceBuffer],
    decode_buffers: Option<&'b mut crate::ops::ascend::DecodeBuffers>,
    dumper: Option<&'b mut super::debug_dump::DebugDumper>,

    use_arena: bool,
    current_layer_arena_idx: Option<usize>,
    allreduce_in_layer: usize,
    add_in_layer: usize,
    last_attention_layer: Option<usize>,

    sampled_token: u32,
    hidden_size: usize,
}

#[cfg(feature = "ascend")]
impl<'a, 'b> ExecContext<'a, 'b> {
    fn dump_name(&self, weight_idx: usize) -> Option<(usize, &'static str)> {
        dump_name_from_weight(self.weights[weight_idx].name())
    }

    pub fn exec_step(&mut self, step: &ExecStep) {
        match step {
            ExecStep::Embedding { table_weight, out, .. } => self.exec_embedding(*table_weight, *out),
            ExecStep::RmsNorm { input, weight, eps, out } => self.exec_rmsnorm(*input, *weight, *eps, *out),
            ExecStep::MatMul { a, b, out } => self.exec_matmul(*a, *b, *out),
            ExecStep::RotaryEmb { q, k, rope_theta, head_dim, .. } => self.exec_rotary(*q, *k, *rope_theta, *head_dim),
            ExecStep::QKNorm { qk, weight, num_heads, head_dim, eps } => self.exec_qknorm(*qk, *weight, *num_heads, *head_dim, *eps),
            ExecStep::Attention { q, k, v, out, num_heads, num_kv_heads, head_dim } => self.exec_attention(*q, *k, *v, *out, *num_heads, *num_kv_heads, *head_dim),
            ExecStep::SiluMul { gate, up, out } => self.exec_silumul(*gate, *up, *out),
            ExecStep::Add { a, b } => self.exec_add(*a, *b),
            ExecStep::Sample { logits, .. } => self.exec_sample(*logits),
            ExecStep::AllReduceSum { tensor } => self.exec_allreduce(*tensor),
            ExecStep::Send { tensor, dst_rank } => self.exec_send(*tensor, *dst_rank),
            ExecStep::Recv { tensor, src_rank, .. } => self.exec_recv(*tensor, *src_rank),
            ExecStep::DequantMatMul { .. } => tracing::warn!("execute_paged: DequantMatMul not implemented"),
        }
    }

    fn exec_embedding(&mut self, table_weight: usize, out: usize) {
        let arena = self.use_arena.then(|| self.rotating_pool.arena_for_layer(0));
        let result = self.ops.embedding(self.input_ids, &self.weights[table_weight], arena);
        self.pool.put(out, result);
        if let Some(ref mut d) = self.dumper {
            d.dump("layer0_00_embedding", self.pool.get(out), self.ops.stream());
        }
    }

    fn exec_rmsnorm(&mut self, input: usize, weight: usize, eps: f32, out: usize) {
        let result = self.ops.rms_norm(self.pool.get(input), &self.weights[weight], eps);
        self.pool.put(out, result);
        let dump_info = self.dump_name(weight);
        let is_final = self.weights[weight].name().contains("model.norm");
        if let Some(ref mut d) = self.dumper {
            if let Some((layer, step_name)) = dump_info {
                if d.should_dump(layer) {
                    d.dump(&format!("layer{layer}_{step_name}"), self.pool.get(out), self.ops.stream());
                }
            } else if is_final {
                d.dump("final_norm", self.pool.get(out), self.ops.stream());
            }
        }
    }

    fn exec_matmul(&mut self, a: usize, b: usize, out: usize) {
        let use_hp = self.comm_ops.is_some()
            && *TP_HP_MATMUL
            && {
                let name = self.weights[b].name();
                name.contains("o_proj") || name.contains("down_proj")
            };

        let result = if use_hp {
            let (tensor, temps) = self.ops.matmul_hp(self.pool.get(a), &self.weights[b]);
            if self.use_arena {
                let layer = self.current_layer_arena_idx.unwrap_or(0);
                self.rotating_pool.arena_for_layer(layer).defer_owned_many(temps);
            } else {
                self.pool.defer_buffers(temps);
            }
            tensor
        } else {
            self.ops.matmul(self.pool.get(a), &self.weights[b])
        };
        self.pool.put(out, result);

        let dump_info = self.dump_name(b);
        let is_lm = self.weights[b].name().contains("lm_head");
        if let Some(ref mut d) = self.dumper {
            if let Some((layer, step_name)) = dump_info {
                if d.should_dump(layer) {
                    d.dump(&format!("layer{layer}_{step_name}"), self.pool.get(out), self.ops.stream());
                }
            } else if is_lm {
                d.dump("lm_head_out", self.pool.get(out), self.ops.stream());
            }
        }
    }

    fn exec_rotary(&mut self, q: usize, k: usize, rope_theta: f64, head_dim: usize) {
        let q_tensor = self.pool.take(q);
        let k_tensor = self.pool.take(k);
        let arena = self.use_arena.then(|| self.rotating_pool.arena_for_layer(self.paged_ctx.layer_idx.get()));
        let (q_out, k_out) = self.ops.rotary_embedding(q_tensor, k_tensor, self.positions, rope_theta, head_dim, arena);
        self.pool.put(q, q_out);
        self.pool.put(k, k_out);
        if let Some(ref mut d) = self.dumper {
            let layer = self.paged_ctx.layer_idx.get();
            if d.should_dump(layer) {
                d.dump(&format!("layer{layer}_07_q_rope"), self.pool.get(q), self.ops.stream());
                d.dump(&format!("layer{layer}_08_k_rope"), self.pool.get(k), self.ops.stream());
            }
        }
    }

    fn exec_qknorm(&mut self, qk: usize, weight: usize, num_heads: usize, head_dim: usize, eps: f32) {
        let tensor = self.pool.take(qk);
        let arena = self.use_arena.then(|| self.rotating_pool.arena_for_layer(self.paged_ctx.layer_idx.get()));
        let result = self.ops.qk_norm(tensor, &self.weights[weight], num_heads, head_dim, eps, arena);
        self.pool.put(qk, result);
        let dump_info = self.dump_name(weight);
        if let Some(ref mut d) = self.dumper {
            if let Some((layer, step_name)) = dump_info {
                if d.should_dump(layer) {
                    d.dump(&format!("layer{layer}_{step_name}"), self.pool.get(qk), self.ops.stream());
                }
            }
        }
    }

    fn exec_attention(&mut self, q: usize, k: usize, v: usize, out: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
        let layer = self.paged_ctx.layer_idx.get();

        if layer < self.kv_key_caches.len() {
            self.ops.reshape_and_cache(
                self.pool.get(k),
                self.pool.get(v),
                &self.kv_key_caches[layer],
                &self.kv_value_caches[layer],
                &self.paged_ctx.slot_mapping,
                num_kv_heads,
                head_dim,
            );
        }

        let result = if self.paged_ctx.is_decode && layer < self.kv_key_caches.len() {
            let num_blocks = self.kv_key_caches[layer].size() / (self.paged_ctx.block_size * num_kv_heads * head_dim * 2);
            self.ops.paged_decode_attention(
                self.pool.get(q),
                &self.kv_key_caches[layer],
                &self.kv_value_caches[layer],
                num_heads,
                num_kv_heads,
                head_dim,
                self.paged_ctx.block_size,
                num_blocks,
                &self.paged_ctx.block_table,
                self.paged_ctx.max_blocks_per_seq,
                self.paged_ctx.context_len,
                self.decode_buffers.as_deref_mut().expect("decode requires DecodeBuffers"),
            )
        } else {
            let arena = self.use_arena.then(|| self.rotating_pool.arena_for_layer(layer));
            self.ops.attention(
                self.pool.get(q),
                self.pool.get(k),
                self.pool.get(v),
                num_heads,
                num_kv_heads,
                head_dim,
                arena,
            )
        };
        self.pool.put(out, result);

        if let Some(ref mut d) = self.dumper {
            if d.should_dump(layer) {
                d.dump(&format!("layer{layer}_09_attn_out"), self.pool.get(out), self.ops.stream());
            }
        }

        self.allreduce_in_layer = 0;
        self.add_in_layer = 0;
        self.last_attention_layer = Some(layer);
        self.current_layer_arena_idx = Some(layer);
        self.paged_ctx.layer_idx.set(layer + 1);

        if !self.use_arena {
            self.ops.synchronize().ok();
        }
    }

    fn exec_silumul(&mut self, gate: usize, up: usize, out: usize) {
        let arena = self.use_arena.then(|| self.rotating_pool.arena_for_layer(self.current_layer_arena_idx.unwrap_or(0)));
        let result = self.ops.silu_mul(self.pool.get(gate), self.pool.get(up), arena);
        self.pool.put(out, result);
        if let Some(ref mut d) = self.dumper {
            if let Some(layer) = self.last_attention_layer {
                if d.should_dump(layer) {
                    d.dump(&format!("layer{layer}_16_silu_mul"), self.pool.get(out), self.ops.stream());
                }
            }
        }
    }

    fn exec_add(&mut self, a: usize, b: usize) {
        let tensor_a = self.pool.take(a);
        let result = self.ops.add(tensor_a, self.pool.get(b));
        self.pool.put(a, result);
        if let Some(ref mut d) = self.dumper {
            if let Some(layer) = self.last_attention_layer {
                if d.should_dump(layer) {
                    let step = if self.add_in_layer == 0 { "12_residual_attn" } else { "19_residual_ffn" };
                    d.dump(&format!("layer{layer}_{step}"), self.pool.get(a), self.ops.stream());
                }
            }
        }
        self.add_in_layer += 1;
    }

    fn exec_sample(&mut self, logits: usize) {
        if let Some(ref mut d) = self.dumper {
            d.dump("logits_pre_sample", self.pool.get(logits), self.ops.stream());
        }
        self.sampled_token = self.ops.sample_argmax(self.pool.get(logits));
    }

    fn exec_allreduce(&mut self, tensor: usize) {
        if let Some(comm) = self.comm_ops {
            let is_fp32 = self.pool.get(tensor).dtype() == crate::model::tensor::DType::Float32;
            if is_fp32 {
                let fp32_tensor = self.pool.take(tensor);
                let bf16_result = self.ops.cast_device_tensor(&fp32_tensor, crate::model::tensor::DType::BFloat16);
                if self.use_arena {
                    let layer = self.current_layer_arena_idx.unwrap_or(0);
                    self.rotating_pool.arena_for_layer(layer).defer_owned(fp32_tensor.into_buf());
                } else {
                    self.pool.defer_buffers(vec![fp32_tensor.into_buf()]);
                }
                self.pool.put(tensor, bf16_result);
            }

            comm.all_reduce_sum_inplace(self.pool.get(tensor));

            if !self.use_arena {
                self.ops.synchronize().ok();
                self.pool.release_deferred_after_sync();
            }

            if let Some(ref mut d) = self.dumper {
                if let Some(layer) = self.last_attention_layer {
                    if d.should_dump(layer) {
                        let step = if self.allreduce_in_layer == 0 { "11_o_proj_allreduce" } else { "18_down_proj_allreduce" };
                        d.dump(&format!("layer{layer}_{step}"), self.pool.get(tensor), self.ops.stream());
                    }
                }
            }
        } else {
            tracing::warn!("AllReduceSum: no comm ops configured, skipping");
        }
        self.allreduce_in_layer += 1;
    }

    fn exec_send(&mut self, tensor: usize, dst_rank: usize) {
        if let Some(comm) = self.comm_ops {
            comm.send_tensor(self.pool.get(tensor), dst_rank);
        } else {
            tracing::warn!("Send: no comm ops configured, skipping");
        }
    }

    fn exec_recv(&mut self, tensor: usize, src_rank: usize) {
        if let Some(comm) = self.comm_ops {
            let shape = vec![self.input_ids.len(), self.hidden_size];
            let received = comm.recv_tensor(&shape, crate::model::tensor::DType::Float16, src_rank);
            self.pool.put(tensor, received);
        } else {
            tracing::warn!("Recv: no comm ops configured, skipping");
        }
    }
}

impl CompiledPlan {
    pub fn new(plan: ExecutionPlan) -> Self {
        Self { plan }
    }

    /// Execute the compiled plan for one forward pass (paged KV cache variant).
    ///
    /// Branches the Attention step:
    /// - Prefill: Uses FlashAttention with full Q/K/V, writes K/V to paged cache
    /// - Decode: Uses FlashAttention (all tokens), writes K/V to paged cache
    ///
    /// The `paged_ctx` carries block_table, slot_mapping, and Prefill/Decode flag.
    /// The `kv_key_caches` / `kv_value_caches` are per-layer device buffers for paged KV.
    #[cfg(feature = "ascend")]
    pub fn execute_paged(
        &self,
        ops: &crate::ops::ascend::AscendComputeOps,
        comm_ops: Option<&crate::ops::ascend_comm::AscendCommOps>,
        pool: &mut crate::model::device_tensor::TensorPool,
        rotating_pool: &mut crate::model::scratch_arena::RotatingPool,
        weights: &[crate::model::device_tensor::WeightTensor],
        input_ids: &[u32],
        positions: &[u32],
        paged_ctx: &PagedKVContext,
        kv_key_caches: &[ascend::memory::DeviceBuffer],
        kv_value_caches: &[ascend::memory::DeviceBuffer],
        mut decode_buffers: Option<&mut crate::ops::ascend::DecodeBuffers>,
    ) -> u32 {
        use crate::model::scratch_arena::POOL_DEPTH;

        let cfg = &self.plan.config;
        let perf_breakdown = *PERF_BREAKDOWN;
        let call_start = if perf_breakdown {
            Some(Instant::now())
        } else {
            None
        };
        let mut perf_timer = crate::engine::perf::PerfTimer::new(perf_breakdown);

        if let Some(ref mut db) = decode_buffers {
            db.block_table_dirty = true;
        }

        let use_arena = POOL_DEPTH > 1;
        static ARENA_STATS_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);

        let dump_decode = *DUMP_DECODE;
        let mut dumper = if !paged_ctx.is_decode || dump_decode {
            super::debug_dump::DebugDumper::from_env()
        } else {
            None
        };

        let mut ctx = ExecContext {
            ops,
            comm_ops,
            pool,
            rotating_pool,
            weights,
            input_ids,
            positions,
            paged_ctx,
            kv_key_caches,
            kv_value_caches,
            decode_buffers,
            dumper: dumper.as_mut(),
            use_arena,
            current_layer_arena_idx: None,
            allreduce_in_layer: 0,
            add_in_layer: 0,
            last_attention_layer: None,
            sampled_token: 0,
            hidden_size: cfg.hidden_size,
        };

        for step in &self.plan.steps {
            perf_timer.time_step(step, || {
                ctx.exec_step(step);
            });
        }

        if let Some(ref mut d) = ctx.dumper {
            d.finalize();
        }

        if use_arena && !ARENA_STATS_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!("RotatingPool arena stats after first forward pass:");
            ctx.rotating_pool.log_stats();
        }

        perf_timer.time_sync(|| {
            ctx.ops.synchronize().ok();
            ctx.pool.release_deferred_after_sync();
        });

        if let Some(t0) = call_start {
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            perf_timer.log_breakdown(
                total_ms,
                paged_ctx.is_decode,
                input_ids.len(),
                paged_ctx.context_len,
            );
        }

        ctx.sampled_token
    }

    /// Get plan metadata.
    #[allow(dead_code)]
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::Qwen3Config;

    #[test]
    fn test_compile_single_device() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config.clone());
        let parallel = ParallelConfig::single_device();
        let quant = QuantConfig::none();

        let plan = compile_plan(&model, &parallel, &quant);

        // Should have Embedding + 28 layers of ops + final norm + lm_head + sample
        assert!(
            plan.num_steps() > 100,
            "Expected >100 steps, got {}",
            plan.num_steps()
        );

        // No communication ops in single device mode
        assert_eq!(
            plan.count_step_type(|s| matches!(s, ExecStep::AllReduceSum { .. })),
            0
        );
        assert_eq!(
            plan.count_step_type(|s| matches!(s, ExecStep::Send { .. })),
            0
        );
        assert_eq!(
            plan.count_step_type(|s| matches!(s, ExecStep::Recv { .. })),
            0
        );

        // Has exactly 1 Embedding, 1 Sample
        assert_eq!(
            plan.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })),
            1
        );
        assert_eq!(
            plan.count_step_type(|s| matches!(s, ExecStep::Sample { .. })),
            1
        );
    }

    #[test]
    fn test_compile_tp() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        let parallel = ParallelConfig::tensor_parallel(4, 0);
        let quant = QuantConfig::none();

        let plan = compile_plan(&model, &parallel, &quant);

        // TP: 2 AllReduce per layer (after attn and MLP) = 28 * 2 = 56
        let ar_count = plan.count_step_type(|s| matches!(s, ExecStep::AllReduceSum { .. }));
        assert_eq!(
            ar_count, 56,
            "Expected 56 AllReduce (28 layers * 2), got {ar_count}"
        );
    }

    #[test]
    fn test_compile_pp() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config.clone());

        // First stage: has Embedding, no Sample, has Send
        let plan0 = compile_plan(
            &model,
            &ParallelConfig::pipeline_parallel(2, 0),
            &QuantConfig::none(),
        );
        assert_eq!(
            plan0.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })),
            1
        );
        assert_eq!(
            plan0.count_step_type(|s| matches!(s, ExecStep::Sample { .. })),
            0
        );
        assert_eq!(
            plan0.count_step_type(|s| matches!(s, ExecStep::Send { .. })),
            1
        );

        // Last stage: has Recv, no Embedding, has Sample
        let plan1 = compile_plan(
            &model,
            &ParallelConfig::pipeline_parallel(2, 1),
            &QuantConfig::none(),
        );
        assert_eq!(
            plan1.count_step_type(|s| matches!(s, ExecStep::Recv { .. })),
            1
        );
        assert_eq!(
            plan1.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })),
            0
        );
        assert_eq!(
            plan1.count_step_type(|s| matches!(s, ExecStep::Sample { .. })),
            1
        );
    }

    #[test]
    fn test_compile_quantized() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config);
        let parallel = ParallelConfig::single_device();
        let quant = QuantConfig::int8_per_tensor();

        let plan = compile_plan(&model, &parallel, &quant);

        // Should have DequantMatMul steps for attention and MLP weights
        let dq_count = plan.count_step_type(|s| matches!(s, ExecStep::DequantMatMul { .. }));
        assert!(
            dq_count > 0,
            "Expected DequantMatMul steps with quantization"
        );

        // LM head should still be plain MatMul (excluded from INT8)
        // (Last MatMul before Sample should be plain MatMul)
    }

    #[cfg(feature = "ascend")]
    #[test]
    fn test_execute_plan_no_panic() {
        use crate::model::device_tensor::TensorPool;

        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config.clone());
        let parallel = ParallelConfig::single_device();
        let quant = QuantConfig::none();

        let plan = compile_plan(&model, &parallel, &quant);
        let compiled = plan.compile();
        let mut pool = TensorPool::new(compiled.plan().num_buffers);
        let mut kv_cache = SequenceKVCache::new(&config, 2048);

        let token = compiled.execute(&ops, &mut pool, &[1, 2, 3], &[0, 1, 2], &mut kv_cache);

        assert_eq!(token, 0); // StubOps returns 0
    }
}
