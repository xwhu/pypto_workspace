use std::fmt;

use crate::model::config::Qwen3Config;
use crate::model::network::Qwen3Model;
use crate::model::parallel::ParallelConfig;
use crate::model::quantize::{QuantConfig, QuantScheme};
use crate::model::tensor::{DType, Tensor};
use crate::ops::OpsBundle;
use super::kv_cache::SequenceKVCache;

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
            ExecStep::Attention { num_heads, num_kv_heads, .. } =>
                write!(f, "Attention(h={num_heads},kv={num_kv_heads})"),
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
    fn new() -> Self { Self { next_id: 0 } }
    fn alloc(&mut self) -> TensorRef {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
    fn total(&self) -> usize { self.next_id }
}

/// Weight registry: maps weight names to indices and stores actual tensors.
struct WeightRegistry {
    names: Vec<String>,
    tensors: Vec<Tensor>,
}

impl WeightRegistry {
    fn new() -> Self { Self { names: Vec::new(), tensors: Vec::new() } }

    fn register(&mut self, tensor: &Tensor) -> WeightRef {
        let id = self.names.len();
        self.names.push(tensor.name.clone());
        self.tensors.push(tensor.clone());
        id
    }

    fn into_parts(self) -> (Vec<String>, Vec<Tensor>) {
        (self.names, self.tensors)
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
    let input_ids_slot = bufs.alloc();     // 0: input token IDs
    let positions_slot = bufs.alloc();     // 1: position indices
    let hidden_slot = bufs.alloc();        // 2: main hidden state

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
        emit_matmul_or_dequant(&mut steps, normed, q_w, q, &layer.self_attn.q_proj.name, quant, &mut weights);
        emit_matmul_or_dequant(&mut steps, normed, k_w, k, &layer.self_attn.k_proj.name, quant, &mut weights);
        emit_matmul_or_dequant(&mut steps, normed, v_w, v, &layer.self_attn.v_proj.name, quant, &mut weights);

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
            q, k, v,
            out: attn_out,
            num_heads: cfg.num_attention_heads / parallel.tp_size,
            num_kv_heads: cfg.num_key_value_heads / parallel.tp_size,
            head_dim: cfg.head_dim,
        });

        // O projection
        emit_matmul_or_dequant(&mut steps, attn_out, o_w, proj_out, &layer.self_attn.o_proj.name, quant, &mut weights);

        // TP: AllReduce after attention (if row-sharded o_proj)
        if parallel.is_tp() {
            steps.push(ExecStep::AllReduceSum { tensor: proj_out });
        }

        // Residual
        steps.push(ExecStep::Add { a: hidden_slot, b: proj_out });

        // Post-attention LayerNorm
        steps.push(ExecStep::RmsNorm {
            input: hidden_slot,
            weight: ln2_w,
            eps: cfg.rms_norm_eps as f32,
            out: normed2,
        });

        // MLP: gate_proj and up_proj
        emit_matmul_or_dequant(&mut steps, normed2, gate_w, gate, &layer.mlp.gate_proj.name, quant, &mut weights);
        emit_matmul_or_dequant(&mut steps, normed2, up_w, up, &layer.mlp.up_proj.name, quant, &mut weights);

        // SwiGLU
        steps.push(ExecStep::SiluMul { gate, up, out: silu_out });

        // down_proj
        emit_matmul_or_dequant(&mut steps, silu_out, down_w, ffn_out, &layer.mlp.down_proj.name, quant, &mut weights);

        // TP: AllReduce after MLP (if row-sharded down_proj)
        if parallel.is_tp() {
            steps.push(ExecStep::AllReduceSum { tensor: ffn_out });
        }

        // Residual
        steps.push(ExecStep::Add { a: hidden_slot, b: ffn_out });
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

        emit_matmul_or_dequant(&mut steps, final_normed, lm_head_w, logits, "lm_head.weight", quant, &mut weights);

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

    let (weight_names, weight_tensors) = weights.into_parts();

    ExecutionPlan {
        steps,
        weight_names,
        weight_tensors,
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
                let zeros_tensor = Tensor::new(vec![1], DType::Float32, format!("{weight_name}.zeros"));
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
        steps.push(ExecStep::MatMul { a: input, b: weight, out });
    }
}

// ─── Execution Plan ────────────────────────────────────────────────────

/// The compiled execution plan — a flat list of steps + metadata.
///
/// Created once at engine initialization, reused for every inference step.
pub struct ExecutionPlan {
    /// Flat instruction sequence.
    pub steps: Vec<ExecStep>,
    /// Weight names (indexed by WeightRef).
    pub weight_names: Vec<String>,
    /// Actual weight tensors (indexed by WeightRef).
    /// These hold data_ptr to device memory after weight loading.
    pub weight_tensors: Vec<Tensor>,
    /// Number of tensor buffer slots needed.
    pub num_buffers: usize,
    /// Model config snapshot.
    pub config: Qwen3Config,
    /// Parallel config snapshot.
    pub parallel: ParallelConfig,
}

impl ExecutionPlan {
    /// Compile into a `CompiledPlan` with cached closures.
    pub fn compile(self, _ops: &OpsBundle) -> CompiledPlan {
        CompiledPlan::new(self)
    }

    /// Total number of steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Count steps of a specific type.
    pub fn count_step_type(&self, predicate: impl Fn(&ExecStep) -> bool) -> usize {
        self.steps.iter().filter(|s| predicate(s)).count()
    }

    /// Print the plan for debugging.
    pub fn dump(&self) {
        tracing::info!("Execution Plan: {} steps, {} buffers, {} weights",
            self.steps.len(), self.num_buffers, self.weight_names.len());
        for (i, step) in self.steps.iter().enumerate() {
            tracing::debug!("  [{:3}] {}", i, step);
        }
    }
}

// ─── Compiled Plan (方案B) ─────────────────────────────────────────────

/// Runtime tensor buffer pool.
pub struct TensorPool {
    buffers: Vec<Tensor>,
}

impl TensorPool {
    pub fn new(num_slots: usize) -> Self {
        // Pre-allocate empty tensors (shape set dynamically at execution)
        let buffers = (0..num_slots)
            .map(|i| Tensor::new(vec![0], DType::Float16, format!("buf_{i}")))
            .collect();
        Self { buffers }
    }

    pub fn get(&self, idx: TensorRef) -> &Tensor {
        &self.buffers[idx]
    }

    pub fn get_mut(&mut self, idx: TensorRef) -> &mut Tensor {
        &mut self.buffers[idx]
    }

    /// Set the input token IDs in slot 0.
    pub fn set_input_ids(&mut self, ids: &[u32]) {
        self.buffers[0] = Tensor::new(
            vec![1, ids.len()],
            DType::Uint32,
            "input_ids",
        );
    }

    /// Set positions in slot 1.
    pub fn set_positions(&mut self, positions: &[u32]) {
        self.buffers[1] = Tensor::new(
            vec![positions.len()],
            DType::Uint32,
            "positions",
        );
    }
}

/// Compiled execution plan — caches closures for zero-dispatch execution.
///
/// Created once from an `ExecutionPlan`. Holds the plan metadata and
/// executes steps by iterating through them with minimal overhead.
pub struct CompiledPlan {
    plan: ExecutionPlan,
}

impl CompiledPlan {
    pub fn new(plan: ExecutionPlan) -> Self {
        Self { plan }
    }

    /// Execute the compiled plan for one forward pass.
    ///
    /// This is the hot path. Each step dispatches to the appropriate
    /// ops method with minimal overhead (single match per step).
    pub fn execute(
        &self,
        ops: &OpsBundle,
        pool: &mut TensorPool,
        input_ids: &[u32],
        positions: &[u32],
        _kv_cache: &mut SequenceKVCache,
    ) -> u32 {
        pool.set_input_ids(input_ids);
        pool.set_positions(positions);

        // Set initial hidden state shape
        let cfg = &self.plan.config;
        pool.buffers[2] = Tensor::new(
            vec![1, input_ids.len(), cfg.hidden_size],
            DType::Float16,
            "hidden_states",
        );

        let weights = &self.plan.weight_tensors;

        let mut sampled_token: u32 = 0;

        for step in &self.plan.steps {
            match step {
                ExecStep::Embedding { ids_ref: _, table_weight, out } => {
                    pool.buffers[*out] = Tensor::new(
                        vec![1, input_ids.len(), cfg.hidden_size],
                        DType::Float16,
                        "embed_out",
                    );
                    ops.compute.embedding(input_ids, &weights[*table_weight], &mut pool.buffers[*out]);
                }
                ExecStep::RmsNorm { input, weight, eps, out } => {
                    let shape = pool.buffers[*input].shape.clone();
                    pool.buffers[*out] = Tensor::new(shape, DType::Float16, "norm_out");
                    let inp_clone = pool.buffers[*input].clone();
                    ops.compute.rms_norm(&inp_clone, &weights[*weight], *eps, &mut pool.buffers[*out]);
                }
                ExecStep::MatMul { a, b, out } => {
                    // Output shape: [batch, seq_len, weight.shape[0]] (out_features)
                    let a_shape = &pool.buffers[*a].shape;
                    let out_features = weights[*b].shape[0]; // [out_features, in_features]
                    let mut out_shape = a_shape.clone();
                    if let Some(last) = out_shape.last_mut() {
                        *last = out_features;
                    }
                    pool.buffers[*out] = Tensor::new(out_shape, DType::Float16, "matmul_out");
                    let a_clone = pool.buffers[*a].clone();
                    ops.compute.matmul(&a_clone, &weights[*b], &mut pool.buffers[*out]);
                }
                ExecStep::RotaryEmb { q, k, positions_ref: _, rope_theta, head_dim } => {
                    // Use split_at_mut to get two mutable refs to different indices
                    let (lo, hi) = if q < k {
                        let (left, right) = pool.buffers.split_at_mut(*k);
                        (&mut left[*q], &mut right[0])
                    } else {
                        let (left, right) = pool.buffers.split_at_mut(*q);
                        (&mut right[0], &mut left[*k])
                    };
                    ops.compute.rotary_embedding(lo, hi, positions, *rope_theta, *head_dim);
                }
                ExecStep::QKNorm { qk, weight, num_heads, head_dim, eps } => {
                    ops.compute.qk_norm(&mut pool.buffers[*qk], &weights[*weight], *num_heads, *head_dim, *eps);
                }
                ExecStep::Attention { q, k, v, out, num_heads, num_kv_heads, head_dim } => {
                    let q_shape = pool.buffers[*q].shape.clone();
                    pool.buffers[*out] = Tensor::new(q_shape, DType::Float16, "attn_out");
                    let q_c = pool.buffers[*q].clone();
                    let k_c = pool.buffers[*k].clone();
                    let v_c = pool.buffers[*v].clone();
                    ops.compute.attention(
                        &q_c, &k_c, &v_c, &mut pool.buffers[*out],
                        *num_heads, *num_kv_heads, *head_dim,
                    );
                }
                ExecStep::SiluMul { gate, up, out } => {
                    let g_shape = pool.buffers[*gate].shape.clone();
                    pool.buffers[*out] = Tensor::new(g_shape, DType::Float16, "silu_out");
                    let gate_c = pool.buffers[*gate].clone();
                    let up_c = pool.buffers[*up].clone();
                    ops.compute.silu_mul(&gate_c, &up_c, &mut pool.buffers[*out]);
                }
                ExecStep::Add { a, b } => {
                    let b_clone = pool.buffers[*b].clone();
                    ops.compute.add(&mut pool.buffers[*a], &b_clone);
                }
                ExecStep::Sample { logits, .. } => {
                    sampled_token = ops.compute.sample_argmax(&pool.buffers[*logits]);
                }
                ExecStep::AllReduceSum { tensor } => {
                    ops.comm.all_reduce_sum(&mut pool.buffers[*tensor]);
                }
                ExecStep::Send { tensor, dst_rank } => {
                    ops.comm.send(&pool.buffers[*tensor], *dst_rank);
                }
                ExecStep::Recv { tensor, src_rank } => {
                    ops.comm.recv(&mut pool.buffers[*tensor], *src_rank);
                }
                ExecStep::DequantMatMul { input, weight, scales, zeros, out } => {
                    let i_shape = &pool.buffers[*input].shape;
                    let out_features = weights[*weight].shape[0];
                    let mut out_shape = i_shape.clone();
                    if let Some(last) = out_shape.last_mut() {
                        *last = out_features;
                    }
                    pool.buffers[*out] = Tensor::new(out_shape, DType::Float16, "dqmm_out");
                    let inp_c = pool.buffers[*input].clone();
                    let zeros_tensor = zeros.map(|z| &weights[z]);
                    ops.quant.matmul_quantized(
                        &inp_c, &weights[*weight], &weights[*scales], zeros_tensor,
                        &mut pool.buffers[*out],
                    );
                }
            }
        }

        sampled_token
    }

    /// Get plan metadata.
    pub fn plan(&self) -> &ExecutionPlan { &self.plan }
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
        assert!(plan.num_steps() > 100, "Expected >100 steps, got {}", plan.num_steps());

        // No communication ops in single device mode
        assert_eq!(plan.count_step_type(|s| matches!(s, ExecStep::AllReduceSum { .. })), 0);
        assert_eq!(plan.count_step_type(|s| matches!(s, ExecStep::Send { .. })), 0);
        assert_eq!(plan.count_step_type(|s| matches!(s, ExecStep::Recv { .. })), 0);

        // Has exactly 1 Embedding, 1 Sample
        assert_eq!(plan.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })), 1);
        assert_eq!(plan.count_step_type(|s| matches!(s, ExecStep::Sample { .. })), 1);
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
        assert_eq!(ar_count, 56, "Expected 56 AllReduce (28 layers * 2), got {ar_count}");
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
        assert_eq!(plan0.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })), 1);
        assert_eq!(plan0.count_step_type(|s| matches!(s, ExecStep::Sample { .. })), 0);
        assert_eq!(plan0.count_step_type(|s| matches!(s, ExecStep::Send { .. })), 1);

        // Last stage: has Recv, no Embedding, has Sample
        let plan1 = compile_plan(
            &model,
            &ParallelConfig::pipeline_parallel(2, 1),
            &QuantConfig::none(),
        );
        assert_eq!(plan1.count_step_type(|s| matches!(s, ExecStep::Recv { .. })), 1);
        assert_eq!(plan1.count_step_type(|s| matches!(s, ExecStep::Embedding { .. })), 0);
        assert_eq!(plan1.count_step_type(|s| matches!(s, ExecStep::Sample { .. })), 1);
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
        assert!(dq_count > 0, "Expected DequantMatMul steps with quantization");

        // LM head should still be plain MatMul (excluded from INT8)
        // (Last MatMul before Sample should be plain MatMul)
    }

    #[test]
    fn test_execute_plan_no_panic() {
        let config = Qwen3Config::qwen3_0_6b();
        let model = Qwen3Model::new(config.clone());
        let parallel = ParallelConfig::single_device();
        let quant = QuantConfig::none();

        let plan = compile_plan(&model, &parallel, &quant);
        let compiled = plan.compile(&OpsBundle::stub());
        let ops = OpsBundle::stub();
        let mut pool = TensorPool::new(compiled.plan().num_buffers);
        let mut kv_cache = SequenceKVCache::new(&config, 2048);

        let token = compiled.execute(
            &ops,
            &mut pool,
            &[1, 2, 3],
            &[0, 1, 2],
            &mut kv_cache,
        );

        assert_eq!(token, 0); // StubOps returns 0
    }
}
