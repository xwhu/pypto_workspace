use std::time::Instant;
use crate::engine::plan::ExecStep;

/// Statistics collected during execution of a forward pass or sampled tokens.
#[derive(Default, Debug, Clone)]
pub struct PerfStats {
    pub embedding_ms: f64,
    pub rmsnorm_ms: f64,
    pub matmul_ms: f64,
    pub rotary_ms: f64,
    pub qknorm_ms: f64,
    pub attention_ms: f64,
    pub silu_ms: f64,
    pub add_ms: f64,
    pub sample_ms: f64,
    pub allreduce_ms: f64,
    pub send_ms: f64,
    pub recv_ms: f64,
    pub dequant_ms: f64,
    pub sync_ms: f64,
}

/// A timer to track execution time of various operations.
pub struct PerfTimer {
    pub stats: PerfStats,
    enabled: bool,
}

impl PerfTimer {
    /// Create a new PerfTimer. Times actions only if `enabled` is true.
    pub fn new(enabled: bool) -> Self {
        Self {
            stats: PerfStats::default(),
            enabled,
        }
    }

    /// Measure execution time for a generic ExecStep.
    #[inline(always)]
    pub fn time_step<F, R>(&mut self, step_type: &ExecStep, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        let t0 = Instant::now();
        let result = f();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

        match step_type {
            ExecStep::Embedding { .. } => self.stats.embedding_ms += elapsed,
            ExecStep::RmsNorm { .. } => self.stats.rmsnorm_ms += elapsed,
            ExecStep::MatMul { .. } => self.stats.matmul_ms += elapsed,
            ExecStep::RotaryEmb { .. } => self.stats.rotary_ms += elapsed,
            ExecStep::QKNorm { .. } => self.stats.qknorm_ms += elapsed,
            ExecStep::Attention { .. } => self.stats.attention_ms += elapsed,
            ExecStep::SiluMul { .. } => self.stats.silu_ms += elapsed,
            ExecStep::Add { .. } => self.stats.add_ms += elapsed,
            ExecStep::Sample { .. } => self.stats.sample_ms += elapsed,
            ExecStep::AllReduceSum { .. } => self.stats.allreduce_ms += elapsed,
            ExecStep::Send { .. } => self.stats.send_ms += elapsed,
            ExecStep::Recv { .. } => self.stats.recv_ms += elapsed,
            ExecStep::DequantMatMul { .. } => self.stats.dequant_ms += elapsed,
        }
        result
    }

    /// Measure synchronization time.
    #[inline(always)]
    pub fn time_sync<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        let t0 = Instant::now();
        let result = f();
        self.stats.sync_ms += t0.elapsed().as_secs_f64() * 1000.0;
        result
    }

    /// Measure sampling logic time.
    #[inline(always)]
    pub fn time_sample<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        let t0 = Instant::now();
        let result = f();
        self.stats.sample_ms += t0.elapsed().as_secs_f64() * 1000.0;
        result
    }

    /// Print performance breakdown using the tracing macro.
    pub fn log_breakdown(&self, total_ms: f64, is_decode: bool, input_len: usize, context_len: usize) {
        if !self.enabled {
            return;
        }
        let s = &self.stats;
        let known_ms = s.embedding_ms
            + s.rmsnorm_ms
            + s.matmul_ms
            + s.rotary_ms
            + s.qknorm_ms
            + s.attention_ms
            + s.silu_ms
            + s.add_ms
            + s.sample_ms
            + s.allreduce_ms
            + s.send_ms
            + s.recv_ms
            + s.dequant_ms
            + s.sync_ms;
        let other_ms = (total_ms - known_ms).max(0.0);

        tracing::info!(
            "ExecBreakdown: is_decode={}, input_len={}, context_len={}, total_ms={:.2}, sync_tail_ms={:.2}, emb_ms={:.2}, rms_ms={:.2}, matmul_ms={:.2}, rotary_ms={:.2}, qknorm_ms={:.2}, attn_ms={:.2}, silu_ms={:.2}, add_ms={:.2}, sample_ms={:.2}, allreduce_ms={:.2}, send_ms={:.2}, recv_ms={:.2}, dequant_ms={:.2}, other_ms={:.2}",
            is_decode,
            input_len,
            context_len,
            total_ms,
            s.sync_ms,
            s.embedding_ms,
            s.rmsnorm_ms,
            s.matmul_ms,
            s.rotary_ms,
            s.qknorm_ms,
            s.attention_ms,
            s.silu_ms,
            s.add_ms,
            s.sample_ms,
            s.allreduce_ms,
            s.send_ms,
            s.recv_ms,
            s.dequant_ms,
            other_ms
        );
    }
}
