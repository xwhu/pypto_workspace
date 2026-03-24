use serde::{Deserialize, Serialize};

/// A request for text completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Input prompt text.
    pub prompt: String,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    128
}

/// Response for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text (stub: token IDs as comma-separated string).
    pub text: String,

    /// Number of prompt tokens.
    pub prompt_tokens: usize,

    /// Number of completion tokens.
    pub completion_tokens: usize,

    /// Model name.
    pub model: String,
}

/// A sequence in the scheduler's queue.
#[derive(Debug)]
pub struct SchedulerSequence {
    /// Unique sequence ID.
    pub seq_id: u64,

    /// Input token IDs.
    pub token_ids: Vec<u32>,

    /// Generation config.
    pub max_new_tokens: usize,
}

/// Basic FIFO scheduler.
///
/// This is a minimal scheduler for the initial framework.
/// It processes one request at a time (no batching).
/// The v8 design calls for a continuous batching scheduler with
/// hierarchical routing — that would replace this.
pub struct Scheduler {
    next_seq_id: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { next_seq_id: 0 }
    }

    /// Schedule a new request.
    pub fn add_request(&mut self, token_ids: Vec<u32>, max_new_tokens: usize) -> SchedulerSequence {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        SchedulerSequence {
            seq_id,
            token_ids,
            max_new_tokens,
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenizer backed by HuggingFace `tokenizers` crate.
///
/// Loads `tokenizer.json` from the model weights directory for
/// proper encode/decode of Qwen3 tokens.
pub struct Qwen3Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Qwen3Tokenizer {
    /// Load tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", path, e))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(e) => {
                tracing::error!("Tokenizer encode failed: {}", e);
                vec![]
            }
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self.inner.decode(ids, true) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("Tokenizer decode failed: {}", e);
                format!("[decode error: {}]", e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_add_request() {
        let mut scheduler = Scheduler::new();
        let seq = scheduler.add_request(vec![1, 2, 3], 10);
        assert_eq!(seq.seq_id, 0);
        assert_eq!(seq.token_ids, vec![1, 2, 3]);

        let seq2 = scheduler.add_request(vec![4, 5], 5);
        assert_eq!(seq2.seq_id, 1);
    }
}
