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

/// Stub tokenizer — converts text to/from token IDs.
///
/// In a real implementation, this would use the Qwen3 tokenizer
/// (SentencePiece/tiktoken based). For the stub, we just split on
/// whitespace and assign sequential IDs.
pub struct StubTokenizer;

impl StubTokenizer {
    /// Encode text to token IDs (stub: space-split, each word → hash).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                // Simple hash to get a token ID in valid range
                let hash: u32 = word.bytes().fold(0u32, |acc, b| {
                    acc.wrapping_mul(31).wrapping_add(b as u32)
                });
                // Ensure within vocab range [1, 151935] (0 reserved)
                (hash % 151935) + 1
            })
            .collect()
    }

    /// Decode token IDs to text (stub: just format as "[tok_0, tok_1, ...]").
    pub fn decode(&self, ids: &[u32]) -> String {
        if ids.is_empty() {
            return String::new();
        }
        format!(
            "[generated {} tokens: {}]",
            ids.len(),
            ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
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

    #[test]
    fn test_stub_tokenizer() {
        let tokenizer = StubTokenizer;
        let ids = tokenizer.encode("Hello world");
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().all(|&id| id > 0 && id <= 151935));

        let text = tokenizer.decode(&[1, 2, 3]);
        assert!(text.contains("3 tokens"));
    }
}
