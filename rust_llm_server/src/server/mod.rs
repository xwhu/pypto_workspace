use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures_core::Stream;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::engine::engine::{GenerationConfig, StreamToken};
use crate::scheduler::{
    apply_chat_template, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage, CompletionRequest,
    CompletionResponse, Qwen3Tokenizer, UsageInfo,
};

/// Shared application state for the HTTP server.
pub struct AppState {
    pub engine: crate::engine::engine::Engine,
    pub tokenizer: Qwen3Tokenizer,
}

/// Build the Axum router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/completions", post(completion_handler))
        .route("/v1/chat/completions", post(chat_completion_handler))
        .route("/v1/models", get(models_handler))
        .with_state(state)
}

/// GET /health — server health check.
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": state.engine.model_info().to_string(),
    }))
}

/// POST /v1/completions — text completion endpoint (non-streaming & streaming).
async fn completion_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    tracing::info!(
        "Completion request: prompt={:?}, max_tokens={}, stream={}",
        req.prompt,
        req.max_tokens,
        req.stream,
    );

    let prompt_ids = state.tokenizer.encode(&req.prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Empty prompt after tokenization".to_string(),
        ));
    }

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        ..Default::default()
    };

    if req.stream {
        let model_name = state.engine.config().model_type.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamToken>();
        let engine_state = Arc::clone(&state);

        tokio::task::spawn_blocking(move || {
            engine_state
                .engine
                .generate_streaming(&prompt_ids, &gen_config, tx);
        });

        let stream = completion_stream(rx, Arc::clone(&state), model_name);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let result = state.engine.generate(&prompt_ids, &gen_config);
        tracing::info!(
            "Completion perf: prompt_tokens={}, completion_tokens={}, ttft_ms={}, tpot_ms={}",
            result.prompt_tokens,
            result.completion_tokens,
            result
                .ttft_ms
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            result
                .tpot_ms
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
        );
        let text = state.tokenizer.decode(&result.token_ids);

        Ok(Json(CompletionResponse {
            text,
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            model: state.engine.config().model_type.clone(),
        })
        .into_response())
    }
}

/// POST /v1/chat/completions — OpenAI-compatible chat completion (non-streaming & streaming).
async fn chat_completion_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    tracing::info!(
        "Chat completion: model={}, messages={}, max_tokens={}, temp={}, top_p={}, stream={}",
        req.model,
        req.messages.len(),
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.stream,
    );

    // Apply Qwen3 chat template
    let prompt = apply_chat_template(&req.messages);
    tracing::debug!(
        "Chat template prompt: {:?}",
        &prompt[..prompt.len().min(200)]
    );

    // Tokenize
    let prompt_ids = state.tokenizer.encode(&prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Empty prompt after tokenization".to_string(),
        ));
    }
    tracing::info!("Prompt tokens: {}", prompt_ids.len());

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        ..Default::default()
    };

    if req.stream {
        let model_name = req.model.clone();
        let response_id = format!("chatcmpl-{}", uuid_simple());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamToken>();
        let engine_state = Arc::clone(&state);

        tokio::task::spawn_blocking(move || {
            engine_state
                .engine
                .generate_streaming(&prompt_ids, &gen_config, tx);
        });

        let stream = chat_completion_stream(rx, Arc::clone(&state), model_name, response_id);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Generate
        let result = state.engine.generate(&prompt_ids, &gen_config);
        tracing::info!(
            "Chat perf: prompt_tokens={}, completion_tokens={}, ttft_ms={}, tpot_ms={}",
            result.prompt_tokens,
            result.completion_tokens,
            result
                .ttft_ms
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            result
                .tpot_ms
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
        );

        // Decode — strip <|im_end|> if present
        let mut text = state.tokenizer.decode(&result.token_ids);
        if let Some(pos) = text.find("<|im_end|>") {
            text.truncate(pos);
        }

        let finish_reason = if result.token_ids.last() == Some(&gen_config.eos_token_id) {
            "stop"
        } else {
            "length"
        };

        Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid_simple()),
            object: "chat.completion".to_string(),
            model: req.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: finish_reason.to_string(),
            }],
            usage: UsageInfo {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
            },
        })
        .into_response())
    }
}

/// Build an SSE stream for `/v1/completions` with `stream: true`.
fn completion_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamToken>,
    state: Arc<AppState>,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let token_stream = UnboundedReceiverStream::new(rx);

    token_stream
        .map(move |st| match st {
            StreamToken::Token(token_id) => {
                let text = state.tokenizer.decode(&[token_id]);
                let data = json!({
                    "object": "text_completion",
                    "model": model,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "finish_reason": null,
                    }]
                });
                Ok(Event::default().data(data.to_string()))
            }
            StreamToken::Finish(reason) => {
                let data = json!({
                    "object": "text_completion",
                    "model": model,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "finish_reason": reason,
                    }]
                });
                Ok(Event::default().data(data.to_string()))
            }
        })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))))
}

/// Build an SSE stream for `/v1/chat/completions` with `stream: true`.
fn chat_completion_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamToken>,
    state: Arc<AppState>,
    model: String,
    id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let token_stream = UnboundedReceiverStream::new(rx);

    // First event: role announcement
    let first_id = id.clone();
    let first_model = model.clone();
    let role_event = tokio_stream::once(Ok(Event::default().data(
        serde_json::to_string(&ChatCompletionChunk {
            id: first_id,
            object: "chat.completion.chunk".to_string(),
            model: first_model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        })
        .unwrap(),
    )));

    // Content + finish events
    let content_stream = token_stream.map(move |st| {
        let chunk = match st {
            StreamToken::Token(token_id) => {
                let mut text = state.tokenizer.decode(&[token_id]);
                // Strip <|im_end|> from streamed tokens
                if let Some(pos) = text.find("<|im_end|>") {
                    text.truncate(pos);
                }
                ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                }
            }
            StreamToken::Finish(reason) => ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                model: model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(reason),
                }],
            },
        };
        Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
    });

    role_event
        .chain(content_stream)
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))))
}

/// GET /v1/models — list available models.
async fn models_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "data": [{
            "id": state.engine.config().model_type,
            "object": "model",
            "owned_by": "local",
            "info": state.engine.model_info().to_string(),
        }]
    }))
}

/// Simple pseudo-UUID for response IDs.
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:04x}", t.as_secs(), t.subsec_millis())
}

/// Start the HTTP server.
pub async fn serve(state: Arc<AppState>, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = build_router(state);
    let addr = format!("0.0.0.0:{port}");

    tracing::info!("Starting server on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
