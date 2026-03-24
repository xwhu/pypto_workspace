use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

use crate::engine::engine::{Engine, GenerationConfig};
use crate::scheduler::{
    CompletionRequest, CompletionResponse, Qwen3Tokenizer,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatMessage, UsageInfo, apply_chat_template,
};

/// Shared application state for the HTTP server.
pub struct AppState {
    pub engine: Engine,
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

/// POST /v1/completions — text completion endpoint.
async fn completion_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    tracing::info!("Completion request: prompt={:?}, max_tokens={}", req.prompt, req.max_tokens);

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

    let result = state.engine.generate(&prompt_ids, &gen_config);
    let text = state.tokenizer.decode(&result.token_ids);

    Ok(Json(CompletionResponse {
        text,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        model: state.engine.config().model_type.clone(),
    }))
}

/// POST /v1/chat/completions — OpenAI-compatible chat completion endpoint.
async fn chat_completion_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    tracing::info!(
        "Chat completion: model={}, messages={}, max_tokens={}, temp={}, top_p={}",
        req.model, req.messages.len(), req.max_tokens, req.temperature, req.top_p
    );

    // Apply Qwen3 chat template
    let prompt = apply_chat_template(&req.messages);
    tracing::debug!("Chat template prompt: {:?}", &prompt[..prompt.len().min(200)]);

    // Tokenize
    let prompt_ids = state.tokenizer.encode(&prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Empty prompt after tokenization".to_string(),
        ));
    }
    tracing::info!("Prompt tokens: {}", prompt_ids.len());

    // Generate
    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        ..Default::default()
    };

    let result = state.engine.generate(&prompt_ids, &gen_config);

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
    }))
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
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
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
