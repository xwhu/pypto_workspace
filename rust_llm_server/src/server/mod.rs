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
use crate::scheduler::{CompletionRequest, CompletionResponse, StubTokenizer};

/// Shared application state for the HTTP server.
pub struct AppState {
    pub engine: Engine,
    pub tokenizer: StubTokenizer,
}

/// Build the Axum router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/completions", post(completion_handler))
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
///
/// Accepts a prompt and returns generated text (stub tokens).
async fn completion_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    tracing::info!("Completion request: prompt={:?}, max_tokens={}", req.prompt, req.max_tokens);

    // Tokenize input
    let prompt_ids = state.tokenizer.encode(&req.prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Empty prompt after tokenization".to_string(),
        ));
    }

    // Generate
    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        ..Default::default()
    };

    let result = state.engine.generate(&prompt_ids, &gen_config);

    // Decode output
    let text = state.tokenizer.decode(&result.token_ids);

    Ok(Json(CompletionResponse {
        text,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        model: state.engine.config().model_type.clone(),
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

/// Start the HTTP server.
pub async fn serve(state: Arc<AppState>, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = build_router(state);
    let addr = format!("0.0.0.0:{port}");

    tracing::info!("Starting server on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
