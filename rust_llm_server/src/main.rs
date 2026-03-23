mod model;
mod ops;
mod engine;
mod scheduler;
mod server;

use std::sync::Arc;

use clap::Parser;

use model::config::Qwen3Config;
use model::network::Qwen3Model;
use ops::StubOps;
use engine::engine::Engine;
use scheduler::StubTokenizer;
use server::AppState;

/// Rust LLM inference server for Qwen3 models.
#[derive(Parser, Debug)]
#[command(name = "rust-llm-server")]
#[command(about = "LLM inference server framework for Qwen3 models")]
struct Cli {
    /// Path to model config.json (if omitted, uses Qwen3-8B defaults).
    #[arg(long)]
    config: Option<String>,

    /// Model variant shortcut: "0.6b", "4b", "8b".
    #[arg(long, default_value = "8b")]
    model: String,

    /// Server port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_llm_server=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Load or create config
    let config = if let Some(config_path) = &cli.config {
        tracing::info!("Loading config from {config_path}");
        Qwen3Config::from_json(config_path)?
    } else {
        match cli.model.as_str() {
            "0.6b" => {
                tracing::info!("Using Qwen3-0.6B default config");
                Qwen3Config::qwen3_0_6b()
            }
            "4b" => {
                tracing::info!("Using Qwen3-4B default config");
                Qwen3Config::qwen3_4b()
            }
            "8b" => {
                tracing::info!("Using Qwen3-8B default config");
                Qwen3Config::qwen3_8b()
            }
            other => {
                return Err(format!("Unknown model variant: {other}. Use 0.6b, 4b, or 8b.").into());
            }
        }
    };

    // Build model
    let model = Qwen3Model::new(config);
    tracing::info!(
        "Model loaded: {} layers, {}B parameters",
        model.num_layers(),
        model.param_count() as f64 / 1e9
    );

    // Create engine with stub operators
    let ops = Arc::new(StubOps);
    let engine = Engine::new(model, ops);

    tracing::info!("Engine initialized: {}", engine.model_info());

    // Create app state
    let state = Arc::new(AppState {
        engine,
        tokenizer: StubTokenizer,
    });

    // Start server
    tracing::info!("Server starting on port {}", cli.port);
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Rust LLM Server — Qwen3 Framework            ║");
    println!("║  Model: {}                                     ", state.engine.config().model_type);
    println!("║  Operators: STUB (no-op)                       ║");
    println!("║  Endpoints:                                    ║");
    println!("║    GET  /health                                ║");
    println!("║    POST /v1/completions                        ║");
    println!("║    GET  /v1/models                             ║");
    println!("║  Port: {}                                    ", cli.port);
    println!("╚════════════════════════════════════════════════╝");

    server::serve(state, cli.port).await?;

    Ok(())
}
