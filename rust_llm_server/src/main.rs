mod model;
mod ops;
mod engine;
mod scheduler;
mod server;

use std::sync::Arc;

use clap::Parser;

use model::config::Qwen3Config;
use model::network::Qwen3Model;
use model::parallel::ParallelConfig;
use model::quantize::QuantConfig;
use ops::OpsBundle;
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

    /// Tensor parallelism degree.
    #[arg(long, default_value_t = 1)]
    tp: usize,

    /// Pipeline parallelism degree.
    #[arg(long, default_value_t = 1)]
    pp: usize,

    /// This device's TP rank.
    #[arg(long, default_value_t = 0)]
    tp_rank: usize,

    /// This device's PP rank.
    #[arg(long, default_value_t = 0)]
    pp_rank: usize,

    /// Quantization: "none", "int8", "awq-int4".
    #[arg(long, default_value = "none")]
    quant: String,
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
            "0.6b" => { tracing::info!("Using Qwen3-0.6B default config"); Qwen3Config::qwen3_0_6b() }
            "4b" => { tracing::info!("Using Qwen3-4B default config"); Qwen3Config::qwen3_4b() }
            "8b" => { tracing::info!("Using Qwen3-8B default config"); Qwen3Config::qwen3_8b() }
            other => return Err(format!("Unknown model variant: {other}. Use 0.6b, 4b, or 8b.").into()),
        }
    };

    // Build parallel config
    let parallel = ParallelConfig {
        tp_size: cli.tp,
        pp_size: cli.pp,
        tp_rank: cli.tp_rank,
        pp_rank: cli.pp_rank,
    };

    // Build quant config
    let quant = match cli.quant.as_str() {
        "none" => QuantConfig::none(),
        "int8" => QuantConfig::int8_per_tensor(),
        "awq-int4" => QuantConfig::awq_int4(128),
        other => return Err(format!("Unknown quant: {other}. Use none, int8, or awq-int4.").into()),
    };

    // Build model
    let model = Qwen3Model::new(config);
    tracing::info!(
        "Model: {} layers, {}B parameters",
        model.num_layers(),
        model.param_count() as f64 / 1e9
    );

    // Create engine with compiled execution plan
    let engine = Engine::new(model, OpsBundle::stub(), parallel, quant);
    tracing::info!("Engine: {}", engine.model_info());

    // Create app state
    let state = Arc::new(AppState {
        engine,
        tokenizer: StubTokenizer,
    });

    // Start server
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Rust LLM Server — Qwen3 (Compiled Plan)      ║");
    println!("║  {}",  state.engine.model_info());
    println!("║  Operators: STUB (no-op)                       ║");
    println!("║  Endpoints:                                    ║");
    println!("║    GET  /health          POST /v1/completions  ║");
    println!("║    GET  /v1/models                             ║");
    println!("║  Port: {}                                    ", cli.port);
    println!("╚════════════════════════════════════════════════╝");

    server::serve(state, cli.port).await?;
    Ok(())
}
