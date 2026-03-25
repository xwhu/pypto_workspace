mod engine;
mod model;
mod ops;
mod scheduler;
mod server;

use std::sync::Arc;

use clap::Parser;

use engine::engine::Engine;
use model::config::Qwen3Config;
use model::network::Qwen3Model;
use model::parallel::ParallelConfig;
use model::quantize::QuantConfig;
use model::weights::SafetensorsLoader;
use ops::OpsBundle;
use scheduler::Qwen3Tokenizer;
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

    /// Backend: "stub" (no-op) or "ascend" (Ascend NPU via CANN).
    /// The "ascend" backend requires building with --features ascend.
    #[arg(long, default_value = "stub")]
    backend: String,

    /// Ascend NPU device ID (only used with --backend ascend).
    /// If not specified, reads ASCEND_DEVICE_ID env var (default: 0).
    #[arg(long)]
    device_id: Option<i32>,

    /// Path to model weights directory (containing *.safetensors files).
    /// If not specified, the server runs with uninitialized weights (shape-only).
    #[arg(long)]
    weights: Option<String>,
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
    // Priority: --config > auto-detect from --weights dir > hardcoded defaults
    let config = if let Some(config_path) = &cli.config {
        tracing::info!("Loading config from {config_path}");
        Qwen3Config::from_json(config_path)?
    } else if let Some(ref weights_dir) = cli.weights {
        // Auto-detect config.json in weights directory
        let auto_config = std::path::Path::new(weights_dir).join("config.json");
        if auto_config.exists() {
            tracing::info!(
                "Auto-detected config from weights dir: {}",
                auto_config.display()
            );
            Qwen3Config::from_json(auto_config.to_str().unwrap())?
        } else {
            tracing::info!(
                "No config.json in weights dir, using --model {} defaults",
                cli.model
            );
            match cli.model.as_str() {
                "0.6b" => Qwen3Config::qwen3_0_6b(),
                "4b" => Qwen3Config::qwen3_4b(),
                "8b" => Qwen3Config::qwen3_8b(),
                other => {
                    return Err(
                        format!("Unknown model variant: {other}. Use 0.6b, 4b, or 8b.").into(),
                    )
                }
            }
        }
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
                return Err(format!("Unknown model variant: {other}. Use 0.6b, 4b, or 8b.").into())
            }
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
        other => {
            return Err(format!("Unknown quant: {other}. Use none, int8, or awq-int4.").into())
        }
    };

    // Select backend
    let (ops, backend_label) = match cli.backend.as_str() {
        "stub" => {
            tracing::info!("Using STUB backend (no-op operators)");
            (OpsBundle::stub(), "STUB (no-op)")
        }
        #[cfg(feature = "ascend")]
        "ascend" => {
            tracing::info!("Using ASCEND NPU backend");
            let ops = OpsBundle::ascend(cli.device_id)
                .map_err(|e| format!("Failed to init Ascend backend: {}", e))?;
            (ops, "ASCEND NPU (CANN)")
        }
        #[cfg(not(feature = "ascend"))]
        "ascend" => {
            return Err(
                "Ascend backend requested but not compiled. Rebuild with: cargo build --features ascend".into()
            );
        }
        other => return Err(format!("Unknown backend: {other}. Use stub or ascend.").into()),
    };

    // Build model
    let mut model = Qwen3Model::new(config);
    tracing::info!(
        "Model: {} layers, {}B parameters",
        model.num_layers(),
        model.param_count() as f64 / 1e9
    );

    // Load weights if path is provided
    if let Some(weights_dir) = &cli.weights {
        let loader = SafetensorsLoader::from_dir(std::path::Path::new(weights_dir))?;
        model::weights::load_weights(&mut model, &loader)?;

        // Upload to device if using ascend backend
        #[cfg(feature = "ascend")]
        if cli.backend == "ascend" {
            // We need a stream for uploading — create a temporary one
            let upload_stream = ascend::Stream::new()
                .map_err(|e| format!("Failed to create upload stream: {}", e))?;
            model::weights::upload_weights_to_device(&mut model, &upload_stream)?;
        }
    } else {
        tracing::warn!("No --weights specified, running with uninitialized weights");
    }

    // Create engine with compiled execution plan
    #[cfg(feature = "ascend")]
    let engine = if cli.backend == "ascend" {
        // Create a separate AscendComputeOps for the v2 typed execution path
        let ascend_ops = crate::ops::ascend::AscendComputeOps::new(cli.device_id)
            .map_err(|e| format!("Failed to init Ascend ops for v2 path: {}", e))?;
        Engine::new_ascend(model, ascend_ops, ops, parallel, quant)
    } else {
        Engine::new(model, ops, parallel, quant)
    };
    #[cfg(not(feature = "ascend"))]
    let engine = Engine::new(model, ops, parallel, quant);
    tracing::info!("Engine: {}", engine.model_info());

    // Load tokenizer from weights directory
    let tokenizer = if let Some(ref weights_dir) = cli.weights {
        let tokenizer_path = std::path::Path::new(weights_dir).join("tokenizer.json");
        tracing::info!("Loading tokenizer from: {}", tokenizer_path.display());
        Qwen3Tokenizer::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?
    } else {
        return Err("--weights is required to load tokenizer.json".into());
    };

    // Create app state
    let state = Arc::new(AppState { engine, tokenizer });

    // Start server
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Rust LLM Server — Qwen3 (Compiled Plan)      ║");
    println!("║  {}", state.engine.model_info());
    println!("║  Operators: {:38}║", backend_label);
    println!("║  Endpoints:                                    ║");
    println!("║    GET  /health          POST /v1/completions  ║");
    println!("║    GET  /v1/models                             ║");
    println!("║  Port: {:41}║", cli.port);
    println!("╚════════════════════════════════════════════════╝");

    server::serve(state, cli.port).await?;
    Ok(())
}
