//! Debug tensor dump for TP precision comparison.
//!
//! When the environment variable `TP_DEBUG_DUMP_DIR` is set, this module
//! dumps intermediate tensors from the first N layers (default 2) as
//! raw binary files. A `meta.json` sidecar records shape/dtype for each file.
//!
//! Usage:
//! ```bash
//! TP_DEBUG_DUMP_DIR=/tmp/tp1_dump cargo run --features ascend -- ...
//! ```
//!
//! Then use `scripts/compare_tp_dumps.py` to diff TP=1 vs TP=2 outputs.

#[cfg(feature = "ascend")]
use crate::model::device_tensor::DeviceTensor;

use std::collections::BTreeMap;
use std::path::PathBuf;

/// Maximum number of layers to dump (0-indexed, exclusive).
#[allow(dead_code)]
const DEFAULT_MAX_LAYERS: usize = 2;

/// Metadata for one dumped tensor file.
#[derive(serde::Serialize)]
#[allow(dead_code)]
struct DumpEntry {
    file: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
}

/// Debug dumper that writes device tensors to binary files.
///
/// Created once per `execute_paged()` call. If `TP_DEBUG_DUMP_DIR` is not
/// set, `from_env()` returns `None` and all dump calls are no-ops.
#[allow(dead_code)]
pub struct DebugDumper {
    dir: PathBuf,
    max_layers: usize,
    entries: BTreeMap<String, DumpEntry>,
}

#[allow(dead_code)]
impl DebugDumper {
    /// Read `TP_DEBUG_DUMP_DIR` from environment. Returns `None` if unset.
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var("TP_DEBUG_DUMP_DIR").ok()?;
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).ok()?;

        let max_layers = std::env::var("TP_DEBUG_DUMP_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_LAYERS);

        tracing::info!(
            "DebugDumper enabled: dir={}, max_layers={}",
            dir.display(),
            max_layers
        );

        Some(Self {
            dir,
            max_layers,
            entries: BTreeMap::new(),
        })
    }

    /// Whether this layer should be dumped.
    pub fn should_dump(&self, layer_idx: usize) -> bool {
        layer_idx < self.max_layers
    }

    /// Dump a device tensor to a binary file.
    ///
    /// Synchronizes the stream, copies data from device to host, and writes
    /// the raw bytes to `{dir}/{name}.bin`. Also records metadata.
    #[cfg(feature = "ascend")]
    pub fn dump(
        &mut self,
        name: &str,
        tensor: &DeviceTensor,
        stream: &ascend::Stream,
    ) {
        let size = tensor.size_bytes();
        let mut host = vec![0u8; size];

        // Synchronize to ensure all prior compute ops have finished
        stream.synchronize().expect("DebugDumper: stream sync failed");

        // Copy from device to host
        tensor
            .buf
            .copy_to_host(&mut host)
            .expect("DebugDumper: copy_to_host failed");

        // Write binary file
        let filename = format!("{name}.bin");
        let path = self.dir.join(&filename);
        std::fs::write(&path, &host).unwrap_or_else(|e| {
            tracing::error!("DebugDumper: failed to write {}: {}", path.display(), e);
        });

        // Record metadata
        let dtype_str = match tensor.dtype() {
            crate::model::tensor::DType::Float16 => "float16",
            crate::model::tensor::DType::BFloat16 => "bfloat16",
            crate::model::tensor::DType::Float32 => "float32",
            crate::model::tensor::DType::Int32 => "int32",
            crate::model::tensor::DType::Uint32 => "uint32",
            crate::model::tensor::DType::Int8 => "int8",
            crate::model::tensor::DType::Int4 => "int4",
        };

        self.entries.insert(
            name.to_string(),
            DumpEntry {
                file: filename,
                shape: tensor.shape().to_vec(),
                dtype: dtype_str.to_string(),
                size_bytes: size,
            },
        );

        tracing::debug!(
            "DebugDumper: wrote {} ({:?} {:?}, {} bytes)",
            name,
            tensor.shape(),
            dtype_str,
            size
        );
    }

    /// Write `meta.json` with all recorded entries.
    pub fn finalize(&self) {
        let path = self.dir.join("meta.json");
        let json = serde_json::to_string_pretty(&self.entries)
            .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
        std::fs::write(&path, json).unwrap_or_else(|e| {
            tracing::error!("DebugDumper: failed to write meta.json: {}", e);
        });
        tracing::info!(
            "DebugDumper: finalized {} entries to {}",
            self.entries.len(),
            path.display()
        );
    }
}
