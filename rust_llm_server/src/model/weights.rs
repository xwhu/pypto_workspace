//! Weight loading from safetensors files.
//!
//! Supports loading Qwen3 weights from HuggingFace safetensors format,
//! including multi-shard models (model-00001-of-00004.safetensors).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;

use super::network::Qwen3Model;
use super::parallel::{ParallelConfig, Qwen3ShardMap, ShardStrategy};
use super::tensor::DType;

// ─── TensorInfo ────────────────────────────────────────────────────────

/// Metadata about a tensor in a weight file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_size: usize,
}

// ─── WeightLoader trait ────────────────────────────────────────────────

/// Abstract interface for loading model weights from files.
///
/// Implementations handle specific file formats (safetensors, GGUF, etc.)
pub trait WeightLoader {
    /// List all tensor names in the weight file(s).
    fn tensor_names(&self) -> Vec<String>;

    /// Get metadata for a specific tensor.
    fn tensor_info(&self, name: &str) -> Option<TensorInfo>;

    /// Read the raw bytes for a tensor (host memory, zero-copy if mmap'd).
    fn read_tensor_bytes(&self, name: &str) -> Option<&[u8]>;
}

// ─── SafetensorsLoader ─────────────────────────────────────────────────

/// A single memory-mapped safetensors file.
struct MappedSafetensors {
    _path: PathBuf,
    mmap: Mmap,
}

/// Loads model weights from one or more safetensors files.
///
/// Uses memory mapping for zero-copy reads — the OS handles paging
/// data in from disk as needed, avoiding 2× memory for large models.
pub struct SafetensorsLoader {
    /// Memory-mapped files, kept alive for mmap lifetime.
    files: Vec<MappedSafetensors>,
    /// Maps tensor name → (file_index, byte_offset, byte_length).
    index: HashMap<String, (usize, usize, usize)>,
}

/// Convert safetensors dtype string to our DType.
fn parse_safetensors_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F16 => DType::Float16,
        safetensors::Dtype::BF16 => DType::BFloat16,
        safetensors::Dtype::F32 => DType::Float32,
        safetensors::Dtype::I32 => DType::Int32,
        safetensors::Dtype::U32 => DType::Uint32,
        safetensors::Dtype::I8 => DType::Int8,
        _ => DType::Float16, // fallback
    }
}

impl SafetensorsLoader {
    /// Load all `*.safetensors` files from a directory.
    ///
    /// Files are sorted by name to ensure deterministic loading order
    /// (important for multi-shard models like model-00001-of-00004.safetensors).
    pub fn from_dir(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut paths: Vec<PathBuf> = fs::read_dir(model_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path
                    .extension()
                    .map(|e| e == "safetensors")
                    .unwrap_or(false)
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if paths.is_empty() {
            return Err(format!("No *.safetensors files found in {}", model_dir.display()).into());
        }

        paths.sort();
        tracing::info!(
            "Found {} safetensors file(s) in {}",
            paths.len(),
            model_dir.display()
        );

        let mut files = Vec::new();
        let mut index = HashMap::new();

        for (file_idx, path) in paths.iter().enumerate() {
            let file = fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            // Parse the safetensors header to build our index
            let st = SafeTensors::deserialize(&mmap)?;
            for (name, info) in st.tensors() {
                let data = info.data();
                // We need to find the offset of this data within the mmap
                let data_ptr = data.as_ptr() as usize;
                let mmap_ptr = mmap.as_ptr() as usize;
                let offset = data_ptr - mmap_ptr;
                let length = data.len();

                tracing::trace!(
                    "  {} → shape={:?} dtype={:?} offset={} len={}",
                    name,
                    info.shape(),
                    info.dtype(),
                    offset,
                    length
                );

                index.insert(name.to_string(), (file_idx, offset, length));
            }

            tracing::info!(
                "  [{}] {} — {} tensors",
                file_idx,
                path.file_name().unwrap_or_default().to_string_lossy(),
                st.len()
            );

            files.push(MappedSafetensors {
                _path: path.clone(),
                mmap,
            });
        }

        tracing::info!("Total: {} weight tensors indexed", index.len());

        Ok(Self { files, index })
    }

    /// Load from a single safetensors file.
    #[allow(dead_code)]
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let _parent = path.parent().unwrap_or(Path::new("."));
        // If a single file is specified, load just that directory
        // but we'll use the direct approach
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let st = SafeTensors::deserialize(&mmap)?;
        let mut index = HashMap::new();

        for (name, info) in st.tensors() {
            let data = info.data();
            let data_ptr = data.as_ptr() as usize;
            let mmap_ptr = mmap.as_ptr() as usize;
            let offset = data_ptr - mmap_ptr;
            let length = data.len();
            index.insert(name.to_string(), (0, offset, length));
        }

        tracing::info!("Loaded {} tensors from {}", index.len(), path.display());

        let files = vec![MappedSafetensors {
            _path: path.to_path_buf(),
            mmap,
        }];

        Ok(Self { files, index })
    }

    /// Get the SafeTensors view for a file (for metadata queries).
    fn get_safetensors(
        &self,
        file_idx: usize,
    ) -> Result<SafeTensors<'_>, Box<dyn std::error::Error>> {
        Ok(SafeTensors::deserialize(&self.files[file_idx].mmap)?)
    }
}

impl WeightLoader for SafetensorsLoader {
    fn tensor_names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        let (file_idx, _offset, length) = self.index.get(name)?;
        let st = self.get_safetensors(*file_idx).ok()?;
        let info = st.tensor(name).ok()?;
        Some(TensorInfo {
            name: name.to_string(),
            shape: info.shape().to_vec(),
            dtype: parse_safetensors_dtype(info.dtype()),
            byte_size: *length,
        })
    }

    fn read_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let (file_idx, offset, length) = self.index.get(name)?;
        let mmap = &self.files[*file_idx].mmap;
        Some(&mmap[*offset..*offset + *length])
    }
}

// ─── Weight Loading into Model ─────────────────────────────────────────

/// Load weights from a WeightLoader into a Qwen3Model, copying to host.
///
/// This sets each Tensor's `host_data` field with the raw weight bytes.
/// For NPU execution, call `upload_weights_to_device` afterwards.
///
/// Auto-detects dtype from safetensors metadata and updates model tensors
/// if they differ from the hardcoded defaults.
pub fn load_weights(
    model: &mut Qwen3Model,
    loader: &dyn WeightLoader,
) -> Result<WeightLoadStats, Box<dyn std::error::Error>> {
    let mut stats = WeightLoadStats::default();

    for tensor in model.weight_tensors_mut() {
        match loader.tensor_info(&tensor.name) {
            Some(file_info) => {
                // Warn on dtype mismatch but trust model config
                // (user may have manually converted weights to a different dtype)
                if tensor.dtype != file_info.dtype {
                    tracing::warn!(
                        "dtype mismatch for {}: model={:?}, file={:?} (auto-upgrading model tensor to file dtype)",
                        tensor.name,
                        tensor.dtype,
                        file_info.dtype
                    );
                }

                // Validate shape consistency
                if tensor.shape != file_info.shape {
                    tracing::error!(
                        "Shape mismatch for {}: model expects {:?}, file has {:?}. \
                         Hint: use --config <path>/config.json to load the model's real config.",
                        tensor.name,
                        tensor.shape,
                        file_info.shape
                    );
                    stats.shape_mismatches += 1;
                }

                // Read bytes
                if let Some(bytes) = loader.read_tensor_bytes(&tensor.name) {
                    let expected = tensor.size_bytes();
                    if bytes.len() != expected {
                        tracing::warn!(
                            "Size mismatch for {}: model expects {} bytes, file has {} bytes",
                            tensor.name,
                            expected,
                            bytes.len()
                        );
                    }

                    tensor.dtype = file_info.dtype;
                    tensor.host_data = Some(bytes.to_vec());
                    stats.loaded += 1;
                    stats.bytes += bytes.len();
                }
            }
            None => {
                tracing::warn!("Weight not found in file: {}", tensor.name);
                stats.missing += 1;
                stats.missing_names.push(tensor.name.clone());
            }
        }
    }

    // Report extra tensors in file that aren't in our model
    let model_tensor_names: std::collections::HashSet<String> = model
        .weight_tensors_mut()
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let file_tensors = loader.tensor_names();
    let extra: Vec<&String> = file_tensors
        .iter()
        .filter(|n| !model_tensor_names.contains(n.as_str()))
        .collect();
    if !extra.is_empty() {
        tracing::info!(
            "{} extra tensors in file not mapped to model (e.g., {:?})",
            extra.len(),
            &extra[..extra.len().min(5)]
        );
    }

    tracing::info!(
        "Loaded {}/{} weight tensors ({:.2} GB), {} missing, {} shape mismatches",
        stats.loaded,
        stats.loaded + stats.missing,
        stats.bytes as f64 / 1e9,
        stats.missing,
        stats.shape_mismatches,
    );

    if !stats.missing_names.is_empty() {
        tracing::warn!(
            "Missing weights: {:?}",
            &stats.missing_names[..stats.missing_names.len().min(5)]
        );
    }

    if stats.shape_mismatches > 0 {
        return Err(format!(
            "{} shape mismatches detected. Use --config to load the correct config.json from the weights directory.",
            stats.shape_mismatches
        ).into());
    }

    Ok(stats)
}

/// Determine the shard strategy for a weight tensor by its name.
fn shard_strategy_for_name(name: &str) -> ShardStrategy {
    if name.ends_with("q_proj.weight") {
        Qwen3ShardMap::q_proj()
    } else if name.ends_with("k_proj.weight") {
        Qwen3ShardMap::k_proj()
    } else if name.ends_with("v_proj.weight") {
        Qwen3ShardMap::v_proj()
    } else if name.ends_with("o_proj.weight") {
        Qwen3ShardMap::o_proj()
    } else if name.ends_with("gate_proj.weight") {
        Qwen3ShardMap::gate_proj()
    } else if name.ends_with("up_proj.weight") {
        Qwen3ShardMap::up_proj()
    } else if name.ends_with("down_proj.weight") {
        Qwen3ShardMap::down_proj()
    } else if name.contains("layernorm") || name.contains("norm.weight") {
        Qwen3ShardMap::norm_weight()
    } else if name == "model.embed_tokens.weight" {
        Qwen3ShardMap::embed_tokens()
    } else if name == "lm_head.weight" {
        // Replicate lm_head for simplicity (avoids AllGather)
        ShardStrategy::Replicate
    } else {
        ShardStrategy::Replicate
    }
}

/// Extract a shard of weight bytes according to the shard strategy.
///
/// For `ShardColumns` (split dim 0 of `[out, in]`): contiguous slice.
/// For `ShardRows` (split dim 1 of `[out, in]`): non-contiguous gather.
/// For `Replicate`: returns the full bytes.
fn extract_shard(
    bytes: &[u8],
    shape: &[usize],
    strategy: ShardStrategy,
    tp_rank: usize,
    tp_size: usize,
    dtype_size: usize,
) -> Vec<u8> {
    if tp_size <= 1 {
        return bytes.to_vec();
    }

    match strategy {
        ShardStrategy::Replicate => bytes.to_vec(),

        ShardStrategy::ShardColumns => {
            // Split dim 0 (rows in storage). For [out_features, in_features]:
            // Each rank gets rows [rank * out/tp .. (rank+1) * out/tp]
            // This is contiguous in row-major layout.
            if shape.len() < 2 {
                return bytes.to_vec(); // 1D tensors are replicated
            }
            let out_dim = shape[0];
            let in_dim = shape[1];
            let shard_out = out_dim / tp_size;
            let row_bytes = in_dim * dtype_size;
            let start = tp_rank * shard_out * row_bytes;
            let end = start + shard_out * row_bytes;
            bytes[start..end].to_vec()
        }

        ShardStrategy::ShardRows => {
            // Split dim 1 (columns in storage). For [out_features, in_features]:
            // Each rank gets columns [rank * in/tp .. (rank+1) * in/tp]
            // Non-contiguous: need to gather column stripe from each row.
            if shape.len() < 2 {
                return bytes.to_vec();
            }
            let out_dim = shape[0];
            let in_dim = shape[1];
            let shard_in = in_dim / tp_size;
            let row_bytes = in_dim * dtype_size;
            let shard_row_bytes = shard_in * dtype_size;
            let col_offset = tp_rank * shard_row_bytes;

            let mut result = Vec::with_capacity(out_dim * shard_row_bytes);
            for row in 0..out_dim {
                let row_start = row * row_bytes + col_offset;
                result.extend_from_slice(&bytes[row_start..row_start + shard_row_bytes]);
            }
            result
        }
    }
}

/// Load weights from a WeightLoader into a Qwen3Model with TP sharding.
///
/// Each weight tensor is sliced according to its shard strategy before
/// copying to `host_data`. The model should have been created with
/// `Qwen3Model::new_sharded()` so shapes already reflect the shard.
///
/// PP-aware: only loads weights for tensors present in the model
/// (which only contains this PP stage's layers if created with new_sharded).
pub fn load_weights_sharded(
    model: &mut Qwen3Model,
    loader: &dyn WeightLoader,
    parallel: &ParallelConfig,
) -> Result<WeightLoadStats, Box<dyn std::error::Error>> {
    let mut stats = WeightLoadStats::default();

    // If no TP, fall back to standard loading
    if !parallel.is_tp() {
        return load_weights(model, loader);
    }

    for tensor in model.weight_tensors_mut() {
        let strategy = shard_strategy_for_name(&tensor.name);

        match loader.tensor_info(&tensor.name) {
            Some(file_info) => {
                if tensor.dtype != file_info.dtype {
                    tracing::warn!(
                        "dtype mismatch for {}: model={:?}, file={:?} (auto-upgrading model tensor to file dtype)",
                        tensor.name,
                        tensor.dtype,
                        file_info.dtype
                    );
                }

                if let Some(bytes) = loader.read_tensor_bytes(&tensor.name) {
                    let dtype_size = file_info.dtype.size_bytes();
                    let shard_bytes = extract_shard(
                        bytes,
                        &file_info.shape,
                        strategy,
                        parallel.tp_rank,
                        parallel.tp_size,
                        dtype_size,
                    );

                    // Validate shard size matches expected tensor size
                    let expected = tensor.size_bytes();
                    if shard_bytes.len() != expected {
                        tracing::warn!(
                            "Shard size mismatch for {} (strategy={:?}): expected {} bytes, got {} bytes",
                            tensor.name,
                            strategy,
                            expected,
                            shard_bytes.len()
                        );
                    }

                    tensor.dtype = file_info.dtype; // <--- CRITICAL FIX
                    tensor.host_data = Some(shard_bytes);
                    stats.loaded += 1;
                    stats.bytes += tensor.host_data.as_ref().unwrap().len();
                }
            }
            None => {
                // PP-aware: if this tensor isn't in the file but isn't needed
                // for this stage, that's OK. But log it as missing.
                tracing::warn!("Weight not found in file: {}", tensor.name);
                stats.missing += 1;
                stats.missing_names.push(tensor.name.clone());
            }
        }
    }

    tracing::info!(
        "Loaded {}/{} sharded weight tensors ({:.2} GB), {} missing (tp_rank={}/{})",
        stats.loaded,
        stats.loaded + stats.missing,
        stats.bytes as f64 / 1e9,
        stats.missing,
        parallel.tp_rank,
        parallel.tp_size,
    );

    Ok(stats)
}

/// Statistics from a weight loading operation.
#[derive(Debug, Default)]
pub struct WeightLoadStats {
    pub loaded: usize,
    pub missing: usize,
    pub bytes: usize,
    pub shape_mismatches: usize,
    pub missing_names: Vec<String>,
}

// ─── Device Upload (Ascend-specific) ───────────────────────────────────

#[cfg(feature = "ascend")]
pub fn upload_weights_to_device(
    model: &mut Qwen3Model,
    stream: &ascend::Stream,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut uploaded = 0usize;

    for tensor in model.weight_tensors_mut() {
        if let Some(host_bytes) = tensor.host_data.take() {
            let mut buf = ascend::DeviceBuffer::alloc(host_bytes.len())?;
            buf.copy_from_host(&host_bytes)?;

            tensor.data_ptr = Some(buf.ptr() as usize);
            tensor.device_buf = Some(buf);
            uploaded += 1;
        }
    }

    stream.synchronize()?;
    tracing::info!("Uploaded {} weight tensors to NPU device memory", uploaded);
    Ok(uploaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_conversion() {
        assert_eq!(
            parse_safetensors_dtype(safetensors::Dtype::F16),
            DType::Float16
        );
        assert_eq!(
            parse_safetensors_dtype(safetensors::Dtype::BF16),
            DType::BFloat16
        );
        assert_eq!(
            parse_safetensors_dtype(safetensors::Dtype::F32),
            DType::Float32
        );
    }

    #[test]
    fn test_shard_strategy_for_name() {
        assert_eq!(
            shard_strategy_for_name("model.layers.0.self_attn.q_proj.weight"),
            ShardStrategy::ShardColumns
        );
        assert_eq!(
            shard_strategy_for_name("model.layers.0.self_attn.o_proj.weight"),
            ShardStrategy::ShardRows
        );
        assert_eq!(
            shard_strategy_for_name("model.layers.0.input_layernorm.weight"),
            ShardStrategy::Replicate
        );
        assert_eq!(
            shard_strategy_for_name("model.embed_tokens.weight"),
            ShardStrategy::Replicate
        );
        assert_eq!(
            shard_strategy_for_name("lm_head.weight"),
            ShardStrategy::Replicate
        );
    }

    #[test]
    fn test_extract_shard_columns() {
        // 4x4 matrix (FP16 = 2 bytes per element), TP=2
        // Row-major: [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        // ShardColumns splits dim 0: rank 0 gets rows 0-1, rank 1 gets rows 2-3
        let data: Vec<u8> = (0..32).collect(); // 4*4*2 = 32 bytes
        let shape = vec![4, 4];

        let shard0 = extract_shard(&data, &shape, ShardStrategy::ShardColumns, 0, 2, 2);
        assert_eq!(shard0.len(), 16); // 2*4*2 = 16 bytes
        assert_eq!(&shard0[..8], &data[..8]); // first 2 rows

        let shard1 = extract_shard(&data, &shape, ShardStrategy::ShardColumns, 1, 2, 2);
        assert_eq!(shard1.len(), 16);
        assert_eq!(&shard1[..8], &data[16..24]); // last 2 rows
    }

    #[test]
    fn test_extract_shard_rows() {
        // 2x4 matrix (1 byte per element for simplicity), TP=2
        // Row-major: [[0,1,2,3], [4,5,6,7]]
        // ShardRows splits dim 1: rank 0 gets cols 0-1, rank 1 gets cols 2-3
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let shape = vec![2, 4];

        let shard0 = extract_shard(&data, &shape, ShardStrategy::ShardRows, 0, 2, 1);
        assert_eq!(shard0, vec![0, 1, 4, 5]); // cols 0-1 from each row

        let shard1 = extract_shard(&data, &shape, ShardStrategy::ShardRows, 1, 2, 1);
        assert_eq!(shard1, vec![2, 3, 6, 7]); // cols 2-3 from each row
    }

    #[test]
    fn test_extract_shard_replicate() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let shape = vec![4];
        let shard = extract_shard(&data, &shape, ShardStrategy::Replicate, 0, 4, 1);
        assert_eq!(shard, data);
    }
}
