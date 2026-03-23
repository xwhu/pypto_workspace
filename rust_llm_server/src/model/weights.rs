//! Weight loading from safetensors files.
//!
//! Supports loading Qwen3 weights from HuggingFace safetensors format,
//! including multi-shard models (model-00001-of-00004.safetensors).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;

use super::tensor::{DType, Tensor};
use super::network::Qwen3Model;

// ─── TensorInfo ────────────────────────────────────────────────────────

/// Metadata about a tensor in a weight file.
#[derive(Debug, Clone)]
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
                if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if paths.is_empty() {
            return Err(format!(
                "No *.safetensors files found in {}",
                model_dir.display()
            ).into());
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
                    name, info.shape(), info.dtype(), offset, length
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
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let parent = path.parent().unwrap_or(Path::new("."));
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

        tracing::info!(
            "Loaded {} tensors from {}",
            index.len(),
            path.display()
        );

        let files = vec![MappedSafetensors {
            _path: path.to_path_buf(),
            mmap,
        }];

        Ok(Self { files, index })
    }

    /// Get the SafeTensors view for a file (for metadata queries).
    fn get_safetensors(&self, file_idx: usize) -> Result<SafeTensors<'_>, Box<dyn std::error::Error>> {
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
                // Auto-detect dtype from file (e.g., BF16 weights when model defaults to FP16)
                if tensor.dtype != file_info.dtype {
                    tracing::info!(
                        "dtype override for {}: {:?} -> {:?} (from file)",
                        tensor.name, tensor.dtype, file_info.dtype
                    );
                    tensor.dtype = file_info.dtype;
                }

                // Validate shape consistency
                if tensor.shape != file_info.shape {
                    tracing::error!(
                        "Shape mismatch for {}: model expects {:?}, file has {:?}. \
                         Hint: use --config <path>/config.json to load the model's real config.",
                        tensor.name, tensor.shape, file_info.shape
                    );
                    stats.shape_mismatches += 1;
                }

                // Read bytes
                if let Some(bytes) = loader.read_tensor_bytes(&tensor.name) {
                    let expected = tensor.size_bytes();
                    if bytes.len() != expected {
                        tracing::warn!(
                            "Size mismatch for {}: model expects {} bytes, file has {} bytes",
                            tensor.name, expected, bytes.len()
                        );
                    }

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
        tracing::warn!("Missing weights: {:?}", &stats.missing_names[..stats.missing_names.len().min(5)]);
    }

    if stats.shape_mismatches > 0 {
        return Err(format!(
            "{} shape mismatches detected. Use --config to load the correct config.json from the weights directory.",
            stats.shape_mismatches
        ).into());
    }

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
        assert_eq!(parse_safetensors_dtype(safetensors::Dtype::F16), DType::Float16);
        assert_eq!(parse_safetensors_dtype(safetensors::Dtype::BF16), DType::BFloat16);
        assert_eq!(parse_safetensors_dtype(safetensors::Dtype::F32), DType::Float32);
    }
}
