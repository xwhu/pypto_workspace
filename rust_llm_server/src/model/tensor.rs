use std::fmt;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float16,
    BFloat16,
    Float32,
    Int32,
    Uint32,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 | DType::Int32 | DType::Uint32 => 4,
        }
    }
}

/// Lightweight tensor descriptor.
///
/// This does not hold actual data — it describes the shape, dtype, and
/// optionally points to device memory. Used to define data flow through
/// the model graph and track shapes during the forward pass.
#[derive(Clone)]
pub struct Tensor {
    /// Shape dimensions (e.g., [batch, seq_len, hidden_size]).
    pub shape: Vec<usize>,

    /// Element data type.
    pub dtype: DType,

    /// Optional device memory pointer (opaque, device-specific).
    /// `None` for shape-only descriptors (used during graph construction).
    pub data_ptr: Option<usize>,

    /// Human-readable name for debugging.
    pub name: String,
}

impl Tensor {
    /// Create a new tensor descriptor with no data.
    pub fn new(shape: Vec<usize>, dtype: DType, name: impl Into<String>) -> Self {
        Self {
            shape,
            dtype,
            data_ptr: None,
            name: name.into(),
        }
    }

    /// Create a tensor descriptor for a weight matrix.
    pub fn weight(rows: usize, cols: usize, name: impl Into<String>) -> Self {
        Self::new(vec![rows, cols], DType::Float16, name)
    }

    /// Create a tensor descriptor for an embedding table.
    pub fn embedding_table(vocab_size: usize, hidden_size: usize, name: impl Into<String>) -> Self {
        Self::new(vec![vocab_size, hidden_size], DType::Float16, name)
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Create an output tensor with the given shape, inheriting dtype.
    pub fn with_shape(&self, shape: Vec<usize>, name: impl Into<String>) -> Self {
        Self {
            shape,
            dtype: self.dtype,
            data_ptr: None,
            name: name.into(),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, {:?}, name={:?})",
            self.shape, self.dtype, self.name
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}]", self.name, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_numel() {
        let t = Tensor::new(vec![2, 3, 4], DType::Float16, "test");
        assert_eq!(t.numel(), 24);
        assert_eq!(t.size_bytes(), 48);
    }

    #[test]
    fn test_weight_tensor() {
        let w = Tensor::weight(4096, 4096, "q_proj");
        assert_eq!(w.shape, vec![4096, 4096]);
        assert_eq!(w.numel(), 4096 * 4096);
    }
}
