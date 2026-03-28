//! HCCL process group initialization for distributed inference.
//!
//! Creates separate HCCL communicators for TP and PP groups.
//! Each communicator is initialized with a unique `HcclRootInfo` shared
//! via a temporary file.

use super::DistributedConfig;
use ascend::comm::{get_root_info, read_root_info_from_file, write_root_info_to_file, HcclCommunicator};
use hccl_sys::HcclRootInfo;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Holds the HCCL communicators for each parallelism dimension.
pub struct ProcessGroups {
    /// Communicator for TP group (AllReduce, AllGather). None if tp_size == 1.
    pub tp_comm: Option<HcclCommunicator>,
    /// Communicator for PP group (Send, Recv). None if pp_size == 1.
    pub pp_comm: Option<HcclCommunicator>,
}

/// Initialize HCCL communicators for all parallelism groups.
///
/// This function must be called collectively by all processes. It:
/// 1. Creates a TP communicator (if tp_size > 1)
/// 2. Creates a PP communicator (if pp_size > 1)
///
/// Root info is shared between ranks via temporary files in `root_info_dir`.
/// The rank with the lowest world_rank in each group generates the root info.
///
/// # Arguments
/// - `dist`: Distributed configuration for this process.
/// - `root_info_dir`: Directory to store temporary root info files.
pub fn init_process_groups(
    dist: &DistributedConfig,
    root_info_dir: &Path,
) -> Result<ProcessGroups, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(root_info_dir)?;

    // Initialize TP communicator
    let tp_comm = if dist.tp_size > 1 {
        let tp_ranks = dist.tp_group_ranks();
        let tp_group_id = format!("tp_pp{}_dp{}", dist.pp_rank, dist.dp_rank);
        let root_info_path = root_info_dir.join(format!("hccl_root_{}.bin", tp_group_id));

        tracing::info!(
            "Initializing TP communicator: group={}, ranks={:?}, my_rank={}",
            tp_group_id,
            tp_ranks,
            dist.tp_group_rank()
        );

        let root_info = share_root_info(
            &root_info_path,
            dist.tp_group_rank() == 0,
        )?;

        let comm = HcclCommunicator::init_rank(
            dist.tp_size as u32,
            &root_info,
            dist.tp_group_rank() as u32,
        )?;
        tracing::info!("TP communicator initialized successfully");
        Some(comm)
    } else {
        None
    };

    // Initialize PP communicator
    let pp_comm = if dist.pp_size > 1 {
        let pp_ranks = dist.pp_group_ranks();
        let pp_group_id = format!("pp_tp{}_dp{}", dist.tp_rank, dist.dp_rank);
        let root_info_path = root_info_dir.join(format!("hccl_root_{}.bin", pp_group_id));

        tracing::info!(
            "Initializing PP communicator: group={}, ranks={:?}, my_rank={}",
            pp_group_id,
            pp_ranks,
            dist.pp_group_rank()
        );

        let root_info = share_root_info(
            &root_info_path,
            dist.pp_group_rank() == 0,
        )?;

        let comm = HcclCommunicator::init_rank(
            dist.pp_size as u32,
            &root_info,
            dist.pp_group_rank() as u32,
        )?;
        tracing::info!("PP communicator initialized successfully");
        Some(comm)
    } else {
        None
    };

    Ok(ProcessGroups { tp_comm, pp_comm })
}

/// Share root info between ranks via a temporary file.
///
/// - If `is_root`: generates root info and writes it to `path`.
/// - If not root: polls `path` until it appears (with timeout).
fn share_root_info(
    path: &Path,
    is_root: bool,
) -> Result<HcclRootInfo, Box<dyn std::error::Error>> {
    if is_root {
        // Remove stale file from previous runs
        let _ = std::fs::remove_file(path);

        let root_info = get_root_info()?;
        write_root_info_to_file(&root_info, path)?;
        tracing::debug!("Root info written to {}", path.display());
        Ok(root_info)
    } else {
        // Poll for the file to appear (root rank may not have written it yet)
        let max_wait = Duration::from_secs(60);
        let poll_interval = Duration::from_millis(100);
        let start = std::time::Instant::now();

        loop {
            if path.exists() {
                let root_info = read_root_info_from_file(path)?;
                tracing::debug!("Root info read from {}", path.display());
                return Ok(root_info);
            }

            if start.elapsed() > max_wait {
                return Err(format!(
                    "Timeout waiting for root info at {} (waited {:?})",
                    path.display(),
                    max_wait
                )
                .into());
            }

            thread::sleep(poll_interval);
        }
    }
}

/// Clean up root info files after initialization.
pub fn cleanup_root_info(root_info_dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(root_info_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("hccl_root_"))
                .unwrap_or(false)
            {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}
