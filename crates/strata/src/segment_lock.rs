//! Sealed Segment Filesystem Immutability (#836 Vector 4).
//!
//! Marks sealed `.strata` segment files read-only and applies OS-level
//! immutability protections:
//! - Linux: sets `0444` permissions and attempts `FS_IOC_SETFLAGS`
//!   (`FS_IMMUTABLE_FL`) when `CAP_LINUX_IMMUTABLE` is available.
//! - Windows: sets `FILE_ATTRIBUTE_READONLY` (via `Permissions::set_readonly`)
//!   so unauthenticated writes/truncations fail at the filesystem layer.
//!
//! Records the achieved protection level honestly in [`SealReceipt`] without
//! failing when unprivileged user accounts lack kernel immutable capability.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Achieved filesystem immutability level for a sealed segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealLevel {
    /// Linux `FS_IMMUTABLE_FL` set (requires `CAP_LINUX_IMMUTABLE`).
    KernelImmutable,
    /// OS read-only attribute / `0444` mode applied.
    ReadOnlyAttr,
}

/// Receipt emitted after sealing a Strata segment file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealReceipt {
    pub path: PathBuf,
    pub level: SealLevel,
    pub blake3_hex: String,
    pub readonly_set: bool,
}

/// Compute BLAKE3 digest of a segment file.
fn hash_file_blake3(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Seal a Strata segment file against modification or truncation.
pub fn seal_segment(path: &Path) -> io::Result<SealReceipt> {
    let blake3_hex = hash_file_blake3(path)?;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_readonly(true);
    fs::set_permissions(path, perms)?;

    #[cfg(target_os = "linux")]
    let level = {
        use std::os::unix::fs::PermissionsExt;
        let mut unix_perms = fs::metadata(path)?.permissions();
        unix_perms.set_mode(0o444);
        fs::set_permissions(path, unix_perms)?;
        if try_set_linux_immutable(path, true) {
            SealLevel::KernelImmutable
        } else {
            SealLevel::ReadOnlyAttr
        }
    };

    #[cfg(not(target_os = "linux"))]
    let level = SealLevel::ReadOnlyAttr;

    Ok(SealReceipt {
        path: path.to_path_buf(),
        level,
        blake3_hex,
        readonly_set: true,
    })
}

/// Verify that a sealed segment file still has its read-only protection intact.
pub fn verify_segment_seal(path: &Path) -> io::Result<bool> {
    let meta = fs::metadata(path)?;
    Ok(meta.permissions().readonly())
}

/// Owner-ceremony unlock for deliberate segment maintenance or migration.
pub fn unseal_segment_owner(path: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let _ = try_set_linux_immutable(path, false);
    }

    let mut perms = fs::metadata(path)?.permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    fs::set_permissions(path, perms)
}

#[cfg(target_os = "linux")]
fn try_set_linux_immutable(path: &Path, enable: bool) -> bool {
    let flag = if enable { "+i" } else { "-i" };
    std::process::Command::new("chattr")
        .arg(flag)
        .arg(path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn seal_blocks_subsequent_write_and_verifies() {
        let dir = std::env::temp_dir().join(format!("strata-seal-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        let seg = dir.join("00000001-test.seg");
        {
            let mut f = File::create(&seg).expect("create segment");
            f.write_all(b"STRATA-FRAME-DATA").expect("write segment");
        }

        let receipt = seal_segment(&seg).expect("seal segment");
        assert!(receipt.readonly_set);
        assert_eq!(receipt.blake3_hex.len(), 64);
        assert!(verify_segment_seal(&seg).expect("verify seal"));

        // Opening for write must fail on a sealed segment.
        let write_attempt = fs::OpenOptions::new().write(true).open(&seg);
        assert!(
            write_attempt.is_err(),
            "sealed segment must reject write open"
        );

        unseal_segment_owner(&seg).expect("owner unseal");
        assert!(!verify_segment_seal(&seg).expect("verify unsealed"));
        let _ = fs::remove_dir_all(&dir);
    }
}
