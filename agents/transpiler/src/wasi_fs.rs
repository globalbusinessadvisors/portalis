//! WASI Filesystem Integration
//!
//! Provides a unified filesystem API that works across:
//! - Native Rust (std::fs)
//! - WASM with WASI (wasi crate)
//! - Browser WASM (virtual filesystem via IndexedDB)
//!
//! This module bridges Python's file operations to WASM-compatible implementations.

use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

/// Unified File handle that works across platforms
pub struct WasiFile {
    #[cfg(not(target_arch = "wasm32"))]
    inner: std::fs::File,

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    inner: wasi::File,

    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    inner: VirtualFile,
}

/// Virtual file for browser environment (no WASI)
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
struct VirtualFile {
    path: String,
    content: Vec<u8>,
    position: usize,
}

/// Filesystem operations abstraction
pub struct WasiFs;

impl WasiFs {
    /// Open a file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<WasiFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = std::fs::File::open(path.as_ref())
                .context("Failed to open file")?;
            Ok(WasiFile { inner: file })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI filesystem access
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;
            let file = wasi::File::open(path_str)
                .map_err(|e| anyhow::anyhow!("WASI file open failed: {:?}", e))?;
            Ok(WasiFile { inner: file })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem (would use IndexedDB in real impl)
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;

            // Placeholder: In real implementation, read from IndexedDB
            let content = vec![]; // Would fetch from browser storage

            Ok(WasiFile {
                inner: VirtualFile {
                    path: path_str.to_string(),
                    content,
                    position: 0,
                }
            })
        }
    }

    /// Create a new file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<WasiFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = std::fs::File::create(path.as_ref())
                .context("Failed to create file")?;
            Ok(WasiFile { inner: file })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;
            let file = wasi::File::create(path_str)
                .map_err(|e| anyhow::anyhow!("WASI file create failed: {:?}", e))?;
            Ok(WasiFile { inner: file })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;

            Ok(WasiFile {
                inner: VirtualFile {
                    path: path_str.to_string(),
                    content: Vec::new(),
                    position: 0,
                }
            })
        }
    }

    /// Read entire file to string
    pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::read_to_string(path.as_ref())
                .context("Failed to read file to string")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let mut file = Self::open(path)?;
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            Ok(content)
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: would read from IndexedDB
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;

            // Placeholder
            Ok(format!("// Virtual file: {}", path_str))
        }
    }

    /// Write string to file
    pub fn write<P: AsRef<Path>>(path: P, contents: &str) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::write(path.as_ref(), contents)
                .context("Failed to write file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let mut file = Self::create(path)?;
            file.write_all(contents.as_bytes())?;
            Ok(())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: would write to IndexedDB
            let _path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;

            // Placeholder: In real impl, store in IndexedDB
            Ok(())
        }
    }

    /// Check if path exists
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            path.as_ref().exists()
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI path check
            if let Some(path_str) = path.as_ref().to_str() {
                wasi::path_exists(path_str).unwrap_or(false)
            } else {
                false
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: check virtual filesystem
            // Placeholder: would query IndexedDB
            false
        }
    }

    /// Check if path is a file
    pub fn is_file<P: AsRef<Path>>(path: P) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            path.as_ref().is_file()
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            if let Some(path_str) = path.as_ref().to_str() {
                wasi::is_file(path_str).unwrap_or(false)
            } else {
                false
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem check
            false
        }
    }

    /// Check if path is a directory
    pub fn is_dir<P: AsRef<Path>>(path: P) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            path.as_ref().is_dir()
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            if let Some(path_str) = path.as_ref().to_str() {
                wasi::is_dir(path_str).unwrap_or(false)
            } else {
                false
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            false
        }
    }

    /// Create a directory
    pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::create_dir(path.as_ref())
                .context("Failed to create directory")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;
            wasi::create_dir(path_str)
                .map_err(|e| anyhow::anyhow!("WASI create_dir failed: {:?}", e))
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: create virtual directory
            Ok(())
        }
    }

    /// Remove a file
    pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::remove_file(path.as_ref())
                .context("Failed to remove file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;
            wasi::remove_file(path_str)
                .map_err(|e| anyhow::anyhow!("WASI remove_file failed: {:?}", e))
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: remove from virtual filesystem
            Ok(())
        }
    }
}

impl WasiFile {
    /// Read file contents to string
    pub fn read_to_string(&mut self, buf: &mut String) -> Result<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::io::Read;
            self.inner.read_to_string(buf)
                .context("Failed to read file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let bytes = self.inner.read_all()
                .map_err(|e| anyhow::anyhow!("WASI read failed: {:?}", e))?;
            let content = String::from_utf8(bytes)
                .context("Invalid UTF-8 in file")?;
            let len = content.len();
            buf.push_str(&content);
            Ok(len)
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let content = String::from_utf8(self.inner.content.clone())
                .context("Invalid UTF-8")?;
            let len = content.len();
            buf.push_str(&content);
            Ok(len)
        }
    }

    /// Write all bytes to file
    pub fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::io::Write;
            self.inner.write_all(buf)
                .context("Failed to write to file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            self.inner.write(buf)
                .map_err(|e| anyhow::anyhow!("WASI write failed: {:?}", e))?;
            Ok(())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            self.inner.content = buf.to_vec();
            // Would persist to IndexedDB here
            Ok(())
        }
    }
}

/// Path operations that work across platforms
pub struct WasiPath;

impl WasiPath {
    /// Create a new PathBuf
    pub fn new<P: AsRef<Path>>(path: P) -> PathBuf {
        path.as_ref().to_path_buf()
    }

    /// Join paths
    pub fn join<P: AsRef<Path>, Q: AsRef<Path>>(base: P, path: Q) -> PathBuf {
        base.as_ref().join(path.as_ref())
    }

    /// Get file name
    pub fn file_name<P: AsRef<Path>>(path: P) -> Option<String> {
        path.as_ref()
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    }

    /// Get parent directory
    pub fn parent<P: AsRef<Path>>(path: P) -> Option<PathBuf> {
        path.as_ref().parent().map(|p| p.to_path_buf())
    }

    /// Get file extension
    pub fn extension<P: AsRef<Path>>(path: P) -> Option<String> {
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_path_operations() {
        let path = WasiPath::new("/tmp/test.txt");
        assert_eq!(WasiPath::file_name(&path), Some("test.txt".to_string()));
        assert_eq!(WasiPath::extension(&path), Some("txt".to_string()));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_path_join() {
        let base = WasiPath::new("/tmp");
        let full = WasiPath::join(&base, "test.txt");
        assert!(full.to_str().unwrap().contains("test.txt"));
    }
}
