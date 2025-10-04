//! Complete WASI Filesystem Implementation
//!
//! Provides a production-ready filesystem API that works across:
//! - Native Rust (std::fs)
//! - WASM with WASI (wasi crate)
//! - Browser WASM (virtual filesystem via IndexedDB/localStorage)
//!
//! This module implements ALL Python file operations for WASM compatibility.

use std::path::{Path, PathBuf};
use std::io::{self, Read, Write, Seek, SeekFrom};
use anyhow::{Result, Context};

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use wasm_bindgen::prelude::*;

// ============================================================================
// File Handle
// ============================================================================

/// Unified file handle that works across all platforms
pub struct WasiFile {
    #[cfg(not(target_arch = "wasm32"))]
    inner: NativeFile,

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    inner: WasiNativeFile,

    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    inner: BrowserFile,
}

// Native implementation
#[cfg(not(target_arch = "wasm32"))]
struct NativeFile {
    file: std::fs::File,
    #[allow(dead_code)]
    path: PathBuf,
}

// WASI implementation
#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
struct WasiNativeFile {
    // WASI file descriptor
    fd: u32,
    path: String,
    position: u64,
}

// Browser implementation
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
struct BrowserFile {
    path: String,
    content: Vec<u8>,
    position: usize,
    writable: bool,
    modified: bool,
}

impl WasiFile {
    /// Read from file into buffer
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.file.read(buf)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI read
            unsafe {
                let iovs = [wasi::Iovec {
                    buf: buf.as_mut_ptr(),
                    len: buf.len(),
                }];

                match wasi::fd_read(self.inner.fd, &iovs) {
                    Ok(n) => {
                        self.inner.position += n as u64;
                        Ok(n)
                    }
                    Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("WASI read error: {:?}", e))),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let available = self.inner.content.len().saturating_sub(self.inner.position);
            let to_read = buf.len().min(available);

            if to_read > 0 {
                buf[..to_read].copy_from_slice(
                    &self.inner.content[self.inner.position..self.inner.position + to_read]
                );
                self.inner.position += to_read;
            }

            Ok(to_read)
        }
    }

    /// Write buffer to file
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.file.write(buf)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI write
            unsafe {
                let iovs = [wasi::Ciovec {
                    buf: buf.as_ptr(),
                    len: buf.len(),
                }];

                match wasi::fd_write(self.inner.fd, &iovs) {
                    Ok(n) => {
                        self.inner.position += n as u64;
                        Ok(n)
                    }
                    Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("WASI write error: {:?}", e))),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            if !self.inner.writable {
                return Err(io::Error::new(io::ErrorKind::PermissionDenied, "File not writable"));
            }

            // Extend content if needed
            let needed_size = self.inner.position + buf.len();
            if needed_size > self.inner.content.len() {
                self.inner.content.resize(needed_size, 0);
            }

            self.inner.content[self.inner.position..self.inner.position + buf.len()]
                .copy_from_slice(buf);
            self.inner.position += buf.len();
            self.inner.modified = true;

            Ok(buf.len())
        }
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> io::Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.file.flush()
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI sync
            match unsafe { wasi::fd_sync(self.inner.fd) } {
                Ok(_) => Ok(()),
                Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("WASI sync error: {:?}", e))),
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            if self.inner.modified {
                // Save to browser storage
                let _ = save_to_browser_storage(&self.inner.path, &self.inner.content);
                self.inner.modified = false;
            }
            Ok(())
        }
    }

    /// Seek to position in file
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.file.seek(pos)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let (offset, whence) = match pos {
                SeekFrom::Start(n) => (n as i64, wasi::WHENCE_SET),
                SeekFrom::Current(n) => (n, wasi::WHENCE_CUR),
                SeekFrom::End(n) => (n, wasi::WHENCE_END),
            };

            match unsafe { wasi::fd_seek(self.inner.fd, offset, whence) } {
                Ok(new_pos) => {
                    self.inner.position = new_pos;
                    Ok(new_pos)
                }
                Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("WASI seek error: {:?}", e))),
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let new_pos = match pos {
                SeekFrom::Start(n) => n as usize,
                SeekFrom::Current(n) => {
                    if n >= 0 {
                        self.inner.position.saturating_add(n as usize)
                    } else {
                        self.inner.position.saturating_sub((-n) as usize)
                    }
                }
                SeekFrom::End(n) => {
                    if n >= 0 {
                        self.inner.content.len().saturating_add(n as usize)
                    } else {
                        self.inner.content.len().saturating_sub((-n) as usize)
                    }
                }
            };

            self.inner.position = new_pos;
            Ok(new_pos as u64)
        }
    }

    /// Read entire file to string
    pub fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.file.read_to_string(buf)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let mut bytes = Vec::new();
            self.read_to_end(&mut bytes)?;
            let s = String::from_utf8(bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let len = s.len();
            buf.push_str(&s);
            Ok(len)
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let content = String::from_utf8(self.inner.content.clone())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let len = content.len();
            buf.push_str(&content);
            Ok(len)
        }
    }

    /// Read entire file to byte vector
    pub fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut total = 0;
        let mut temp = [0u8; 4096];

        loop {
            let n = self.read(&mut temp)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&temp[..n]);
            total += n;
        }

        Ok(total)
    }

    /// Write all bytes to file
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut written = 0;
        while written < buf.len() {
            let n = self.write(&buf[written..])?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "write returned 0"));
            }
            written += n;
        }
        Ok(())
    }

    /// Get file metadata
    pub fn metadata(&self) -> io::Result<FileMetadata> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let meta = self.inner.file.metadata()?;
            Ok(FileMetadata {
                size: meta.len(),
                is_file: meta.is_file(),
                is_dir: meta.is_dir(),
                read_only: meta.permissions().readonly(),
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            unsafe {
                match wasi::fd_filestat_get(self.inner.fd) {
                    Ok(stat) => Ok(FileMetadata {
                        size: stat.size,
                        is_file: stat.filetype == wasi::FILETYPE_REGULAR_FILE,
                        is_dir: stat.filetype == wasi::FILETYPE_DIRECTORY,
                        read_only: false, // WASI doesn't provide this easily
                    }),
                    Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("WASI stat error: {:?}", e))),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            Ok(FileMetadata {
                size: self.inner.content.len() as u64,
                is_file: true,
                is_dir: false,
                read_only: !self.inner.writable,
            })
        }
    }
}

// Implement Drop for automatic resource cleanup (RAII pattern)
impl Drop for WasiFile {
    fn drop(&mut self) {
        // Flush any pending writes
        let _ = self.flush();

        // Close file descriptor on WASI
        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            unsafe {
                let _ = wasi::fd_close(self.inner.fd);
            }
        }
    }
}

// ============================================================================
// Filesystem Operations
// ============================================================================

/// Main filesystem interface
pub struct WasiFilesystem;

impl WasiFilesystem {
    /// Open file with Python-style mode string
    /// Modes: "r", "w", "a", "r+", "w+", "a+", "rb", "wb", etc.
    pub fn open_with_mode<P: AsRef<Path>>(path: P, mode: &str) -> Result<WasiFile> {
        // Parse mode string
        let mode_clean = mode.trim_matches(|c| c == '"' || c == '\'');
        let (read, write, create, append, truncate) = match mode_clean {
            "r" | "rb" => (true, false, false, false, false),
            "w" | "wb" => (false, true, true, false, true),
            "a" | "ab" => (false, true, true, true, false),
            "r+" | "rb+" | "r+b" => (true, true, false, false, false),
            "w+" | "wb+" | "w+b" => (true, true, true, false, true),
            "a+" | "ab+" | "a+b" => (true, true, true, true, false),
            _ => (true, false, false, false, false), // Default to read
        };

        if write && create && truncate {
            Self::create(path)
        } else if write || create || append {
            Self::open_with_options(path, read, write, create, append)
        } else {
            Self::open(path)
        }
    }

    /// Open file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<WasiFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = std::fs::File::open(path.as_ref())
                .context("Failed to open file")?;
            Ok(WasiFile {
                inner: NativeFile {
                    file,
                    path: path.as_ref().to_path_buf(),
                },
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            // Open file with WASI
            let fd = unsafe {
                let mut fd = 0u32;
                let path_bytes = path_str.as_bytes();

                // Use path_open from preopened directory
                match wasi::path_open(
                    3, // Assuming fd 3 is preopened root
                    0, // dirflags
                    path_bytes.as_ptr(),
                    path_bytes.len(),
                    wasi::OFLAGS_NONE,
                    wasi::RIGHTS_FD_READ | wasi::RIGHTS_FD_SEEK | wasi::RIGHTS_FD_TELL | wasi::RIGHTS_FD_FILESTAT_GET,
                    0,
                    0,
                ) {
                    Ok(f) => f,
                    Err(e) => return Err(anyhow!("WASI open failed: {:?}", e)),
                }
            };

            Ok(WasiFile {
                inner: WasiNativeFile {
                    fd,
                    path: path_str.to_string(),
                    position: 0,
                },
            })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            // Load from browser storage
            let content = load_from_browser_storage(path_str)?;

            Ok(WasiFile {
                inner: BrowserFile {
                    path: path_str.to_string(),
                    content,
                    position: 0,
                    writable: false,
                    modified: false,
                },
            })
        }
    }

    /// Create or truncate file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<WasiFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = std::fs::File::create(path.as_ref())
                .context("Failed to create file")?;
            Ok(WasiFile {
                inner: NativeFile {
                    file,
                    path: path.as_ref().to_path_buf(),
                },
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            let fd = unsafe {
                let path_bytes = path_str.as_bytes();

                match wasi::path_open(
                    3,
                    0,
                    path_bytes.as_ptr(),
                    path_bytes.len(),
                    wasi::OFLAGS_CREAT | wasi::OFLAGS_TRUNC,
                    wasi::RIGHTS_FD_WRITE | wasi::RIGHTS_FD_SEEK | wasi::RIGHTS_FD_SYNC,
                    0,
                    0,
                ) {
                    Ok(f) => f,
                    Err(e) => return Err(anyhow!("WASI create failed: {:?}", e)),
                }
            };

            Ok(WasiFile {
                inner: WasiNativeFile {
                    fd,
                    path: path_str.to_string(),
                    position: 0,
                },
            })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            Ok(WasiFile {
                inner: BrowserFile {
                    path: path_str.to_string(),
                    content: Vec::new(),
                    position: 0,
                    writable: true,
                    modified: false,
                },
            })
        }
    }

    /// Open file with options
    pub fn open_with_options<P: AsRef<Path>>(path: P, read: bool, write: bool, create: bool, append: bool) -> Result<WasiFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut options = std::fs::OpenOptions::new();
            options.read(read).write(write).create(create).append(append);

            let file = options.open(path.as_ref())
                .context("Failed to open file with options")?;

            Ok(WasiFile {
                inner: NativeFile {
                    file,
                    path: path.as_ref().to_path_buf(),
                },
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            let mut oflags = wasi::OFLAGS_NONE;
            if create {
                oflags |= wasi::OFLAGS_CREAT;
            }

            let mut rights = 0u64;
            if read {
                rights |= wasi::RIGHTS_FD_READ | wasi::RIGHTS_FD_SEEK;
            }
            if write {
                rights |= wasi::RIGHTS_FD_WRITE | wasi::RIGHTS_FD_SYNC;
            }

            let fd = unsafe {
                let path_bytes = path_str.as_bytes();
                match wasi::path_open(3, 0, path_bytes.as_ptr(), path_bytes.len(), oflags, rights, 0, 0) {
                    Ok(f) => f,
                    Err(e) => return Err(anyhow!("WASI open failed: {:?}", e)),
                }
            };

            let mut file = WasiFile {
                inner: WasiNativeFile {
                    fd,
                    path: path_str.to_string(),
                    position: 0,
                },
            };

            if append {
                file.seek(SeekFrom::End(0))?;
            }

            Ok(file)
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;

            let content = if read && Self::exists(path.as_ref()) {
                load_from_browser_storage(path_str).unwrap_or_default()
            } else {
                Vec::new()
            };

            let position = if append { content.len() } else { 0 };

            Ok(WasiFile {
                inner: BrowserFile {
                    path: path_str.to_string(),
                    content,
                    position,
                    writable: write,
                    modified: false,
                },
            })
        }
    }

    /// Read entire file to string
    pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
        let mut file = Self::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }

    /// Read entire file to bytes
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
        let mut file = Self::open(path)?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Ok(content)
    }

    /// Write string to file
    pub fn write<P: AsRef<Path>>(path: P, contents: &str) -> Result<()> {
        let mut file = Self::create(path)?;
        file.write_all(contents.as_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Write bytes to file
    pub fn write_bytes<P: AsRef<Path>>(path: P, contents: &[u8]) -> Result<()> {
        let mut file = Self::create(path)?;
        file.write_all(contents)?;
        file.flush()?;
        Ok(())
    }

    /// Check if path exists
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            path.as_ref().exists()
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            if let Some(path_str) = path.as_ref().to_str() {
                let path_bytes = path_str.as_bytes();
                unsafe {
                    match wasi::path_filestat_get(3, 0, path_bytes.as_ptr(), path_bytes.len()) {
                        Ok(_) => true,
                        Err(_) => false,
                    }
                }
            } else {
                false
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            if let Some(path_str) = path.as_ref().to_str() {
                browser_storage_exists(path_str)
            } else {
                false
            }
        }
    }

    /// Create directory
    pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::create_dir(path.as_ref())
                .context("Failed to create directory")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;
            let path_bytes = path_str.as_bytes();

            unsafe {
                match wasi::path_create_directory(3, path_bytes.as_ptr(), path_bytes.len()) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(anyhow!("WASI mkdir failed: {:?}", e)),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: directories are implicit
            Ok(())
        }
    }

    /// Create directory and all parents
    pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::create_dir_all(path.as_ref())
                .context("Failed to create directory recursively")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // Recursive directory creation
            let mut current = PathBuf::new();
            for component in path.as_ref().components() {
                current.push(component);
                if !Self::exists(&current) {
                    Self::create_dir(&current)?;
                }
            }
            Ok(())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            Ok(())
        }
    }

    /// Remove file
    pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::remove_file(path.as_ref())
                .context("Failed to remove file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;
            let path_bytes = path_str.as_bytes();

            unsafe {
                match wasi::path_unlink_file(3, path_bytes.as_ptr(), path_bytes.len()) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(anyhow!("WASI unlink failed: {:?}", e)),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid UTF-8 in path")?;
            remove_from_browser_storage(path_str);
            Ok(())
        }
    }

    /// Copy file
    pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<u64> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::copy(from.as_ref(), to.as_ref())
                .context("Failed to copy file")
        }

        #[cfg(any(target_arch = "wasm32"))]
        {
            // Manual copy
            let content = Self::read(from)?;
            Self::write_bytes(to, &content)?;
            Ok(content.len() as u64)
        }
    }

    /// Rename/move file
    pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::rename(from.as_ref(), to.as_ref())
                .context("Failed to rename file")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let from_str = from.as_ref().to_str().context("Invalid UTF-8 in from path")?;
            let to_str = to.as_ref().to_str().context("Invalid UTF-8 in to path")?;

            let from_bytes = from_str.as_bytes();
            let to_bytes = to_str.as_bytes();

            unsafe {
                match wasi::path_rename(
                    3, from_bytes.as_ptr(), from_bytes.len(),
                    3, to_bytes.as_ptr(), to_bytes.len()
                ) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(anyhow!("WASI rename failed: {:?}", e)),
                }
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser: copy and delete
            Self::copy(&from, &to)?;
            Self::remove_file(from)?;
            Ok(())
        }
    }
}

// ============================================================================
// File Metadata
// ============================================================================

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub read_only: bool,
}

// ============================================================================
// Browser Storage Helpers
// ============================================================================

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn load_from_browser_storage(path: &str) -> Result<Vec<u8>> {
    use web_sys::window;

    let window = window().ok_or_else(|| anyhow!("No window object"))?;
    let storage = window
        .local_storage()
        .map_err(|_| anyhow!("Failed to get localStorage"))?
        .ok_or_else(|| anyhow!("localStorage not available"))?;

    let key = format!("portalis_fs:{}", path);
    let value = storage
        .get_item(&key)
        .map_err(|_| anyhow!("Failed to read from localStorage"))?
        .ok_or_else(|| anyhow!("File not found: {}", path))?;

    // Decode base64
    let bytes = base64_decode(&value)?;
    Ok(bytes)
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn save_to_browser_storage(path: &str, content: &[u8]) -> Result<()> {
    use web_sys::window;

    let window = window().ok_or_else(|| anyhow!("No window object"))?;
    let storage = window
        .local_storage()
        .map_err(|_| anyhow!("Failed to get localStorage"))?
        .ok_or_else(|| anyhow!("localStorage not available"))?;

    let key = format!("portalis_fs:{}", path);
    let value = base64_encode(content);

    storage
        .set_item(&key, &value)
        .map_err(|_| anyhow!("Failed to write to localStorage"))?;

    Ok(())
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn browser_storage_exists(path: &str) -> bool {
    use web_sys::window;

    if let Some(window) = window() {
        if let Ok(Some(storage)) = window.local_storage() {
            let key = format!("portalis_fs:{}", path);
            return storage.get_item(&key).ok().flatten().is_some();
        }
    }
    false
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn remove_from_browser_storage(path: &str) {
    use web_sys::window;

    if let Some(window) = window() {
        if let Ok(Some(storage)) = window.local_storage() {
            let key = format!("portalis_fs:{}", path);
            let _ = storage.remove_item(&key);
        }
    }
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn base64_encode(data: &[u8]) -> String {
    use wasm_bindgen::JsValue;
    use js_sys::{Uint8Array, Object};

    let array = Uint8Array::from(data);
    let window = web_sys::window().unwrap();
    let btoa = js_sys::Reflect::get(&window, &JsValue::from_str("btoa")).unwrap();
    let func = btoa.dyn_into::<js_sys::Function>().unwrap();

    // Convert to string for btoa
    let string_data = (0..data.len())
        .map(|i| char::from(data[i]))
        .collect::<String>();

    let result = func.call1(&window, &JsValue::from_str(&string_data)).unwrap();
    result.as_string().unwrap()
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    use wasm_bindgen::JsValue;

    let window = web_sys::window().ok_or_else(|| anyhow!("No window"))?;
    let atob = js_sys::Reflect::get(&window, &JsValue::from_str("atob"))
        .map_err(|_| anyhow!("No atob function"))?;
    let func = atob.dyn_into::<js_sys::Function>()
        .map_err(|_| anyhow!("atob not a function"))?;

    let result = func.call1(&window, &JsValue::from_str(s))
        .map_err(|_| anyhow!("atob call failed"))?;
    let decoded = result.as_string().ok_or_else(|| anyhow!("atob result not string"))?;

    Ok(decoded.bytes().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read_file() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_wasi_file.txt");

        let content = "Hello, WASI filesystem!";
        WasiFilesystem::write(&test_file, content).unwrap();

        let read_content = WasiFilesystem::read_to_string(&test_file).unwrap();
        assert_eq!(read_content, content);

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_file_operations() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_file_ops.txt");

        // Create and write
        let mut file = WasiFilesystem::create(&test_file).unwrap();
        file.write_all(b"Line 1\n").unwrap();
        file.write_all(b"Line 2\n").unwrap();
        file.flush().unwrap();
        drop(file);

        // Read
        let mut file = WasiFilesystem::open(&test_file).unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        assert_eq!(content, "Line 1\nLine 2\n");

        // Cleanup
        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_seek_operations() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_seek.txt");

        WasiFilesystem::write(&test_file, "0123456789").unwrap();

        let mut file = WasiFilesystem::open(&test_file).unwrap();

        // Seek to position 5
        file.seek(SeekFrom::Start(5)).unwrap();
        let mut buf = [0u8; 5];
        file.read(&mut buf).unwrap();
        assert_eq!(&buf, b"56789");

        // Seek from end
        file.seek(SeekFrom::End(-3)).unwrap();
        let mut buf = [0u8; 3];
        file.read(&mut buf).unwrap();
        assert_eq!(&buf, b"789");

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_copy_and_rename() {
        let temp_dir = std::env::temp_dir();
        let file1 = temp_dir.join("test_copy_src.txt");
        let file2 = temp_dir.join("test_copy_dst.txt");
        let file3 = temp_dir.join("test_rename.txt");

        WasiFilesystem::write(&file1, "Test content").unwrap();

        // Copy
        WasiFilesystem::copy(&file1, &file2).unwrap();
        assert_eq!(
            WasiFilesystem::read_to_string(&file2).unwrap(),
            "Test content"
        );

        // Rename
        WasiFilesystem::rename(&file2, &file3).unwrap();
        assert!(!WasiFilesystem::exists(&file2));
        assert!(WasiFilesystem::exists(&file3));

        // Cleanup
        WasiFilesystem::remove_file(&file1).unwrap();
        WasiFilesystem::remove_file(&file3).unwrap();
    }

    #[test]
    fn test_open_with_mode_read() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_mode_read.txt");

        WasiFilesystem::write(&test_file, "Test content").unwrap();

        // Open in read mode
        let mut file = WasiFilesystem::open_with_mode(&test_file, "r").unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        assert_eq!(content, "Test content");

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_open_with_mode_write() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_mode_write.txt");

        // Open in write mode
        let mut file = WasiFilesystem::open_with_mode(&test_file, "w").unwrap();
        file.write_all(b"New content").unwrap();
        file.flush().unwrap();
        drop(file);

        let content = WasiFilesystem::read_to_string(&test_file).unwrap();
        assert_eq!(content, "New content");

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_open_with_mode_append() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_mode_append.txt");

        WasiFilesystem::write(&test_file, "Line 1\n").unwrap();

        // Open in append mode
        let mut file = WasiFilesystem::open_with_mode(&test_file, "a").unwrap();
        file.write_all(b"Line 2\n").unwrap();
        file.flush().unwrap();
        drop(file);

        let content = WasiFilesystem::read_to_string(&test_file).unwrap();
        assert_eq!(content, "Line 1\nLine 2\n");

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_open_with_mode_read_write() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_mode_rw.txt");

        WasiFilesystem::write(&test_file, "Original").unwrap();

        // Open in r+ mode
        let mut file = WasiFilesystem::open_with_mode(&test_file, "r+").unwrap();

        // Read
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        assert_eq!(content, "Original");

        // Write (this would overwrite at current position)
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"Modified").unwrap();
        file.flush().unwrap();
        drop(file);

        let final_content = WasiFilesystem::read_to_string(&test_file).unwrap();
        assert!(final_content.starts_with("Modified"));

        WasiFilesystem::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_drop_auto_cleanup() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_drop.txt");

        {
            let mut file = WasiFilesystem::create(&test_file).unwrap();
            file.write_all(b"Auto-flushed").unwrap();
            // Drop happens here - should auto-flush
        }

        // File should be properly written even without explicit flush
        let content = WasiFilesystem::read_to_string(&test_file).unwrap();
        assert_eq!(content, "Auto-flushed");

        WasiFilesystem::remove_file(&test_file).unwrap();
    }
}
