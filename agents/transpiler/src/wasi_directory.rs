//! WASI Directory Operations
//!
//! Implements WASI specification for directory-related filesystem operations:
//! - fd_readdir: Read directory entries
//! - path_create_directory: Create a directory
//! - path_remove_directory: Remove a directory
//! - path_filestat_get: Get file/directory metadata
//! - fd_prestat_get/fd_prestat_dir_name: Get preopened directory info
//!
//! This module provides low-level WASI directory operations that work across:
//! - Native Rust (std::fs)
//! - WASM with WASI
//! - Browser WASM (virtual filesystem)

use std::path::{Path, PathBuf};
use anyhow::{Result, Context, anyhow};

/// WASI file type constants
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Unknown = 0,
    BlockDevice = 1,
    CharacterDevice = 2,
    Directory = 3,
    RegularFile = 4,
    SocketDgram = 5,
    SocketStream = 6,
    SymbolicLink = 7,
}

impl FileType {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => FileType::BlockDevice,
            2 => FileType::CharacterDevice,
            3 => FileType::Directory,
            4 => FileType::RegularFile,
            5 => FileType::SocketDgram,
            6 => FileType::SocketStream,
            7 => FileType::SymbolicLink,
            _ => FileType::Unknown,
        }
    }
}

/// Directory entry structure (matches WASI dirent)
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// Inode number (or placeholder)
    pub d_ino: u64,
    /// Next directory entry cookie
    pub d_next: u64,
    /// Length of the name
    pub d_namlen: u32,
    /// Type of file
    pub d_type: FileType,
    /// Name of the entry
    pub name: String,
}

/// File statistics (matches WASI filestat)
#[derive(Debug, Clone)]
pub struct FileStat {
    /// Device ID
    pub dev: u64,
    /// Inode number
    pub ino: u64,
    /// File type
    pub filetype: FileType,
    /// Number of hard links
    pub nlink: u64,
    /// File size in bytes
    pub size: u64,
    /// Last access time (nanoseconds since epoch)
    pub atim: u64,
    /// Last modification time (nanoseconds since epoch)
    pub mtim: u64,
    /// Last status change time (nanoseconds since epoch)
    pub ctim: u64,
}

/// Prestat type for preopened directories
#[derive(Debug, Clone)]
pub enum PrestatType {
    Dir { pr_name_len: u32 },
}

/// Prestat structure
#[derive(Debug, Clone)]
pub struct Prestat {
    pub tag: u8,
    pub u: PrestatType,
}

/// Directory handle with iteration state
pub struct WasiDir {
    path: PathBuf,
    #[cfg(not(target_arch = "wasm32"))]
    entries: Vec<DirEntry>,
    #[allow(dead_code)]
    cursor: u64,
}

impl WasiDir {
    /// Open a directory for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        #[cfg(not(target_arch = "wasm32"))]
        {
            if !path.exists() {
                return Err(anyhow!("Directory does not exist: {:?}", path));
            }

            if !path.is_dir() {
                return Err(anyhow!("Path is not a directory: {:?}", path));
            }

            Ok(WasiDir {
                path,
                entries: Vec::new(),
                cursor: 0,
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI directory opening will be handled by WASI runtime
            Ok(WasiDir {
                path,
                cursor: 0,
            })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            Ok(WasiDir {
                path,
                cursor: 0,
            })
        }
    }

    /// Read directory entries (WASI fd_readdir implementation)
    pub fn read_dir(&mut self, cookie: u64) -> Result<Vec<DirEntry>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::fs;

            // If this is the first read, populate entries
            if self.entries.is_empty() && cookie == 0 {
                let mut entries = Vec::new();
                let mut next_cookie = 1u64;

                let read_dir = fs::read_dir(&self.path)
                    .context("Failed to read directory")?;

                for entry in read_dir {
                    let entry = entry.context("Failed to read directory entry")?;
                    let metadata = entry.metadata()
                        .context("Failed to read entry metadata")?;

                    let file_type = if metadata.is_dir() {
                        FileType::Directory
                    } else if metadata.is_file() {
                        FileType::RegularFile
                    } else if metadata.is_symlink() {
                        FileType::SymbolicLink
                    } else {
                        FileType::Unknown
                    };

                    let name = entry.file_name()
                        .to_string_lossy()
                        .into_owned();

                    let dir_entry = DirEntry {
                        d_ino: 0, // Not available on all platforms
                        d_next: next_cookie,
                        d_namlen: name.len() as u32,
                        d_type: file_type,
                        name,
                    };

                    entries.push(dir_entry);
                    next_cookie += 1;
                }

                self.entries = entries;
            }

            // Return entries starting from cookie
            let start_idx = if cookie == 0 { 0 } else { cookie as usize };
            Ok(self.entries.get(start_idx..).unwrap_or(&[]).to_vec())
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI implementation would use fd_readdir syscall
            // This is a placeholder for actual WASI implementation
            Ok(Vec::new())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            // Would read from IndexedDB or virtual FS
            Ok(Vec::new())
        }
    }
}

/// WASI directory operations
pub struct WasiDirectory;

impl WasiDirectory {
    /// Create a directory (WASI path_create_directory)
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

            // WASI path_create_directory implementation
            // This would call the actual WASI syscall
            wasi::create_dir(path_str)
                .map_err(|e| anyhow!("WASI create_dir failed: {:?}", e))
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            // Would update IndexedDB or virtual FS structure
            Ok(())
        }
    }

    /// Create a directory and all parent directories (like mkdir -p)
    pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::create_dir_all(path.as_ref())
                .context("Failed to create directory recursively")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // Manually create each directory level for WASI
            let path = path.as_ref();
            let mut current = PathBuf::new();

            for component in path.components() {
                current.push(component);
                if !Self::exists(&current) {
                    Self::create_dir(&current)?;
                }
            }

            Ok(())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            Ok(())
        }
    }

    /// Remove an empty directory (WASI path_remove_directory)
    pub fn remove_dir<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::remove_dir(path.as_ref())
                .context("Failed to remove directory")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let path_str = path.as_ref().to_str()
                .context("Invalid path encoding")?;

            wasi::remove_dir(path_str)
                .map_err(|e| anyhow!("WASI remove_dir failed: {:?}", e))
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            Ok(())
        }
    }

    /// Remove a directory and all its contents
    pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::remove_dir_all(path.as_ref())
                .context("Failed to remove directory recursively")
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // Manually remove all contents for WASI
            let path = path.as_ref();
            let mut dir = WasiDir::open(path)?;
            let entries = dir.read_dir(0)?;

            for entry in entries {
                let entry_path = path.join(&entry.name);

                if entry.d_type == FileType::Directory {
                    Self::remove_dir_all(&entry_path)?;
                } else {
                    Self::remove_file(&entry_path)?;
                }
            }

            Self::remove_dir(path)
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            Ok(())
        }
    }

    /// Get file/directory metadata (WASI path_filestat_get)
    pub fn metadata<P: AsRef<Path>>(path: P) -> Result<FileStat> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::fs;
            use std::time::UNIX_EPOCH;

            let metadata = fs::metadata(path.as_ref())
                .context("Failed to get metadata")?;

            let filetype = if metadata.is_dir() {
                FileType::Directory
            } else if metadata.is_file() {
                FileType::RegularFile
            } else if metadata.file_type().is_symlink() {
                FileType::SymbolicLink
            } else {
                FileType::Unknown
            };

            let atim = metadata.accessed()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);

            let mtim = metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);

            let ctim = metadata.created()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(mtim);

            Ok(FileStat {
                dev: 0,
                ino: 0,
                filetype,
                nlink: 1,
                size: metadata.len(),
                atim,
                mtim,
                ctim,
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI path_filestat_get implementation
            // This would call the actual WASI syscall
            // Placeholder for now
            Ok(FileStat {
                dev: 0,
                ino: 0,
                filetype: FileType::Unknown,
                nlink: 1,
                size: 0,
                atim: 0,
                mtim: 0,
                ctim: 0,
            })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            Ok(FileStat {
                dev: 0,
                ino: 0,
                filetype: FileType::Unknown,
                nlink: 1,
                size: 0,
                atim: 0,
                mtim: 0,
                ctim: 0,
            })
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
            Self::metadata(path).is_ok()
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            // Browser virtual filesystem
            false
        }
    }

    /// Get preopened directory information (WASI fd_prestat_get)
    pub fn prestat_get(_fd: u32) -> Result<Prestat> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Not applicable for native code
            Err(anyhow!("Prestat only available in WASI environment"))
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI fd_prestat_get implementation
            // This would call the actual WASI syscall
            // For now, return a placeholder
            Ok(Prestat {
                tag: 0,
                u: PrestatType::Dir { pr_name_len: 0 },
            })
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            Err(anyhow!("Prestat not available in browser environment"))
        }
    }

    /// Get preopened directory name (WASI fd_prestat_dir_name)
    pub fn prestat_dir_name(_fd: u32) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(anyhow!("Prestat only available in WASI environment"))
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI fd_prestat_dir_name implementation
            // This would call the actual WASI syscall
            Ok(String::new())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            Err(anyhow!("Prestat not available in browser environment"))
        }
    }

    /// Read directory entries as a list
    pub fn read_dir<P: AsRef<Path>>(path: P) -> Result<Vec<DirEntry>> {
        let mut dir = WasiDir::open(path)?;
        dir.read_dir(0)
    }

    /// List directory contents (convenience wrapper)
    pub fn list_dir<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let entries = Self::read_dir(path)?;
        Ok(entries.into_iter().map(|e| e.name).collect())
    }

    // Helper methods for internal use

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_str = path.as_ref().to_str()
            .context("Invalid path encoding")?;
        wasi::remove_file(path_str)
            .map_err(|e| anyhow!("WASI remove_file failed: {:?}", e))
    }

    #[allow(dead_code)]
    #[cfg(not(all(target_arch = "wasm32", feature = "wasi")))]
    fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
        std::fs::remove_file(path.as_ref())
            .context("Failed to remove file")
    }
}

/// Iterator for directory entries
pub struct DirIterator {
    #[allow(dead_code)]
    dir: WasiDir,
    entries: Vec<DirEntry>,
    index: usize,
}

impl DirIterator {
    /// Create a new directory iterator
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut dir = WasiDir::open(path)?;
        let entries = dir.read_dir(0)?;

        Ok(DirIterator {
            dir,
            entries,
            index: 0,
        })
    }
}

impl Iterator for DirIterator {
    type Item = Result<DirEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entries.len() {
            let entry = self.entries[self.index].clone();
            self.index += 1;
            Some(Ok(entry))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_create_and_remove_directory() {
        let test_dir = "/tmp/portalis_wasi_test_create_dir";

        // Clean up first
        let _ = fs::remove_dir(test_dir);

        // Create directory
        WasiDirectory::create_dir(test_dir).expect("Failed to create directory");
        assert!(WasiDirectory::exists(test_dir));

        // Remove directory
        WasiDirectory::remove_dir(test_dir).expect("Failed to remove directory");
        assert!(!WasiDirectory::exists(test_dir));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_create_dir_all() {
        let test_dir = "/tmp/portalis_wasi_test/nested/deep/path";

        // Clean up first
        let _ = fs::remove_dir_all("/tmp/portalis_wasi_test");

        // Create nested directories
        WasiDirectory::create_dir_all(test_dir).expect("Failed to create nested directories");
        assert!(WasiDirectory::exists(test_dir));

        // Clean up
        WasiDirectory::remove_dir_all("/tmp/portalis_wasi_test")
            .expect("Failed to remove directory tree");
        assert!(!WasiDirectory::exists("/tmp/portalis_wasi_test"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_read_directory() {
        let test_dir = "/tmp/portalis_wasi_test_readdir";

        // Clean up first
        let _ = fs::remove_dir_all(test_dir);

        // Create directory with some files
        fs::create_dir(test_dir).expect("Failed to create test directory");
        fs::write(format!("{}/file1.txt", test_dir), "content1")
            .expect("Failed to write file1");
        fs::write(format!("{}/file2.txt", test_dir), "content2")
            .expect("Failed to write file2");
        fs::create_dir(format!("{}/subdir", test_dir))
            .expect("Failed to create subdirectory");

        // Read directory
        let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");

        // Should have 3 entries
        assert_eq!(entries.len(), 3);

        // Verify entries
        let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
        assert!(names.contains(&"file1.txt".to_string()));
        assert!(names.contains(&"file2.txt".to_string()));
        assert!(names.contains(&"subdir".to_string()));

        // Clean up
        fs::remove_dir_all(test_dir).expect("Failed to clean up");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_list_dir() {
        let test_dir = "/tmp/portalis_wasi_test_listdir";

        // Clean up first
        let _ = fs::remove_dir_all(test_dir);

        // Create directory with files
        fs::create_dir(test_dir).expect("Failed to create test directory");
        fs::write(format!("{}/a.txt", test_dir), "a").expect("Failed to write a.txt");
        fs::write(format!("{}/b.txt", test_dir), "b").expect("Failed to write b.txt");

        // List directory
        let mut names = WasiDirectory::list_dir(test_dir).expect("Failed to list directory");
        names.sort();

        assert_eq!(names, vec!["a.txt", "b.txt"]);

        // Clean up
        fs::remove_dir_all(test_dir).expect("Failed to clean up");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_metadata() {
        let test_file = "/tmp/portalis_wasi_test_metadata.txt";

        // Create test file
        fs::write(test_file, "test content").expect("Failed to write test file");

        // Get metadata
        let stat = WasiDirectory::metadata(test_file).expect("Failed to get metadata");

        assert_eq!(stat.filetype, FileType::RegularFile);
        assert_eq!(stat.size, 12); // "test content" is 12 bytes
        assert!(stat.mtim > 0);

        // Clean up
        fs::remove_file(test_file).expect("Failed to clean up");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_directory_metadata() {
        let test_dir = "/tmp/portalis_wasi_test_dir_metadata";

        // Clean up first
        let _ = fs::remove_dir(test_dir);

        // Create directory
        fs::create_dir(test_dir).expect("Failed to create directory");

        // Get metadata
        let stat = WasiDirectory::metadata(test_dir).expect("Failed to get metadata");

        assert_eq!(stat.filetype, FileType::Directory);

        // Clean up
        fs::remove_dir(test_dir).expect("Failed to clean up");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_dir_iterator() {
        let test_dir = "/tmp/portalis_wasi_test_iterator";

        // Clean up first
        let _ = fs::remove_dir_all(test_dir);

        // Create directory with files
        fs::create_dir(test_dir).expect("Failed to create test directory");
        fs::write(format!("{}/1.txt", test_dir), "1").expect("Failed to write 1.txt");
        fs::write(format!("{}/2.txt", test_dir), "2").expect("Failed to write 2.txt");
        fs::write(format!("{}/3.txt", test_dir), "3").expect("Failed to write 3.txt");

        // Iterate over entries
        let iter = DirIterator::new(test_dir).expect("Failed to create iterator");
        let entries: Result<Vec<DirEntry>> = iter.collect();
        let entries = entries.expect("Iterator failed");

        assert_eq!(entries.len(), 3);

        // Clean up
        fs::remove_dir_all(test_dir).expect("Failed to clean up");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_file_type_conversion() {
        assert_eq!(FileType::from_u8(3), FileType::Directory);
        assert_eq!(FileType::from_u8(4), FileType::RegularFile);
        assert_eq!(FileType::from_u8(7), FileType::SymbolicLink);
        assert_eq!(FileType::from_u8(99), FileType::Unknown);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_remove_dir_all_with_contents() {
        let test_dir = "/tmp/portalis_wasi_test_remove_all";

        // Clean up first
        let _ = fs::remove_dir_all(test_dir);

        // Create nested structure
        fs::create_dir_all(format!("{}/sub1/sub2", test_dir))
            .expect("Failed to create nested dirs");
        fs::write(format!("{}/file1.txt", test_dir), "1")
            .expect("Failed to write file1");
        fs::write(format!("{}/sub1/file2.txt", test_dir), "2")
            .expect("Failed to write file2");
        fs::write(format!("{}/sub1/sub2/file3.txt", test_dir), "3")
            .expect("Failed to write file3");

        // Remove all
        WasiDirectory::remove_dir_all(test_dir).expect("Failed to remove directory tree");
        assert!(!WasiDirectory::exists(test_dir));
    }
}
