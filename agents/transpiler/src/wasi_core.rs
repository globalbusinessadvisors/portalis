//! Core WASI Filesystem Implementation
//!
//! This module implements the core WASI (WebAssembly System Interface) filesystem
//! functionality following the WASI snapshot_preview1 specification.
//!
//! # Architecture
//!
//! The WASI filesystem is capability-based and sandboxed:
//! - File descriptors are the primary interface
//! - Preopen directories define accessible paths
//! - Path resolution enforces sandbox boundaries
//! - All operations return WASI errno codes
//!
//! # Key Components
//!
//! 1. **File Descriptor Table**: Manages open file handles
//! 2. **Preopen System**: Defines accessible directory capabilities
//! 3. **Path Resolution**: Resolves and validates paths within sandbox
//! 4. **File Operations**: Implements WASI syscalls (fd_read, fd_write, etc.)
//! 5. **Error Mapping**: Maps OS errors to WASI errno values

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// WASI errno codes following snapshot_preview1 specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum WasiErrno {
    /// No error occurred
    Success = 0,
    /// Argument list too long
    TooBig = 1,
    /// Permission denied
    Access = 2,
    /// Address in use
    AddrInUse = 3,
    /// Address not available
    AddrNotAvail = 4,
    /// Address family not supported
    AFNoSupport = 5,
    /// Resource unavailable, try again
    Again = 6,
    /// Connection already in progress
    Already = 7,
    /// Bad file descriptor
    BadF = 8,
    /// Bad message
    BadMsg = 9,
    /// Device or resource busy
    Busy = 10,
    /// Operation canceled
    Canceled = 11,
    /// No child processes
    Child = 12,
    /// Connection aborted
    ConnAborted = 13,
    /// Connection refused
    ConnRefused = 14,
    /// Connection reset
    ConnReset = 15,
    /// Resource deadlock would occur
    Deadlk = 16,
    /// Destination address required
    DestAddrReq = 17,
    /// Mathematics argument out of domain
    Dom = 18,
    /// Reserved
    DQuot = 19,
    /// File exists
    Exist = 20,
    /// Bad address
    Fault = 21,
    /// File too large
    FBig = 22,
    /// Host is unreachable
    HostUnreach = 23,
    /// Identifier removed
    Idrm = 24,
    /// Illegal byte sequence
    IlSeq = 25,
    /// Operation in progress
    InProgress = 26,
    /// Interrupted function
    Intr = 27,
    /// Invalid argument
    Inval = 28,
    /// I/O error
    Io = 29,
    /// Socket is connected
    IsConn = 30,
    /// Is a directory
    IsDir = 31,
    /// Too many levels of symbolic links
    Loop = 32,
    /// File descriptor value too large
    MFile = 33,
    /// Too many links
    MLink = 34,
    /// Message too large
    MsgSize = 35,
    /// Reserved
    Multihop = 36,
    /// Filename too long
    NameTooLong = 37,
    /// Network is down
    NetDown = 38,
    /// Connection aborted by network
    NetReset = 39,
    /// Network unreachable
    NetUnreach = 40,
    /// Too many files open in system
    NFile = 41,
    /// No buffer space available
    NoBufS = 42,
    /// No such device
    NoDev = 43,
    /// No such file or directory
    NoEnt = 44,
    /// Executable file format error
    NoExec = 45,
    /// No locks available
    NoLck = 46,
    /// Reserved
    NoLink = 47,
    /// Not enough space
    NoMem = 48,
    /// No message of the desired type
    NoMsg = 49,
    /// Protocol not available
    NoProtoOpt = 50,
    /// No space left on device
    NoSpc = 51,
    /// Function not supported
    NoSys = 52,
    /// The socket is not connected
    NotConn = 53,
    /// Not a directory or a symbolic link to a directory
    NotDir = 54,
    /// Directory not empty
    NotEmpty = 55,
    /// State not recoverable
    NotRecoverable = 56,
    /// Not a socket
    NotSock = 57,
    /// Not supported, or operation not supported on socket
    NotSup = 58,
    /// Inappropriate I/O control operation
    NoTty = 59,
    /// No such device or address
    NxIo = 60,
    /// Value too large to be stored in data type
    Overflow = 61,
    /// Previous owner died
    OwnerDead = 62,
    /// Operation not permitted
    Perm = 63,
    /// Broken pipe
    Pipe = 64,
    /// Protocol error
    Proto = 65,
    /// Protocol not supported
    ProtoNoSupport = 66,
    /// Protocol wrong type for socket
    Prototype = 67,
    /// Result too large
    Range = 68,
    /// Read-only file system
    RoFs = 69,
    /// Invalid seek
    SPipe = 70,
    /// No such process
    Srch = 71,
    /// Reserved
    Stale = 72,
    /// Connection timed out
    TimedOut = 73,
    /// Text file busy
    TxtBsy = 74,
    /// Cross-device link
    XDev = 75,
    /// Extension: Capabilities insufficient
    NotCapable = 76,
}

impl WasiErrno {
    /// Convert WasiErrno to u16 for WASM ABI
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Convert io::Error to WasiErrno
    pub fn from_io_error(error: &io::Error) -> Self {
        use io::ErrorKind;
        match error.kind() {
            ErrorKind::NotFound => WasiErrno::NoEnt,
            ErrorKind::PermissionDenied => WasiErrno::Access,
            ErrorKind::ConnectionRefused => WasiErrno::ConnRefused,
            ErrorKind::ConnectionReset => WasiErrno::ConnReset,
            ErrorKind::ConnectionAborted => WasiErrno::ConnAborted,
            ErrorKind::NotConnected => WasiErrno::NotConn,
            ErrorKind::AddrInUse => WasiErrno::AddrInUse,
            ErrorKind::AddrNotAvailable => WasiErrno::AddrNotAvail,
            ErrorKind::BrokenPipe => WasiErrno::Pipe,
            ErrorKind::AlreadyExists => WasiErrno::Exist,
            ErrorKind::WouldBlock => WasiErrno::Again,
            ErrorKind::InvalidInput => WasiErrno::Inval,
            ErrorKind::InvalidData => WasiErrno::Io,
            ErrorKind::TimedOut => WasiErrno::TimedOut,
            ErrorKind::WriteZero => WasiErrno::Io,
            ErrorKind::Interrupted => WasiErrno::Intr,
            ErrorKind::UnexpectedEof => WasiErrno::Io,
            _ => WasiErrno::Io,
        }
    }
}

/// File descriptor type
pub type Fd = u32;

/// Standard file descriptors
pub const STDIN_FD: Fd = 0;
pub const STDOUT_FD: Fd = 1;
pub const STDERR_FD: Fd = 2;

/// Starting FD for preopen directories
pub const PREOPEN_START_FD: Fd = 3;

/// Open flags for path_open
#[derive(Debug, Clone, Copy)]
pub struct OpenFlags {
    pub create: bool,
    pub directory: bool,
    pub excl: bool,
    pub trunc: bool,
}

impl Default for OpenFlags {
    fn default() -> Self {
        Self {
            create: false,
            directory: false,
            excl: false,
            trunc: false,
        }
    }
}

/// File descriptor rights
#[derive(Debug, Clone, Copy)]
pub struct Rights {
    pub fd_read: bool,
    pub fd_write: bool,
    pub fd_seek: bool,
    pub fd_tell: bool,
    pub fd_sync: bool,
    pub path_open: bool,
    pub path_create_file: bool,
    pub path_create_directory: bool,
    pub path_unlink_file: bool,
}

impl Rights {
    /// Full rights (for preopens)
    pub fn all() -> Self {
        Self {
            fd_read: true,
            fd_write: true,
            fd_seek: true,
            fd_tell: true,
            fd_sync: true,
            path_open: true,
            path_create_file: true,
            path_create_directory: true,
            path_unlink_file: true,
        }
    }

    /// Read-only rights
    pub fn read_only() -> Self {
        Self {
            fd_read: true,
            fd_write: false,
            fd_seek: true,
            fd_tell: true,
            fd_sync: false,
            path_open: true,
            path_create_file: false,
            path_create_directory: false,
            path_unlink_file: false,
        }
    }

    /// Write-only rights
    pub fn write_only() -> Self {
        Self {
            fd_read: false,
            fd_write: true,
            fd_seek: true,
            fd_tell: true,
            fd_sync: true,
            path_open: true,
            path_create_file: true,
            path_create_directory: true,
            path_unlink_file: true,
        }
    }

    /// Read-write rights
    pub fn read_write() -> Self {
        Self {
            fd_read: true,
            fd_write: true,
            fd_seek: true,
            fd_tell: true,
            fd_sync: true,
            path_open: true,
            path_create_file: true,
            path_create_directory: true,
            path_unlink_file: true,
        }
    }
}

/// File descriptor entry
#[derive(Debug)]
pub struct FdEntry {
    /// The underlying file handle (None for directories and special FDs)
    pub file: Option<File>,
    /// Path this FD refers to (for directories and preopens)
    pub path: PathBuf,
    /// Whether this is a preopen directory
    pub is_preopen: bool,
    /// Access rights for this FD
    pub rights: Rights,
    /// Inheriting rights (for path_open from this FD)
    pub rights_inheriting: Rights,
}

/// File Descriptor Table
///
/// Manages all open file descriptors for the WASI instance.
/// Thread-safe via RwLock for concurrent access.
#[derive(Debug)]
pub struct FdTable {
    /// Map of FD -> FdEntry
    entries: Arc<RwLock<HashMap<Fd, FdEntry>>>,
    /// Next available FD
    next_fd: Arc<RwLock<Fd>>,
}

impl FdTable {
    /// Create a new file descriptor table
    pub fn new() -> Self {
        let table = Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            next_fd: Arc::new(RwLock::new(PREOPEN_START_FD)),
        };

        // Note: stdin/stdout/stderr are not initialized here
        // They should be provided by the WASM runtime

        table
    }

    /// Insert a preopen directory
    ///
    /// Preopens are the root capabilities that define what paths
    /// the WASM module can access.
    pub fn insert_preopen(&self, path: PathBuf) -> Result<Fd, WasiErrno> {
        // Validate that path exists and is a directory
        if !path.exists() {
            return Err(WasiErrno::NoEnt);
        }
        if !path.is_dir() {
            return Err(WasiErrno::NotDir);
        }

        let mut next_fd = self.next_fd.write().unwrap();
        let fd = *next_fd;
        *next_fd += 1;

        let entry = FdEntry {
            file: None,
            path: path.canonicalize().map_err(|_| WasiErrno::Io)?,
            is_preopen: true,
            rights: Rights::all(),
            rights_inheriting: Rights::all(),
        };

        self.entries.write().unwrap().insert(fd, entry);
        Ok(fd)
    }

    /// Insert a regular file descriptor
    pub fn insert_file(
        &self,
        file: File,
        path: PathBuf,
        rights: Rights,
    ) -> Result<Fd, WasiErrno> {
        let mut next_fd = self.next_fd.write().unwrap();
        let fd = *next_fd;
        *next_fd += 1;

        let entry = FdEntry {
            file: Some(file),
            path,
            is_preopen: false,
            rights,
            rights_inheriting: rights,
        };

        self.entries.write().unwrap().insert(fd, entry);
        Ok(fd)
    }

    /// Get an entry by FD (for reading)
    pub fn get(&self, fd: Fd) -> Result<FdEntry, WasiErrno> {
        self.entries
            .read()
            .unwrap()
            .get(&fd)
            .ok_or(WasiErrno::BadF)
            .map(|entry| {
                // Clone the entry for safe access
                // Note: File can't be cloned, so we return a minimal copy
                FdEntry {
                    file: None, // Don't clone file handle
                    path: entry.path.clone(),
                    is_preopen: entry.is_preopen,
                    rights: entry.rights,
                    rights_inheriting: entry.rights_inheriting,
                }
            })
    }

    /// Get mutable access to an entry
    pub fn get_mut<F, R>(&self, fd: Fd, f: F) -> Result<R, WasiErrno>
    where
        F: FnOnce(&mut FdEntry) -> Result<R, WasiErrno>,
    {
        let mut entries = self.entries.write().unwrap();
        let entry = entries.get_mut(&fd).ok_or(WasiErrno::BadF)?;
        f(entry)
    }

    /// Remove and close a file descriptor
    pub fn remove(&self, fd: Fd) -> Result<(), WasiErrno> {
        // Don't allow closing stdin/stdout/stderr
        if fd < PREOPEN_START_FD {
            return Err(WasiErrno::Inval);
        }

        self.entries
            .write()
            .unwrap()
            .remove(&fd)
            .ok_or(WasiErrno::BadF)
            .map(|_| ())
    }

    /// Check if FD has a specific right
    pub fn check_rights(&self, fd: Fd, check: impl Fn(&Rights) -> bool) -> Result<(), WasiErrno> {
        let entry = self.get(fd)?;
        if check(&entry.rights) {
            Ok(())
        } else {
            Err(WasiErrno::NotCapable)
        }
    }
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Path resolution with sandboxing
///
/// Resolves paths relative to a preopen directory and ensures
/// they don't escape the sandbox.
pub struct PathResolver;

impl PathResolver {
    /// Resolve a path relative to a base directory
    ///
    /// Ensures the resolved path is within the sandbox (doesn't escape base).
    pub fn resolve(base: &Path, path: &Path) -> Result<PathBuf, WasiErrno> {
        // Join the paths
        let joined = base.join(path);

        // Try to canonicalize - if it fails because file doesn't exist,
        // we need to verify parent directory instead
        match joined.canonicalize() {
            Ok(canonical) => {
                // Ensure the canonical path is within base
                if !canonical.starts_with(base) {
                    return Err(WasiErrno::NotCapable);
                }
                Ok(canonical)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // File doesn't exist yet - this is OK for creation
                // Verify the parent directory is within sandbox
                if let Some(parent) = joined.parent() {
                    // Try to canonicalize the parent - if that also doesn't exist,
                    // we need to recursively check up the tree
                    let canonical_parent = match parent.canonicalize() {
                        Ok(p) => p,
                        Err(_) => {
                            // Parent doesn't exist either, check if base is the parent
                            if parent == base {
                                base.to_path_buf()
                            } else {
                                // Recursively resolve parent
                                return Self::resolve(base, path.parent().ok_or(WasiErrno::Inval)?)
                                    .and_then(|parent_resolved| {
                                        Ok(parent_resolved.join(
                                            path.file_name().ok_or(WasiErrno::Inval)?
                                        ))
                                    });
                            }
                        }
                    };

                    if !canonical_parent.starts_with(base) {
                        return Err(WasiErrno::NotCapable);
                    }
                    // Return the joined path (not canonical since it doesn't exist)
                    Ok(canonical_parent.join(
                        joined.file_name().ok_or(WasiErrno::Inval)?
                    ))
                } else {
                    Err(WasiErrno::Inval)
                }
            }
            Err(_) => Err(WasiErrno::Io),
        }
    }

    /// Find the appropriate preopen for a given path
    ///
    /// Returns the FD and the relative path from that preopen.
    pub fn find_preopen<'a>(
        table: &FdTable,
        path: &'a Path,
    ) -> Result<(Fd, &'a Path), WasiErrno> {
        let entries = table.entries.read().unwrap();

        // Find all preopens
        let mut preopens: Vec<_> = entries
            .iter()
            .filter(|(_, entry)| entry.is_preopen)
            .collect();

        // Sort by path length (longest first) to find most specific match
        preopens.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.path.as_os_str().len()));

        // Find the first preopen that contains this path
        for (fd, entry) in preopens {
            if let Ok(rel_path) = path.strip_prefix(&entry.path) {
                return Ok((*fd, rel_path));
            }
        }

        Err(WasiErrno::NotCapable)
    }
}

/// WASI Filesystem Core
///
/// Implements the core WASI filesystem operations following snapshot_preview1.
pub struct WasiFilesystem {
    /// File descriptor table
    pub fd_table: FdTable,
}

impl WasiFilesystem {
    /// Create a new WASI filesystem instance
    pub fn new() -> Self {
        Self {
            fd_table: FdTable::new(),
        }
    }

    /// Add a preopen directory
    ///
    /// This defines a capability - a directory that the WASM module
    /// is allowed to access.
    pub fn add_preopen(&self, path: impl AsRef<Path>) -> Result<Fd, WasiErrno> {
        let path = path.as_ref().to_path_buf();
        self.fd_table.insert_preopen(path)
    }

    /// fd_read: Read from a file descriptor
    ///
    /// # Arguments
    /// * `fd` - File descriptor to read from
    /// * `buf` - Buffer to read into
    ///
    /// # Returns
    /// Number of bytes read, or error
    pub fn fd_read(&self, fd: Fd, buf: &mut [u8]) -> Result<usize, WasiErrno> {
        // Check read permission
        self.fd_table.check_rights(fd, |r| r.fd_read)?;

        // Read from the file
        self.fd_table.get_mut(fd, |entry| {
            if let Some(ref mut file) = entry.file {
                file.read(buf).map_err(|e| WasiErrno::from_io_error(&e))
            } else {
                Err(WasiErrno::IsDir)
            }
        })
    }

    /// fd_write: Write to a file descriptor
    ///
    /// # Arguments
    /// * `fd` - File descriptor to write to
    /// * `buf` - Buffer containing data to write
    ///
    /// # Returns
    /// Number of bytes written, or error
    pub fn fd_write(&self, fd: Fd, buf: &[u8]) -> Result<usize, WasiErrno> {
        // Check write permission
        self.fd_table.check_rights(fd, |r| r.fd_write)?;

        // Write to the file
        self.fd_table.get_mut(fd, |entry| {
            if let Some(ref mut file) = entry.file {
                file.write(buf).map_err(|e| WasiErrno::from_io_error(&e))
            } else {
                Err(WasiErrno::IsDir)
            }
        })
    }

    /// fd_seek: Move the file position
    ///
    /// # Arguments
    /// * `fd` - File descriptor
    /// * `offset` - Offset to seek by
    /// * `whence` - Reference point (0=start, 1=current, 2=end)
    ///
    /// # Returns
    /// New file position, or error
    pub fn fd_seek(&self, fd: Fd, offset: i64, whence: u8) -> Result<u64, WasiErrno> {
        // Check seek permission
        self.fd_table.check_rights(fd, |r| r.fd_seek)?;

        // Convert whence to SeekFrom
        let seek_from = match whence {
            0 => SeekFrom::Start(offset as u64),
            1 => SeekFrom::Current(offset),
            2 => SeekFrom::End(offset),
            _ => return Err(WasiErrno::Inval),
        };

        // Perform the seek
        self.fd_table.get_mut(fd, |entry| {
            if let Some(ref mut file) = entry.file {
                file.seek(seek_from)
                    .map_err(|e| WasiErrno::from_io_error(&e))
            } else {
                Err(WasiErrno::IsDir)
            }
        })
    }

    /// fd_tell: Get current file position
    ///
    /// # Arguments
    /// * `fd` - File descriptor
    ///
    /// # Returns
    /// Current file position, or error
    pub fn fd_tell(&self, fd: Fd) -> Result<u64, WasiErrno> {
        // Check tell permission
        self.fd_table.check_rights(fd, |r| r.fd_tell)?;

        // Get current position
        self.fd_table.get_mut(fd, |entry| {
            if let Some(ref mut file) = entry.file {
                file.stream_position()
                    .map_err(|e| WasiErrno::from_io_error(&e))
            } else {
                Err(WasiErrno::IsDir)
            }
        })
    }

    /// fd_close: Close a file descriptor
    ///
    /// # Arguments
    /// * `fd` - File descriptor to close
    ///
    /// # Returns
    /// Success or error
    pub fn fd_close(&self, fd: Fd) -> Result<(), WasiErrno> {
        self.fd_table.remove(fd)
    }

    /// path_open: Open a file by path
    ///
    /// # Arguments
    /// * `dirfd` - Directory FD to resolve path relative to
    /// * `path` - Path to open (relative to dirfd)
    /// * `flags` - Open flags (create, truncate, etc.)
    /// * `rights` - Requested rights for the new FD
    ///
    /// # Returns
    /// New file descriptor, or error
    pub fn path_open(
        &self,
        dirfd: Fd,
        path: &Path,
        flags: OpenFlags,
        rights: Rights,
    ) -> Result<Fd, WasiErrno> {
        // Get the directory entry
        let dir_entry = self.fd_table.get(dirfd)?;

        // Check that dirfd has path_open right
        if !dir_entry.rights.path_open {
            return Err(WasiErrno::NotCapable);
        }

        // Resolve the path within the sandbox
        let full_path = PathResolver::resolve(&dir_entry.path, path)?;

        // Check if we need create permission
        if flags.create && !dir_entry.rights.path_create_file {
            return Err(WasiErrno::NotCapable);
        }

        // Build OpenOptions
        let mut open_opts = OpenOptions::new();
        open_opts.read(rights.fd_read);
        open_opts.write(rights.fd_write || flags.create); // Need write to create
        open_opts.create(flags.create);
        open_opts.truncate(flags.trunc);

        if flags.excl {
            open_opts.create_new(true);
        }

        // Open the file
        let file = open_opts
            .open(&full_path)
            .map_err(|e| WasiErrno::from_io_error(&e))?;

        // Insert into FD table
        self.fd_table.insert_file(file, full_path, rights)
    }

    /// path_create_directory: Create a directory
    ///
    /// # Arguments
    /// * `dirfd` - Directory FD to resolve path relative to
    /// * `path` - Path of directory to create
    ///
    /// # Returns
    /// Success or error
    pub fn path_create_directory(&self, dirfd: Fd, path: &Path) -> Result<(), WasiErrno> {
        // Get the directory entry
        let dir_entry = self.fd_table.get(dirfd)?;

        // Check permission
        if !dir_entry.rights.path_create_directory {
            return Err(WasiErrno::NotCapable);
        }

        // Resolve the path within the sandbox
        // For create, if parent doesn't exist yet, we need special handling
        let full_path = if let Some(parent) = path.parent() {
            if parent.as_os_str().is_empty() {
                // No parent component, create directly in base
                dir_entry.path.join(path)
            } else {
                // Parent exists, resolve it
                let parent_path = PathResolver::resolve(&dir_entry.path, parent)?;
                let name = path.file_name().ok_or(WasiErrno::Inval)?;
                parent_path.join(name)
            }
        } else {
            // No parent at all, create in base
            dir_entry.path.join(path)
        };

        // Create the directory
        std::fs::create_dir(&full_path).map_err(|e| WasiErrno::from_io_error(&e))
    }

    /// path_unlink_file: Remove a file
    ///
    /// # Arguments
    /// * `dirfd` - Directory FD to resolve path relative to
    /// * `path` - Path of file to remove
    ///
    /// # Returns
    /// Success or error
    pub fn path_unlink_file(&self, dirfd: Fd, path: &Path) -> Result<(), WasiErrno> {
        // Get the directory entry
        let dir_entry = self.fd_table.get(dirfd)?;

        // Check permission
        if !dir_entry.rights.path_unlink_file {
            return Err(WasiErrno::NotCapable);
        }

        // Resolve the path within the sandbox
        let full_path = PathResolver::resolve(&dir_entry.path, path)?;

        // Remove the file
        std::fs::remove_file(&full_path).map_err(|e| WasiErrno::from_io_error(&e))
    }

    /// path_filestat_get: Get file metadata
    ///
    /// # Arguments
    /// * `dirfd` - Directory FD to resolve path relative to
    /// * `path` - Path of file to stat
    ///
    /// # Returns
    /// File metadata, or error
    pub fn path_filestat_get(&self, dirfd: Fd, path: &Path) -> Result<FileStats, WasiErrno> {
        // Get the directory entry
        let dir_entry = self.fd_table.get(dirfd)?;

        // Resolve the path within the sandbox
        let full_path = PathResolver::resolve(&dir_entry.path, path)?;

        // Get metadata
        let metadata = std::fs::metadata(&full_path).map_err(|e| WasiErrno::from_io_error(&e))?;

        Ok(FileStats {
            size: metadata.len(),
            is_dir: metadata.is_dir(),
            is_file: metadata.is_file(),
        })
    }
}

impl Default for WasiFilesystem {
    fn default() -> Self {
        Self::new()
    }
}

/// File statistics
#[derive(Debug, Clone)]
pub struct FileStats {
    pub size: u64,
    pub is_dir: bool,
    pub is_file: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn test_errno_conversion() {
        let err = io::Error::from(io::ErrorKind::NotFound);
        assert_eq!(WasiErrno::from_io_error(&err), WasiErrno::NoEnt);

        let err = io::Error::from(io::ErrorKind::PermissionDenied);
        assert_eq!(WasiErrno::from_io_error(&err), WasiErrno::Access);
    }

    #[test]
    fn test_fd_table_creation() {
        let table = FdTable::new();
        // Should start with no entries
        assert!(table.get(PREOPEN_START_FD).is_err());
    }

    #[test]
    fn test_preopen_insertion() {
        let table = FdTable::new();
        let temp_dir = std::env::temp_dir();

        let fd = table.insert_preopen(temp_dir.clone()).unwrap();
        assert!(fd >= PREOPEN_START_FD);

        let entry = table.get(fd).unwrap();
        assert!(entry.is_preopen);
        assert!(entry.rights.fd_read);
    }

    #[test]
    fn test_rights_checking() {
        let table = FdTable::new();
        let temp_dir = std::env::temp_dir();
        let fd = table.insert_preopen(temp_dir).unwrap();

        // Should have read rights
        assert!(table.check_rights(fd, |r| r.fd_read).is_ok());
        // Should have write rights
        assert!(table.check_rights(fd, |r| r.fd_write).is_ok());
    }

    #[test]
    fn test_path_resolution() {
        let base = PathBuf::from("/tmp");
        let path = PathBuf::from("test.txt");

        // This will fail if /tmp doesn't exist, which is fine for this test
        // The important thing is testing the logic
        let result = PathResolver::resolve(&base, &path);

        // Result should either succeed or fail with a valid error
        match result {
            Ok(resolved) => {
                // Should be /tmp/test.txt (canonicalized)
                assert!(resolved.to_string_lossy().contains("tmp"));
            }
            Err(e) => {
                // Should be a reasonable error (NotFound, etc.)
                assert!(matches!(e, WasiErrno::NoEnt | WasiErrno::Io));
            }
        }
    }

    #[test]
    fn test_sandbox_escape_prevention() {
        let base = PathBuf::from("/tmp/sandbox");
        let escape_path = PathBuf::from("../etc/passwd");

        // Create sandbox dir for test
        let _ = fs::create_dir_all(&base);

        // Should prevent escaping the sandbox
        let result = PathResolver::resolve(&base, &escape_path);

        // Should fail because it tries to escape
        assert!(result.is_err());
    }

    #[test]
    fn test_wasi_filesystem_creation() {
        let wasi_fs = WasiFilesystem::new();
        // Should be able to add preopens
        let temp_dir = std::env::temp_dir();
        let result = wasi_fs.add_preopen(temp_dir);
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_operations_end_to_end() {
        let wasi_fs = WasiFilesystem::new();

        // Create a test directory
        let test_dir = std::env::temp_dir().join("wasi_core_test");
        let _ = fs::create_dir(&test_dir);

        // Add as preopen
        let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

        // Open a file for writing
        let test_file = Path::new("test.txt");
        let fd = wasi_fs
            .path_open(
                dirfd,
                test_file,
                OpenFlags {
                    create: true,
                    trunc: true,
                    ..Default::default()
                },
                Rights::read_write(),
            )
            .unwrap();

        // Write data
        let data = b"Hello, WASI!";
        let written = wasi_fs.fd_write(fd, data).unwrap();
        assert_eq!(written, data.len());

        // Seek to beginning
        let pos = wasi_fs.fd_seek(fd, 0, 0).unwrap();
        assert_eq!(pos, 0);

        // Read data back
        let mut buf = vec![0u8; data.len()];
        let read = wasi_fs.fd_read(fd, &mut buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buf, data);

        // Close file
        wasi_fs.fd_close(fd).unwrap();

        // Clean up
        let _ = fs::remove_file(test_dir.join("test.txt"));
        let _ = fs::remove_dir(test_dir);
    }

    #[test]
    fn test_directory_operations() {
        let wasi_fs = WasiFilesystem::new();

        // Create a test directory
        let test_dir = std::env::temp_dir().join("wasi_core_test_dir");
        let _ = fs::create_dir(&test_dir);

        // Add as preopen
        let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

        // Create a subdirectory
        let subdir = Path::new("subdir");
        wasi_fs.path_create_directory(dirfd, subdir).unwrap();

        // Verify it exists
        let stats = wasi_fs.path_filestat_get(dirfd, subdir).unwrap();
        assert!(stats.is_dir);

        // Clean up
        let _ = fs::remove_dir(test_dir.join("subdir"));
        let _ = fs::remove_dir(test_dir);
    }

    #[test]
    fn test_permission_enforcement() {
        let wasi_fs = WasiFilesystem::new();
        let test_dir = std::env::temp_dir().join("wasi_core_test_perms");
        let _ = fs::create_dir(&test_dir);

        let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

        // Open file with read-only rights
        let test_file = Path::new("readonly.txt");

        // First create the file
        fs::File::create(test_dir.join(test_file))
            .unwrap()
            .write_all(b"test")
            .unwrap();

        let fd = wasi_fs
            .path_open(dirfd, test_file, OpenFlags::default(), Rights::read_only())
            .unwrap();

        // Writing should fail
        let result = wasi_fs.fd_write(fd, b"data");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), WasiErrno::NotCapable);

        // Reading should succeed
        let mut buf = vec![0u8; 4];
        let result = wasi_fs.fd_read(fd, &mut buf);
        assert!(result.is_ok());

        // Clean up
        wasi_fs.fd_close(fd).unwrap();
        let _ = fs::remove_file(test_dir.join(test_file));
        let _ = fs::remove_dir(test_dir);
    }
}
