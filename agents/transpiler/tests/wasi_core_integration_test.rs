//! WASI Core Filesystem Integration Tests
//!
//! Comprehensive tests for the core WASI filesystem implementation

use portalis_transpiler::wasi_core::{
    WasiFilesystem, WasiErrno, OpenFlags, Rights, Fd,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn setup_test_dir(name: &str) -> PathBuf {
    let test_dir = std::env::temp_dir().join(name);
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("Failed to create test dir");
    test_dir
}

fn cleanup_test_dir(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

#[test]
fn test_basic_file_lifecycle() {
    let test_dir = setup_test_dir("wasi_core_lifecycle");
    let wasi_fs = WasiFilesystem::new();

    // Add preopen
    let dirfd = wasi_fs.add_preopen(&test_dir).expect("Failed to add preopen");

    // Create and write to a file
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("test.txt"),
            OpenFlags {
                create: true,
                trunc: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .expect("Failed to open file");

    let data = b"Hello, WASI Core!";
    let written = wasi_fs.fd_write(fd, data).expect("Failed to write");
    assert_eq!(written, data.len());

    // Seek to start
    wasi_fs.fd_seek(fd, 0, 0).expect("Failed to seek");

    // Read back
    let mut buf = vec![0u8; data.len()];
    let read = wasi_fs.fd_read(fd, &mut buf).expect("Failed to read");
    assert_eq!(read, data.len());
    assert_eq!(&buf, data);

    // Close
    wasi_fs.fd_close(fd).expect("Failed to close");

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_multiple_preopens() {
    let test_dir1 = setup_test_dir("wasi_core_preopen1");
    let test_dir2 = setup_test_dir("wasi_core_preopen2");

    let wasi_fs = WasiFilesystem::new();

    let fd1 = wasi_fs.add_preopen(&test_dir1).expect("Failed to add preopen 1");
    let fd2 = wasi_fs.add_preopen(&test_dir2).expect("Failed to add preopen 2");

    // FDs should be different
    assert_ne!(fd1, fd2);

    // Should be able to open files in both
    let file1 = wasi_fs
        .path_open(
            fd1,
            Path::new("file1.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .expect("Failed to open in preopen 1");

    let file2 = wasi_fs
        .path_open(
            fd2,
            Path::new("file2.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .expect("Failed to open in preopen 2");

    assert_ne!(file1, file2);

    wasi_fs.fd_close(file1).unwrap();
    wasi_fs.fd_close(file2).unwrap();

    cleanup_test_dir(&test_dir1);
    cleanup_test_dir(&test_dir2);
}

#[test]
fn test_sandbox_enforcement() {
    let test_dir = setup_test_dir("wasi_core_sandbox");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Try to escape sandbox
    let result = wasi_fs.path_open(
        dirfd,
        Path::new("../../../etc/passwd"),
        OpenFlags::default(),
        Rights::read_only(),
    );

    // Should fail with NotCapable
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::NotCapable);

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_permission_isolation() {
    let test_dir = setup_test_dir("wasi_core_permissions");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create a file first
    fs::File::create(test_dir.join("readonly.txt"))
        .unwrap()
        .write_all(b"test data")
        .unwrap();

    // Open with read-only rights
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("readonly.txt"),
            OpenFlags::default(),
            Rights::read_only(),
        )
        .expect("Failed to open");

    // Read should work
    let mut buf = vec![0u8; 9];
    assert!(wasi_fs.fd_read(fd, &mut buf).is_ok());
    assert_eq!(&buf, b"test data");

    // Write should fail
    let result = wasi_fs.fd_write(fd, b"new data");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::NotCapable);

    wasi_fs.fd_close(fd).unwrap();
    cleanup_test_dir(&test_dir);
}

#[test]
fn test_directory_creation_and_nesting() {
    let test_dir = setup_test_dir("wasi_core_dirs");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create a subdirectory
    wasi_fs
        .path_create_directory(dirfd, Path::new("subdir"))
        .expect("Failed to create subdir");

    // Verify it exists
    let stats = wasi_fs
        .path_filestat_get(dirfd, Path::new("subdir"))
        .expect("Failed to stat subdir");
    assert!(stats.is_dir);

    // Create a file in the subdirectory
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("subdir/nested.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .expect("Failed to create nested file");

    wasi_fs.fd_write(fd, b"nested data").unwrap();
    wasi_fs.fd_close(fd).unwrap();

    // Verify file exists
    let stats = wasi_fs
        .path_filestat_get(dirfd, Path::new("subdir/nested.txt"))
        .expect("Failed to stat nested file");
    assert!(stats.is_file);
    assert_eq!(stats.size, 11);

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_seek_operations() {
    let test_dir = setup_test_dir("wasi_core_seek");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("seek_test.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();

    // Write some data
    let data = b"0123456789ABCDEF";
    wasi_fs.fd_write(fd, data).unwrap();

    // Seek to position 5 from start
    let pos = wasi_fs.fd_seek(fd, 5, 0).unwrap();
    assert_eq!(pos, 5);

    // Read from position 5
    let mut buf = vec![0u8; 5];
    wasi_fs.fd_read(fd, &mut buf).unwrap();
    assert_eq!(&buf, b"56789");

    // Seek relative (forward 2 from current)
    let pos = wasi_fs.fd_seek(fd, 2, 1).unwrap();
    assert_eq!(pos, 12);

    // Read from position 12
    let mut buf = vec![0u8; 4];
    wasi_fs.fd_read(fd, &mut buf).unwrap();
    assert_eq!(&buf, b"CDEF");

    // Seek from end (-4 bytes)
    let pos = wasi_fs.fd_seek(fd, -4, 2).unwrap();
    assert_eq!(pos, 12);

    // Get current position
    let pos = wasi_fs.fd_tell(fd).unwrap();
    assert_eq!(pos, 12);

    wasi_fs.fd_close(fd).unwrap();
    cleanup_test_dir(&test_dir);
}

#[test]
fn test_file_truncation() {
    let test_dir = setup_test_dir("wasi_core_truncate");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create file with data
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("trunc.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();

    wasi_fs.fd_write(fd, b"Hello, World!").unwrap();
    wasi_fs.fd_close(fd).unwrap();

    // Reopen with truncate
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("trunc.txt"),
            OpenFlags {
                trunc: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();

    // Write new data
    wasi_fs.fd_write(fd, b"New").unwrap();
    wasi_fs.fd_close(fd).unwrap();

    // Verify size
    let stats = wasi_fs
        .path_filestat_get(dirfd, Path::new("trunc.txt"))
        .unwrap();
    assert_eq!(stats.size, 3);

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_exclusive_create() {
    let test_dir = setup_test_dir("wasi_core_excl");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create file with excl flag
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("exclusive.txt"),
            OpenFlags {
                create: true,
                excl: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .expect("Failed to create exclusive file");

    wasi_fs.fd_close(fd).unwrap();

    // Try to create again with excl - should fail
    let result = wasi_fs.path_open(
        dirfd,
        Path::new("exclusive.txt"),
        OpenFlags {
            create: true,
            excl: true,
            ..Default::default()
        },
        Rights::read_write(),
    );

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::Exist);

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_file_deletion() {
    let test_dir = setup_test_dir("wasi_core_delete");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create file
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("delete_me.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();
    wasi_fs.fd_close(fd).unwrap();

    // Verify it exists
    assert!(wasi_fs.path_filestat_get(dirfd, Path::new("delete_me.txt")).is_ok());

    // Delete it
    wasi_fs
        .path_unlink_file(dirfd, Path::new("delete_me.txt"))
        .expect("Failed to delete file");

    // Verify it's gone
    let result = wasi_fs.path_filestat_get(dirfd, Path::new("delete_me.txt"));
    assert!(result.is_err());

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_error_handling_invalid_fd() {
    let wasi_fs = WasiFilesystem::new();

    // Try to read from invalid FD
    let mut buf = vec![0u8; 10];
    let result = wasi_fs.fd_read(999, &mut buf);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::BadF);

    // Try to write to invalid FD
    let result = wasi_fs.fd_write(999, b"data");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::BadF);

    // Try to close invalid FD
    let result = wasi_fs.fd_close(999);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::BadF);
}

#[test]
fn test_error_handling_nonexistent_file() {
    let test_dir = setup_test_dir("wasi_core_nonexistent");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Try to open nonexistent file without create flag
    let result = wasi_fs.path_open(
        dirfd,
        Path::new("does_not_exist.txt"),
        OpenFlags::default(),
        Rights::read_only(),
    );

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::NoEnt);

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_concurrent_file_access() {
    let test_dir = setup_test_dir("wasi_core_concurrent");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create a file
    let fd1 = wasi_fs
        .path_open(
            dirfd,
            Path::new("shared.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();

    wasi_fs.fd_write(fd1, b"First write").unwrap();

    // Open the same file again
    let fd2 = wasi_fs
        .path_open(
            dirfd,
            Path::new("shared.txt"),
            OpenFlags::default(),
            Rights::read_only(),
        )
        .unwrap();

    // Both FDs should be different
    assert_ne!(fd1, fd2);

    // Read from second FD
    let mut buf = vec![0u8; 11];
    wasi_fs.fd_read(fd2, &mut buf).unwrap();
    assert_eq!(&buf, b"First write");

    // Close both
    wasi_fs.fd_close(fd1).unwrap();
    wasi_fs.fd_close(fd2).unwrap();

    cleanup_test_dir(&test_dir);
}

#[test]
fn test_rights_inheritance() {
    let test_dir = setup_test_dir("wasi_core_rights_inherit");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    // Create file with read-only rights
    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("readonly.txt"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_only(),
        )
        .expect("Failed to create file");

    // Write should fail (read-only rights)
    let result = wasi_fs.fd_write(fd, b"data");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), WasiErrno::NotCapable);

    // Read should work
    let mut buf = vec![0u8; 10];
    let result = wasi_fs.fd_read(fd, &mut buf);
    assert!(result.is_ok());

    wasi_fs.fd_close(fd).unwrap();
    cleanup_test_dir(&test_dir);
}

#[test]
fn test_large_file_operations() {
    let test_dir = setup_test_dir("wasi_core_large_file");
    let wasi_fs = WasiFilesystem::new();
    let dirfd = wasi_fs.add_preopen(&test_dir).unwrap();

    let fd = wasi_fs
        .path_open(
            dirfd,
            Path::new("large.bin"),
            OpenFlags {
                create: true,
                ..Default::default()
            },
            Rights::read_write(),
        )
        .unwrap();

    // Write 1MB of data in chunks
    let chunk = vec![0xABu8; 1024]; // 1KB chunks
    for _ in 0..1024 {
        wasi_fs.fd_write(fd, &chunk).unwrap();
    }

    // Seek to middle
    wasi_fs.fd_seek(fd, 512 * 1024, 0).unwrap();

    // Read and verify
    let mut buf = vec![0u8; 1024];
    wasi_fs.fd_read(fd, &mut buf).unwrap();
    assert_eq!(buf, chunk);

    wasi_fs.fd_close(fd).unwrap();

    // Verify file size
    let stats = wasi_fs.path_filestat_get(dirfd, Path::new("large.bin")).unwrap();
    assert_eq!(stats.size, 1024 * 1024);

    cleanup_test_dir(&test_dir);
}
