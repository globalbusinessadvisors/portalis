//! Comprehensive WASI Directory Operations Tests
//!
//! Tests all directory-related functionality including:
//! - Directory creation and removal
//! - Directory reading and iteration
//! - File metadata retrieval
//! - Permission checks
//! - Error handling

use portalis_transpiler::wasi_directory::{
    WasiDirectory, WasiDir, DirEntry, DirIterator, FileStat, FileType
};
use std::fs;

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_create_directory() {
    let test_dir = "/tmp/portalis_test_create_directory";

    // Clean up first
    let _ = fs::remove_dir(test_dir);

    // Create directory
    WasiDirectory::create_dir(test_dir).expect("Failed to create directory");

    // Verify it exists
    assert!(WasiDirectory::exists(test_dir));

    // Get metadata
    let metadata = WasiDirectory::metadata(test_dir).expect("Failed to get metadata");
    assert_eq!(metadata.filetype, FileType::Directory);

    // Clean up
    WasiDirectory::remove_dir(test_dir).expect("Failed to remove directory");
    assert!(!WasiDirectory::exists(test_dir));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_create_nested_directories() {
    let test_dir = "/tmp/portalis_test_nested/level1/level2/level3";

    // Clean up first
    let _ = fs::remove_dir_all("/tmp/portalis_test_nested");

    // Create nested directories
    WasiDirectory::create_dir_all(test_dir).expect("Failed to create nested directories");

    // Verify all levels exist
    assert!(WasiDirectory::exists("/tmp/portalis_test_nested"));
    assert!(WasiDirectory::exists("/tmp/portalis_test_nested/level1"));
    assert!(WasiDirectory::exists("/tmp/portalis_test_nested/level1/level2"));
    assert!(WasiDirectory::exists(test_dir));

    // Clean up
    WasiDirectory::remove_dir_all("/tmp/portalis_test_nested")
        .expect("Failed to remove directory tree");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_read_empty_directory() {
    let test_dir = "/tmp/portalis_test_read_empty";

    // Clean up first
    let _ = fs::remove_dir(test_dir);

    // Create empty directory
    fs::create_dir(test_dir).expect("Failed to create directory");

    // Read directory
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), 0);

    // Clean up
    fs::remove_dir(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_read_directory_with_files() {
    let test_dir = "/tmp/portalis_test_read_files";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with files
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/file1.txt", test_dir), "content1")
        .expect("Failed to write file1");
    fs::write(format!("{}/file2.txt", test_dir), "content2")
        .expect("Failed to write file2");
    fs::write(format!("{}/file3.txt", test_dir), "content3")
        .expect("Failed to write file3");

    // Read directory
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), 3);

    // Verify all entries are files
    for entry in &entries {
        assert_eq!(entry.d_type, FileType::RegularFile);
        assert!(entry.name.ends_with(".txt"));
    }

    // Verify names
    let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"file1.txt".to_string()));
    assert!(names.contains(&"file2.txt".to_string()));
    assert!(names.contains(&"file3.txt".to_string()));

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_read_directory_with_subdirectories() {
    let test_dir = "/tmp/portalis_test_read_subdirs";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with subdirectories
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::create_dir(format!("{}/subdir1", test_dir)).expect("Failed to create subdir1");
    fs::create_dir(format!("{}/subdir2", test_dir)).expect("Failed to create subdir2");
    fs::write(format!("{}/file.txt", test_dir), "content")
        .expect("Failed to write file");

    // Read directory
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), 3);

    // Count directories and files
    let dir_count = entries.iter().filter(|e| e.d_type == FileType::Directory).count();
    let file_count = entries.iter().filter(|e| e.d_type == FileType::RegularFile).count();

    assert_eq!(dir_count, 2);
    assert_eq!(file_count, 1);

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_list_directory() {
    let test_dir = "/tmp/portalis_test_list_dir";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with files
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/alpha.txt", test_dir), "a").expect("Failed to write alpha.txt");
    fs::write(format!("{}/beta.txt", test_dir), "b").expect("Failed to write beta.txt");
    fs::write(format!("{}/gamma.txt", test_dir), "c").expect("Failed to write gamma.txt");

    // List directory
    let mut names = WasiDirectory::list_dir(test_dir).expect("Failed to list directory");
    names.sort();

    assert_eq!(names, vec!["alpha.txt", "beta.txt", "gamma.txt"]);

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_directory_iterator() {
    let test_dir = "/tmp/portalis_test_dir_iterator";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with files
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/1.txt", test_dir), "1").expect("Failed to write 1.txt");
    fs::write(format!("{}/2.txt", test_dir), "2").expect("Failed to write 2.txt");
    fs::write(format!("{}/3.txt", test_dir), "3").expect("Failed to write 3.txt");

    // Create iterator
    let iter = DirIterator::new(test_dir).expect("Failed to create iterator");

    // Collect entries
    let entries: Result<Vec<DirEntry>, _> = iter.collect();
    let entries = entries.expect("Iterator failed");

    assert_eq!(entries.len(), 3);

    // Verify all entries
    for entry in entries {
        assert!(entry.name.ends_with(".txt"));
        assert_eq!(entry.d_type, FileType::RegularFile);
    }

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_file_metadata() {
    let test_file = "/tmp/portalis_test_file_metadata.txt";
    let test_content = "Hello, WASI!";

    // Create test file
    fs::write(test_file, test_content).expect("Failed to write test file");

    // Get metadata
    let metadata = WasiDirectory::metadata(test_file).expect("Failed to get metadata");

    // Verify metadata
    assert_eq!(metadata.filetype, FileType::RegularFile);
    assert_eq!(metadata.size, test_content.len() as u64);
    assert!(metadata.mtim > 0);
    assert!(metadata.atim > 0);

    // Clean up
    fs::remove_file(test_file).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_directory_metadata() {
    let test_dir = "/tmp/portalis_test_dir_metadata";

    // Clean up first
    let _ = fs::remove_dir(test_dir);

    // Create directory
    fs::create_dir(test_dir).expect("Failed to create directory");

    // Get metadata
    let metadata = WasiDirectory::metadata(test_dir).expect("Failed to get metadata");

    // Verify metadata
    assert_eq!(metadata.filetype, FileType::Directory);

    // Clean up
    fs::remove_dir(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_remove_dir_all() {
    let test_dir = "/tmp/portalis_test_remove_all";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create complex directory structure
    fs::create_dir_all(format!("{}/sub1/sub2", test_dir))
        .expect("Failed to create nested dirs");
    fs::write(format!("{}/file1.txt", test_dir), "1")
        .expect("Failed to write file1");
    fs::write(format!("{}/sub1/file2.txt", test_dir), "2")
        .expect("Failed to write file2");
    fs::write(format!("{}/sub1/sub2/file3.txt", test_dir), "3")
        .expect("Failed to write file3");

    // Verify structure exists
    assert!(WasiDirectory::exists(test_dir));
    assert!(WasiDirectory::exists(format!("{}/sub1", test_dir)));
    assert!(WasiDirectory::exists(format!("{}/sub1/sub2", test_dir)));

    // Remove all
    WasiDirectory::remove_dir_all(test_dir).expect("Failed to remove directory tree");

    // Verify everything is gone
    assert!(!WasiDirectory::exists(test_dir));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_error_create_existing_directory() {
    let test_dir = "/tmp/portalis_test_error_existing";

    // Clean up first
    let _ = fs::remove_dir(test_dir);

    // Create directory
    fs::create_dir(test_dir).expect("Failed to create directory");

    // Try to create again - should fail
    let result = WasiDirectory::create_dir(test_dir);
    assert!(result.is_err());

    // Clean up
    fs::remove_dir(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_error_remove_nonexistent_directory() {
    let test_dir = "/tmp/portalis_test_error_nonexistent";

    // Make sure it doesn't exist
    let _ = fs::remove_dir(test_dir);

    // Try to remove - should fail
    let result = WasiDirectory::remove_dir(test_dir);
    assert!(result.is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_error_remove_nonempty_directory() {
    let test_dir = "/tmp/portalis_test_error_nonempty";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with file
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/file.txt", test_dir), "content")
        .expect("Failed to write file");

    // Try to remove - should fail (not empty)
    let result = WasiDirectory::remove_dir(test_dir);
    assert!(result.is_err());

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_error_read_nonexistent_directory() {
    let test_dir = "/tmp/portalis_test_error_read_nonexistent";

    // Make sure it doesn't exist
    let _ = fs::remove_dir_all(test_dir);

    // Try to read - should fail
    let result = WasiDirectory::read_dir(test_dir);
    assert!(result.is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_dir_entry_structure() {
    let test_dir = "/tmp/portalis_test_dir_entry";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with a file
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/test.txt", test_dir), "content")
        .expect("Failed to write file");

    // Read directory
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), 1);

    let entry = &entries[0];
    assert_eq!(entry.name, "test.txt");
    assert_eq!(entry.d_type, FileType::RegularFile);
    assert_eq!(entry.d_namlen, "test.txt".len() as u32);
    assert!(entry.d_next > 0);

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_file_type_detection() {
    let test_dir = "/tmp/portalis_test_file_types";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory structure
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/file.txt", test_dir), "content")
        .expect("Failed to write file");
    fs::create_dir(format!("{}/subdir", test_dir))
        .expect("Failed to create subdir");

    // Read entries
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), 2);

    // Find and check file
    let file_entry = entries.iter().find(|e| e.name == "file.txt").unwrap();
    assert_eq!(file_entry.d_type, FileType::RegularFile);

    // Find and check directory
    let dir_entry = entries.iter().find(|e| e.name == "subdir").unwrap();
    assert_eq!(dir_entry.d_type, FileType::Directory);

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_large_directory() {
    let test_dir = "/tmp/portalis_test_large_dir";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with many files
    fs::create_dir(test_dir).expect("Failed to create directory");

    let num_files = 100;
    for i in 0..num_files {
        fs::write(format!("{}/file_{:03}.txt", test_dir, i), format!("content {}", i))
            .expect(&format!("Failed to write file {}", i));
    }

    // Read directory
    let entries = WasiDirectory::read_dir(test_dir).expect("Failed to read directory");
    assert_eq!(entries.len(), num_files);

    // Verify all entries are present
    for i in 0..num_files {
        let filename = format!("file_{:03}.txt", i);
        assert!(entries.iter().any(|e| e.name == filename));
    }

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}

#[test]
fn test_file_type_from_u8() {
    assert_eq!(FileType::from_u8(0), FileType::Unknown);
    assert_eq!(FileType::from_u8(1), FileType::BlockDevice);
    assert_eq!(FileType::from_u8(2), FileType::CharacterDevice);
    assert_eq!(FileType::from_u8(3), FileType::Directory);
    assert_eq!(FileType::from_u8(4), FileType::RegularFile);
    assert_eq!(FileType::from_u8(5), FileType::SocketDgram);
    assert_eq!(FileType::from_u8(6), FileType::SocketStream);
    assert_eq!(FileType::from_u8(7), FileType::SymbolicLink);
    assert_eq!(FileType::from_u8(255), FileType::Unknown);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_exists_check() {
    let test_file = "/tmp/portalis_test_exists.txt";
    let test_dir = "/tmp/portalis_test_exists_dir";

    // Clean up
    let _ = fs::remove_file(test_file);
    let _ = fs::remove_dir(test_dir);

    // Check non-existent
    assert!(!WasiDirectory::exists(test_file));
    assert!(!WasiDirectory::exists(test_dir));

    // Create file and directory
    fs::write(test_file, "content").expect("Failed to write file");
    fs::create_dir(test_dir).expect("Failed to create directory");

    // Check exist
    assert!(WasiDirectory::exists(test_file));
    assert!(WasiDirectory::exists(test_dir));

    // Clean up
    fs::remove_file(test_file).expect("Failed to clean up file");
    fs::remove_dir(test_dir).expect("Failed to clean up dir");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_wasi_dir_open() {
    let test_dir = "/tmp/portalis_test_wasi_dir_open";

    // Clean up first
    let _ = fs::remove_dir(test_dir);

    // Create directory
    fs::create_dir(test_dir).expect("Failed to create directory");

    // Open directory
    let dir = WasiDir::open(test_dir);
    assert!(dir.is_ok());

    // Clean up
    fs::remove_dir(test_dir).expect("Failed to clean up");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_wasi_dir_open_nonexistent() {
    let test_dir = "/tmp/portalis_test_wasi_dir_nonexistent";

    // Make sure it doesn't exist
    let _ = fs::remove_dir(test_dir);

    // Try to open - should fail
    let result = WasiDir::open(test_dir);
    assert!(result.is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_wasi_dir_read_multiple_times() {
    let test_dir = "/tmp/portalis_test_wasi_dir_multiple_reads";

    // Clean up first
    let _ = fs::remove_dir_all(test_dir);

    // Create directory with files
    fs::create_dir(test_dir).expect("Failed to create directory");
    fs::write(format!("{}/file1.txt", test_dir), "1").expect("Failed to write file1");
    fs::write(format!("{}/file2.txt", test_dir), "2").expect("Failed to write file2");

    // Open directory
    let mut dir = WasiDir::open(test_dir).expect("Failed to open directory");

    // Read first time
    let entries1 = dir.read_dir(0).expect("Failed to read directory");
    assert_eq!(entries1.len(), 2);

    // Read second time - should return same results
    let entries2 = dir.read_dir(0).expect("Failed to read directory second time");
    assert_eq!(entries2.len(), 2);

    // Clean up
    fs::remove_dir_all(test_dir).expect("Failed to clean up");
}
