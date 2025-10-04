//! WASI Directory Operations - Usage Examples
//!
//! This example demonstrates how to use the WASI directory operations
//! in the Portalis transpiler.

use portalis_transpiler::wasi_directory::{WasiDirectory, DirIterator, FileType};
use portalis_transpiler::wasi_fs::WasiFs;
use anyhow::Result;

fn main() -> Result<()> {
    println!("WASI Directory Operations Examples\n");

    // Example 1: Create a directory
    example_create_directory()?;

    // Example 2: Create nested directories
    example_create_nested_directories()?;

    // Example 3: Read directory contents
    example_read_directory()?;

    // Example 4: List directory with iteration
    example_iterate_directory()?;

    // Example 5: Get file metadata
    example_file_metadata()?;

    // Example 6: Remove directory tree
    example_remove_directory_tree()?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

fn example_create_directory() -> Result<()> {
    println!("Example 1: Create a directory");

    let dir_path = "/tmp/portalis_example_dir";

    // Create directory
    WasiDirectory::create_dir(dir_path)?;
    println!("  Created directory: {}", dir_path);

    // Check if it exists
    if WasiDirectory::exists(dir_path) {
        println!("  Directory exists: ✓");
    }

    // Clean up
    WasiDirectory::remove_dir(dir_path)?;
    println!("  Cleaned up\n");

    Ok(())
}

fn example_create_nested_directories() -> Result<()> {
    println!("Example 2: Create nested directories");

    let nested_path = "/tmp/portalis_example/level1/level2/level3";

    // Create all parent directories
    WasiDirectory::create_dir_all(nested_path)?;
    println!("  Created nested path: {}", nested_path);

    // Verify all levels exist
    if WasiDirectory::exists("/tmp/portalis_example") &&
       WasiDirectory::exists("/tmp/portalis_example/level1") &&
       WasiDirectory::exists(nested_path) {
        println!("  All levels exist: ✓");
    }

    // Clean up
    WasiDirectory::remove_dir_all("/tmp/portalis_example")?;
    println!("  Cleaned up\n");

    Ok(())
}

fn example_read_directory() -> Result<()> {
    println!("Example 3: Read directory contents");

    let dir_path = "/tmp/portalis_example_read";

    // Create directory with some files
    WasiDirectory::create_dir(dir_path)?;
    WasiFs::write(format!("{}/file1.txt", dir_path), "Content 1")?;
    WasiFs::write(format!("{}/file2.txt", dir_path), "Content 2")?;
    WasiFs::write(format!("{}/file3.txt", dir_path), "Content 3")?;
    WasiDirectory::create_dir(format!("{}/subdir", dir_path))?;

    // Read directory entries
    let entries = WasiDirectory::read_dir(dir_path)?;
    println!("  Found {} entries:", entries.len());

    for entry in entries {
        let type_str = match entry.d_type {
            FileType::RegularFile => "File",
            FileType::Directory => "Dir ",
            _ => "????"
        };
        println!("    [{}] {}", type_str, entry.name);
    }

    // Clean up
    WasiDirectory::remove_dir_all(dir_path)?;
    println!("  Cleaned up\n");

    Ok(())
}

fn example_iterate_directory() -> Result<()> {
    println!("Example 4: Iterate directory with DirIterator");

    let dir_path = "/tmp/portalis_example_iter";

    // Create directory with files
    WasiDirectory::create_dir(dir_path)?;
    for i in 1..=5 {
        WasiFs::write(format!("{}/item_{}.txt", dir_path, i), &format!("Item {}", i))?;
    }

    // Use iterator
    let iter = DirIterator::new(dir_path)?;
    println!("  Iterating over entries:");

    for (idx, result) in iter.enumerate() {
        let entry = result?;
        println!("    {}. {} ({} bytes name)", idx + 1, entry.name, entry.d_namlen);
    }

    // Alternative: Use list_dir for just names
    let names = WasiDirectory::list_dir(dir_path)?;
    println!("  Names only: {:?}", names);

    // Clean up
    WasiDirectory::remove_dir_all(dir_path)?;
    println!("  Cleaned up\n");

    Ok(())
}

fn example_file_metadata() -> Result<()> {
    println!("Example 5: Get file metadata");

    let file_path = "/tmp/portalis_example_metadata.txt";
    let dir_path = "/tmp/portalis_example_metadata_dir";

    // Create a file
    let content = "This is test content for metadata example.";
    WasiFs::write(file_path, content)?;

    // Get file metadata
    let file_stat = WasiDirectory::metadata(file_path)?;
    println!("  File metadata:");
    println!("    Type: {:?}", file_stat.filetype);
    println!("    Size: {} bytes", file_stat.size);
    println!("    Modified time: {} ns since epoch", file_stat.mtim);

    // Create a directory
    WasiDirectory::create_dir(dir_path)?;

    // Get directory metadata
    let dir_stat = WasiDirectory::metadata(dir_path)?;
    println!("  Directory metadata:");
    println!("    Type: {:?}", dir_stat.filetype);

    // Clean up
    WasiFs::remove_file(file_path)?;
    WasiDirectory::remove_dir(dir_path)?;
    println!("  Cleaned up\n");

    Ok(())
}

fn example_remove_directory_tree() -> Result<()> {
    println!("Example 6: Remove directory tree");

    let root_path = "/tmp/portalis_example_tree";

    // Create complex directory structure
    WasiDirectory::create_dir_all(format!("{}/a/b/c", root_path))?;
    WasiDirectory::create_dir_all(format!("{}/a/d", root_path))?;
    WasiDirectory::create_dir_all(format!("{}/e/f", root_path))?;

    // Add files at various levels
    WasiFs::write(format!("{}/root.txt", root_path), "root")?;
    WasiFs::write(format!("{}/a/a.txt", root_path), "a")?;
    WasiFs::write(format!("{}/a/b/b.txt", root_path), "b")?;
    WasiFs::write(format!("{}/a/b/c/c.txt", root_path), "c")?;
    WasiFs::write(format!("{}/e/f/f.txt", root_path), "f")?;

    println!("  Created complex directory tree at {}", root_path);

    // Count total entries
    let entries = count_all_entries(root_path)?;
    println!("  Total entries (recursive): {}", entries);

    // Remove entire tree
    WasiDirectory::remove_dir_all(root_path)?;
    println!("  Removed entire tree: ✓");

    // Verify it's gone
    if !WasiDirectory::exists(root_path) {
        println!("  Verification: Tree is gone ✓\n");
    }

    Ok(())
}

// Helper function to recursively count entries
fn count_all_entries(path: &str) -> Result<usize> {
    let mut count = 0;

    let entries = WasiDirectory::read_dir(path)?;
    for entry in entries {
        count += 1;
        if entry.d_type == FileType::Directory {
            let subpath = format!("{}/{}", path, entry.name);
            count += count_all_entries(&subpath)?;
        }
    }

    Ok(count)
}
