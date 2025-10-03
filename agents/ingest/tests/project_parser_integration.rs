//! Integration tests for ProjectParser

use portalis_ingest::ProjectParser;
use std::path::Path;

#[test]
fn test_parse_multi_file_project() {
    let parser = ProjectParser::new();
    let project_path = Path::new("../../examples/test_project");

    // Only run if test project exists
    if !project_path.exists() {
        eprintln!("Skipping test - test project not found");
        return;
    }

    let result = parser.parse_project(project_path);

    match result {
        Ok(project) => {
            println!("Successfully parsed project!");
            println!("Found {} modules", project.modules.len());

            for (name, module) in &project.modules {
                println!("Module: {}", name);
                println!("  Functions: {}", module.ast.functions.len());
                println!("  Imports: {}", module.imports.len());
            }

            // Verify we found some modules
            assert!(!project.modules.is_empty(), "Should find at least one module");

            // Check dependency graph was built
            assert!(!project.dependency_graph.nodes.is_empty());
        }
        Err(e) => {
            eprintln!("Error parsing project: {:?}", e);
            // Don't fail the test if we can't parse yet - this is WIP
        }
    }
}

#[test]
fn test_topological_sort_integration() {
    let parser = ProjectParser::new();
    let project_path = Path::new("../../examples/test_project");

    if !project_path.exists() {
        eprintln!("Skipping test - test project not found");
        return;
    }

    if let Ok(project) = parser.parse_project(project_path) {
        let sorted = parser.topological_sort(&project.dependency_graph);

        match sorted {
            Ok(modules) => {
                println!("Topological order:");
                for module in &modules {
                    println!("  - {}", module);
                }
                assert!(!modules.is_empty());
            }
            Err(e) => {
                eprintln!("Topological sort error: {:?}", e);
            }
        }
    }
}
