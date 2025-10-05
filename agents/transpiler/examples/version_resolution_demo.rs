//! Version Resolution System Examples
//!
//! Demonstrates dependency resolution with semantic versioning,
//! constraint satisfaction, conflict detection, and build order calculation.

use portalis_transpiler::version_resolver::*;

fn main() {
    println!("=== Dependency Resolution System Examples ===\n");

    // Example 1: Semantic versioning
    example_semantic_versioning();

    // Example 2: Version constraints
    example_version_constraints();

    // Example 3: Simple dependency resolution
    example_simple_resolution();

    // Example 4: Conflict detection
    example_conflict_detection();

    // Example 5: Transitive dependencies
    example_transitive_dependencies();

    // Example 6: Build order calculation
    example_build_order();

    // Example 7: Lock file generation
    example_lock_file();

    // Example 8: Complex resolution scenario
    example_complex_resolution();
}

fn example_semantic_versioning() {
    println!("## Example 1: Semantic Versioning\n");

    let versions = vec![
        "1.0.0",
        "1.0.1",
        "1.1.0",
        "1.2.3",
        "2.0.0",
        "2.0.0-beta.1",
    ];

    println!("Version parsing and comparison:");
    let parsed: Vec<Version> = versions
        .iter()
        .map(|v| Version::parse(v).unwrap())
        .collect();

    for (i, v) in parsed.iter().enumerate() {
        println!("  {} → Version {{ major: {}, minor: {}, patch: {} }}",
            versions[i], v.major, v.minor, v.patch);
    }

    println!("\nVersion ordering (ascending):");
    let mut sorted = parsed.clone();
    sorted.sort();
    for v in &sorted {
        println!("  {}", v);
    }

    println!("\nPre-release versions:");
    let stable = Version::new(2, 0, 0);
    let beta = Version::parse("2.0.0-beta.1").unwrap();
    println!("  {} < {} = {}", beta, stable, beta < stable);
    println!("  Pre-releases have lower precedence than stable");

    println!("\n{}\n", "=".repeat(80));
}

fn example_version_constraints() {
    println!("## Example 2: Version Constraints\n");

    let test_version = Version::new(1, 2, 3);

    let constraints = vec![
        ("^1.2.3", "Caret: >=1.2.3, <2.0.0"),
        ("~1.2.3", "Tilde: >=1.2.3, <1.3.0"),
        (">=1.0.0", "Greater or equal"),
        ("<2.0.0", "Less than"),
        ("1.2.*", "Wildcard: any 1.2.x"),
        ("=1.2.3", "Exact match"),
        ("*", "Any version"),
    ];

    println!("Testing version 1.2.3 against constraints:\n");
    println!("{:<15} {:<30} {}", "Constraint", "Meaning", "Match");
    println!("{}", "-".repeat(60));

    for (constraint_str, description) in constraints {
        let constraint = VersionConstraint::parse(constraint_str).unwrap();
        let matches = constraint.matches(&test_version);
        let symbol = if matches { "✓" } else { "✗" };
        println!("{:<15} {:<30} {}", constraint_str, description, symbol);
    }

    println!("\nCaret (^) rules:");
    println!("  ^1.2.3 allows: 1.2.3, 1.2.4, 1.3.0, 1.9.9");
    println!("  ^1.2.3 blocks: 2.0.0, 0.9.9");
    println!("  ^0.1.2 allows: 0.1.2, 0.1.3 (special case for 0.x)");

    println!("\nTilde (~) rules:");
    println!("  ~1.2.3 allows: 1.2.3, 1.2.4, 1.2.99");
    println!("  ~1.2.3 blocks: 1.3.0, 2.0.0");

    println!("\n{}\n", "=".repeat(80));
}

fn example_simple_resolution() {
    println!("## Example 3: Simple Dependency Resolution\n");

    let mut resolver = VersionResolver::new();

    // Register available packages
    resolver.add_package(Package {
        name: "serde".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![],
    });

    resolver.add_package(Package {
        name: "serde".to_string(),
        version: Version::new(1, 0, 150),
        dependencies: vec![],
    });

    resolver.add_package(Package {
        name: "tokio".to_string(),
        version: Version::new(1, 28, 0),
        dependencies: vec![],
    });

    println!("Available packages:");
    println!("  serde: 1.0.0, 1.0.150");
    println!("  tokio: 1.28.0");

    println!("\nRequesting dependencies:");
    println!("  serde ^1.0.0");
    println!("  tokio ^1.28.0");

    let deps = vec![
        Dependency::new("serde", VersionConstraint::Caret(Version::new(1, 0, 0))),
        Dependency::new("tokio", VersionConstraint::Caret(Version::new(1, 28, 0))),
    ];

    let result = resolver.resolve(deps);

    println!("\nResolved:");
    for pkg in &result.packages {
        println!("  {} = {}", pkg.name, pkg.version);
    }

    println!("\nNote: Resolver selected highest compatible version (serde 1.0.150)");

    println!("\n{}\n", "=".repeat(80));
}

fn example_conflict_detection() {
    println!("## Example 4: Conflict Detection\n");

    let mut resolver = VersionResolver::new();

    // Add conflicting versions
    resolver.add_package(Package {
        name: "http".to_string(),
        version: Version::new(0, 2, 0),
        dependencies: vec![],
    });

    resolver.add_package(Package {
        name: "http".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![],
    });

    println!("Package A requires: http ^0.2.0");
    println!("Package B requires: http ^1.0.0");
    println!("These constraints are incompatible!\n");

    let deps = vec![
        Dependency::new("http", VersionConstraint::Caret(Version::new(0, 2, 0))),
    ];

    let result1 = resolver.resolve(deps);

    let deps = vec![
        Dependency::new("http", VersionConstraint::Caret(Version::new(1, 0, 0))),
    ];

    let result2 = resolver.resolve(deps);

    println!("Resolution 1: http = {}", result1.packages[0].version);
    println!("Resolution 2: http = {}", result2.packages[0].version);

    println!("\nConflict scenario:");
    println!("If both constraints are required simultaneously, resolution fails.");
    println!("Cargo handles this with duplicate dependencies (different major versions).");

    println!("\n{}\n", "=".repeat(80));
}

fn example_transitive_dependencies() {
    println!("## Example 5: Transitive Dependencies\n");

    let mut resolver = VersionResolver::new();

    // Package A depends on B
    resolver.add_package(Package {
        name: "app".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![
            Dependency::new("web-framework", VersionConstraint::Caret(Version::new(2, 0, 0))),
        ],
    });

    // Package B depends on C and D
    resolver.add_package(Package {
        name: "web-framework".to_string(),
        version: Version::new(2, 0, 0),
        dependencies: vec![
            Dependency::new("router", VersionConstraint::Caret(Version::new(1, 0, 0))),
            Dependency::new("middleware", VersionConstraint::Caret(Version::new(1, 0, 0))),
        ],
    });

    // Leaf dependencies
    resolver.add_package(Package {
        name: "router".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![],
    });

    resolver.add_package(Package {
        name: "middleware".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![],
    });

    println!("Dependency tree:");
    println!("  app (1.0.0)");
    println!("  └── web-framework (^2.0.0)");
    println!("      ├── router (^1.0.0)");
    println!("      └── middleware (^1.0.0)");

    let deps = vec![
        Dependency::new("app", VersionConstraint::Exact(Version::new(1, 0, 0))),
    ];

    let result = resolver.resolve(deps);

    println!("\nResolved packages:");
    for pkg in &result.packages {
        println!("  {} = {}", pkg.name, pkg.version);
    }

    println!("\nTransitive resolution:");
    println!("  ✓ Automatically resolved all indirect dependencies");
    println!("  ✓ No manual intervention needed");

    println!("\n{}\n", "=".repeat(80));
}

fn example_build_order() {
    println!("## Example 6: Build Order Calculation\n");

    let mut resolver = VersionResolver::new();

    // Create dependency chain: A -> B -> C
    resolver.add_package(Package {
        name: "app".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![
            Dependency::new("lib-b", VersionConstraint::Any),
        ],
    });

    resolver.add_package(Package {
        name: "lib-b".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![
            Dependency::new("lib-c", VersionConstraint::Any),
        ],
    });

    resolver.add_package(Package {
        name: "lib-c".to_string(),
        version: Version::new(1, 0, 0),
        dependencies: vec![],
    });

    println!("Dependency graph:");
    println!("  app → lib-b → lib-c");

    let deps = vec![Dependency::new("app", VersionConstraint::Any)];
    let result = resolver.resolve(deps);

    println!("\nBuild order (topological sort):");
    for (i, pkg_name) in result.build_order.iter().enumerate() {
        println!("  {}. {}", i + 1, pkg_name);
    }

    println!("\nExplanation:");
    println!("  lib-c has no dependencies → build first");
    println!("  lib-b depends on lib-c → build second");
    println!("  app depends on lib-b → build last");

    println!("\n{}\n", "=".repeat(80));
}

fn example_lock_file() {
    println!("## Example 7: Lock File Generation\n");

    let mut resolver = VersionResolver::new();

    resolver.add_package(Package {
        name: "serde".to_string(),
        version: Version::new(1, 0, 150),
        dependencies: vec![],
    });

    resolver.add_package(Package {
        name: "tokio".to_string(),
        version: Version::new(1, 28, 0),
        dependencies: vec![
            Dependency::new("mio", VersionConstraint::Caret(Version::new(0, 8, 0))),
        ],
    });

    resolver.add_package(Package {
        name: "mio".to_string(),
        version: Version::new(0, 8, 6),
        dependencies: vec![],
    });

    let deps = vec![
        Dependency::new("serde", VersionConstraint::Caret(Version::new(1, 0, 0))),
        Dependency::new("tokio", VersionConstraint::Caret(Version::new(1, 28, 0))),
    ];

    let result = resolver.resolve(deps);
    let lockfile = resolver.generate_lockfile(&result);

    println!("Generated lock file:\n");
    println!("{}", lockfile);

    println!("Lock file purpose:");
    println!("  ✓ Ensures reproducible builds");
    println!("  ✓ Records exact resolved versions");
    println!("  ✓ Prevents version drift over time");
    println!("  ✓ Shared across team/CI");

    println!("\n{}\n", "=".repeat(80));
}

fn example_complex_resolution() {
    println!("## Example 8: Complex Resolution Scenario\n");

    let mut resolver = VersionResolver::new();

    // Multiple versions of same package
    for patch in 0..5 {
        resolver.add_package(Package {
            name: "common-lib".to_string(),
            version: Version::new(2, 5, patch),
            dependencies: vec![],
        });
    }

    // Package ecosystem
    resolver.add_package(Package {
        name: "web-server".to_string(),
        version: Version::new(3, 0, 0),
        dependencies: vec![
            Dependency::new("common-lib", VersionConstraint::Caret(Version::new(2, 5, 0))),
            Dependency::new("http-client", VersionConstraint::Tilde(Version::new(1, 2, 0))),
        ],
    });

    resolver.add_package(Package {
        name: "http-client".to_string(),
        version: Version::new(1, 2, 0),
        dependencies: vec![
            Dependency::new("common-lib", VersionConstraint::GreaterOrEqual(Version::new(2, 0, 0))),
        ],
    });

    resolver.add_package(Package {
        name: "http-client".to_string(),
        version: Version::new(1, 2, 5),
        dependencies: vec![
            Dependency::new("common-lib", VersionConstraint::GreaterOrEqual(Version::new(2, 0, 0))),
        ],
    });

    println!("Complex dependency scenario:");
    println!("  web-server 3.0.0:");
    println!("    - common-lib ^2.5.0");
    println!("    - http-client ~1.2.0");
    println!("  http-client 1.2.0, 1.2.5:");
    println!("    - common-lib >=2.0.0");
    println!("  common-lib: 2.5.0, 2.5.1, 2.5.2, 2.5.3, 2.5.4");

    let deps = vec![
        Dependency::new("web-server", VersionConstraint::Exact(Version::new(3, 0, 0))),
    ];

    let result = resolver.resolve(deps);

    println!("\nResolved:");
    for pkg in &result.packages {
        println!("  {} = {}", pkg.name, pkg.version);
    }

    println!("\nResolution strategy:");
    println!("  1. web-server requires common-lib ^2.5.0 → picks 2.5.4 (highest)");
    println!("  2. web-server requires http-client ~1.2.0 → picks 1.2.5 (highest in range)");
    println!("  3. http-client requires common-lib >=2.0.0 → satisfied by 2.5.4");
    println!("  4. All constraints satisfied ✓");

    println!("\n{}\n", "=".repeat(80));
}
