// Version Compatibility Demo
// Demonstrates semantic versioning and dependency version resolution

use portalis_transpiler::version_compatibility::{
    VersionCompatibilityChecker, VersionRequirement, ResolutionStrategy,
};
use semver::Version;
use std::collections::HashMap;

fn main() {
    println!("=== Version Compatibility Demo ===\n");
    println!("Demonstrates: Semantic Versioning, Conflict Resolution, Version Constraints\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Basic Version Comparison
    demo_version_comparison();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Version Requirement Satisfaction
    demo_version_requirements();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Resolving Single Crate with Multiple Requirements
    demo_single_crate_resolution();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Resolving Multiple Crates
    demo_multiple_crates_resolution();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Version Conflict Scenarios
    demo_conflict_scenarios();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Resolution Strategies
    demo_resolution_strategies();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Version compatibility demonstration complete!");
}

fn demo_version_comparison() {
    println!("\n=== Demo 1: Basic Version Comparison ===\n");

    let checker = VersionCompatibilityChecker::new();

    let comparisons = vec![
        ("1.0.0", "1.5.0"),
        ("2.0.0", "1.9.9"),
        ("1.0.0", "1.0.0"),
        ("0.1.0", "0.2.0"),
    ];

    println!("Version Comparisons:");
    for (v1, v2) in comparisons {
        match checker.get_highest(v1, v2) {
            Ok(highest) => println!("  {} vs {} → Highest: {}", v1, v2, highest),
            Err(e) => println!("  {} vs {} → Error: {}", v1, v2, e),
        }
    }

    println!("\nVersion Compatibility (same major version):");
    let compat_checks = vec![
        ("1.0.0", "1.9.9"),
        ("1.0.0", "2.0.0"),
        ("0.1.0", "0.1.9"),
        ("0.1.0", "0.2.0"),
    ];

    for (v1, v2) in compat_checks {
        let compatible = checker.are_compatible(v1, v2);
        let status = if compatible { "✅ Compatible" } else { "❌ Incompatible" };
        println!("  {} <-> {} → {}", v1, v2, status);
    }
}

fn demo_version_requirements() {
    println!("\n=== Demo 2: Version Requirement Satisfaction ===\n");

    let requirements = vec![
        ("^1.0", "1.0.0", true),
        ("^1.0", "1.5.2", true),
        ("^1.0", "2.0.0", false),
        (">=1.5", "1.4.0", false),
        (">=1.5", "1.5.0", true),
        (">=1.5", "2.0.0", true),
        ("~1.2", "1.2.0", true),
        ("~1.2", "1.2.9", true),
        ("~1.2", "1.3.0", false),
    ];

    println!("Version Requirement Satisfaction:");
    for (req, version, expected) in requirements {
        match VersionCompatibilityChecker::satisfies(version, req) {
            Ok(satisfies) => {
                let symbol = if satisfies { "✅" } else { "❌" };
                let note = if satisfies == expected { "" } else { " (UNEXPECTED!)" };
                println!("  {} satisfies {} → {}{}", version, req, symbol, note);
            }
            Err(e) => println!("  {} satisfies {} → Error: {}", version, req, e),
        }
    }
}

fn demo_single_crate_resolution() {
    println!("\n=== Demo 3: Resolving Single Crate with Multiple Requirements ===\n");

    let mut checker = VersionCompatibilityChecker::new();

    // Simulate available versions for serde
    checker.set_available_versions(
        "serde".to_string(),
        vec![
            Version::parse("1.0.0").unwrap(),
            Version::parse("1.0.5").unwrap(),
            Version::parse("1.0.10").unwrap(),
            Version::parse("1.1.0").unwrap(),
            Version::parse("1.2.0").unwrap(),
            Version::parse("2.0.0").unwrap(),
        ],
    );

    let scenarios = vec![
        vec![
            ("serde", "^1.0", "module_a"),
            ("serde", ">=1.0.5", "module_b"),
        ],
        vec![
            ("serde", "^1.0", "module_a"),
            ("serde", "^1.1", "module_b"),
        ],
        vec![
            ("serde", "1.0.10", "module_a"),
            ("serde", "^1.0", "module_b"),
        ],
    ];

    for (idx, scenario) in scenarios.iter().enumerate() {
        println!("Scenario {}: Multiple modules require 'serde'", idx + 1);

        let requirements: Vec<VersionRequirement> = scenario
            .iter()
            .map(|(crate_name, req, module)| {
                VersionRequirement::new(
                    crate_name.to_string(),
                    req.to_string(),
                    module.to_string(),
                )
            })
            .collect();

        for req in &requirements {
            println!("  - {} requires: {}", req.source_module, req.requirement);
        }

        match checker.resolve_crate("serde", requirements) {
            Ok(resolution) => {
                println!("  ✅ Resolved: {}", resolution.resolved_version);
                println!("     Strategy: {:?}", resolution.strategy);
                if resolution.had_conflict {
                    println!("     ⚠️  Had version conflicts");
                }
                for note in &resolution.notes {
                    println!("     Note: {}", note);
                }
            }
            Err(e) => {
                println!("  ❌ Failed: {}", e);
            }
        }
        println!();
    }
}

fn demo_multiple_crates_resolution() {
    println!("\n=== Demo 4: Resolving Multiple Crates ===\n");

    let mut checker = VersionCompatibilityChecker::new();

    // Set up available versions for multiple crates
    checker.set_available_versions(
        "serde".to_string(),
        vec![
            Version::parse("1.0.0").unwrap(),
            Version::parse("1.0.150").unwrap(),
        ],
    );

    checker.set_available_versions(
        "tokio".to_string(),
        vec![
            Version::parse("1.0.0").unwrap(),
            Version::parse("1.25.0").unwrap(),
            Version::parse("1.30.0").unwrap(),
        ],
    );

    checker.set_available_versions(
        "reqwest".to_string(),
        vec![
            Version::parse("0.11.0").unwrap(),
            Version::parse("0.11.20").unwrap(),
        ],
    );

    // Create requirements for multiple crates
    let mut requirements_map: HashMap<String, Vec<VersionRequirement>> = HashMap::new();

    requirements_map.insert(
        "serde".to_string(),
        vec![
            VersionRequirement::new("serde".to_string(), "^1.0".to_string(), "module_json".to_string()),
            VersionRequirement::new("serde".to_string(), ">=1.0.100".to_string(), "module_config".to_string()),
        ],
    );

    requirements_map.insert(
        "tokio".to_string(),
        vec![
            VersionRequirement::new("tokio".to_string(), "^1.20".to_string(), "module_async".to_string()),
            VersionRequirement::new("tokio".to_string(), ">=1.25".to_string(), "module_runtime".to_string()),
        ],
    );

    requirements_map.insert(
        "reqwest".to_string(),
        vec![
            VersionRequirement::new("reqwest".to_string(), "^0.11".to_string(), "module_http".to_string()),
        ],
    );

    println!("Resolving {} crates with dependencies:", requirements_map.len());
    for (crate_name, reqs) in &requirements_map {
        println!("\n  {}: {} requirements", crate_name, reqs.len());
        for req in reqs {
            println!("    - {} requires: {}", req.source_module, req.requirement);
        }
    }

    println!("\nResolution Results:");
    match checker.resolve_all(requirements_map) {
        Ok(resolutions) => {
            for (crate_name, resolution) in resolutions {
                let conflict_marker = if resolution.had_conflict { "⚠️ " } else { "✅ " };
                println!("  {}{}: {}", conflict_marker, crate_name, resolution.resolved_version);
            }
        }
        Err(errors) => {
            println!("  ❌ Resolution failed:");
            for error in errors {
                println!("     {}", error);
            }
        }
    }
}

fn demo_conflict_scenarios() {
    println!("\n=== Demo 5: Version Conflict Scenarios ===\n");

    let mut checker = VersionCompatibilityChecker::new();

    checker.set_available_versions(
        "conflicting_crate".to_string(),
        vec![
            Version::parse("1.0.0").unwrap(),
            Version::parse("2.0.0").unwrap(),
            Version::parse("3.0.0").unwrap(),
        ],
    );

    println!("Scenario 1: Incompatible major version requirements");
    let requirements = vec![
        VersionRequirement::new(
            "conflicting_crate".to_string(),
            "^1.0".to_string(),
            "old_module".to_string(),
        ),
        VersionRequirement::new(
            "conflicting_crate".to_string(),
            "^2.0".to_string(),
            "new_module".to_string(),
        ),
    ];

    println!("  - old_module requires: ^1.0");
    println!("  - new_module requires: ^2.0");

    match checker.resolve_crate("conflicting_crate", requirements) {
        Ok(resolution) => {
            println!("  ⚠️  Resolved: {} (unexpected!)", resolution.resolved_version);
        }
        Err(e) => {
            println!("  ✅ Correctly detected conflict: {}", e);
        }
    }

    println!("\nScenario 2: Exact version conflict");
    let requirements = vec![
        VersionRequirement::new(
            "conflicting_crate".to_string(),
            "1.0.0".to_string(),
            "module_a".to_string(),
        ),
        VersionRequirement::new(
            "conflicting_crate".to_string(),
            "2.0.0".to_string(),
            "module_b".to_string(),
        ),
    ];

    println!("  - module_a requires: 1.0.0 (exact)");
    println!("  - module_b requires: 2.0.0 (exact)");

    match checker.resolve_crate("conflicting_crate", requirements) {
        Ok(resolution) => {
            println!("  ⚠️  Resolved: {} (unexpected!)", resolution.resolved_version);
        }
        Err(e) => {
            println!("  ✅ Correctly detected conflict: {}", e);
        }
    }
}

fn demo_resolution_strategies() {
    println!("\n=== Demo 6: Resolution Strategies ===\n");

    // Set up available versions
    let available = vec![
        Version::parse("1.0.0").unwrap(),
        Version::parse("1.5.0").unwrap(),
        Version::parse("1.9.0").unwrap(),
    ];

    let requirement = VersionRequirement::new(
        "example_crate".to_string(),
        "^1.0".to_string(),
        "test_module".to_string(),
    );

    println!("Available versions: 1.0.0, 1.5.0, 1.9.0");
    println!("Requirement: ^1.0\n");

    let strategies = vec![
        ResolutionStrategy::Highest,
        ResolutionStrategy::Lowest,
        ResolutionStrategy::Compatible,
    ];

    for strategy in strategies {
        let mut checker = VersionCompatibilityChecker::with_strategy(strategy);
        checker.set_available_versions("example_crate".to_string(), available.clone());

        match checker.resolve_crate("example_crate", vec![requirement.clone()]) {
            Ok(resolution) => {
                println!("  {:?}: {}", strategy, resolution.resolved_version);
            }
            Err(e) => {
                println!("  {:?}: Error - {}", strategy, e);
            }
        }
    }

    println!("\nExplanation:");
    println!("  - Highest: Selects the newest compatible version (1.9.0)");
    println!("  - Lowest: Selects the oldest compatible version (1.0.0)");
    println!("  - Compatible: Same as Highest by default");
}
