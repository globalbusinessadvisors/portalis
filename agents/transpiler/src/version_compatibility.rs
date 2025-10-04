//! Version Compatibility - Semantic versioning and dependency version resolution
//!
//! This module provides:
//! 1. Semantic version parsing and comparison using semver crate
//! 2. Version constraint resolution (^, ~, >=, etc.)
//! 3. Dependency version conflict detection and resolution
//! 4. Compatible version range calculation
//! 5. Minimum version selection strategies

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Version resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Use the highest compatible version
    Highest,
    /// Use the lowest compatible version
    Lowest,
    /// Use exact version matching
    Exact,
    /// Use version that satisfies all constraints
    Compatible,
}

impl Default for ResolutionStrategy {
    fn default() -> Self {
        Self::Highest
    }
}

/// Version requirement with source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionRequirement {
    /// Crate name
    pub crate_name: String,
    /// Version requirement string (e.g., "^1.0", ">=0.5", "1.2.3")
    pub requirement: String,
    /// Parsed version requirement
    #[serde(skip)]
    pub parsed_req: Option<VersionReq>,
    /// Source module that requires this version
    pub source_module: String,
    /// Optional: specific version if known
    pub exact_version: Option<Version>,
}

impl VersionRequirement {
    /// Create new version requirement
    pub fn new(crate_name: String, requirement: String, source_module: String) -> Self {
        let parsed_req = VersionReq::parse(&requirement).ok();
        let exact_version = Version::parse(&requirement).ok();

        Self {
            crate_name,
            requirement,
            parsed_req,
            source_module,
            exact_version,
        }
    }

    /// Check if a version satisfies this requirement
    pub fn satisfies(&self, version: &Version) -> bool {
        if let Some(ref req) = self.parsed_req {
            req.matches(version)
        } else if let Some(ref exact) = self.exact_version {
            exact == version
        } else {
            // Fallback: exact string match
            self.requirement == version.to_string()
        }
    }
}

/// Version conflict resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionResolution {
    /// Crate name
    pub crate_name: String,
    /// Resolved version
    pub resolved_version: Version,
    /// All requirements that contributed
    pub requirements: Vec<VersionRequirement>,
    /// Whether there was a conflict
    pub had_conflict: bool,
    /// Resolution strategy used
    pub strategy: ResolutionStrategy,
    /// Notes about the resolution
    pub notes: Vec<String>,
}

/// Version compatibility checker
pub struct VersionCompatibilityChecker {
    /// Resolution strategy
    strategy: ResolutionStrategy,
    /// Known available versions for crates (for simulation/testing)
    available_versions: HashMap<String, Vec<Version>>,
}

impl VersionCompatibilityChecker {
    /// Create new checker with default strategy
    pub fn new() -> Self {
        Self {
            strategy: ResolutionStrategy::default(),
            available_versions: HashMap::new(),
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            available_versions: HashMap::new(),
        }
    }

    /// Set available versions for a crate (useful for testing)
    pub fn set_available_versions(&mut self, crate_name: String, versions: Vec<Version>) {
        self.available_versions.insert(crate_name, versions);
    }

    /// Resolve version requirements for a single crate
    pub fn resolve_crate(
        &self,
        crate_name: &str,
        requirements: Vec<VersionRequirement>,
    ) -> Result<VersionResolution, String> {
        if requirements.is_empty() {
            return Err(format!("No requirements provided for {}", crate_name));
        }

        // Collect all version constraints
        let mut constraints: Vec<&VersionReq> = Vec::new();
        let mut exact_versions: Vec<&Version> = Vec::new();
        let mut notes = Vec::new();

        for req in &requirements {
            if let Some(ref parsed) = req.parsed_req {
                constraints.push(parsed);
            } else if let Some(ref exact) = req.exact_version {
                exact_versions.push(exact);
            } else {
                notes.push(format!(
                    "Could not parse version requirement '{}' from {}",
                    req.requirement, req.source_module
                ));
            }
        }

        // Check if there are conflicting exact versions
        if exact_versions.len() > 1 {
            let unique: std::collections::HashSet<_> = exact_versions.iter().collect();
            if unique.len() > 1 {
                let conflict_str: Vec<String> = exact_versions.iter().map(|v| v.to_string()).collect();
                return Err(format!(
                    "Conflicting exact versions for {}: {}",
                    crate_name,
                    conflict_str.join(", ")
                ));
            }
        }

        // If we have an exact version and it satisfies all constraints, use it
        if let Some(&exact) = exact_versions.first() {
            let satisfies_all = constraints.iter().all(|req| req.matches(exact));
            if satisfies_all {
                return Ok(VersionResolution {
                    crate_name: crate_name.to_string(),
                    resolved_version: exact.clone(),
                    requirements,
                    had_conflict: false,
                    strategy: ResolutionStrategy::Exact,
                    notes,
                });
            } else {
                notes.push(format!(
                    "Exact version {} does not satisfy all constraints",
                    exact
                ));
            }
        }

        // Find a version that satisfies all constraints
        let resolved_version = if let Some(versions) = self.available_versions.get(crate_name) {
            // Use provided available versions
            self.find_compatible_version(versions, &constraints)?
        } else {
            // Synthesize a compatible version from requirements
            self.synthesize_compatible_version(&requirements, &constraints)?
        };

        let had_conflict = requirements.len() > 1 || !notes.is_empty();

        Ok(VersionResolution {
            crate_name: crate_name.to_string(),
            resolved_version,
            requirements,
            had_conflict,
            strategy: self.strategy,
            notes,
        })
    }

    /// Find a compatible version from available versions
    fn find_compatible_version(
        &self,
        available: &[Version],
        constraints: &[&VersionReq],
    ) -> Result<Version, String> {
        // Filter versions that satisfy all constraints
        let mut compatible: Vec<Version> = available
            .iter()
            .filter(|v| constraints.iter().all(|req| req.matches(v)))
            .cloned()
            .collect();

        if compatible.is_empty() {
            return Err(format!(
                "No compatible version found. Constraints: {:?}",
                constraints.iter().map(|r| r.to_string()).collect::<Vec<_>>()
            ));
        }

        // Sort based on strategy
        compatible.sort();
        match self.strategy {
            ResolutionStrategy::Highest | ResolutionStrategy::Compatible => {
                Ok(compatible.last().unwrap().clone())
            }
            ResolutionStrategy::Lowest => Ok(compatible.first().unwrap().clone()),
            ResolutionStrategy::Exact => Ok(compatible.first().unwrap().clone()),
        }
    }

    /// Synthesize a compatible version from requirements
    fn synthesize_compatible_version(
        &self,
        requirements: &[VersionRequirement],
        constraints: &[&VersionReq],
    ) -> Result<Version, String> {
        // Try to extract a version from the first parseable requirement
        for req in requirements {
            if let Ok(version) = Version::parse(&req.requirement) {
                // Check if it satisfies all constraints
                if constraints.iter().all(|c| c.matches(&version)) {
                    return Ok(version);
                }
            }
        }

        // Try to find the highest minimum version from constraints
        let mut candidates = Vec::new();
        for req in requirements {
            // Parse common patterns: "^1.0", "~1.2", ">=1.5", "1.2.3"
            let req_str = &req.requirement;

            if let Ok(version) = Version::parse(req_str) {
                candidates.push(version);
            } else if req_str.starts_with('^') {
                if let Ok(version) = Version::parse(&req_str[1..]) {
                    candidates.push(version);
                }
            } else if req_str.starts_with('~') {
                if let Ok(version) = Version::parse(&req_str[1..]) {
                    candidates.push(version);
                }
            } else if req_str.starts_with(">=") {
                if let Ok(version) = Version::parse(&req_str[2..].trim()) {
                    candidates.push(version);
                }
            }
        }

        if candidates.is_empty() {
            return Err("Could not determine compatible version from requirements".to_string());
        }

        // Sort and pick based on strategy
        candidates.sort();
        match self.strategy {
            ResolutionStrategy::Highest | ResolutionStrategy::Compatible => {
                Ok(candidates.last().unwrap().clone())
            }
            ResolutionStrategy::Lowest => Ok(candidates.first().unwrap().clone()),
            ResolutionStrategy::Exact => Ok(candidates.first().unwrap().clone()),
        }
    }

    /// Resolve all dependencies with version requirements
    pub fn resolve_all(
        &self,
        requirements_map: HashMap<String, Vec<VersionRequirement>>,
    ) -> Result<HashMap<String, VersionResolution>, Vec<String>> {
        let mut resolutions = HashMap::new();
        let mut errors = Vec::new();

        for (crate_name, requirements) in requirements_map {
            match self.resolve_crate(&crate_name, requirements) {
                Ok(resolution) => {
                    resolutions.insert(crate_name, resolution);
                }
                Err(e) => {
                    errors.push(format!("{}: {}", crate_name, e));
                }
            }
        }

        if errors.is_empty() {
            Ok(resolutions)
        } else {
            Err(errors)
        }
    }

    /// Check if two versions are compatible
    pub fn are_compatible(&self, v1: &str, v2: &str) -> bool {
        let version1 = match Version::parse(v1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let version2 = match Version::parse(v2) {
            Ok(v) => v,
            Err(_) => return false,
        };

        // Versions are compatible if they have the same major version (for >= 1.0.0)
        // or same major and minor version (for < 1.0.0)
        if version1.major >= 1 && version2.major >= 1 {
            version1.major == version2.major
        } else {
            version1.major == version2.major && version1.minor == version2.minor
        }
    }

    /// Get the highest version between two versions
    pub fn get_highest(&self, v1: &str, v2: &str) -> Result<String, String> {
        let version1 = Version::parse(v1)
            .map_err(|e| format!("Invalid version {}: {}", v1, e))?;
        let version2 = Version::parse(v2)
            .map_err(|e| format!("Invalid version {}: {}", v2, e))?;

        if version1 >= version2 {
            Ok(v1.to_string())
        } else {
            Ok(v2.to_string())
        }
    }

    /// Get the lowest version between two versions
    pub fn get_lowest(&self, v1: &str, v2: &str) -> Result<String, String> {
        let version1 = Version::parse(v1)
            .map_err(|e| format!("Invalid version {}: {}", v1, e))?;
        let version2 = Version::parse(v2)
            .map_err(|e| format!("Invalid version {}: {}", v2, e))?;

        if version1 <= version2 {
            Ok(v1.to_string())
        } else {
            Ok(v2.to_string())
        }
    }

    /// Parse version requirement string to VersionReq
    pub fn parse_requirement(requirement: &str) -> Result<VersionReq, String> {
        VersionReq::parse(requirement)
            .map_err(|e| format!("Failed to parse version requirement '{}': {}", requirement, e))
    }

    /// Check if version satisfies requirement
    pub fn satisfies(version: &str, requirement: &str) -> Result<bool, String> {
        let v = Version::parse(version)
            .map_err(|e| format!("Invalid version {}: {}", version, e))?;
        let req = Self::parse_requirement(requirement)?;
        Ok(req.matches(&v))
    }
}

impl Default for VersionCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let req = VersionRequirement::new(
            "serde".to_string(),
            "1.0.0".to_string(),
            "test_module".to_string(),
        );
        assert!(req.exact_version.is_some());
        assert_eq!(req.exact_version.unwrap().to_string(), "1.0.0");
    }

    #[test]
    fn test_version_satisfies() {
        let req = VersionRequirement::new(
            "serde".to_string(),
            "^1.0".to_string(),
            "test_module".to_string(),
        );

        let v1_0 = Version::parse("1.0.0").unwrap();
        let v1_5 = Version::parse("1.5.2").unwrap();
        let v2_0 = Version::parse("2.0.0").unwrap();

        assert!(req.satisfies(&v1_0));
        assert!(req.satisfies(&v1_5));
        assert!(!req.satisfies(&v2_0));
    }

    #[test]
    fn test_are_compatible() {
        let checker = VersionCompatibilityChecker::new();

        assert!(checker.are_compatible("1.0.0", "1.5.0"));
        assert!(!checker.are_compatible("1.0.0", "2.0.0"));
        assert!(checker.are_compatible("0.1.0", "0.1.5"));
        assert!(!checker.are_compatible("0.1.0", "0.2.0"));
    }

    #[test]
    fn test_get_highest() {
        let checker = VersionCompatibilityChecker::new();

        assert_eq!(checker.get_highest("1.0.0", "1.5.0").unwrap(), "1.5.0");
        assert_eq!(checker.get_highest("2.0.0", "1.5.0").unwrap(), "2.0.0");
        assert_eq!(checker.get_highest("1.0.0", "1.0.0").unwrap(), "1.0.0");
    }

    #[test]
    fn test_resolve_single_requirement() {
        let checker = VersionCompatibilityChecker::new();

        let req = VersionRequirement::new(
            "serde".to_string(),
            "1.0.0".to_string(),
            "test_module".to_string(),
        );

        let resolution = checker.resolve_crate("serde", vec![req]).unwrap();

        assert_eq!(resolution.resolved_version.to_string(), "1.0.0");
        assert!(!resolution.had_conflict);
    }

    #[test]
    fn test_resolve_multiple_compatible_requirements() {
        let mut checker = VersionCompatibilityChecker::new();

        // Set available versions
        checker.set_available_versions(
            "serde".to_string(),
            vec![
                Version::parse("1.0.0").unwrap(),
                Version::parse("1.0.5").unwrap(),
                Version::parse("1.1.0").unwrap(),
                Version::parse("2.0.0").unwrap(),
            ],
        );

        let req1 = VersionRequirement::new(
            "serde".to_string(),
            "^1.0".to_string(),
            "module_a".to_string(),
        );
        let req2 = VersionRequirement::new(
            "serde".to_string(),
            ">=1.0.5".to_string(),
            "module_b".to_string(),
        );

        let resolution = checker.resolve_crate("serde", vec![req1, req2]).unwrap();

        // Should pick 1.1.0 (highest that satisfies both ^1.0 and >=1.0.5)
        assert_eq!(resolution.resolved_version.to_string(), "1.1.0");
        assert!(resolution.had_conflict);
    }

    #[test]
    fn test_resolve_incompatible_requirements() {
        let mut checker = VersionCompatibilityChecker::new();

        checker.set_available_versions(
            "serde".to_string(),
            vec![
                Version::parse("1.0.0").unwrap(),
                Version::parse("2.0.0").unwrap(),
            ],
        );

        let req1 = VersionRequirement::new(
            "serde".to_string(),
            "^1.0".to_string(),
            "module_a".to_string(),
        );
        let req2 = VersionRequirement::new(
            "serde".to_string(),
            "^2.0".to_string(),
            "module_b".to_string(),
        );

        let result = checker.resolve_crate("serde", vec![req1, req2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolution_strategy_lowest() {
        let mut checker = VersionCompatibilityChecker::with_strategy(ResolutionStrategy::Lowest);

        checker.set_available_versions(
            "serde".to_string(),
            vec![
                Version::parse("1.0.0").unwrap(),
                Version::parse("1.0.5").unwrap(),
                Version::parse("1.1.0").unwrap(),
            ],
        );

        let req = VersionRequirement::new(
            "serde".to_string(),
            "^1.0".to_string(),
            "test_module".to_string(),
        );

        let resolution = checker.resolve_crate("serde", vec![req]).unwrap();

        // Should pick 1.0.0 (lowest that satisfies ^1.0)
        assert_eq!(resolution.resolved_version.to_string(), "1.0.0");
    }

    #[test]
    fn test_satisfies_static() {
        assert!(VersionCompatibilityChecker::satisfies("1.5.0", "^1.0").unwrap());
        assert!(VersionCompatibilityChecker::satisfies("1.0.0", ">=1.0").unwrap());
        assert!(!VersionCompatibilityChecker::satisfies("0.9.0", "^1.0").unwrap());
    }

    #[test]
    fn test_parse_requirement() {
        assert!(VersionCompatibilityChecker::parse_requirement("^1.0").is_ok());
        assert!(VersionCompatibilityChecker::parse_requirement(">=1.0.0").is_ok());
        assert!(VersionCompatibilityChecker::parse_requirement("~1.2.3").is_ok());
        assert!(VersionCompatibilityChecker::parse_requirement("1.0.0").is_ok());
    }
}
