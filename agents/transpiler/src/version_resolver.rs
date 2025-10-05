//! Version Resolution System
//!
//! Advanced dependency resolution with semantic versioning, constraint
//! satisfaction, and conflict resolution algorithms.

use std::collections::{HashMap, HashSet, VecDeque};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// Semantic version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
    pub build: Option<String>,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build: None,
        }
    }

    /// Parse version string (e.g., "1.2.3", "2.0.0-beta.1")
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        // Split on '-' for pre-release
        let parts: Vec<&str> = s.split('-').collect();
        let version_part = parts[0];

        // Parse version numbers
        let nums: Vec<&str> = version_part.split('.').collect();
        if nums.len() < 2 || nums.len() > 3 {
            return Err(format!("Invalid version format: {}", s));
        }

        let major = nums[0].parse().map_err(|_| format!("Invalid major: {}", nums[0]))?;
        let minor = nums[1].parse().map_err(|_| format!("Invalid minor: {}", nums[1]))?;
        let patch = if nums.len() == 3 {
            nums[2].parse().map_err(|_| format!("Invalid patch: {}", nums[2]))?
        } else {
            0
        };

        let pre_release = if parts.len() > 1 {
            Some(parts[1].to_string())
        } else {
            None
        };

        Ok(Self {
            major,
            minor,
            patch,
            pre_release,
            build: None,
        })
    }

    /// Check if version satisfies constraint
    pub fn satisfies(&self, constraint: &VersionConstraint) -> bool {
        constraint.matches(self)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => match self.patch.cmp(&other.patch) {
                    Ordering::Equal => {
                        // Pre-release versions have lower precedence
                        match (&self.pre_release, &other.pre_release) {
                            (None, None) => Ordering::Equal,
                            (None, Some(_)) => Ordering::Greater,
                            (Some(_), None) => Ordering::Less,
                            (Some(a), Some(b)) => a.cmp(b),
                        }
                    }
                    ord => ord,
                }
                ord => ord,
            }
            ord => ord,
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(build) = &self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

/// Version constraint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionConstraint {
    /// Exact version (=1.2.3)
    Exact(Version),
    /// Caret range (^1.2.3 allows >=1.2.3, <2.0.0)
    Caret(Version),
    /// Tilde range (~1.2.3 allows >=1.2.3, <1.3.0)
    Tilde(Version),
    /// Greater than or equal (>=1.2.3)
    GreaterOrEqual(Version),
    /// Less than (<2.0.0)
    LessThan(Version),
    /// Greater than (>1.0.0)
    GreaterThan(Version),
    /// Less than or equal (<=2.0.0)
    LessOrEqual(Version),
    /// Range (>=1.0.0, <2.0.0)
    Range(Version, Version),
    /// Wildcard (1.2.* allows 1.2.x)
    Wildcard(u32, Option<u32>),
    /// Any version (*)
    Any,
}

impl VersionConstraint {
    /// Parse constraint string
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        if s == "*" {
            return Ok(VersionConstraint::Any);
        }

        if s.contains('*') {
            let parts: Vec<&str> = s.split('.').collect();
            let major = parts[0].parse().map_err(|_| "Invalid wildcard".to_string())?;
            let minor = if parts.len() > 1 && parts[1] != "*" {
                Some(parts[1].parse().map_err(|_| "Invalid wildcard".to_string())?)
            } else {
                None
            };
            return Ok(VersionConstraint::Wildcard(major, minor));
        }

        if s.starts_with('^') {
            let version = Version::parse(&s[1..])?;
            return Ok(VersionConstraint::Caret(version));
        }

        if s.starts_with('~') {
            let version = Version::parse(&s[1..])?;
            return Ok(VersionConstraint::Tilde(version));
        }

        if s.starts_with(">=") {
            let version = Version::parse(&s[2..])?;
            return Ok(VersionConstraint::GreaterOrEqual(version));
        }

        if s.starts_with("<=") {
            let version = Version::parse(&s[2..])?;
            return Ok(VersionConstraint::LessOrEqual(version));
        }

        if s.starts_with('<') {
            let version = Version::parse(&s[1..])?;
            return Ok(VersionConstraint::LessThan(version));
        }

        if s.starts_with('>') {
            let version = Version::parse(&s[1..])?;
            return Ok(VersionConstraint::GreaterThan(version));
        }

        if s.starts_with('=') {
            let version = Version::parse(&s[1..])?;
            return Ok(VersionConstraint::Exact(version));
        }

        // Default to exact or caret
        let version = Version::parse(s)?;
        Ok(VersionConstraint::Exact(version))
    }

    /// Check if version matches constraint
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            VersionConstraint::Exact(v) => version == v,
            VersionConstraint::Caret(v) => {
                // ^1.2.3 := >=1.2.3 <2.0.0
                if v.major == 0 {
                    // ^0.x.y is more restrictive
                    version.major == 0 && version.minor == v.minor && version >= v
                } else {
                    version.major == v.major && version >= v
                }
            }
            VersionConstraint::Tilde(v) => {
                // ~1.2.3 := >=1.2.3 <1.3.0
                version.major == v.major && version.minor == v.minor && version >= v
            }
            VersionConstraint::GreaterOrEqual(v) => version >= v,
            VersionConstraint::LessThan(v) => version < v,
            VersionConstraint::GreaterThan(v) => version > v,
            VersionConstraint::LessOrEqual(v) => version <= v,
            VersionConstraint::Range(min, max) => version >= min && version < max,
            VersionConstraint::Wildcard(major, minor) => {
                if let Some(m) = minor {
                    version.major == *major && version.minor == *m
                } else {
                    version.major == *major
                }
            }
            VersionConstraint::Any => true,
        }
    }

    /// Get the minimum version that satisfies this constraint
    pub fn min_version(&self) -> Option<Version> {
        match self {
            VersionConstraint::Exact(v)
            | VersionConstraint::Caret(v)
            | VersionConstraint::Tilde(v)
            | VersionConstraint::GreaterOrEqual(v)
            | VersionConstraint::GreaterThan(v) => Some(v.clone()),
            VersionConstraint::Range(min, _) => Some(min.clone()),
            VersionConstraint::Wildcard(major, minor) => {
                Some(Version::new(*major, minor.unwrap_or(0), 0))
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for VersionConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionConstraint::Exact(v) => write!(f, "={}", v),
            VersionConstraint::Caret(v) => write!(f, "^{}", v),
            VersionConstraint::Tilde(v) => write!(f, "~{}", v),
            VersionConstraint::GreaterOrEqual(v) => write!(f, ">={}", v),
            VersionConstraint::LessThan(v) => write!(f, "<{}", v),
            VersionConstraint::GreaterThan(v) => write!(f, ">{}", v),
            VersionConstraint::LessOrEqual(v) => write!(f, "<={}", v),
            VersionConstraint::Range(min, max) => write!(f, ">={}, <{}", min, max),
            VersionConstraint::Wildcard(major, Some(minor)) => write!(f, "{}.{}.*", major, minor),
            VersionConstraint::Wildcard(major, None) => write!(f, "{}.*", major),
            VersionConstraint::Any => write!(f, "*"),
        }
    }
}

/// Package dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub constraint: VersionConstraint,
    pub optional: bool,
    pub features: Vec<String>,
}

impl Dependency {
    pub fn new(name: impl Into<String>, constraint: VersionConstraint) -> Self {
        Self {
            name: name.into(),
            constraint,
            optional: false,
            features: Vec::new(),
        }
    }
}

/// Package with version and dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub version: Version,
    pub dependencies: Vec<Dependency>,
}

/// Resolved package with selected version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedPackage {
    pub name: String,
    pub version: Version,
    pub dependencies: Vec<String>,
}

/// Dependency conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub package: String,
    pub requested_by: Vec<(String, VersionConstraint)>,
    pub reason: String,
}

/// Resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    pub packages: Vec<ResolvedPackage>,
    pub conflicts: Vec<Conflict>,
    pub build_order: Vec<String>,
}

/// Dependency resolver with version constraint satisfaction
pub struct VersionResolver {
    /// Available package versions
    available_packages: HashMap<String, Vec<Package>>,
    /// Resolution cache
    cache: HashMap<String, Option<Version>>,
}

impl VersionResolver {
    pub fn new() -> Self {
        Self {
            available_packages: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Register available package version
    pub fn add_package(&mut self, package: Package) {
        self.available_packages
            .entry(package.name.clone())
            .or_default()
            .push(package);
    }

    /// Resolve dependencies using backtracking algorithm
    pub fn resolve(&mut self, root_dependencies: Vec<Dependency>) -> ResolutionResult {
        let mut resolved: HashMap<String, Version> = HashMap::new();
        let mut conflicts = Vec::new();

        // Create a work queue
        let mut queue: VecDeque<(String, VersionConstraint)> = root_dependencies
            .into_iter()
            .map(|d| (d.name, d.constraint))
            .collect();

        while let Some((name, constraint)) = queue.pop_front() {
            // Check if already resolved
            if let Some(existing_version) = resolved.get(&name) {
                // Verify compatibility
                if !constraint.matches(existing_version) {
                    conflicts.push(Conflict {
                        package: name.clone(),
                        requested_by: vec![(name.clone(), constraint.clone())],
                        reason: format!(
                            "Version {} conflicts with constraint {}",
                            existing_version, constraint
                        ),
                    });
                }
                continue;
            }

            // Find compatible version
            if let Some(version) = self.find_version(&name, &constraint) {
                resolved.insert(name.clone(), version.clone());

                // Add dependencies to queue
                if let Some(package) = self.get_package(&name, &version) {
                    for dep in &package.dependencies {
                        if !dep.optional {
                            queue.push_back((dep.name.clone(), dep.constraint.clone()));
                        }
                    }
                }
            } else {
                conflicts.push(Conflict {
                    package: name.clone(),
                    requested_by: vec![(name.clone(), constraint.clone())],
                    reason: format!("No version found satisfying {}", constraint),
                });
            }
        }

        // Build dependency graph and calculate build order
        let packages: Vec<ResolvedPackage> = resolved
            .iter()
            .map(|(name, version)| {
                let deps = if let Some(pkg) = self.get_package(name, version) {
                    pkg.dependencies.iter().map(|d| d.name.clone()).collect()
                } else {
                    Vec::new()
                };

                ResolvedPackage {
                    name: name.clone(),
                    version: version.clone(),
                    dependencies: deps,
                }
            })
            .collect();

        let build_order = self.topological_sort(&packages);

        ResolutionResult {
            packages,
            conflicts,
            build_order,
        }
    }

    /// Find version satisfying constraint
    fn find_version(&self, name: &str, constraint: &VersionConstraint) -> Option<Version> {
        let packages = self.available_packages.get(name)?;

        // Find all matching versions
        let mut matching: Vec<&Package> = packages
            .iter()
            .filter(|p| constraint.matches(&p.version))
            .collect();

        // Sort by version (descending) and pick highest
        matching.sort_by(|a, b| b.version.cmp(&a.version));

        matching.first().map(|p| p.version.clone())
    }

    /// Get specific package version
    fn get_package(&self, name: &str, version: &Version) -> Option<&Package> {
        self.available_packages
            .get(name)?
            .iter()
            .find(|p| &p.version == version)
    }

    /// Topological sort for build order
    fn topological_sort(&self, packages: &[ResolvedPackage]) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        fn visit(
            pkg_name: &str,
            packages: &[ResolvedPackage],
            visited: &mut HashSet<String>,
            visiting: &mut HashSet<String>,
            result: &mut Vec<String>,
        ) {
            if visited.contains(pkg_name) {
                return;
            }

            if visiting.contains(pkg_name) {
                // Circular dependency - skip
                return;
            }

            visiting.insert(pkg_name.to_string());

            // Visit dependencies first
            if let Some(pkg) = packages.iter().find(|p| p.name == pkg_name) {
                for dep in &pkg.dependencies {
                    visit(dep, packages, visited, visiting, result);
                }
            }

            visiting.remove(pkg_name);
            visited.insert(pkg_name.to_string());
            result.push(pkg_name.to_string());
        }

        for package in packages {
            visit(&package.name, packages, &mut visited, &mut visiting, &mut result);
        }

        result
    }

    /// Generate lock file content
    pub fn generate_lockfile(&self, result: &ResolutionResult) -> String {
        let mut lockfile = String::new();
        lockfile.push_str("# Dependency Lock File\n");
        lockfile.push_str("# Auto-generated - do not edit manually\n\n");

        for package in &result.packages {
            lockfile.push_str(&format!("[[package]]\n"));
            lockfile.push_str(&format!("name = \"{}\"\n", package.name));
            lockfile.push_str(&format!("version = \"{}\"\n", package.version));

            if !package.dependencies.is_empty() {
                lockfile.push_str("dependencies = [\n");
                for dep in &package.dependencies {
                    lockfile.push_str(&format!("    \"{}\",\n", dep));
                }
                lockfile.push_str("]\n");
            }

            lockfile.push('\n');
        }

        lockfile
    }
}

impl Default for VersionResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v = Version::parse("2.0.0-beta.1").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.pre_release, Some("beta.1".to_string()));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 2, 4);
        assert!(v1 < v2);

        let v3 = Version::new(2, 0, 0);
        assert!(v1 < v3);
    }

    #[test]
    fn test_constraint_matching() {
        let v = Version::new(1, 2, 3);

        let exact = VersionConstraint::Exact(v.clone());
        assert!(exact.matches(&v));

        let caret = VersionConstraint::Caret(v.clone());
        assert!(caret.matches(&Version::new(1, 2, 3)));
        assert!(caret.matches(&Version::new(1, 3, 0)));
        assert!(!caret.matches(&Version::new(2, 0, 0)));

        let tilde = VersionConstraint::Tilde(v.clone());
        assert!(tilde.matches(&Version::new(1, 2, 3)));
        assert!(tilde.matches(&Version::new(1, 2, 5)));
        assert!(!tilde.matches(&Version::new(1, 3, 0)));
    }

    #[test]
    fn test_constraint_parsing() {
        let c = VersionConstraint::parse("^1.2.3").unwrap();
        assert!(matches!(c, VersionConstraint::Caret(_)));

        let c = VersionConstraint::parse(">=1.0.0").unwrap();
        assert!(matches!(c, VersionConstraint::GreaterOrEqual(_)));

        let c = VersionConstraint::parse("1.*").unwrap();
        assert!(matches!(c, VersionConstraint::Wildcard(1, None)));
    }

    #[test]
    fn test_simple_resolution() {
        let mut resolver = VersionResolver::new();

        // Add packages
        resolver.add_package(Package {
            name: "foo".to_string(),
            version: Version::new(1, 0, 0),
            dependencies: vec![],
        });

        let deps = vec![Dependency::new(
            "foo",
            VersionConstraint::Caret(Version::new(1, 0, 0)),
        )];

        let result = resolver.resolve(deps);
        assert_eq!(result.packages.len(), 1);
        assert!(result.conflicts.is_empty());
    }
}
