#!/bin/bash
set -euo pipefail

# Portalis Changelog Generator
# Generates changelog from git commit history

CHANGELOG_FILE="${CHANGELOG_FILE:-CHANGELOG.md}"
VERSION="${VERSION:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-markdown}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the latest version tag
get_latest_tag() {
    git describe --tags --abbrev=0 2>/dev/null || echo ""
}

# Get the previous version tag
get_previous_tag() {
    local current_tag=$1
    if [ -z "$current_tag" ]; then
        echo ""
    else
        git describe --tags --abbrev=0 "${current_tag}^" 2>/dev/null || echo ""
    fi
}

# Get all tags sorted by version
get_all_tags() {
    git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' || echo ""
}

# Parse commit message for conventional commits
parse_commit_type() {
    local message=$1

    if [[ $message =~ ^feat(\(.*\))?:\ .+ ]]; then
        echo "feature"
    elif [[ $message =~ ^fix(\(.*\))?:\ .+ ]]; then
        echo "fix"
    elif [[ $message =~ ^docs(\(.*\))?:\ .+ ]]; then
        echo "docs"
    elif [[ $message =~ ^perf(\(.*\))?:\ .+ ]]; then
        echo "performance"
    elif [[ $message =~ ^refactor(\(.*\))?:\ .+ ]]; then
        echo "refactor"
    elif [[ $message =~ ^test(\(.*\))?:\ .+ ]]; then
        echo "test"
    elif [[ $message =~ ^chore(\(.*\))?:\ .+ ]]; then
        echo "chore"
    elif [[ $message =~ ^ci(\(.*\))?:\ .+ ]]; then
        echo "ci"
    elif [[ $message =~ ^build(\(.*\))?:\ .+ ]]; then
        echo "build"
    elif [[ $message =~ ^deps(\(.*\))?:\ .+ ]]; then
        echo "dependencies"
    else
        echo "other"
    fi
}

# Extract commit scope
extract_scope() {
    local message=$1
    if [[ $message =~ \(([^)]+)\): ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Clean commit message
clean_message() {
    local message=$1
    # Remove conventional commit prefix
    message=$(echo "$message" | sed -E 's/^(feat|fix|docs|perf|refactor|test|chore|ci|build|deps)(\([^)]+\))?:\ //')
    # Remove merge commit prefixes
    message=$(echo "$message" | sed 's/^Merge pull request #[0-9]* from //')
    message=$(echo "$message" | sed 's/^Merge branch //')
    echo "$message"
}

# Generate changelog for a version range
generate_version_changelog() {
    local from_tag=$1
    local to_tag=$2
    local version=$3

    log_info "Generating changelog from $from_tag to $to_tag"

    # Associative arrays for categorizing commits
    declare -A features=()
    declare -A fixes=()
    declare -A docs=()
    declare -A performance=()
    declare -A refactors=()
    declare -A tests=()
    declare -A dependencies=()
    declare -A ci=()
    declare -A build=()
    declare -A other=()

    # Get commits in range
    local range
    if [ -z "$from_tag" ]; then
        range="$to_tag"
    else
        range="$from_tag..$to_tag"
    fi

    # Read commits
    while IFS='|' read -r hash date author message; do
        # Skip merge commits
        if [[ $message =~ ^Merge ]]; then
            continue
        fi

        local commit_type=$(parse_commit_type "$message")
        local scope=$(extract_scope "$message")
        local clean_msg=$(clean_message "$message")
        local commit_line="- $clean_msg ([${hash:0:7}](https://github.com/portalis/portalis/commit/$hash))"

        if [ -n "$scope" ]; then
            commit_line="- **$scope**: $clean_msg ([${hash:0:7}](https://github.com/portalis/portalis/commit/$hash))"
        fi

        case "$commit_type" in
            feature)
                features["$hash"]="$commit_line"
                ;;
            fix)
                fixes["$hash"]="$commit_line"
                ;;
            docs)
                docs["$hash"]="$commit_line"
                ;;
            performance)
                performance["$hash"]="$commit_line"
                ;;
            refactor)
                refactors["$hash"]="$commit_line"
                ;;
            test)
                tests["$hash"]="$commit_line"
                ;;
            dependencies)
                dependencies["$hash"]="$commit_line"
                ;;
            ci)
                ci["$hash"]="$commit_line"
                ;;
            build)
                build["$hash"]="$commit_line"
                ;;
            *)
                other["$hash"]="$commit_line"
                ;;
        esac
    done < <(git log --pretty=format:"%H|%ad|%an|%s" --date=short "$range" --no-merges)

    # Generate changelog section
    echo ""
    echo "## [$version] - $(date +%Y-%m-%d)"
    echo ""

    if [ ${#features[@]} -gt 0 ]; then
        echo "### Features"
        echo ""
        for hash in "${!features[@]}"; do
            echo "${features[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#fixes[@]} -gt 0 ]; then
        echo "### Bug Fixes"
        echo ""
        for hash in "${!fixes[@]}"; do
            echo "${fixes[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#performance[@]} -gt 0 ]; then
        echo "### Performance Improvements"
        echo ""
        for hash in "${!performance[@]}"; do
            echo "${performance[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#refactors[@]} -gt 0 ]; then
        echo "### Refactoring"
        echo ""
        for hash in "${!refactors[@]}"; do
            echo "${refactors[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#docs[@]} -gt 0 ]; then
        echo "### Documentation"
        echo ""
        for hash in "${!docs[@]}"; do
            echo "${docs[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#tests[@]} -gt 0 ]; then
        echo "### Tests"
        echo ""
        for hash in "${!tests[@]}"; do
            echo "${tests[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#dependencies[@]} -gt 0 ]; then
        echo "### Dependencies"
        echo ""
        for hash in "${!dependencies[@]}"; do
            echo "${dependencies[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#ci[@]} -gt 0 ] || [ ${#build[@]} -gt 0 ]; then
        echo "### CI/CD"
        echo ""
        for hash in "${!ci[@]}"; do
            echo "${ci[$hash]}"
        done | sort
        for hash in "${!build[@]}"; do
            echo "${build[$hash]}"
        done | sort
        echo ""
    fi

    if [ ${#other[@]} -gt 0 ]; then
        echo "### Other Changes"
        echo ""
        for hash in "${!other[@]}"; do
            echo "${other[$hash]}"
        done | sort
        echo ""
    fi
}

# Generate full changelog
generate_full_changelog() {
    log_info "Generating full changelog..."

    # Create header
    cat > "$CHANGELOG_FILE" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

EOF

    # Get all version tags
    local tags=($(get_all_tags))

    if [ ${#tags[@]} -eq 0 ]; then
        log_warn "No version tags found"

        # Generate changelog for unreleased commits
        generate_version_changelog "" "HEAD" "Unreleased" >> "$CHANGELOG_FILE"
    else
        # Generate unreleased section
        local latest_tag="${tags[0]}"
        local unreleased_count=$(git rev-list "${latest_tag}..HEAD" --count)

        if [ "$unreleased_count" -gt 0 ]; then
            log_info "Found $unreleased_count unreleased commits"
            generate_version_changelog "$latest_tag" "HEAD" "Unreleased" >> "$CHANGELOG_FILE"
        fi

        # Generate changelog for each version
        for i in "${!tags[@]}"; do
            local current_tag="${tags[$i]}"
            local previous_tag=""

            if [ $((i + 1)) -lt ${#tags[@]} ]; then
                previous_tag="${tags[$((i + 1))]}"
            fi

            local version="${current_tag#v}"
            generate_version_changelog "$previous_tag" "$current_tag" "$version" >> "$CHANGELOG_FILE"
        done
    fi

    log_info "Changelog generated: $CHANGELOG_FILE"
}

# Generate changelog for specific version
generate_version_only() {
    local version=$1
    local tag="v$version"

    log_info "Generating changelog for version $version"

    # Find previous tag
    local previous_tag=$(get_previous_tag "$tag")

    # Generate changelog section
    generate_version_changelog "$previous_tag" "$tag" "$version"
}

# Update existing changelog with new version
update_changelog() {
    local version=$1

    log_info "Updating changelog with version $version"

    if [ ! -f "$CHANGELOG_FILE" ]; then
        log_warn "Changelog file not found, creating new one"
        generate_full_changelog
        return
    fi

    # Create temporary file with new version
    local temp_file=$(mktemp)
    local tag="v$version"
    local previous_tag=$(get_previous_tag "$tag")

    # Extract header
    sed -n '1,/^## \[/p' "$CHANGELOG_FILE" | head -n -1 > "$temp_file"

    # Add new version
    generate_version_changelog "$previous_tag" "$tag" "$version" >> "$temp_file"

    # Add rest of changelog
    sed -n '/^## \[/,$p' "$CHANGELOG_FILE" >> "$temp_file"

    # Replace original
    mv "$temp_file" "$CHANGELOG_FILE"

    log_info "Changelog updated: $CHANGELOG_FILE"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate or update CHANGELOG.md from git commit history.

OPTIONS:
    -h, --help              Show this help message
    -f, --full              Generate full changelog for all versions
    -v, --version VERSION   Generate changelog for specific version
    -u, --update VERSION    Update changelog with new version
    -o, --output FILE       Output file (default: CHANGELOG.md)

EXAMPLES:
    # Generate full changelog
    $0 --full

    # Generate changelog for specific version
    $0 --version 1.0.0

    # Update changelog with new version
    $0 --update 1.2.0

    # Custom output file
    $0 --full --output HISTORY.md

COMMIT MESSAGE FORMAT:
    This script works best with conventional commits:
    - feat: New feature
    - fix: Bug fix
    - docs: Documentation changes
    - perf: Performance improvements
    - refactor: Code refactoring
    - test: Test changes
    - chore: Maintenance tasks
    - ci: CI/CD changes
    - build: Build system changes
    - deps: Dependency updates

EOF
}

# Main
main() {
    local mode=""
    local version=""

    if [ $# -eq 0 ]; then
        usage
        exit 0
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -f|--full)
                mode="full"
                shift
                ;;
            -v|--version)
                mode="version"
                version="$2"
                shift 2
                ;;
            -u|--update)
                mode="update"
                version="$2"
                shift 2
                ;;
            -o|--output)
                CHANGELOG_FILE="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    case "$mode" in
        full)
            generate_full_changelog
            ;;
        version)
            if [ -z "$version" ]; then
                log_error "Version required for --version"
                exit 1
            fi
            generate_version_only "$version"
            ;;
        update)
            if [ -z "$version" ]; then
                log_error "Version required for --update"
                exit 1
            fi
            update_changelog "$version"
            ;;
        *)
            log_error "No mode specified"
            usage
            exit 1
            ;;
    esac
}

main "$@"
