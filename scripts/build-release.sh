#!/bin/bash
set -euo pipefail

# Portalis Release Build Script
# Cross-compilation and binary optimization for multiple platforms

VERSION="${VERSION:-0.1.0}"
BUILD_DIR="${BUILD_DIR:-./target}"
RELEASE_DIR="${RELEASE_DIR:-./release}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Detect host platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=Linux;;
        Darwin*)    PLATFORM=Mac;;
        MINGW*)     PLATFORM=Windows;;
        *)          PLATFORM="UNKNOWN"
    esac
    log_info "Detected platform: $PLATFORM"
}

# Install cross-compilation tools
install_cross_tools() {
    log_info "Installing cross-compilation tools..."

    if ! command -v cross &> /dev/null; then
        log_info "Installing cross..."
        cargo install cross --locked
    else
        log_info "cross already installed"
    fi

    # Install additional targets
    log_info "Installing additional Rust targets..."
    rustup target add x86_64-unknown-linux-gnu
    rustup target add x86_64-unknown-linux-musl
    rustup target add aarch64-unknown-linux-gnu
    rustup target add x86_64-apple-darwin
    rustup target add aarch64-apple-darwin
    rustup target add x86_64-pc-windows-msvc
    rustup target add wasm32-unknown-unknown
    rustup target add wasm32-wasi
}

# Build for a specific target
build_target() {
    local target=$1
    local use_cross=${2:-false}

    log_info "Building for target: $target"

    if [ "$use_cross" = true ]; then
        log_info "Using cross for $target"
        cross build --release --target "$target" --workspace
    else
        log_info "Using cargo for $target"
        cargo build --release --target "$target" --workspace
    fi

    if [ $? -eq 0 ]; then
        log_info "Successfully built for $target"
        return 0
    else
        log_error "Failed to build for $target"
        return 1
    fi
}

# Strip debug symbols from binary
strip_binary() {
    local binary=$1
    local target=$2

    if [ ! -f "$binary" ]; then
        log_warn "Binary not found: $binary"
        return 1
    fi

    log_info "Stripping debug symbols from $binary"

    case "$target" in
        *linux*)
            strip "$binary" || log_warn "Failed to strip $binary"
            ;;
        *darwin*)
            strip "$binary" || log_warn "Failed to strip $binary"
            ;;
        *windows*)
            # Windows binaries - no strip
            log_info "Skipping strip for Windows binary"
            ;;
    esac
}

# Optimize binary with UPX (optional)
optimize_binary() {
    local binary=$1

    if ! command -v upx &> /dev/null; then
        log_warn "UPX not found, skipping binary compression"
        return 0
    fi

    log_info "Compressing binary with UPX: $binary"
    upx --best --lzma "$binary" || log_warn "UPX compression failed"
}

# Package release for a target
package_release() {
    local target=$1
    local target_dir="$BUILD_DIR/$target/release"
    local package_name="portalis-$VERSION-$target"

    log_info "Packaging release for $target"

    mkdir -p "$RELEASE_DIR"

    case "$target" in
        *windows*)
            # Windows: Create ZIP archive
            local archive="$RELEASE_DIR/$package_name.zip"
            log_info "Creating Windows archive: $archive"

            (cd "$target_dir" && zip -r "$archive" \
                portalis.exe \
                portalis-cli.exe \
                2>/dev/null) || log_warn "Some binaries not found for $target"
            ;;
        *)
            # Unix: Create tar.gz archive
            local archive="$RELEASE_DIR/$package_name.tar.gz"
            log_info "Creating Unix archive: $archive"

            (cd "$target_dir" && tar -czf "$archive" \
                portalis \
                portalis-cli \
                2>/dev/null) || log_warn "Some binaries not found for $target"
            ;;
    esac

    if [ -f "$archive" ]; then
        log_info "Created release archive: $archive"
        ls -lh "$archive"
    else
        log_error "Failed to create archive for $target"
        return 1
    fi
}

# Generate checksums
generate_checksums() {
    log_info "Generating checksums..."

    (cd "$RELEASE_DIR" && sha256sum *.tar.gz *.zip 2>/dev/null > SHA256SUMS) || \
        log_warn "Failed to generate checksums"

    if [ -f "$RELEASE_DIR/SHA256SUMS" ]; then
        log_info "Checksums generated:"
        cat "$RELEASE_DIR/SHA256SUMS"
    fi
}

# Build all targets
build_all() {
    log_info "Building all release targets..."

    # Linux x86_64 (GNU)
    if build_target "x86_64-unknown-linux-gnu" false; then
        strip_binary "$BUILD_DIR/x86_64-unknown-linux-gnu/release/portalis" "x86_64-unknown-linux-gnu"
        strip_binary "$BUILD_DIR/x86_64-unknown-linux-gnu/release/portalis-cli" "x86_64-unknown-linux-gnu"
        package_release "x86_64-unknown-linux-gnu"
    fi

    # Linux x86_64 (MUSL - static)
    if build_target "x86_64-unknown-linux-musl" true; then
        strip_binary "$BUILD_DIR/x86_64-unknown-linux-musl/release/portalis" "x86_64-unknown-linux-musl"
        strip_binary "$BUILD_DIR/x86_64-unknown-linux-musl/release/portalis-cli" "x86_64-unknown-linux-musl"
        package_release "x86_64-unknown-linux-musl"
    fi

    # Linux ARM64
    if build_target "aarch64-unknown-linux-gnu" true; then
        package_release "aarch64-unknown-linux-gnu"
    fi

    # macOS x86_64
    if [ "$PLATFORM" = "Mac" ]; then
        if build_target "x86_64-apple-darwin" false; then
            strip_binary "$BUILD_DIR/x86_64-apple-darwin/release/portalis" "x86_64-apple-darwin"
            strip_binary "$BUILD_DIR/x86_64-apple-darwin/release/portalis-cli" "x86_64-apple-darwin"
            package_release "x86_64-apple-darwin"
        fi

        # macOS ARM64 (Apple Silicon)
        if build_target "aarch64-apple-darwin" false; then
            strip_binary "$BUILD_DIR/aarch64-apple-darwin/release/portalis" "aarch64-apple-darwin"
            strip_binary "$BUILD_DIR/aarch64-apple-darwin/release/portalis-cli" "aarch64-apple-darwin"
            package_release "aarch64-apple-darwin"
        fi
    fi

    # Windows x86_64
    if build_target "x86_64-pc-windows-msvc" true; then
        package_release "x86_64-pc-windows-msvc"
    fi

    # Generate checksums for all packages
    generate_checksums

    log_info "Release build completed!"
    log_info "Release artifacts in: $RELEASE_DIR"
    ls -lh "$RELEASE_DIR"
}

# Build specific target
build_single() {
    local target=$1
    log_info "Building single target: $target"

    if build_target "$target" false; then
        strip_binary "$BUILD_DIR/$target/release/portalis" "$target" || true
        strip_binary "$BUILD_DIR/$target/release/portalis-cli" "$target" || true
        package_release "$target"
        generate_checksums
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Portalis release binaries for multiple platforms.

OPTIONS:
    -h, --help              Show this help message
    -a, --all               Build for all supported targets
    -t, --target TARGET     Build for specific target
    -i, --install-tools     Install cross-compilation tools
    -v, --version VERSION   Set version (default: $VERSION)
    -o, --output DIR        Set output directory (default: $RELEASE_DIR)

TARGETS:
    x86_64-unknown-linux-gnu
    x86_64-unknown-linux-musl
    aarch64-unknown-linux-gnu
    x86_64-apple-darwin
    aarch64-apple-darwin
    x86_64-pc-windows-msvc

EXAMPLES:
    # Build for all platforms
    $0 --all

    # Build for specific platform
    $0 --target x86_64-unknown-linux-gnu

    # Install tools and build
    $0 --install-tools --all

    # Set custom version
    $0 --version 1.0.0 --all

EOF
}

# Main
main() {
    detect_platform

    local install_tools=false
    local build_all_targets=false
    local specific_target=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -a|--all)
                build_all_targets=true
                shift
                ;;
            -t|--target)
                specific_target="$2"
                shift 2
                ;;
            -i|--install-tools)
                install_tools=true
                shift
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -o|--output)
                RELEASE_DIR="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [ "$install_tools" = true ]; then
        install_cross_tools
    fi

    if [ "$build_all_targets" = true ]; then
        build_all
    elif [ -n "$specific_target" ]; then
        build_single "$specific_target"
    else
        log_error "No build target specified"
        usage
        exit 1
    fi
}

main "$@"
