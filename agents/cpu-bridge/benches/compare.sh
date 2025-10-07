#!/bin/bash
# Benchmark comparison utility for CPU Bridge
# Usage: ./compare.sh [baseline] [comparison]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default baselines
BASELINE="${1:-main}"
COMPARISON="${2:-current}"

echo -e "${BLUE}=== CPU Bridge Benchmark Comparison ===${NC}"
echo ""
echo "Baseline:    $BASELINE"
echo "Comparison:  $COMPARISON"
echo ""

# Check if critcmp is installed
if ! command -v critcmp &> /dev/null; then
    echo -e "${YELLOW}Installing critcmp for comparison...${NC}"
    cargo install critcmp
fi

# Function to run benchmarks and save baseline
run_benchmark() {
    local name=$1
    echo -e "${BLUE}Running benchmarks for: $name${NC}"

    cd "$PROJECT_ROOT"
    cargo bench --package portalis-cpu-bridge -- --save-baseline "$name" 2>&1 | tee "benchmark-$name.log"

    echo -e "${GREEN}✓ Saved baseline: $name${NC}"
    echo ""
}

# Function to compare benchmarks
compare_benchmarks() {
    local base=$1
    local comp=$2

    echo -e "${BLUE}Comparing $base vs $comp${NC}"
    echo ""

    cd "$PROJECT_ROOT/target/criterion"

    if [ -d "$base" ] || [ -d "$comp" ]; then
        critcmp "$base" "$comp"
    else
        echo -e "${RED}Error: Baseline '$base' or '$comp' not found${NC}"
        echo "Available baselines:"
        ls -d */ 2>/dev/null | grep -v "^report" || echo "  (none)"
        exit 1
    fi
}

# Function to show performance summary
show_summary() {
    echo ""
    echo -e "${BLUE}=== Performance Summary ===${NC}"
    echo ""

    # Extract key metrics from criterion output
    local criterion_dir="$PROJECT_ROOT/target/criterion"

    if [ -d "$criterion_dir" ]; then
        echo "Latest Benchmark Results:"
        echo ""

        # Single file performance
        if [ -f "$criterion_dir/single_file_translation/small_1kb/new/estimates.json" ]; then
            local mean=$(jq -r '.mean.point_estimate / 1000000' "$criterion_dir/single_file_translation/small_1kb/new/estimates.json" 2>/dev/null || echo "N/A")
            printf "  Single File (1KB):        %.2f ms\n" "$mean" 2>/dev/null || echo "  Single File (1KB):        N/A"
        fi

        # Thread scaling
        if [ -f "$criterion_dir/thread_scaling/threads/16/new/estimates.json" ]; then
            local mean=$(jq -r '.mean.point_estimate / 1000000' "$criterion_dir/thread_scaling/threads/16/new/estimates.json" 2>/dev/null || echo "N/A")
            printf "  100 Files (16 threads):   %.2f ms\n" "$mean" 2>/dev/null || echo "  100 Files (16 threads):   N/A"
        fi

        echo ""
    fi

    # Check against targets
    echo "Performance Targets:"
    echo "  ✓ Single file (1KB, 4-core):    < 50ms"
    echo "  ✓ Small batch (10 files, 16c):  < 70ms"
    echo "  ✓ Medium batch (100 files, 16c): < 500ms"
    echo ""
}

# Function to generate HTML report
generate_report() {
    echo -e "${BLUE}Generating HTML report...${NC}"

    cd "$PROJECT_ROOT"
    cargo install cargo-criterion 2>/dev/null || true
    cargo criterion --package portalis-cpu-bridge

    local report_path="$PROJECT_ROOT/target/criterion/report/index.html"
    if [ -f "$report_path" ]; then
        echo -e "${GREEN}✓ Report generated: $report_path${NC}"

        # Try to open in browser
        if command -v xdg-open &> /dev/null; then
            xdg-open "$report_path" &
        elif command -v open &> /dev/null; then
            open "$report_path" &
        fi
    fi
}

# Function to check for regressions
check_regressions() {
    local base=$1
    local comp=$2
    local threshold=${3:-5}  # 5% regression threshold

    echo -e "${BLUE}Checking for regressions (threshold: ${threshold}%)${NC}"
    echo ""

    cd "$PROJECT_ROOT/target/criterion"

    # Use critcmp to find regressions
    local output=$(critcmp --threshold "$threshold" "$base" "$comp" 2>/dev/null || echo "")

    if echo "$output" | grep -q "regression"; then
        echo -e "${RED}⚠ Performance regressions detected!${NC}"
        echo "$output" | grep -A 5 "regression"
        return 1
    else
        echo -e "${GREEN}✓ No significant regressions${NC}"
        return 0
    fi
}

# Function to export results
export_results() {
    local format=${1:-csv}
    local output_file="benchmark-comparison.$format"

    echo -e "${BLUE}Exporting results to $output_file${NC}"

    cd "$PROJECT_ROOT/target/criterion"

    if [ "$format" == "csv" ]; then
        critcmp --export "$BASELINE" "$COMPARISON" > "$PROJECT_ROOT/$output_file"
    fi

    echo -e "${GREEN}✓ Exported to $output_file${NC}"
}

# Function to run thread scaling analysis
analyze_scaling() {
    echo -e "${BLUE}=== Thread Scaling Analysis ===${NC}"
    echo ""

    cd "$PROJECT_ROOT"

    echo "Running thread scaling benchmarks..."
    for threads in 1 2 4 8 16; do
        echo -e "${YELLOW}Testing with $threads threads...${NC}"
        RAYON_NUM_THREADS=$threads cargo bench --package portalis-cpu-bridge -- thread_scaling/threads/$threads --quiet
    done

    echo ""
    echo "Scaling Efficiency:"
    # Calculate parallel efficiency
    local criterion_dir="$PROJECT_ROOT/target/criterion/thread_scaling"

    if [ -d "$criterion_dir" ]; then
        local baseline=$(jq -r '.mean.point_estimate' "$criterion_dir/threads/1/new/estimates.json" 2>/dev/null || echo "0")

        for threads in 2 4 8 16; do
            if [ -f "$criterion_dir/threads/$threads/new/estimates.json" ]; then
                local time=$(jq -r '.mean.point_estimate' "$criterion_dir/threads/$threads/new/estimates.json")
                if [ "$baseline" != "0" ] && [ "$time" != "0" ]; then
                    local speedup=$(echo "scale=2; $baseline / $time" | bc)
                    local efficiency=$(echo "scale=1; ($speedup / $threads) * 100" | bc)
                    printf "  %2d threads: %.2fx speedup (%.1f%% efficiency)\n" "$threads" "$speedup" "$efficiency"
                fi
            fi
        done
    fi

    echo ""
}

# Main menu
show_menu() {
    echo -e "${BLUE}Select operation:${NC}"
    echo "  1) Run new benchmark and save"
    echo "  2) Compare two baselines"
    echo "  3) Show performance summary"
    echo "  4) Generate HTML report"
    echo "  5) Check for regressions"
    echo "  6) Export comparison to CSV"
    echo "  7) Analyze thread scaling"
    echo "  8) Full benchmark suite"
    echo "  9) Exit"
    echo ""
    read -p "Choice: " choice

    case $choice in
        1)
            read -p "Baseline name: " name
            run_benchmark "$name"
            ;;
        2)
            read -p "Base baseline: " base
            read -p "Comparison baseline: " comp
            compare_benchmarks "$base" "$comp"
            ;;
        3)
            show_summary
            ;;
        4)
            generate_report
            ;;
        5)
            read -p "Base baseline: " base
            read -p "Comparison baseline: " comp
            read -p "Threshold %: " threshold
            check_regressions "$base" "$comp" "$threshold"
            ;;
        6)
            export_results "csv"
            ;;
        7)
            analyze_scaling
            ;;
        8)
            echo -e "${BLUE}Running full benchmark suite...${NC}"
            run_benchmark "current"
            show_summary
            generate_report
            ;;
        9)
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac

    echo ""
    read -p "Press enter to continue..."
    show_menu
}

# If no arguments, show interactive menu
if [ $# -eq 0 ]; then
    show_menu
else
    # Run comparison from command line
    if [ "$BASELINE" == "run" ]; then
        run_benchmark "$COMPARISON"
        show_summary
    else
        compare_benchmarks "$BASELINE" "$COMPARISON"
        check_regressions "$BASELINE" "$COMPARISON" 5 || true
    fi
fi
