#ifndef PORTALIS_AST_PARSER_CUH
#define PORTALIS_AST_PARSER_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace portalis {
namespace cuda {

// AST Node structure for GPU processing
struct ASTNode {
    uint32_t node_type;      // Node type identifier
    uint32_t parent_idx;     // Parent node index
    uint32_t first_child;    // First child index
    uint32_t next_sibling;   // Next sibling index
    uint32_t token_start;    // Start position in source
    uint32_t token_end;      // End position in source
    uint32_t line_number;    // Line number in source
    uint32_t col_number;     // Column number in source
    float confidence;        // Parse confidence score
    uint32_t metadata_idx;   // Index into metadata array
};

// Token structure for parallel tokenization
struct Token {
    uint32_t type;           // Token type
    uint32_t start;          // Start position
    uint32_t length;         // Token length
    uint32_t line;           // Line number
    uint32_t column;         // Column number
    uint32_t flags;          // Additional flags
};

// Parser configuration
struct ParserConfig {
    uint32_t max_nodes;      // Maximum AST nodes
    uint32_t max_tokens;     // Maximum tokens
    uint32_t max_depth;      // Maximum tree depth
    uint32_t batch_size;     // Number of files to parse
    bool enable_async;       // Enable async parsing
    bool collect_metrics;    // Collect performance metrics
};

// Performance metrics
struct ParserMetrics {
    float tokenization_time_ms;
    float parsing_time_ms;
    float total_time_ms;
    uint32_t nodes_created;
    uint32_t tokens_processed;
    float gpu_utilization;
};

// Main API functions
extern "C" {
    // Initialize parser
    cudaError_t initializeASTParser(ParserConfig* config);

    // Parse single source file
    cudaError_t parseSource(
        const char* source,
        uint32_t source_length,
        ASTNode** nodes_out,
        uint32_t* node_count_out,
        ParserMetrics* metrics_out
    );

    // Batch parse multiple files
    cudaError_t batchParseSource(
        const char** sources,
        uint32_t* source_lengths,
        uint32_t batch_size,
        ASTNode*** nodes_out,
        uint32_t** node_counts_out,
        ParserMetrics* metrics_out
    );

    // Free parser resources
    cudaError_t cleanupASTParser();

    // Get GPU memory usage
    cudaError_t getParserMemoryUsage(size_t* bytes_used, size_t* bytes_total);
}

// Kernel declarations
__global__ void tokenizeSourceKernel(
    const char* source,
    uint32_t source_length,
    Token* tokens_out,
    uint32_t* token_count,
    uint32_t max_tokens
);

__global__ void buildASTKernel(
    const Token* tokens,
    uint32_t token_count,
    ASTNode* nodes_out,
    uint32_t* node_count,
    uint32_t max_nodes
);

__global__ void validateASTKernel(
    const ASTNode* nodes,
    uint32_t node_count,
    bool* valid_out,
    uint32_t* error_indices
);

__global__ void optimizeASTKernel(
    ASTNode* nodes,
    uint32_t* node_count,
    uint32_t max_nodes
);

} // namespace cuda
} // namespace portalis

#endif // PORTALIS_AST_PARSER_CUH
