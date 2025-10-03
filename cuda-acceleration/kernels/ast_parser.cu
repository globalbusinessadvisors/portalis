#include "ast_parser.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstring>

namespace portalis {
namespace cuda {

namespace cg = cooperative_groups;

// Device constants
__constant__ char TOKEN_DELIMITERS[256];
__constant__ uint32_t PYTHON_KEYWORDS[64];
__constant__ uint32_t OPERATOR_TOKENS[128];

// Global state
static ParserConfig g_config;
static bool g_initialized = false;

// Token type enumeration
enum TokenType {
    TOK_KEYWORD = 1,
    TOK_IDENTIFIER = 2,
    TOK_NUMBER = 3,
    TOK_STRING = 4,
    TOK_OPERATOR = 5,
    TOK_DELIMITER = 6,
    TOK_INDENT = 7,
    TOK_DEDENT = 8,
    TOK_NEWLINE = 9,
    TOK_COMMENT = 10,
    TOK_EOF = 11
};

// Node type enumeration
enum NodeType {
    NODE_MODULE = 1,
    NODE_FUNCTION_DEF = 2,
    NODE_CLASS_DEF = 3,
    NODE_IF_STMT = 4,
    NODE_FOR_STMT = 5,
    NODE_WHILE_STMT = 6,
    NODE_RETURN_STMT = 7,
    NODE_ASSIGN = 8,
    NODE_EXPR = 9,
    NODE_CALL = 10,
    NODE_ATTRIBUTE = 11,
    NODE_SUBSCRIPT = 12,
    NODE_LIST = 13,
    NODE_DICT = 14,
    NODE_LAMBDA = 15
};

// Device helper functions
__device__ inline bool isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\r';
}

__device__ inline bool isAlpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

__device__ inline bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

__device__ inline bool isAlphaNum(char c) {
    return isAlpha(c) || isDigit(c);
}

// Parallel tokenization kernel
__global__ void tokenizeSourceKernel(
    const char* source,
    uint32_t source_length,
    Token* tokens_out,
    uint32_t* token_count,
    uint32_t max_tokens
) {
    // Calculate this thread's chunk of source code
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = blockDim.x * gridDim.x;

    // Shared memory for thread-local tokens
    __shared__ Token local_tokens[256];
    __shared__ uint32_t local_count;

    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();

    // Each thread processes a section of the source
    uint32_t chunk_size = (source_length + total_threads - 1) / total_threads;
    uint32_t start = tid * chunk_size;
    uint32_t end = min(start + chunk_size, source_length);

    if (start >= source_length) return;

    // State variables
    uint32_t pos = start;
    uint32_t line = 1;
    uint32_t column = 1;
    uint32_t indent_level = 0;

    // Find start of next valid token (handle split points)
    if (tid > 0) {
        // Backtrack to previous whitespace or delimiter
        while (pos > 0 && isAlphaNum(source[pos])) {
            pos--;
        }
        if (pos > 0) pos++;
    }

    // Tokenize this chunk
    while (pos < end && pos < source_length) {
        char current = source[pos];

        // Skip whitespace (except newlines)
        if (isWhitespace(current)) {
            if (current == '\t') {
                column += 4;
            } else {
                column++;
            }
            pos++;
            continue;
        }

        // Handle newlines and indentation
        if (current == '\n') {
            Token tok;
            tok.type = TOK_NEWLINE;
            tok.start = pos;
            tok.length = 1;
            tok.line = line;
            tok.column = column;
            tok.flags = 0;

            uint32_t idx = atomicAdd(&local_count, 1);
            if (idx < 256) {
                local_tokens[idx] = tok;
            }

            line++;
            column = 1;
            pos++;

            // Calculate indentation level
            uint32_t new_indent = 0;
            while (pos < source_length && isWhitespace(source[pos])) {
                if (source[pos] == ' ') new_indent++;
                else if (source[pos] == '\t') new_indent += 4;
                pos++;
            }

            // Emit INDENT/DEDENT tokens
            if (new_indent > indent_level) {
                idx = atomicAdd(&local_count, 1);
                if (idx < 256) {
                    local_tokens[idx].type = TOK_INDENT;
                    local_tokens[idx].start = pos;
                    local_tokens[idx].length = 0;
                    local_tokens[idx].line = line;
                    local_tokens[idx].column = column;
                }
            } else if (new_indent < indent_level) {
                idx = atomicAdd(&local_count, 1);
                if (idx < 256) {
                    local_tokens[idx].type = TOK_DEDENT;
                    local_tokens[idx].start = pos;
                    local_tokens[idx].length = 0;
                    local_tokens[idx].line = line;
                    local_tokens[idx].column = column;
                }
            }
            indent_level = new_indent;
            continue;
        }

        // Handle comments
        if (current == '#') {
            uint32_t comment_start = pos;
            while (pos < source_length && source[pos] != '\n') {
                pos++;
            }

            Token tok;
            tok.type = TOK_COMMENT;
            tok.start = comment_start;
            tok.length = pos - comment_start;
            tok.line = line;
            tok.column = column;
            tok.flags = 0;

            uint32_t idx = atomicAdd(&local_count, 1);
            if (idx < 256) {
                local_tokens[idx] = tok;
            }
            continue;
        }

        // Handle strings
        if (current == '"' || current == '\'') {
            char quote = current;
            uint32_t str_start = pos;
            pos++;

            // Check for triple-quoted strings
            bool triple = false;
            if (pos + 1 < source_length &&
                source[pos] == quote &&
                source[pos + 1] == quote) {
                triple = true;
                pos += 2;
            }

            // Find string end
            while (pos < source_length) {
                if (source[pos] == '\\') {
                    pos += 2; // Skip escaped character
                    continue;
                }

                if (triple) {
                    if (pos + 2 < source_length &&
                        source[pos] == quote &&
                        source[pos + 1] == quote &&
                        source[pos + 2] == quote) {
                        pos += 3;
                        break;
                    }
                } else {
                    if (source[pos] == quote) {
                        pos++;
                        break;
                    }
                }

                if (source[pos] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
                pos++;
            }

            Token tok;
            tok.type = TOK_STRING;
            tok.start = str_start;
            tok.length = pos - str_start;
            tok.line = line;
            tok.column = column;
            tok.flags = triple ? 1 : 0;

            uint32_t idx = atomicAdd(&local_count, 1);
            if (idx < 256) {
                local_tokens[idx] = tok;
            }
            continue;
        }

        // Handle numbers
        if (isDigit(current)) {
            uint32_t num_start = pos;
            bool is_float = false;

            while (pos < source_length) {
                if (isDigit(source[pos])) {
                    pos++;
                } else if (source[pos] == '.' && !is_float) {
                    is_float = true;
                    pos++;
                } else if ((source[pos] == 'e' || source[pos] == 'E') && !is_float) {
                    is_float = true;
                    pos++;
                    if (pos < source_length && (source[pos] == '+' || source[pos] == '-')) {
                        pos++;
                    }
                } else {
                    break;
                }
            }

            Token tok;
            tok.type = TOK_NUMBER;
            tok.start = num_start;
            tok.length = pos - num_start;
            tok.line = line;
            tok.column = column;
            tok.flags = is_float ? 1 : 0;

            uint32_t idx = atomicAdd(&local_count, 1);
            if (idx < 256) {
                local_tokens[idx] = tok;
            }
            column += tok.length;
            continue;
        }

        // Handle identifiers and keywords
        if (isAlpha(current)) {
            uint32_t id_start = pos;
            while (pos < source_length && isAlphaNum(source[pos])) {
                pos++;
            }

            uint32_t length = pos - id_start;

            // Check if it's a keyword (simplified check)
            bool is_keyword = false;
            // TODO: Implement keyword lookup

            Token tok;
            tok.type = is_keyword ? TOK_KEYWORD : TOK_IDENTIFIER;
            tok.start = id_start;
            tok.length = length;
            tok.line = line;
            tok.column = column;
            tok.flags = 0;

            uint32_t idx = atomicAdd(&local_count, 1);
            if (idx < 256) {
                local_tokens[idx] = tok;
            }
            column += length;
            continue;
        }

        // Handle operators and delimiters
        Token tok;
        tok.type = TOK_OPERATOR;
        tok.start = pos;
        tok.length = 1;
        tok.line = line;
        tok.column = column;
        tok.flags = 0;

        // Check for multi-character operators
        if (pos + 1 < source_length) {
            char next = source[pos + 1];
            if ((current == '=' && next == '=') ||
                (current == '!' && next == '=') ||
                (current == '<' && next == '=') ||
                (current == '>' && next == '=') ||
                (current == '*' && next == '*') ||
                (current == '/' && next == '/') ||
                (current == '-' && next == '>')) {
                tok.length = 2;
                pos++;
            }
        }

        uint32_t idx = atomicAdd(&local_count, 1);
        if (idx < 256) {
            local_tokens[idx] = tok;
        }

        pos++;
        column += tok.length;
    }

    __syncthreads();

    // Write local tokens to global memory
    if (threadIdx.x == 0) {
        uint32_t global_offset = atomicAdd(token_count, local_count);
        for (uint32_t i = 0; i < local_count && global_offset + i < max_tokens; i++) {
            tokens_out[global_offset + i] = local_tokens[i];
        }
    }
}

// AST construction kernel
__global__ void buildASTKernel(
    const Token* tokens,
    uint32_t token_count,
    ASTNode* nodes_out,
    uint32_t* node_count,
    uint32_t max_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid != 0) return; // For now, use single-threaded parsing

    // Simplified recursive descent parser
    // This would be implemented with a proper parsing algorithm

    // Create root module node
    ASTNode root;
    root.node_type = NODE_MODULE;
    root.parent_idx = 0xFFFFFFFF;
    root.first_child = 0xFFFFFFFF;
    root.next_sibling = 0xFFFFFFFF;
    root.token_start = 0;
    root.token_end = token_count;
    root.line_number = 1;
    root.col_number = 1;
    root.confidence = 1.0f;
    root.metadata_idx = 0;

    nodes_out[0] = root;
    *node_count = 1;

    // TODO: Implement full parser
    // This would involve:
    // 1. Statement parsing
    // 2. Expression parsing
    // 3. Precedence climbing
    // 4. Error recovery
}

// AST validation kernel
__global__ void validateASTKernel(
    const ASTNode* nodes,
    uint32_t node_count,
    bool* valid_out,
    uint32_t* error_indices
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= node_count) return;

    const ASTNode& node = nodes[tid];
    bool is_valid = true;

    // Validate node structure
    if (node.parent_idx != 0xFFFFFFFF && node.parent_idx >= node_count) {
        is_valid = false;
    }

    if (node.first_child != 0xFFFFFFFF && node.first_child >= node_count) {
        is_valid = false;
    }

    if (node.next_sibling != 0xFFFFFFFF && node.next_sibling >= node_count) {
        is_valid = false;
    }

    if (!is_valid) {
        uint32_t err_idx = atomicAdd(error_indices, 1);
        valid_out[tid] = false;
    } else {
        valid_out[tid] = true;
    }
}

// AST optimization kernel
__global__ void optimizeASTKernel(
    ASTNode* nodes,
    uint32_t* node_count,
    uint32_t max_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= *node_count) return;

    // Perform basic optimizations:
    // 1. Constant folding
    // 2. Dead code elimination
    // 3. Tree balancing

    // TODO: Implement optimizations
}

// Host API implementation
extern "C" {

cudaError_t initializeASTParser(ParserConfig* config) {
    if (g_initialized) {
        return cudaSuccess;
    }

    g_config = *config;

    // Initialize constant memory
    // TODO: Load Python keywords and operators

    g_initialized = true;
    return cudaSuccess;
}

cudaError_t parseSource(
    const char* source,
    uint32_t source_length,
    ASTNode** nodes_out,
    uint32_t* node_count_out,
    ParserMetrics* metrics_out
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    char* d_source;
    Token* d_tokens;
    uint32_t* d_token_count;
    ASTNode* d_nodes;
    uint32_t* d_node_count;

    cudaMalloc(&d_source, source_length);
    cudaMalloc(&d_tokens, g_config.max_tokens * sizeof(Token));
    cudaMalloc(&d_token_count, sizeof(uint32_t));
    cudaMalloc(&d_nodes, g_config.max_nodes * sizeof(ASTNode));
    cudaMalloc(&d_node_count, sizeof(uint32_t));

    // Copy source to device
    cudaMemcpy(d_source, source, source_length, cudaMemcpyHostToDevice);
    cudaMemset(d_token_count, 0, sizeof(uint32_t));
    cudaMemset(d_node_count, 0, sizeof(uint32_t));

    // Tokenize
    int block_size = 256;
    int grid_size = (source_length + block_size - 1) / block_size;

    cudaEvent_t tok_start, tok_stop;
    cudaEventCreate(&tok_start);
    cudaEventCreate(&tok_stop);
    cudaEventRecord(tok_start);

    tokenizeSourceKernel<<<grid_size, block_size>>>(
        d_source, source_length, d_tokens, d_token_count, g_config.max_tokens
    );

    cudaEventRecord(tok_stop);
    cudaEventSynchronize(tok_stop);

    float tokenization_time;
    cudaEventElapsedTime(&tokenization_time, tok_start, tok_stop);

    // Get token count
    uint32_t token_count;
    cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Build AST
    cudaEvent_t parse_start, parse_stop;
    cudaEventCreate(&parse_start);
    cudaEventCreate(&parse_stop);
    cudaEventRecord(parse_start);

    buildASTKernel<<<1, 1>>>(
        d_tokens, token_count, d_nodes, d_node_count, g_config.max_nodes
    );

    cudaEventRecord(parse_stop);
    cudaEventSynchronize(parse_stop);

    float parsing_time;
    cudaEventElapsedTime(&parsing_time, parse_start, parse_stop);

    // Get node count
    uint32_t node_count;
    cudaMemcpy(&node_count, d_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Copy results to host
    *nodes_out = new ASTNode[node_count];
    cudaMemcpy(*nodes_out, d_nodes, node_count * sizeof(ASTNode), cudaMemcpyDeviceToHost);
    *node_count_out = node_count;

    // Collect metrics
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);

    if (metrics_out) {
        metrics_out->tokenization_time_ms = tokenization_time;
        metrics_out->parsing_time_ms = parsing_time;
        metrics_out->total_time_ms = total_time;
        metrics_out->nodes_created = node_count;
        metrics_out->tokens_processed = token_count;

        // Get GPU utilization (simplified)
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics_out->gpu_utilization = 1.0f - (float)free_mem / total_mem;
    }

    // Cleanup
    cudaFree(d_source);
    cudaFree(d_tokens);
    cudaFree(d_token_count);
    cudaFree(d_nodes);
    cudaFree(d_node_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(tok_start);
    cudaEventDestroy(tok_stop);
    cudaEventDestroy(parse_start);
    cudaEventDestroy(parse_stop);

    return cudaSuccess;
}

cudaError_t batchParseSource(
    const char** sources,
    uint32_t* source_lengths,
    uint32_t batch_size,
    ASTNode*** nodes_out,
    uint32_t** node_counts_out,
    ParserMetrics* metrics_out
) {
    // Allocate output arrays
    *nodes_out = new ASTNode*[batch_size];
    *node_counts_out = new uint32_t[batch_size];

    // Process each source file
    for (uint32_t i = 0; i < batch_size; i++) {
        ParserMetrics file_metrics;
        cudaError_t err = parseSource(
            sources[i],
            source_lengths[i],
            &(*nodes_out)[i],
            &(*node_counts_out)[i],
            &file_metrics
        );

        if (err != cudaSuccess) {
            return err;
        }

        if (metrics_out && i == 0) {
            *metrics_out = file_metrics;
        }
    }

    return cudaSuccess;
}

cudaError_t cleanupASTParser() {
    g_initialized = false;
    return cudaSuccess;
}

cudaError_t getParserMemoryUsage(size_t* bytes_used, size_t* bytes_total) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    *bytes_total = total_mem;
    *bytes_used = total_mem - free_mem;
    return cudaSuccess;
}

} // extern "C"

} // namespace cuda
} // namespace portalis
