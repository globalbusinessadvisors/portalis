#ifndef PORTALIS_EMBEDDING_GENERATOR_CUH
#define PORTALIS_EMBEDDING_GENERATOR_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <stdint.h>

namespace portalis {
namespace cuda {

// Embedding configuration
struct EmbeddingConfig {
    uint32_t vocab_size;          // Vocabulary size
    uint32_t embedding_dim;       // Embedding dimension
    uint32_t max_sequence_length; // Maximum sequence length
    uint32_t batch_size;          // Batch size for processing
    float dropout_rate;           // Dropout rate
    bool use_fp16;                // Use FP16 for faster computation
};

// Code snippet structure
struct CodeSnippet {
    uint32_t* token_ids;          // Token IDs
    uint32_t length;              // Sequence length
    float* embedding;             // Output embedding (host pointer)
};

// Similarity result
struct SimilarityResult {
    uint32_t query_idx;           // Query index
    uint32_t match_idx;           // Match index
    float similarity_score;       // Cosine similarity score
    float confidence;             // Confidence score
};

// Performance metrics
struct EmbeddingMetrics {
    float encoding_time_ms;       // Time for encoding
    float similarity_time_ms;     // Time for similarity computation
    float total_time_ms;          // Total time
    uint32_t sequences_processed; // Number of sequences processed
    float throughput_seq_per_sec; // Throughput in sequences/second
    float gpu_memory_used_mb;     // GPU memory used in MB
};

// Main API functions
extern "C" {
    // Initialize embedding generator
    cudaError_t initializeEmbeddingGenerator(
        EmbeddingConfig* config,
        const float* pretrained_embeddings  // Pre-trained embedding matrix
    );

    // Generate embeddings for code snippets
    cudaError_t generateEmbeddings(
        const uint32_t* token_ids,        // Token IDs [batch_size x max_seq_len]
        const uint32_t* sequence_lengths, // Actual lengths [batch_size]
        uint32_t batch_size,
        float* embeddings_out,            // Output [batch_size x embedding_dim]
        EmbeddingMetrics* metrics_out
    );

    // Compute pairwise similarity between code snippets
    cudaError_t computeSimilarity(
        const float* query_embeddings,    // Query embeddings [num_queries x dim]
        uint32_t num_queries,
        const float* candidate_embeddings,// Candidate embeddings [num_candidates x dim]
        uint32_t num_candidates,
        SimilarityResult* results_out,    // Top-K results per query
        uint32_t top_k,
        EmbeddingMetrics* metrics_out
    );

    // Find most similar code snippets (KNN search)
    cudaError_t findSimilarSnippets(
        const uint32_t* query_tokens,
        const uint32_t* query_lengths,
        uint32_t num_queries,
        const uint32_t* corpus_tokens,    // Large corpus of code
        const uint32_t* corpus_lengths,
        uint32_t corpus_size,
        SimilarityResult* results_out,
        uint32_t top_k,
        EmbeddingMetrics* metrics_out
    );

    // Cluster code snippets based on similarity
    cudaError_t clusterCodeSnippets(
        const uint32_t* token_ids,
        const uint32_t* sequence_lengths,
        uint32_t num_snippets,
        uint32_t num_clusters,
        uint32_t* cluster_assignments,    // Output cluster IDs
        float* cluster_centers,           // Output cluster centers
        EmbeddingMetrics* metrics_out
    );

    // Cleanup embedding generator
    cudaError_t cleanupEmbeddingGenerator();

    // Get memory usage
    cudaError_t getEmbeddingMemoryUsage(size_t* bytes_used, size_t* bytes_total);
}

// Kernel declarations
__global__ void embedTokensKernel(
    const uint32_t* token_ids,
    const float* embedding_matrix,
    uint32_t vocab_size,
    uint32_t embedding_dim,
    uint32_t batch_size,
    uint32_t max_seq_len,
    float* output_embeddings
);

__global__ void poolEmbeddingsKernel(
    const float* sequence_embeddings,    // [batch x seq_len x dim]
    const uint32_t* sequence_lengths,
    uint32_t batch_size,
    uint32_t max_seq_len,
    uint32_t embedding_dim,
    float* pooled_embeddings             // [batch x dim]
);

__global__ void computeCosineSimKernel(
    const float* queries,                // [num_queries x dim]
    const float* candidates,             // [num_candidates x dim]
    uint32_t num_queries,
    uint32_t num_candidates,
    uint32_t embedding_dim,
    float* similarity_matrix             // [num_queries x num_candidates]
);

__global__ void topKSimilarityKernel(
    const float* similarity_matrix,
    uint32_t num_queries,
    uint32_t num_candidates,
    uint32_t k,
    SimilarityResult* results_out
);

__global__ void normalizeEmbeddingsKernel(
    float* embeddings,
    uint32_t num_embeddings,
    uint32_t embedding_dim
);

__global__ void kMeansUpdateKernel(
    const float* embeddings,
    const uint32_t* cluster_assignments,
    uint32_t num_embeddings,
    uint32_t embedding_dim,
    uint32_t num_clusters,
    float* cluster_centers
);

__global__ void kMeansAssignKernel(
    const float* embeddings,
    const float* cluster_centers,
    uint32_t num_embeddings,
    uint32_t embedding_dim,
    uint32_t num_clusters,
    uint32_t* cluster_assignments,
    bool* changed
);

} // namespace cuda
} // namespace portalis

#endif // PORTALIS_EMBEDDING_GENERATOR_CUH
