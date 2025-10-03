#include "embedding_generator.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cmath>
#include <algorithm>

namespace portalis {
namespace cuda {

namespace cg = cooperative_groups;

// Global state
static EmbeddingConfig g_config;
static float* d_embedding_matrix = nullptr;
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_initialized = false;

// Device helper: dot product
__device__ inline float dotProduct(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Device helper: L2 norm
__device__ inline float l2Norm(const float* vec, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrtf(sum);
}

// Embed tokens using pre-trained embedding matrix
__global__ void embedTokensKernel(
    const uint32_t* token_ids,           // [batch x max_seq_len]
    const float* embedding_matrix,        // [vocab_size x embedding_dim]
    uint32_t vocab_size,
    uint32_t embedding_dim,
    uint32_t batch_size,
    uint32_t max_seq_len,
    float* output_embeddings              // [batch x max_seq_len x embedding_dim]
) {
    uint32_t batch_idx = blockIdx.x;
    uint32_t seq_idx = blockIdx.y;
    uint32_t dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= max_seq_len || dim_idx >= embedding_dim) {
        return;
    }

    uint32_t token_id = token_ids[batch_idx * max_seq_len + seq_idx];

    if (token_id >= vocab_size) {
        // Handle unknown token
        output_embeddings[batch_idx * max_seq_len * embedding_dim +
                         seq_idx * embedding_dim + dim_idx] = 0.0f;
        return;
    }

    // Copy embedding from pre-trained matrix
    float embedding_value = embedding_matrix[token_id * embedding_dim + dim_idx];
    output_embeddings[batch_idx * max_seq_len * embedding_dim +
                     seq_idx * embedding_dim + dim_idx] = embedding_value;
}

// Pool sequence embeddings (mean pooling)
__global__ void poolEmbeddingsKernel(
    const float* sequence_embeddings,     // [batch x seq_len x dim]
    const uint32_t* sequence_lengths,
    uint32_t batch_size,
    uint32_t max_seq_len,
    uint32_t embedding_dim,
    float* pooled_embeddings              // [batch x dim]
) {
    uint32_t batch_idx = blockIdx.x;
    uint32_t dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || dim_idx >= embedding_dim) {
        return;
    }

    uint32_t actual_length = sequence_lengths[batch_idx];
    if (actual_length == 0) {
        pooled_embeddings[batch_idx * embedding_dim + dim_idx] = 0.0f;
        return;
    }

    // Mean pooling over sequence length
    float sum = 0.0f;
    for (uint32_t seq_idx = 0; seq_idx < actual_length; seq_idx++) {
        sum += sequence_embeddings[batch_idx * max_seq_len * embedding_dim +
                                  seq_idx * embedding_dim + dim_idx];
    }

    pooled_embeddings[batch_idx * embedding_dim + dim_idx] = sum / actual_length;
}

// Normalize embeddings to unit length
__global__ void normalizeEmbeddingsKernel(
    float* embeddings,
    uint32_t num_embeddings,
    uint32_t embedding_dim
) {
    uint32_t emb_idx = blockIdx.x;
    uint32_t dim_idx = threadIdx.x;

    if (emb_idx >= num_embeddings || dim_idx >= embedding_dim) {
        return;
    }

    __shared__ float shared_norm;

    // Compute norm (only first thread)
    if (dim_idx == 0) {
        float* vec = &embeddings[emb_idx * embedding_dim];
        shared_norm = l2Norm(vec, embedding_dim);
        if (shared_norm < 1e-8f) {
            shared_norm = 1.0f;  // Avoid division by zero
        }
    }
    __syncthreads();

    // Normalize
    embeddings[emb_idx * embedding_dim + dim_idx] /= shared_norm;
}

// Compute cosine similarity matrix
__global__ void computeCosineSimKernel(
    const float* queries,                 // [num_queries x dim]
    const float* candidates,              // [num_candidates x dim]
    uint32_t num_queries,
    uint32_t num_candidates,
    uint32_t embedding_dim,
    float* similarity_matrix              // [num_queries x num_candidates]
) {
    uint32_t query_idx = blockIdx.x;
    uint32_t cand_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (query_idx >= num_queries || cand_idx >= num_candidates) {
        return;
    }

    __shared__ float shared_sum;

    if (tid == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();

    // Parallel dot product
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < embedding_dim; i += blockDim.x) {
        local_sum += queries[query_idx * embedding_dim + i] *
                     candidates[cand_idx * embedding_dim + i];
    }

    // Reduce across threads
    atomicAdd(&shared_sum, local_sum);
    __syncthreads();

    if (tid == 0) {
        // Assuming normalized embeddings, dot product = cosine similarity
        similarity_matrix[query_idx * num_candidates + cand_idx] = shared_sum;
    }
}

// Extract top-K similar items
__global__ void topKSimilarityKernel(
    const float* similarity_matrix,
    uint32_t num_queries,
    uint32_t num_candidates,
    uint32_t k,
    SimilarityResult* results_out
) {
    uint32_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_idx >= num_queries) {
        return;
    }

    const float* row = &similarity_matrix[query_idx * num_candidates];

    // Simple selection sort for top-K (can be optimized with heap)
    for (uint32_t top = 0; top < k && top < num_candidates; top++) {
        uint32_t best_idx = 0;
        float best_score = -1.0f;

        for (uint32_t i = 0; i < num_candidates; i++) {
            // Skip already selected
            bool already_selected = false;
            for (uint32_t j = 0; j < top; j++) {
                if (results_out[query_idx * k + j].match_idx == i) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && row[i] > best_score) {
                best_score = row[i];
                best_idx = i;
            }
        }

        results_out[query_idx * k + top].query_idx = query_idx;
        results_out[query_idx * k + top].match_idx = best_idx;
        results_out[query_idx * k + top].similarity_score = best_score;
        results_out[query_idx * k + top].confidence = best_score;  // Simplified
    }
}

// K-means clustering: assign points to nearest cluster
__global__ void kMeansAssignKernel(
    const float* embeddings,
    const float* cluster_centers,
    uint32_t num_embeddings,
    uint32_t embedding_dim,
    uint32_t num_clusters,
    uint32_t* cluster_assignments,
    bool* changed
) {
    uint32_t emb_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (emb_idx >= num_embeddings) {
        return;
    }

    const float* embedding = &embeddings[emb_idx * embedding_dim];

    // Find nearest cluster
    uint32_t best_cluster = 0;
    float best_distance = 1e9f;

    for (uint32_t c = 0; c < num_clusters; c++) {
        const float* center = &cluster_centers[c * embedding_dim];

        float distance = 0.0f;
        for (uint32_t d = 0; d < embedding_dim; d++) {
            float diff = embedding[d] - center[d];
            distance += diff * diff;
        }

        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = c;
        }
    }

    // Update assignment
    uint32_t old_assignment = cluster_assignments[emb_idx];
    if (old_assignment != best_cluster) {
        cluster_assignments[emb_idx] = best_cluster;
        *changed = true;
    }
}

// K-means clustering: update cluster centers
__global__ void kMeansUpdateKernel(
    const float* embeddings,
    const uint32_t* cluster_assignments,
    uint32_t num_embeddings,
    uint32_t embedding_dim,
    uint32_t num_clusters,
    float* cluster_centers
) {
    uint32_t cluster_idx = blockIdx.x;
    uint32_t dim_idx = threadIdx.x;

    if (cluster_idx >= num_clusters || dim_idx >= embedding_dim) {
        return;
    }

    __shared__ uint32_t cluster_count;
    __shared__ float cluster_sum[1024];  // Max embedding_dim = 1024

    if (dim_idx == 0) {
        cluster_count = 0;
    }
    __syncthreads();

    // Count members and sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < num_embeddings; i++) {
        if (cluster_assignments[i] == cluster_idx) {
            if (dim_idx == 0) {
                atomicAdd(&cluster_count, 1);
            }
            sum += embeddings[i * embedding_dim + dim_idx];
        }
    }
    __syncthreads();

    // Compute mean
    if (cluster_count > 0) {
        cluster_centers[cluster_idx * embedding_dim + dim_idx] = sum / cluster_count;
    }
}

// Host API implementation
extern "C" {

cudaError_t initializeEmbeddingGenerator(
    EmbeddingConfig* config,
    const float* pretrained_embeddings
) {
    if (g_initialized) {
        cleanupEmbeddingGenerator();
    }

    g_config = *config;

    // Allocate device memory for embedding matrix
    size_t embedding_matrix_size = config->vocab_size * config->embedding_dim * sizeof(float);
    cudaMalloc(&d_embedding_matrix, embedding_matrix_size);

    // Copy pre-trained embeddings to device
    if (pretrained_embeddings) {
        cudaMemcpy(d_embedding_matrix, pretrained_embeddings,
                  embedding_matrix_size, cudaMemcpyHostToDevice);
    } else {
        // Initialize with random values
        cudaMemset(d_embedding_matrix, 0, embedding_matrix_size);
    }

    // Create cuBLAS handle
    cublasCreate(&g_cublas_handle);

    g_initialized = true;
    return cudaSuccess;
}

cudaError_t generateEmbeddings(
    const uint32_t* token_ids,
    const uint32_t* sequence_lengths,
    uint32_t batch_size,
    float* embeddings_out,
    EmbeddingMetrics* metrics_out
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    uint32_t* d_token_ids;
    uint32_t* d_sequence_lengths;
    float* d_sequence_embeddings;
    float* d_pooled_embeddings;

    size_t token_ids_size = batch_size * g_config.max_sequence_length * sizeof(uint32_t);
    size_t lengths_size = batch_size * sizeof(uint32_t);
    size_t seq_emb_size = batch_size * g_config.max_sequence_length * g_config.embedding_dim * sizeof(float);
    size_t pooled_emb_size = batch_size * g_config.embedding_dim * sizeof(float);

    cudaMalloc(&d_token_ids, token_ids_size);
    cudaMalloc(&d_sequence_lengths, lengths_size);
    cudaMalloc(&d_sequence_embeddings, seq_emb_size);
    cudaMalloc(&d_pooled_embeddings, pooled_emb_size);

    // Copy inputs to device
    cudaMemcpy(d_token_ids, token_ids, token_ids_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequence_lengths, sequence_lengths, lengths_size, cudaMemcpyHostToDevice);

    // Embed tokens
    dim3 embed_grid(batch_size, g_config.max_sequence_length);
    dim3 embed_block(g_config.embedding_dim);

    cudaEvent_t embed_start, embed_stop;
    cudaEventCreate(&embed_start);
    cudaEventCreate(&embed_stop);
    cudaEventRecord(embed_start);

    embedTokensKernel<<<embed_grid, embed_block>>>(
        d_token_ids,
        d_embedding_matrix,
        g_config.vocab_size,
        g_config.embedding_dim,
        batch_size,
        g_config.max_sequence_length,
        d_sequence_embeddings
    );

    cudaEventRecord(embed_stop);
    cudaEventSynchronize(embed_stop);

    float encoding_time;
    cudaEventElapsedTime(&encoding_time, embed_start, embed_stop);

    // Pool embeddings
    dim3 pool_grid(batch_size);
    dim3 pool_block(g_config.embedding_dim);

    poolEmbeddingsKernel<<<pool_grid, pool_block>>>(
        d_sequence_embeddings,
        d_sequence_lengths,
        batch_size,
        g_config.max_sequence_length,
        g_config.embedding_dim,
        d_pooled_embeddings
    );

    // Normalize embeddings
    normalizeEmbeddingsKernel<<<batch_size, g_config.embedding_dim>>>(
        d_pooled_embeddings,
        batch_size,
        g_config.embedding_dim
    );

    // Copy results to host
    cudaMemcpy(embeddings_out, d_pooled_embeddings, pooled_emb_size, cudaMemcpyDeviceToHost);

    // Collect metrics
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);

    if (metrics_out) {
        metrics_out->encoding_time_ms = encoding_time;
        metrics_out->similarity_time_ms = 0.0f;
        metrics_out->total_time_ms = total_time;
        metrics_out->sequences_processed = batch_size;
        metrics_out->throughput_seq_per_sec = (batch_size / total_time) * 1000.0f;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics_out->gpu_memory_used_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);
    }

    // Cleanup
    cudaFree(d_token_ids);
    cudaFree(d_sequence_lengths);
    cudaFree(d_sequence_embeddings);
    cudaFree(d_pooled_embeddings);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(embed_start);
    cudaEventDestroy(embed_stop);

    return cudaSuccess;
}

cudaError_t computeSimilarity(
    const float* query_embeddings,
    uint32_t num_queries,
    const float* candidate_embeddings,
    uint32_t num_candidates,
    SimilarityResult* results_out,
    uint32_t top_k,
    EmbeddingMetrics* metrics_out
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    float* d_queries;
    float* d_candidates;
    float* d_similarity_matrix;
    SimilarityResult* d_results;

    size_t queries_size = num_queries * g_config.embedding_dim * sizeof(float);
    size_t candidates_size = num_candidates * g_config.embedding_dim * sizeof(float);
    size_t sim_matrix_size = num_queries * num_candidates * sizeof(float);
    size_t results_size = num_queries * top_k * sizeof(SimilarityResult);

    cudaMalloc(&d_queries, queries_size);
    cudaMalloc(&d_candidates, candidates_size);
    cudaMalloc(&d_similarity_matrix, sim_matrix_size);
    cudaMalloc(&d_results, results_size);

    // Copy inputs
    cudaMemcpy(d_queries, query_embeddings, queries_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, candidate_embeddings, candidates_size, cudaMemcpyHostToDevice);

    // Compute similarity matrix
    dim3 sim_grid(num_queries, num_candidates);
    dim3 sim_block(256);

    computeCosineSimKernel<<<sim_grid, sim_block>>>(
        d_queries,
        d_candidates,
        num_queries,
        num_candidates,
        g_config.embedding_dim,
        d_similarity_matrix
    );

    // Extract top-K
    int top_k_threads = 256;
    int top_k_blocks = (num_queries + top_k_threads - 1) / top_k_threads;

    topKSimilarityKernel<<<top_k_blocks, top_k_threads>>>(
        d_similarity_matrix,
        num_queries,
        num_candidates,
        top_k,
        d_results
    );

    // Copy results
    cudaMemcpy(results_out, d_results, results_size, cudaMemcpyDeviceToHost);

    // Metrics
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);

    if (metrics_out) {
        metrics_out->similarity_time_ms = total_time;
        metrics_out->total_time_ms = total_time;
        metrics_out->sequences_processed = num_queries;
    }

    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_candidates);
    cudaFree(d_similarity_matrix);
    cudaFree(d_results);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaSuccess;
}

cudaError_t findSimilarSnippets(
    const uint32_t* query_tokens,
    const uint32_t* query_lengths,
    uint32_t num_queries,
    const uint32_t* corpus_tokens,
    const uint32_t* corpus_lengths,
    uint32_t corpus_size,
    SimilarityResult* results_out,
    uint32_t top_k,
    EmbeddingMetrics* metrics_out
) {
    // Generate embeddings for queries
    float* query_embeddings = new float[num_queries * g_config.embedding_dim];
    EmbeddingMetrics query_metrics;

    cudaError_t err = generateEmbeddings(
        query_tokens,
        query_lengths,
        num_queries,
        query_embeddings,
        &query_metrics
    );

    if (err != cudaSuccess) {
        delete[] query_embeddings;
        return err;
    }

    // Generate embeddings for corpus
    float* corpus_embeddings = new float[corpus_size * g_config.embedding_dim];
    EmbeddingMetrics corpus_metrics;

    err = generateEmbeddings(
        corpus_tokens,
        corpus_lengths,
        corpus_size,
        corpus_embeddings,
        &corpus_metrics
    );

    if (err != cudaSuccess) {
        delete[] query_embeddings;
        delete[] corpus_embeddings;
        return err;
    }

    // Compute similarity
    EmbeddingMetrics sim_metrics;
    err = computeSimilarity(
        query_embeddings,
        num_queries,
        corpus_embeddings,
        corpus_size,
        results_out,
        top_k,
        &sim_metrics
    );

    // Aggregate metrics
    if (metrics_out) {
        metrics_out->encoding_time_ms = query_metrics.encoding_time_ms + corpus_metrics.encoding_time_ms;
        metrics_out->similarity_time_ms = sim_metrics.similarity_time_ms;
        metrics_out->total_time_ms = query_metrics.total_time_ms +
                                     corpus_metrics.total_time_ms +
                                     sim_metrics.total_time_ms;
        metrics_out->sequences_processed = num_queries + corpus_size;
    }

    delete[] query_embeddings;
    delete[] corpus_embeddings;

    return err;
}

cudaError_t clusterCodeSnippets(
    const uint32_t* token_ids,
    const uint32_t* sequence_lengths,
    uint32_t num_snippets,
    uint32_t num_clusters,
    uint32_t* cluster_assignments,
    float* cluster_centers,
    EmbeddingMetrics* metrics_out
) {
    // Generate embeddings
    float* embeddings = new float[num_snippets * g_config.embedding_dim];
    EmbeddingMetrics embed_metrics;

    cudaError_t err = generateEmbeddings(
        token_ids,
        sequence_lengths,
        num_snippets,
        embeddings,
        &embed_metrics
    );

    if (err != cudaSuccess) {
        delete[] embeddings;
        return err;
    }

    // Allocate device memory
    float* d_embeddings;
    uint32_t* d_assignments;
    float* d_centers;
    bool* d_changed;

    cudaMalloc(&d_embeddings, num_snippets * g_config.embedding_dim * sizeof(float));
    cudaMalloc(&d_assignments, num_snippets * sizeof(uint32_t));
    cudaMalloc(&d_centers, num_clusters * g_config.embedding_dim * sizeof(float));
    cudaMalloc(&d_changed, sizeof(bool));

    cudaMemcpy(d_embeddings, embeddings,
              num_snippets * g_config.embedding_dim * sizeof(float),
              cudaMemcpyHostToDevice);

    // Initialize cluster centers (k-means++)
    // Simplified: use first k points
    cudaMemcpy(d_centers, embeddings,
              num_clusters * g_config.embedding_dim * sizeof(float),
              cudaMemcpyHostToDevice);

    // Initialize assignments to 0
    cudaMemset(d_assignments, 0, num_snippets * sizeof(uint32_t));

    // K-means iterations
    const int max_iterations = 100;
    int block_size = 256;
    int grid_size = (num_snippets + block_size - 1) / block_size;

    for (int iter = 0; iter < max_iterations; iter++) {
        bool h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Assignment step
        kMeansAssignKernel<<<grid_size, block_size>>>(
            d_embeddings,
            d_centers,
            num_snippets,
            g_config.embedding_dim,
            num_clusters,
            d_assignments,
            d_changed
        );

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        if (!h_changed) {
            break;  // Converged
        }

        // Update step
        kMeansUpdateKernel<<<num_clusters, g_config.embedding_dim>>>(
            d_embeddings,
            d_assignments,
            num_snippets,
            g_config.embedding_dim,
            num_clusters,
            d_centers
        );
    }

    // Copy results
    cudaMemcpy(cluster_assignments, d_assignments,
              num_snippets * sizeof(uint32_t),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_centers, d_centers,
              num_clusters * g_config.embedding_dim * sizeof(float),
              cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] embeddings;
    cudaFree(d_embeddings);
    cudaFree(d_assignments);
    cudaFree(d_centers);
    cudaFree(d_changed);

    return cudaSuccess;
}

cudaError_t cleanupEmbeddingGenerator() {
    if (d_embedding_matrix) {
        cudaFree(d_embedding_matrix);
        d_embedding_matrix = nullptr;
    }

    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }

    g_initialized = false;
    return cudaSuccess;
}

cudaError_t getEmbeddingMemoryUsage(size_t* bytes_used, size_t* bytes_total) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    *bytes_total = total_mem;
    *bytes_used = total_mem - free_mem;
    return cudaSuccess;
}

} // extern "C"

} // namespace cuda
} // namespace portalis
