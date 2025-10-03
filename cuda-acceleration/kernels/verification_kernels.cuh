#ifndef PORTALIS_VERIFICATION_KERNELS_CUH
#define PORTALIS_VERIFICATION_KERNELS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace portalis {
namespace cuda {

// Test case structure
struct TestCase {
    uint32_t* input_data;        // Input test data
    uint32_t input_size;         // Size of input
    uint32_t* expected_output;   // Expected output
    uint32_t output_size;        // Size of expected output
    uint32_t test_id;            // Unique test identifier
    float tolerance;             // Floating point comparison tolerance
};

// Test result
struct TestResult {
    uint32_t test_id;
    bool passed;
    float execution_time_ms;
    uint32_t* actual_output;
    uint32_t output_size;
    char error_message[256];
};

// Verification configuration
struct VerificationConfig {
    uint32_t max_concurrent_tests;
    uint32_t max_input_size;
    uint32_t max_output_size;
    float default_tolerance;
    bool enable_profiling;
};

// Performance metrics
struct VerificationMetrics {
    uint32_t tests_executed;
    uint32_t tests_passed;
    uint32_t tests_failed;
    float total_execution_time_ms;
    float average_test_time_ms;
    float gpu_utilization;
};

// Main API functions
extern "C" {
    // Initialize verification system
    cudaError_t initializeVerification(VerificationConfig* config);

    // Run parallel test suite
    cudaError_t runTestSuite(
        TestCase* test_cases,
        uint32_t num_tests,
        TestResult* results_out,
        VerificationMetrics* metrics_out
    );

    // Compare outputs for parity testing
    cudaError_t compareOutputs(
        const void* python_output,
        const void* rust_output,
        uint32_t output_size,
        float tolerance,
        bool* match_out,
        float* diff_percentage_out
    );

    // Batch parity verification
    cudaError_t batchParityCheck(
        const void** python_outputs,
        const void** rust_outputs,
        uint32_t* output_sizes,
        uint32_t num_outputs,
        float tolerance,
        bool* results_out,
        float* diff_percentages_out
    );

    // Property-based testing
    cudaError_t generatePropertyTests(
        uint32_t function_signature_hash,
        uint32_t num_tests,
        TestCase* tests_out
    );

    // Cleanup verification system
    cudaError_t cleanupVerification();
}

// Kernel declarations
__global__ void parallelTestExecutionKernel(
    TestCase* test_cases,
    uint32_t num_tests,
    TestResult* results
);

__global__ void compareArraysKernel(
    const float* array_a,
    const float* array_b,
    uint32_t size,
    float tolerance,
    bool* match,
    float* diff_percentage
);

__global__ void fuzzyCompareKernel(
    const void* data_a,
    const void* data_b,
    uint32_t size,
    float tolerance,
    uint32_t* mismatch_indices,
    uint32_t* mismatch_count
);

__global__ void propertyTestGeneratorKernel(
    uint32_t seed,
    uint32_t num_tests,
    TestCase* tests_out
);

} // namespace cuda
} // namespace portalis

#endif // PORTALIS_VERIFICATION_KERNELS_CUH
