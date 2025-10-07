# Portalis Platform - Visual Integration Diagrams

**Date**: 2025-10-07
**Companion to**: INTEGRATION_ARCHITECTURE_MAP.md

---

## 1. Complete System Integration Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PORTALIS PLATFORM ARCHITECTURE                      │
│                                                                               │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                         USER INTERFACE LAYER                        │     │
│  │                                                                      │     │
│  │  ┌──────────────────────────────────────────────────────────────┐ │     │
│  │  │  CLI (cli/src/commands/convert.rs)                           │ │     │
│  │  │                                                                │ │     │
│  │  │  Flags:                                                        │ │     │
│  │  │    --simd          → Enable SIMD optimizations [Line 48]     │ │     │
│  │  │    --cpu-only      → Force CPU execution [Line 45]           │ │     │
│  │  │    --hybrid        → GPU+CPU execution [Line 52]             │ │     │
│  │  │    --jobs=N        → Thread count [Line 40]                  │ │     │
│  │  │                                                                │ │     │
│  │  │  Integration: Lines 252-289 (#[cfg(feature = "acceleration")])│ │     │
│  │  └────────────────────────────┬─────────────────────────────────┘ │     │
│  └────────────────────────────────┼───────────────────────────────────┘     │
│                                   │                                          │
│                                   │ AccelerationConfig                       │
│                                   │ (Line 255-257)                           │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    CORE ACCELERATION LAYER                          │     │
│  │                (core/src/acceleration/)                             │     │
│  │                                                                      │     │
│  │  ┌─────────────────────────────────────────────────────────────┐  │     │
│  │  │ StrategyManager<C, G> (executor.rs:246-563)                 │  │     │
│  │  │                                                               │  │     │
│  │  │ ┌───────────────────────────────────────────────────────┐   │  │     │
│  │  │ │  Auto-Selection Logic (Lines 323-366)                 │   │  │     │
│  │  │ │                                                         │   │  │     │
│  │  │ │  Stage 1: GPU Available? ────────────┐                │   │  │     │
│  │  │ │           NO → CpuOnly               │                │   │  │     │
│  │  │ │           YES ↓                      │                │   │  │     │
│  │  │ │                                      │                │   │  │     │
│  │  │ │  Stage 2: Small Workload (< 10)?    │                │   │  │     │
│  │  │ │           YES → CpuOnly             │                │   │  │     │
│  │  │ │           NO ↓                       │                │   │  │     │
│  │  │ │                                      │                │   │  │     │
│  │  │ │  Stage 3: GPU Memory > 80%?         │                │   │  │     │
│  │  │ │           YES → Hybrid (60/40)      │                │   │  │     │
│  │  │ │           NO ↓                       │                │   │  │     │
│  │  │ │                                      │                │   │  │     │
│  │  │ │  Stage 4: Large Parallel (> 50)?    │                │   │  │     │
│  │  │ │           YES → GpuOnly             │                │   │  │     │
│  │  │ │           NO → CpuOnly              │                │   │  │     │
│  │  │ └─────────────────────────────────────────────────────┘   │  │     │
│  │  │                                                               │  │     │
│  │  │ ┌───────────────────────────────────────────────────────┐   │  │     │
│  │  │ │  execute() Method (Lines 369-444)                     │   │  │     │
│  │  │ │                                                         │   │  │     │
│  │  │ │  1. Detect strategy (auto_select_strategy)            │   │  │     │
│  │  │ │  2. Match strategy:                                   │   │  │     │
│  │  │ │     ┌─────────────────────────────────────────────┐  │   │  │     │
│  │  │ │     │ GpuOnly (Lines 394-403)                     │  │   │  │     │
│  │  │ │     │   Try GPU → Fallback CPU on error           │  │   │  │     │
│  │  │ │     │   warn!("GPU failed, falling back")         │  │   │  │     │
│  │  │ │     └─────────────────────────────────────────────┘  │   │  │     │
│  │  │ │     ┌─────────────────────────────────────────────┐  │   │  │     │
│  │  │ │     │ CpuOnly (Line 406)                          │  │   │  │     │
│  │  │ │     │   cpu_bridge.execute_batch(tasks, fn)       │  │   │  │     │
│  │  │ │     └─────────────────────────────────────────────┘  │   │  │     │
│  │  │ │     ┌─────────────────────────────────────────────┐  │   │  │     │
│  │  │ │     │ Hybrid (Lines 408-421)                      │  │   │  │     │
│  │  │ │     │   Split: 60% GPU + 40% CPU                  │  │   │  │     │
│  │  │ │     │   Parallel threads (Lines 514-541)          │  │   │  │     │
│  │  │ │     │   Combine results (Line 543)                │  │   │  │     │
│  │  │ │     └─────────────────────────────────────────────┘  │   │  │     │
│  │  │ │  3. Return ExecutionResult<T>                         │   │  │     │
│  │  │ └───────────────────────────────────────────────────────┘   │  │     │
│  │  └───────────────────────────────────────────────────────────────┘  │     │
│  │                                                                      │     │
│  │  Traits:                                                             │     │
│  │  ┌────────────────────┐         ┌────────────────────┐             │     │
│  │  │ CpuExecutor        │         │ GpuExecutor        │             │     │
│  │  │ (Lines 566-576)    │         │ (Lines 579-593)    │             │     │
│  │  │                    │         │                    │             │     │
│  │  │ execute_batch()    │         │ execute_batch()    │             │     │
│  │  │                    │         │ is_available()     │             │     │
│  │  │                    │         │ memory_available() │             │     │
│  │  └────────┬───────────┘         └────────┬───────────┘             │     │
│  └───────────┼──────────────────────────────┼───────────────────────────┘     │
│              │                              │                                 │
│              │ impl                         │ impl                            │
│              ▼                              ▼                                 │
│  ┌───────────────────────┐      ┌─────────────────────────┐                 │
│  │   CPU BRIDGE          │      │   CUDA BRIDGE           │                 │
│  │ (agents/cpu-bridge/)  │      │ (agents/cuda-bridge/)   │                 │
│  │                       │      │                         │                 │
│  │ ┌───────────────────┐ │      │ [GPU-specific         │                 │
│  │ │ CpuBridge         │ │      │  implementation]       │                 │
│  │ │ (lib.rs:72-321)   │ │      │                        │                 │
│  │ │                   │ │      │ - CUDA kernels         │                 │
│  │ │ Fields:           │ │      │ - GPU memory mgmt      │                 │
│  │ │  • thread_pool    │ │      │ - PCIe transfers       │                 │
│  │ │  • config         │ │      │                        │                 │
│  │ │  • metrics        │ │      └────────────────────────┘                 │
│  │ │                   │ │                                                  │
│  │ │ Methods:          │ │                                                  │
│  │ │  • new()          │ │                                                  │
│  │ │  • with_config()  │ │                                                  │
│  │ │  • parallel_      │ │                                                  │
│  │ │    translate()    │ │                                                  │
│  │ │  • execute_batch()│ │ ◄── Trait impl (Lines 328-360)                  │
│  │ └───────┬───────────┘ │                                                  │
│  │         │             │                                                  │
│  │         │ uses        │                                                  │
│  │         ▼             │                                                  │
│  │ ┌───────────────────┐ │                                                  │
│  │ │ SIMD Module       │ │                                                  │
│  │ │ (simd.rs:1-802)   │ │                                                  │
│  │ │                   │ │                                                  │
│  │ │ ┌───────────────┐ │ │                                                  │
│  │ │ │ Capability    │ │ │                                                  │
│  │ │ │ Detection     │ │ │                                                  │
│  │ │ │ (Lines 89-139)│ │ │                                                  │
│  │ │ │               │ │ │                                                  │
│  │ │ │ x86_64:       │ │ │                                                  │
│  │ │ │  • AVX2       │ │ │                                                  │
│  │ │ │  • SSE4.2     │ │ │                                                  │
│  │ │ │ aarch64:      │ │ │                                                  │
│  │ │ │  • NEON       │ │ │                                                  │
│  │ │ └───────────────┘ │ │                                                  │
│  │ │                   │ │                                                  │
│  │ │ Operations:       │ │                                                  │
│  │ │ ┌───────────────┐ │ │                                                  │
│  │ │ │batch_string_  │ │ │                                                  │
│  │ │ │contains()     │ │ │ → Import detection (3-4x speedup)               │
│  │ │ │(Lines 174-196)│ │ │                                                  │
│  │ │ └───────────────┘ │ │                                                  │
│  │ │ ┌───────────────┐ │ │                                                  │
│  │ │ │parallel_      │ │ │                                                  │
│  │ │ │string_match() │ │ │ → Prefix matching (2-3x speedup)                │
│  │ │ │(Lines 263-285)│ │ │                                                  │
│  │ │ └───────────────┘ │ │                                                  │
│  │ │ ┌───────────────┐ │ │                                                  │
│  │ │ │vectorized_    │ │ │                                                  │
│  │ │ │char_count()   │ │ │ → Character counting (4x speedup)               │
│  │ │ │(Lines 517-544)│ │ │                                                  │
│  │ │ └───────────────┘ │ │                                                  │
│  │ └───────────────────┘ │                                                  │
│  │                       │                                                  │
│  │ Thread Pool:          │                                                  │
│  │ ┌───────────────────┐ │                                                  │
│  │ │ Rayon Work-       │ │                                                  │
│  │ │ Stealing Scheduler│ │                                                  │
│  │ │                   │ │                                                  │
│  │ │ Features:         │ │                                                  │
│  │ │  • Auto-balance   │ │                                                  │
│  │ │  • Lock-free      │ │                                                  │
│  │ │  • Cache-friendly │ │                                                  │
│  │ └───────────────────┘ │                                                  │
│  └───────────────────────┘                                                  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    TRANSPILER AGENT                                 │     │
│  │              (agents/transpiler/src/lib.rs)                         │     │
│  │                                                                      │     │
│  │  ┌────────────────────────────────────────────────────────────┐    │     │
│  │  │ TranspilerAgent Struct (Lines 217-222)                     │    │     │
│  │  │                                                              │    │     │
│  │  │ Fields:                                                      │    │     │
│  │  │   id: AgentId                                               │    │     │
│  │  │   translation_mode: TranslationMode                         │    │     │
│  │  │   #[cfg(feature = "acceleration")]                          │    │     │
│  │  │   acceleration: Option<Arc<StrategyManager<...>>>  ◄───────┐│    │     │
│  │  └──────────────────────────────────────────────────────│─────┘│    │     │
│  │                                                          │      │    │     │
│  │  Constructors:                                          │      │    │     │
│  │  ┌──────────────────────────────────────────────────────▼─────┐│    │     │
│  │  │ with_acceleration(config) (Lines 274-301)                  ││    │     │
│  │  │                                                             ││    │     │
│  │  │ 1. Create CpuBridge from config                            ││    │     │
│  │  │    let cpu_config = CpuConfig::builder()                   ││    │     │
│  │  │        .num_threads(config.cpu_threads.unwrap_or(ncpus))   ││    │     │
│  │  │        .build();                                            ││    │     │
│  │  │                                                             ││    │     │
│  │  │ 2. Create StrategyManager                                  ││    │     │
│  │  │    StrategyManager::with_strategy(                         ││    │     │
│  │  │        Arc::new(CpuBridge::with_config(cpu_config)),       ││    │     │
│  │  │        None,  // No GPU                                    ││    │     │
│  │  │        config.strategy                                     ││    │     │
│  │  │    )                                                        ││    │     │
│  │  └─────────────────────────────────────────────────────────────┘│    │     │
│  │                                                                  │    │     │
│  │  Translation Methods:                                            │    │     │
│  │  ┌────────────────────────────────────────────────────────────┐ │    │     │
│  │  │ translate_batch_accelerated(files) (Lines 409-471)        │ │    │     │
│  │  │                                                             │ │    │     │
│  │  │ if let Some(strategy_manager) = &self.acceleration {      │ │    │     │
│  │  │     let result = strategy_manager.execute(               │ │    │     │
│  │  │         files,                                            │ │    │     │
│  │  │         |python_source| {                                 │ │    │     │
│  │  │             // Translate single file                      │ │    │     │
│  │  │             let rust_code = self.translate_python_module( │ │    │     │
│  │  │                 python_source                             │ │    │     │
│  │  │             )?;                                            │ │    │     │
│  │  │             Ok(TranspilerOutput { rust_code, metadata })  │ │    │     │
│  │  │         }                                                  │ │    │     │
│  │  │     )?;                                                    │ │    │     │
│  │  │     return Ok(result.outputs);                            │ │    │     │
│  │  │ }                                                          │ │    │     │
│  │  │                                                             │ │    │     │
│  │  │ // Fallback: sequential processing (Lines 449-470)        │ │    │     │
│  │  └─────────────────────────────────────────────────────────────┘ │    │     │
│  └──────────────────────────────────────────────────────────────────┘    │     │
│                                                                           │     │
│  ┌────────────────────────────────────────────────────────────────────┐ │     │
│  │              WEBASSEMBLY INTEGRATION                                │ │     │
│  │            (agents/wassette-bridge/)                                │ │     │
│  │                                                                      │ │     │
│  │  ┌────────────────────────────────────────────────────────────┐   │ │     │
│  │  │ WassetteClient (lib.rs:131-229)                            │   │ │     │
│  │  │                                                              │   │ │     │
│  │  │ Configuration:                                              │   │ │     │
│  │  │   • Sandbox enabled (default: true)                        │   │ │     │
│  │  │   • Max memory: 128MB                                      │   │ │     │
│  │  │   • Max exec time: 30s                                     │   │ │     │
│  │  │   • Permissions: ComponentPermissions                      │   │ │     │
│  │  │                                                              │   │ │     │
│  │  │ Methods:                                                    │   │ │     │
│  │  │   ┌──────────────────────────────────────────────┐        │   │ │     │
│  │  │   │ validate_component(path) (Lines 195-216)     │        │   │ │     │
│  │  │   │   → ValidationReport { is_valid, errors }    │        │   │ │     │
│  │  │   └──────────────────────────────────────────────┘        │   │ │     │
│  │  │   ┌──────────────────────────────────────────────┐        │   │ │     │
│  │  │   │ load_component(path) (Lines 157-169)         │        │   │ │     │
│  │  │   │   → ComponentHandle { id, path }             │        │   │ │     │
│  │  │   └──────────────────────────────────────────────┘        │   │ │     │
│  │  │   ┌──────────────────────────────────────────────┐        │   │ │     │
│  │  │   │ execute_component(handle, args)              │        │   │ │     │
│  │  │   │      (Lines 172-192)                          │        │   │ │     │
│  │  │   │   → ExecutionResult { success, output }      │        │   │ │     │
│  │  │   └──────────────────────────────────────────────┘        │   │ │     │
│  │  │                                                              │   │ │     │
│  │  │ Feature flags:                                              │   │ │     │
│  │  │   • runtime: Full Wasmtime integration                     │   │ │     │
│  │  │   • (default): Mock validation only                        │   │ │     │
│  │  └──────────────────────────────────────────────────────────────┘   │ │     │
│  └──────────────────────────────────────────────────────────────────────┘ │     │
│                                                                             │     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Through System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: User Invocation
━━━━━━━━━━━━━━━━━━━━━━
$ portalis convert script.py --simd --jobs=8
                │
                │ Parse CLI arguments
                ▼
        ┌───────────────┐
        │ ConvertCommand│
        │  .execute()   │
        └───────┬───────┘
                │
                │ Line 254: Check --simd flag
                │ Line 259: Extract --jobs value
                │ Line 264: Configure strategy
                ▼

Step 2: Acceleration Configuration (Lines 252-289)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #[cfg(feature = "acceleration")]
        {
            let mut config = AccelerationConfig::cpu_only();  // Line 257
            config.cpu_threads = Some(8);                      // Line 260

            if self.hybrid {
                config.strategy = ExecutionStrategy::Hybrid {
                    gpu_allocation: 70,
                    cpu_allocation: 30,
                };
            }
        }
                │
                │ config
                ▼

Step 3: TranspilerAgent Creation (Line 282)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let transpiler = TranspilerAgent::with_acceleration(config);
                │
                │ Line 274-301
                ▼
        ┌────────────────────────────────┐
        │ TranspilerAgent::              │
        │   with_acceleration(config)    │
        │                                │
        │ 1. Create CpuBridge:           │
        │    CpuConfig::builder()        │
        │      .num_threads(8)           │
        │      .build()                  │
        │                                │
        │ 2. Create StrategyManager:     │
        │    StrategyManager::           │
        │      with_strategy(            │
        │        cpu_bridge,             │
        │        None,  // No GPU        │
        │        ExecutionStrategy::Auto │
        │      )                         │
        │                                │
        │ 3. Store in agent:             │
        │    self.acceleration =         │
        │      Some(Arc::new(manager))   │
        └────────────────┬───────────────┘
                         │
                         ▼

Step 4: Read Python Source (Line 245)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let python_code = std::fs::read_to_string(path)?;
                │
                │ "def add(a, b):\n    return a + b"
                ▼

Step 5: Translation Invocation (Line 291)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let rust_code = transpiler.translate_python_module(&python_code)?;
                │
                │ Line 353-376
                ▼
        ┌────────────────────────────────┐
        │ translate_python_module()      │
        │                                │
        │ match translation_mode {       │
        │   AstBased => {                │
        │     translate_with_ast(code)   │
        │   }                            │
        │ }                              │
        └────────────────┬───────────────┘
                         │
                         │ For batch: Use translate_batch_accelerated
                         ▼

Step 6: Batch Acceleration (Lines 409-471)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if let Some(strategy_manager) = &self.acceleration {
            let result = strategy_manager.execute(
                files,
                |python_source| {
                    // Translate each file
                    let rust_code = self.translate_python_module(
                        python_source
                    )?;
                    Ok(TranspilerOutput { rust_code, metadata })
                }
            )?;
        }
                │
                │ Passes to StrategyManager
                ▼

Step 7: Strategy Execution (executor.rs:369-444)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let workload = WorkloadProfile::from_task_count(files.len());
        let strategy = self.detect_strategy(&workload);

        match strategy {
            CpuOnly => {
                self.cpu_bridge.execute_batch(tasks, process_fn)
            }
            │
            │ CPU path (most common)
            ▼
        ┌────────────────────────────────┐
        │ CpuBridge::execute_batch()     │
        │ (lib.rs:328-360)               │
        │                                │
        │ self.thread_pool.install(|| { │
        │     tasks                      │
        │       .par_iter()              │
        │       .map(|task| {            │
        │           process_fn(task)     │
        │       })                       │
        │       .collect()               │
        │ })                             │
        └────────────────┬───────────────┘
                         │
                         │ Parallel execution across 8 threads
                         ▼

Step 8: Rayon Work Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Thread 1           Thread 2           Thread 3           Thread 4
    ┌────────┐        ┌────────┐        ┌────────┐        ┌────────┐
    │ Task 1 │        │ Task 2 │        │ Task 3 │        │ Task 4 │
    │ Task 5 │        │ Task 6 │        │ Task 7 │        │ Task 8 │
    │ ...    │        │ ...    │        │ ...    │        │ ...    │
    └───┬────┘        └───┬────┘        └───┬────┘        └───┬────┘
        │                 │                 │                 │
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘
                                │
                                │ Work-stealing load balancing
                                ▼

Step 9: SIMD Acceleration (simd.rs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For each Python file during parsing:

    ┌────────────────────────────────────────┐
    │ batch_string_contains()                │
    │   Input: ["import os", "import sys"]   │
    │   Pattern: "import"                    │
    │                                        │
    │ AVX2 Path (x86_64):                    │
    │   Process 32 bytes at once             │
    │   _mm256_loadu_si256()                 │
    │   _mm256_cmpeq_epi8()                  │
    │                                        │
    │ Result: [true, true]                   │
    └────────────────────────────────────────┘
                │
                │ 3-4x faster than scalar
                ▼
    ┌────────────────────────────────────────┐
    │ vectorized_char_count()                │
    │   Input: "def function():\n    pass"   │
    │   Character: '('                       │
    │                                        │
    │ AVX2 Path:                             │
    │   Process 32 chars at once             │
    │   Count matches in parallel            │
    │                                        │
    │ Result: 2                              │
    └────────────────────────────────────────┘
                │
                │ 4x faster than scalar
                ▼

Step 10: AST Translation
━━━━━━━━━━━━━━━━━━━━━━━━

    PythonParser::parse(python_code)
        │
        │ RustPython parser (AST generation)
        ▼
    Python AST Nodes
        │
        │ Memory: ~4KB overhead per file
        │ Future: Arena allocator for 50-70% reduction
        ▼
    PythonToRustTranslator::translate_module(ast)
        │
        │ Generate Rust code from AST
        ▼
    Rust Code String
        │
        │ "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"
        ▼

Step 11: Result Collection
━━━━━━━━━━━━━━━━━━━━━━━━━

    Vec<TranspilerOutput>
        │
        │ Collected from all threads
        │ Metrics recorded (execution time, task count)
        ▼
    ExecutionResult {
        outputs: [output1, output2, ...],
        strategy_used: CpuOnly,
        execution_time: Duration::from_millis(150),
        fallback_occurred: false,
        errors: []
    }
        │
        │ Return to CLI
        ▼

Step 12: Output to Disk (Line 298)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let rust_path = output_dir.join("script.rs");
    std::fs::write(&rust_path, &rust_code)?;
        │
        │ Write translated Rust code
        ▼
    File: ./dist/script.rs
    ┌────────────────────────────────────┐
    │ // Generated by Portalis           │
    │ #![allow(unused)]                  │
    │                                    │
    │ pub fn add(a: i32, b: i32) -> i32 {│
    │     a + b                          │
    │ }                                  │
    └────────────────────────────────────┘
        │
        │ Success!
        ▼
    ✅ Conversion complete!
```

---

## 3. Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY FLOW & OPTIMIZATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1: CLI Input
━━━━━━━━━━━━━━━━━━
    Stack:
    ┌──────────────────┐
    │ ConvertCommand   │ (small struct)
    │   input: PathBuf │
    │   format: enum   │
    │   simd: bool     │
    └──────────────────┘

    Heap:
    ┌──────────────────────────────┐
    │ Python Source Files          │
    │   script.py → String         │ ~5KB per file
    │   (owned, moved to agent)    │
    └──────────────────────────────┘

Stage 2: Transpiler Agent
━━━━━━━━━━━━━━━━━━━━━━━━━
    Stack:
    ┌────────────────────────────┐
    │ TranspilerAgent            │
    │   id: AgentId              │
    │   mode: enum               │
    │   accel: Option<Arc<...>>  │ ← Shared ownership
    └────────────────────────────┘

    Heap (Shared):
    ┌────────────────────────────┐
    │ Arc<StrategyManager>       │
    │   Reference count: 1→2→1   │
    │                            │
    │ ┌────────────────────────┐ │
    │ │ CpuBridge              │ │
    │ │   thread_pool: Rayon   │ │ ← Global singleton
    │ │   config: CpuConfig    │ │
    │ │   metrics: Arc<RwLock> │ │ ← Concurrent access
    │ └────────────────────────┘ │
    └────────────────────────────┘

Stage 3: Task Distribution (Hybrid Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┌────────────────────────────────────┐
    │ Original Tasks: Vec<String>        │
    │   ["src1", "src2", "src3", ...]    │
    │   Memory: N × ~5KB                 │
    └──────────┬─────────────────────────┘
               │
               │ Split for Hybrid (60/40)
               ▼
    ┌─────────────────────┬──────────────────────┐
    │ GPU Tasks (60%)     │ CPU Tasks (40%)      │
    │ ┌─────────────────┐ │ ┌────────────────┐  │
    │ │ Clone (Cow)     │ │ │ Clone (Cow)    │  │
    │ │   OR            │ │ │   OR           │  │
    │ │ Borrowed slice  │ │ │ Borrowed slice │  │
    │ └─────────────────┘ │ └────────────────┘  │
    │                     │                      │
    │ Memory Impact:      │ Memory Impact:       │
    │   Current: Full     │   Current: Full      │
    │   Optimized: Zero   │   Optimized: Zero    │
    └─────────────────────┴──────────────────────┘
                │                    │
                │ Parallel execution │
                ▼                    ▼

Stage 4: Parallel Processing (CPU Path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Rayon Thread Pool (8 threads):

    Thread 1:                Thread 2:
    ┌──────────────────┐    ┌──────────────────┐
    │ Stack Frame:     │    │ Stack Frame:     │
    │   task: &String  │    │   task: &String  │ ← Borrowed ref
    │   (reference)    │    │   (reference)    │   (zero-copy)
    │                  │    │                  │
    │ Parse Python:    │    │ Parse Python:    │
    │ ┌──────────────┐ │    │ ┌──────────────┐ │
    │ │ AST Nodes    │ │    │ │ AST Nodes    │ │
    │ │ ~4KB heap    │ │    │ │ ~4KB heap    │ │ ← Individual
    │ │ allocations  │ │    │ │ allocations  │ │   allocations
    │ └──────────────┘ │    │ └──────────────┘ │
    │                  │    │                  │
    │ Generate Rust:   │    │ Generate Rust:   │
    │ ┌──────────────┐ │    │ ┌──────────────┐ │
    │ │ String       │ │    │ │ String       │ │
    │ │ ~6KB output  │ │    │ │ ~6KB output  │ │
    │ └──────────────┘ │    │ └──────────────┘ │
    └──────────────────┘    └──────────────────┘
            │                       │
            │ Return results        │
            ▼                       ▼
    ┌─────────────────────────────────────────┐
    │ Vec<TranspilerOutput>                   │
    │   Pre-allocated capacity: files.len()   │
    │   No reallocation during collection     │
    └─────────────────────────────────────────┘

Stage 5: SIMD Memory Access
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┌────────────────────────────────────┐
    │ Input String: "import sys, os"     │
    │   Memory: Contiguous bytes         │
    │   Alignment: May be unaligned      │
    └──────────┬─────────────────────────┘
               │
               │ SIMD Operation
               ▼
    AVX2 Register (256-bit):
    ┌────────────────────────────────────┐
    │ [i][m][p][o][r][t][ ][s][y][s][,]  │ ← 32 bytes
    │ [ ][o][s][...] (padding)           │   loaded at once
    └────────────────────────────────────┘
               │
               │ Compare with pattern
               ▼
    Result Mask:
    ┌────────────────────────────────────┐
    │ [1][1][1][1][1][1][0][0][0][0][0]  │ ← 32-bit mask
    │ [0][0][0][...] (padding)           │   (1 = match)
    └────────────────────────────────────┘
               │
               │ Count matches (popcount)
               ▼
    Match count: 6 (for "import")

    Memory Access Pattern:
    ┌────────────────────────────────────┐
    │ Scalar:    1 byte × 32 iterations  │ → 32 loads
    │ AVX2:      32 bytes × 1 iteration  │ → 1 load
    │ Speedup:   ~4x (includes overhead) │
    └────────────────────────────────────┘

Stage 6: Memory Optimization Opportunities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Current State:
    ┌────────────────────────────────────┐
    │ AST Node Allocation                │
    │   • Individual heap allocations    │
    │   • ~4KB overhead per file         │
    │   • Fragmentation over time        │
    └────────────────────────────────────┘
               │
               │ Optimization: Arena Allocator
               ▼
    Optimized State:
    ┌────────────────────────────────────┐
    │ Arena<AstNode>                     │
    │   • Single 64KB allocation         │
    │   • Bump allocator (fast)          │
    │   • All nodes dropped together     │
    │   • Cache-friendly (contiguous)    │
    │                                    │
    │ Benefit:                           │
    │   • 50-70% faster allocation       │
    │   • Better cache locality          │
    │   • Instant deallocation           │
    └────────────────────────────────────┘

    Current State:
    ┌────────────────────────────────────┐
    │ Stdlib Identifier Strings          │
    │   • "std::io" duplicated 50×       │
    │   • Memory: 50 × 8 bytes = 400B    │
    └────────────────────────────────────┘
               │
               │ Optimization: String Interning
               ▼
    Optimized State:
    ┌────────────────────────────────────┐
    │ StringInterner                     │
    │   • "std::io" stored once          │
    │   • Symbol: u32 (4 bytes)          │
    │   • Memory: 8B + (50 × 4B) = 208B  │
    │                                    │
    │ Benefit:                           │
    │   • 48% memory reduction           │
    │   • Faster comparisons (integer)   │
    │   • Copy instead of clone          │
    └────────────────────────────────────┘

Total Memory Footprint (100 files):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input:           100 × 5KB    = 500KB
    AST (current):   100 × 4KB    = 400KB
    AST (arena):     1 × 64KB     = 64KB   (84% reduction)
    Output:          100 × 6KB    = 600KB
    ────────────────────────────────────────
    Total (current): 1.5MB
    Total (opt):     1.164MB      (22% reduction)
```

---

## 4. SIMD Acceleration Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMD VECTORIZATION ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

Platform Detection (simd.rs:89-139)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Runtime Detection:
    ┌────────────────────────────────────┐
    │ detect_cpu_capabilities()          │
    │                                    │
    │ if SIMD_INITIALIZED.load() {       │
    │     return cached_capabilities;    │ ← Fast path (< 1μs)
    │ }                                  │
    │                                    │
    │ #[cfg(target_arch = "x86_64")]     │
    │ {                                  │
    │     AVX2_AVAILABLE =               │
    │       is_x86_feature_detected!()   │ ← CPUID instruction
    │     SSE42_AVAILABLE =              │   (~100 cycles)
    │       is_x86_feature_detected!()   │
    │ }                                  │
    │                                    │
    │ #[cfg(target_arch = "aarch64")]    │
    │ {                                  │
    │     NEON_AVAILABLE = true;         │ ← Always available
    │ }                                  │
    │                                    │
    │ SIMD_INITIALIZED.store(true);      │
    └────────────────────────────────────┘
               │
               │ Cache for subsequent calls
               ▼
    static AVX2_AVAILABLE: bool
    static SSE42_AVAILABLE: bool
    static NEON_AVAILABLE: bool

Operation 1: String Contains (simd.rs:174-196)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input: haystack = ["import os", "use std", "import sys"]
           needle = "import"

    Scalar Path (fallback):
    ┌────────────────────────────────────┐
    │ for s in haystack {                │
    │     if s.contains(needle) {        │ ← Linear scan
    │         result.push(true);         │   O(n×m)
    │     }                              │
    │ }                                  │
    │                                    │
    │ Performance: 100ns per string      │
    └────────────────────────────────────┘

    AVX2 Path (x86_64):
    ┌────────────────────────────────────────────────────┐
    │ #[target_feature(enable = "avx2")]                 │
    │ unsafe fn batch_string_contains_avx2() {           │
    │                                                     │
    │   let pattern_vec =                                │
    │     _mm256_set1_epi8(first_char);  // Broadcast   │
    │                                                     │
    │   // Process 32 bytes at once                      │
    │   for chunk in bytes.chunks(32) {                  │
    │     let data_vec =                                 │
    │       _mm256_loadu_si256(chunk);   // Load 32 bytes│
    │                                                     │
    │     let cmp_vec =                                  │
    │       _mm256_cmpeq_epi8(           // Compare all  │
    │         pattern_vec,                               │
    │         data_vec                                   │
    │       );                                           │
    │                                                     │
    │     let mask =                                     │
    │       _mm256_movemask_epi8(cmp_vec); // Extract    │
    │                                                     │
    │     if mask != 0 { /* found match */ }            │
    │   }                                                 │
    │ }                                                   │
    │                                                     │
    │ Performance: 30ns per string (3.3× faster)        │
    └────────────────────────────────────────────────────┘

    Visual Comparison:
    ┌────────────────────────────────────────────────────┐
    │ Scalar (per character):                            │
    │   i → m → p → o → r → t → (space) → ...           │
    │   ↓   ↓   ↓   ↓   ↓   ↓   ↓                       │
    │   Compare 1 byte at a time                         │
    │                                                     │
    │ AVX2 (vectorized):                                 │
    │   [i][m][p][o][r][t][ ][s][y][s][,][ ][o][s][...] │
    │   └──────────────────────┬──────────────────────┘  │
    │                          ▼                          │
    │          Compare 32 bytes simultaneously            │
    └────────────────────────────────────────────────────┘

Operation 2: String Prefix Match (simd.rs:263-285)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input: strings = ["test_func", "test_var", "example"]
           pattern = "test_"

    AVX2 Implementation (x86_64):
    ┌────────────────────────────────────────────────────┐
    │ Pattern length: 5 bytes ("test_")                  │
    │                                                     │
    │ Load pattern into AVX2 register:                   │
    │   pattern_vec = _mm256_loadu_si256(pattern_bytes) │
    │                                                     │
    │ For each string:                                   │
    │   string_vec = _mm256_loadu_si256(string_bytes)   │
    │                                                     │
    │   Compare registers:                               │
    │   cmp_result = _mm256_cmpeq_epi8(                 │
    │     pattern_vec,                                   │
    │     string_vec                                     │
    │   )                                                │
    │                                                     │
    │   Extract comparison mask:                         │
    │   mask = _mm256_movemask_epi8(cmp_result)         │
    │                                                     │
    │   Check if all pattern bytes match:               │
    │   if (mask & pattern_mask) == pattern_mask {      │
    │     return true;  // Prefix matches                │
    │   }                                                │
    │                                                     │
    │ Performance: 15ns per string (3× faster)          │
    └────────────────────────────────────────────────────┘

    NEON Implementation (ARM64):
    ┌────────────────────────────────────────────────────┐
    │ Pattern length: 5 bytes                            │
    │                                                     │
    │ Load pattern into NEON register (128-bit):         │
    │   pattern_vec = vld1q_u8(pattern_bytes)           │
    │                                                     │
    │ For each string:                                   │
    │   string_vec = vld1q_u8(string_bytes)             │
    │                                                     │
    │   Compare vectors:                                 │
    │   cmp_result = vceqq_u8(                          │
    │     pattern_vec,                                   │
    │     string_vec                                     │
    │   )                                                │
    │                                                     │
    │   Check if all lanes match:                       │
    │   min_val = vminvq_u8(cmp_result)                 │
    │   if min_val == 0xFF {                            │
    │     return true;  // All bytes match               │
    │   }                                                │
    │                                                     │
    │ Performance: 20ns per string (2.5× faster)        │
    └────────────────────────────────────────────────────┘

Operation 3: Character Count (simd.rs:517-544)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input: strings = ["hello world", "test string"]
           char = 'l'

    Scalar Path:
    ┌────────────────────────────────────┐
    │ for s in strings {                 │
    │     count = 0;                     │
    │     for c in s.chars() {           │
    │         if c == target_char {      │
    │             count += 1;            │
    │         }                          │
    │     }                              │
    │     results.push(count);           │
    │ }                                  │
    │                                    │
    │ Performance: 50ns per character    │
    └────────────────────────────────────┘

    AVX2 Path (32 chars at once):
    ┌────────────────────────────────────────────────────┐
    │ Broadcast target character:                        │
    │   ch_vec = _mm256_set1_epi8('l')                  │
    │   (fills all 32 lanes with 'l')                   │
    │                                                     │
    │ Process string in 32-byte chunks:                  │
    │   for chunk in bytes.chunks(32) {                  │
    │     data_vec = _mm256_loadu_si256(chunk)          │
    │                                                     │
    │     // Compare all 32 bytes simultaneously         │
    │     cmp_vec = _mm256_cmpeq_epi8(                  │
    │       data_vec,                                    │
    │       ch_vec                                       │
    │     )                                              │
    │                                                     │
    │     // Extract bit mask (1 = match)               │
    │     mask = _mm256_movemask_epi8(cmp_vec)          │
    │                                                     │
    │     // Count set bits (matches)                   │
    │     count += mask.count_ones()                    │
    │   }                                                │
    │                                                     │
    │ Performance: 12ns per character (4× faster)       │
    └────────────────────────────────────────────────────┘

    Visual: "hello world" counting 'l':
    ┌────────────────────────────────────────────────────┐
    │ Input:    h e l l o   w o r l d                   │
    │           │ │ │ │ │ │ │ │ │ │ │                   │
    │ Target:   l l l l l l l l l l l                   │
    │           │ │ ✓ ✓ │ │ │ │ │ ✓ │  ← Compare       │
    │           │ │ │ │ │ │ │ │ │ │ │                   │
    │ Mask:     0 0 1 1 0 0 0 0 0 1 0                   │
    │                                                     │
    │ count_ones(mask) = 3  ← Result                    │
    └────────────────────────────────────────────────────┘

Performance Summary
━━━━━━━━━━━━━━━━━━━
    ┌────────────────────┬──────────┬──────────┬──────────┐
    │ Operation          │  Scalar  │  AVX2    │ Speedup  │
    ├────────────────────┼──────────┼──────────┼──────────┤
    │ String contains    │  100ns   │   30ns   │  3.3×    │
    │ Prefix match       │   50ns   │   15ns   │  3.0×    │
    │ Char count         │   50ns   │   12ns   │  4.0×    │
    ├────────────────────┼──────────┼──────────┼──────────┤
    │ Overall benefit    │    -     │    -     │  3.5×    │
    └────────────────────┴──────────┴──────────┴──────────┘

    Platform Coverage:
    ┌────────────────────┬──────────┬──────────┬──────────┐
    │ Platform           │  SIMD    │ Width    │  Status  │
    ├────────────────────┼──────────┼──────────┼──────────┤
    │ x86_64 (modern)    │  AVX2    │ 256-bit  │   ✅     │
    │ x86_64 (older)     │  SSE4.2  │ 128-bit  │   ✅     │
    │ ARM64 (all)        │  NEON    │ 128-bit  │   ✅     │
    │ Other              │  Scalar  │  8-bit   │   ✅     │
    └────────────────────┴──────────┴──────────┴──────────┘
```

---

**End of Visual Integration Diagrams**

For detailed file:line references and implementation details, see `INTEGRATION_ARCHITECTURE_MAP.md`.
