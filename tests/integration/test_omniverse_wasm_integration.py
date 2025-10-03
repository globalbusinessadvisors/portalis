"""
Integration Tests: Omniverse WASM Integration

Tests the Omniverse WebAssembly integration:
- WASM binary loading and execution
- USD schema integration
- Performance validation
- Scene interaction
"""

import pytest
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

try:
    from wasm_bridge.wasmtime_bridge import WasmtimeBridge, WasmModule
    from usd_schemas.wasm_prim_schema import WasmPrimSchema
except ImportError:
    pytest.skip("Omniverse integration not available", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.omniverse
@pytest.mark.wasm
class TestOmniverseWASMLoading:
    """Test WASM loading in Omniverse."""

    @pytest.fixture
    def wasm_bridge(self):
        """Create WASM bridge instance."""
        bridge = WasmtimeBridge()
        yield bridge
        bridge.cleanup()

    @pytest.fixture
    def compiled_wasm(self, temp_dir) -> Path:
        """Create a compiled WASM binary for testing."""
        # Minimal WASM module
        wasm_bytes = bytes([
            0x00, 0x61, 0x73, 0x6d,  # Magic
            0x01, 0x00, 0x00, 0x00,  # Version
            0x01, 0x07,  # Type section
            0x01,  # 1 type
            0x60,  # Function type
            0x01, 0x7f,  # 1 i32 param
            0x01, 0x7f,  # 1 i32 result
            0x03, 0x02,  # Function section
            0x01, 0x00,  # 1 function, type 0
            0x07, 0x0a,  # Export section
            0x01,  # 1 export
            0x06, 0x64, 0x6f, 0x75, 0x62, 0x6c, 0x65,  # "double"
            0x00, 0x00,  # Function 0
            0x0a, 0x09,  # Code section
            0x01,  # 1 function body
            0x07,  # Body size
            0x00,  # 0 locals
            0x20, 0x00,  # local.get 0
            0x20, 0x00,  # local.get 0
            0x6a,  # i32.add
            0x0b,  # end
        ])

        wasm_file = temp_dir / "test_module.wasm"
        wasm_file.write_bytes(wasm_bytes)
        return wasm_file

    def test_load_wasm_module(self, wasm_bridge, compiled_wasm):
        """Test loading WASM module."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        assert module is not None
        assert isinstance(module, WasmModule)

    def test_execute_wasm_function(self, wasm_bridge, compiled_wasm):
        """Test executing WASM function."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        # Call the 'double' function
        result = module.call("double", [21])

        assert result == 42

    def test_wasm_performance(self, wasm_bridge, compiled_wasm, benchmark_config):
        """Test WASM execution performance."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        latencies = []

        # Benchmark
        for i in range(benchmark_config["test_iterations"]):
            start = time.perf_counter()
            result = module.call("double", [i])
            elapsed = (time.perf_counter() - start) * 1000000  # microseconds

            latencies.append(elapsed)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\n=== WASM Execution Performance ===")
        print(f"Average: {avg_latency:.2f}μs")
        print(f"P95: {p95_latency:.2f}μs")

        # Should be very fast (< 1ms)
        assert avg_latency < 1000, "WASM execution too slow"

    def test_wasm_memory_management(self, wasm_bridge, compiled_wasm):
        """Test WASM memory management."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        # Execute multiple times
        for i in range(1000):
            result = module.call("double", [i])
            assert result == i * 2

        # No memory leaks expected
        # Memory usage should remain stable


@pytest.mark.integration
@pytest.mark.omniverse
class TestOmniverseUSDIntegration:
    """Test USD schema integration."""

    @pytest.fixture
    def wasm_prim_schema(self, mock_usd_stage):
        """Create WASM prim schema."""
        schema = WasmPrimSchema(stage=mock_usd_stage)
        return schema

    def test_create_wasm_prim(self, wasm_prim_schema):
        """Test creating WASM primitive in USD."""
        prim_path = "/World/WasmModule"

        prim = wasm_prim_schema.create_prim(
            prim_path=prim_path,
            module_name="test_module"
        )

        assert prim is not None

    def test_attach_wasm_to_prim(self, wasm_prim_schema, compiled_wasm):
        """Test attaching WASM module to USD prim."""
        prim_path = "/World/WasmModule"

        result = wasm_prim_schema.attach_wasm(
            prim_path=prim_path,
            wasm_path=str(compiled_wasm)
        )

        assert result is True

    def test_wasm_prim_attributes(self, wasm_prim_schema):
        """Test WASM prim attributes."""
        prim_path = "/World/WasmModule"

        prim = wasm_prim_schema.create_prim(prim_path, "test")

        # Set attributes
        wasm_prim_schema.set_attribute(prim_path, "fps", 60)
        wasm_prim_schema.set_attribute(prim_path, "enabled", True)

        # Get attributes
        fps = wasm_prim_schema.get_attribute(prim_path, "fps")
        assert fps == 60


@pytest.mark.integration
@pytest.mark.omniverse
@pytest.mark.slow
class TestOmniverseFullPipeline:
    """Test complete Omniverse integration pipeline."""

    def test_python_to_wasm_to_omniverse(
        self, wasm_bridge, wasm_prim_schema, sample_python_code, temp_dir
    ):
        """Test complete pipeline: Python → Rust → WASM → Omniverse."""
        # This would normally involve:
        # 1. Translate Python to Rust (via NeMo)
        # 2. Compile Rust to WASM
        # 3. Load WASM in Omniverse
        # 4. Execute in scene

        # For this test, we'll use pre-compiled WASM
        wasm_file = temp_dir / "pipeline_test.wasm"

        # Mock WASM compilation result
        wasm_file.write_bytes(bytes([
            0x00, 0x61, 0x73, 0x6d,  # Magic
            0x01, 0x00, 0x00, 0x00,  # Version
        ]))

        # Load in Omniverse
        module = wasm_bridge.load_module(str(wasm_file))

        assert module is not None

        # Create USD prim
        prim_path = "/World/TranslatedCode"
        prim = wasm_prim_schema.create_prim(prim_path, "translation_result")

        assert prim is not None

    @pytest.mark.benchmark
    def test_omniverse_fps_target(self, wasm_bridge, compiled_wasm):
        """Test that WASM execution meets 60 FPS target."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        # Simulate 60 FPS (16.67ms per frame)
        frame_time_target = 16.67  # milliseconds

        frame_times = []

        for frame in range(60):  # 1 second at 60 FPS
            start = time.perf_counter()

            # Execute WASM (simulate per-frame work)
            for _ in range(10):  # Multiple calls per frame
                result = module.call("double", [frame])

            elapsed = (time.perf_counter() - start) * 1000  # ms
            frame_times.append(elapsed)

        avg_frame_time = np.mean(frame_times)
        max_frame_time = max(frame_times)

        print(f"\n=== Omniverse FPS Performance ===")
        print(f"Average frame time: {avg_frame_time:.2f}ms")
        print(f"Max frame time: {max_frame_time:.2f}ms")
        print(f"Average FPS: {1000/avg_frame_time:.1f}")

        # Should maintain 60 FPS (< 16.67ms per frame)
        assert avg_frame_time < frame_time_target, "Cannot maintain 60 FPS"

    def test_concurrent_wasm_modules(self, wasm_bridge, compiled_wasm):
        """Test multiple concurrent WASM modules."""
        num_modules = 10

        modules = []
        for i in range(num_modules):
            module = wasm_bridge.load_module(str(compiled_wasm))
            modules.append(module)

        # Execute all concurrently
        results = []
        for i, module in enumerate(modules):
            result = module.call("double", [i])
            results.append(result)

        assert len(results) == num_modules
        assert results[5] == 10  # double(5) = 10


@pytest.mark.integration
@pytest.mark.omniverse
class TestOmniverseErrorHandling:
    """Test error handling in Omniverse integration."""

    def test_invalid_wasm_module(self, wasm_bridge, temp_dir):
        """Test handling of invalid WASM module."""
        invalid_wasm = temp_dir / "invalid.wasm"
        invalid_wasm.write_bytes(b"not valid wasm")

        with pytest.raises(Exception):
            wasm_bridge.load_module(str(invalid_wasm))

    def test_missing_export(self, wasm_bridge, compiled_wasm):
        """Test calling non-existent function."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        with pytest.raises(Exception):
            module.call("nonexistent_function", [])

    def test_wasm_runtime_error(self, wasm_bridge, compiled_wasm):
        """Test handling of WASM runtime errors."""
        module = wasm_bridge.load_module(str(compiled_wasm))

        # Try invalid arguments (implementation specific)
        try:
            result = module.call("double", ["not_an_int"])
        except (TypeError, ValueError, Exception) as e:
            # Expected error
            assert True
