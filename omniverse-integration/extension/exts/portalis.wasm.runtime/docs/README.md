# Portalis WASM Runtime for NVIDIA Omniverse

## Overview

The **Portalis WASM Runtime** extension enables execution of WebAssembly (WASM) modules within NVIDIA Omniverse simulations. It provides a seamless bridge between Portalis-generated WASM modules and Omniverse's USD-based simulation environment.

## Features

- **WASM Execution**: Run WASM modules compiled from Python (via Rust) in real-time
- **USD Integration**: Define WASM modules as USD primitives with full attribute support
- **Performance**: >30 FPS execution, <10ms latency for typical operations
- **Monitoring**: Built-in performance metrics and debugging tools
- **Flexible Execution**: Continuous, on-demand, or event-driven modes
- **Industrial Use Cases**: Physics, robotics, sensor processing, digital twins

## Installation

### Method 1: Extension Manager (Recommended)

1. Open NVIDIA Omniverse Create/Code/Kit
2. Navigate to **Window → Extensions**
3. Search for "Portalis WASM Runtime"
4. Click **Install**
5. Enable the extension

### Method 2: Manual Installation

1. Download the extension package
2. Extract to Omniverse extensions folder:
   - Windows: `%LOCALAPPDATA%\ov\pkg\<app>\extensions`
   - Linux: `~/.local/share/ov/pkg/<app>/extensions`
3. Restart Omniverse
4. Enable via Extension Manager

### Method 3: Development Mode

1. Clone repository: `git clone https://github.com/portalis/omniverse-integration.git`
2. Open Extension Manager
3. Click gear icon → **Add Extension Search Path**
4. Add path to `omniverse-integration/extension/exts`
5. Enable "Portalis WASM Runtime"

## Quick Start

### 1. Create a WASM Module

Using Portalis (Python → Rust → WASM):

```bash
# Write Python function
cat > physics.py << EOF
def calculate_force(mass: float, acceleration: float) -> float:
    return mass * acceleration
EOF

# Translate with Portalis
portalis translate physics.py --target wasm

# Output: physics.wasm
```

### 2. Create USD Scene

```python
from pxr import Usd, UsdGeom
from portalis_usd import create_wasm_module_prim

# Create stage
stage = Usd.Stage.CreateNew("my_scene.usd")

# Add WASM module
wasm_prim = create_wasm_module_prim(
    stage=stage,
    path="/World/PhysicsController",
    wasm_path="./physics.wasm",
    module_id="physics",
    entry_function="calculate_force"
)

stage.Save()
```

### 3. Load in Omniverse

1. Open `my_scene.usd` in Omniverse Create
2. WASM module loads automatically
3. Press **Play** to run simulation
4. WASM function executes each frame

### 4. Monitor Performance

1. Open **Window → Portalis WASM Runtime**
2. View loaded modules
3. Check performance metrics
4. Use control buttons (Scan, Reload, Unload)

## USD Schema Reference

### WasmModuleSchema (Base)

All WASM module primitives support these attributes:

| Attribute | Type | Description | Default |
|-----------|------|-------------|---------|
| `wasmPath` | string | Path to .wasm file | "" |
| `moduleId` | string | Unique identifier | "" |
| `entryFunction` | string | Function to call | "main" |
| `enabled` | bool | Enable/disable module | true |
| `executionMode` | token | Execution mode¹ | "continuous" |
| `updateRate` | float | Update frequency (Hz) | 60.0 |
| `performanceMonitoring` | bool | Enable metrics | true |

¹ Execution modes:
- `continuous`: Call every frame at `updateRate`
- `on_demand`: Call only when triggered
- `event_driven`: Call on specific events

### Specialized Schemas

#### WasmPhysicsSchema

For physics simulations:

```python
from portalis_usd import WasmPhysicsSchema

physics = WasmPhysicsSchema.Define(stage, "/World/Physics")
```

Additional attributes:
- `physicsFunction`: Physics update function name
- `forceMultiplier`: Scaling factor for forces
- `gravityOverride`: Custom gravity vector

#### WasmRoboticsSchema

For robot control:

```python
from portalis_usd import WasmRoboticsSchema

robot = WasmRoboticsSchema.Define(stage, "/World/Robot", num_joints=6)
```

Additional attributes:
- `kinematicsFunction`: IK/FK solver function
- `jointTargets`: Target joint angles (array)
- `endEffectorTarget`: Target XYZ position
- `controlMode`: "position", "velocity", or "torque"

## Performance Guidelines

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Frame Rate | >30 FPS | For real-time simulation |
| Latency | <10ms | Per WASM function call |
| Memory | <100MB | Per module |
| Load Time | <5s | Initial module load |

### Optimization Tips

1. **Use Release Builds**
   ```bash
   cargo build --release --target wasm32-unknown-unknown
   ```

2. **Enable LTO** (in Cargo.toml)
   ```toml
   [profile.release]
   lto = true
   opt-level = 3
   ```

3. **Minimize Allocations**
   - Use `#![no_std]` when possible
   - Pre-allocate buffers
   - Avoid dynamic memory in hot paths

4. **Set Appropriate Update Rates**
   - Physics: 60-120 Hz
   - UI updates: 30 Hz
   - Analytics: 1-10 Hz

## API Reference

### Python Extension API

Access extension from Python console:

```python
import omni.ext

# Get extension instance
ext = omni.ext.get_ext_by_id("portalis.wasm.runtime")

# Access WASM bridge
bridge = ext._wasm_bridge

# Load module
module_id = bridge.load_module("my_module.wasm")

# Call function
result = bridge.call_function(module_id, "my_function", arg1, arg2)

# Get stats
stats = bridge.get_performance_stats()
print(f"Average execution: {stats['avg_execution_time_ms']:.2f}ms")
```

### USD Attribute Access

Get/set attributes on WASM prims:

```python
from portalis_usd import WasmModuleSchema

# Get prim
prim = stage.GetPrimAtPath("/World/MyWasmModule")

# Get WASM path
wasm_path = WasmModuleSchema.GetWasmPath(prim)

# Set enabled state
WasmModuleSchema.SetEnabled(prim, True)

# Check if enabled
if WasmModuleSchema.GetEnabled(prim):
    print("Module is enabled")
```

## Troubleshooting

### Module Not Loading

**Symptom**: Extension shows "WASM module not found"

**Solutions**:
1. Check WASM path is correct (absolute or relative to USD)
2. Verify .wasm file exists and has read permissions
3. Check extension log: `Window → Console`
4. Enable debug logging: `logging.getLogger('portalis').setLevel(logging.DEBUG)`

### Low Performance

**Symptom**: FPS <30 or latency >10ms

**Solutions**:
1. Run performance benchmark: `Window → Portalis WASM Runtime → Show Performance Stats`
2. Check `executionMode` is set correctly
3. Reduce `updateRate` if acceptable
4. Optimize WASM module (see Optimization Tips)
5. Profile with Chrome DevTools (WASM profiling)

### WASM Function Errors

**Symptom**: Exception when calling WASM function

**Solutions**:
1. Verify function signature matches call
2. Check function is exported: `wasm-objdump -x module.wasm`
3. Validate arguments are correct types
4. Test WASM module standalone with `wasmtime`

## Examples

### Physics Simulation

```python
# Python source
def calculate_trajectory(v0: float, angle: float, t: float) -> tuple[float, float]:
    import math
    vx = v0 * math.cos(math.radians(angle))
    vy = v0 * math.sin(math.radians(angle))
    x = vx * t
    y = vy * t - 0.5 * 9.81 * t**2
    return (x, y)

# Translate to WASM with Portalis
# portalis translate physics.py

# USD scene
from portalis_usd import WasmPhysicsSchema

physics = WasmPhysicsSchema.Define(stage, "/World/Projectile")
WasmModuleSchema.SetWasmPath(physics, "./physics.wasm")
WasmModuleSchema.SetEntryFunction(physics, "calculate_trajectory")
```

### Robot Inverse Kinematics

```python
# Python IK solver
class RobotArm:
    def solve_ik(self, target_x: float, target_y: float, target_z: float) -> list[float]:
        # IK algorithm
        return joint_angles

# WASM in Omniverse
from portalis_usd import WasmRoboticsSchema

robot = WasmRoboticsSchema.Define(stage, "/World/RobotArm", num_joints=6)
WasmModuleSchema.SetWasmPath(robot, "./robot_ik.wasm")
robot.GetAttribute("endEffectorTarget").Set((1.0, 0.5, 0.3))
```

## Support

### Documentation
- **Full Guide**: [portalis.dev/docs/omniverse](https://portalis.dev/docs/omniverse)
- **API Reference**: [portalis.dev/api](https://portalis.dev/api)
- **Tutorials**: [portalis.dev/tutorials](https://portalis.dev/tutorials)

### Community
- **GitHub Issues**: [github.com/portalis/omniverse-integration/issues](https://github.com/portalis/omniverse-integration/issues)
- **Discord**: [discord.gg/portalis](https://discord.gg/portalis)
- **NVIDIA Forums**: [forums.developer.nvidia.com](https://forums.developer.nvidia.com)

### Commercial Support
- **Email**: support@portalis.dev
- **Enterprise**: enterprise@portalis.dev

## License

MIT License - see LICENSE file for details

## Credits

- **Portalis Team**: Core development
- **NVIDIA**: Omniverse platform and support
- **Bytecode Alliance**: Wasmtime runtime

---

**Version**: 1.0.0
**Omniverse Compatibility**: Kit 105.0+
**Last Updated**: 2025-10-03
