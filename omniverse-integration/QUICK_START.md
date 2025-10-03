# Portalis Omniverse Integration - Quick Start Guide

## 5-Minute Setup

Get started with WASM modules in NVIDIA Omniverse in under 5 minutes.

### Prerequisites

- NVIDIA Omniverse Create, Code, or Kit 105.0+
- Python 3.10+
- Rust toolchain (for compiling WASM)

### Step 1: Install Extension (1 minute)

#### Option A: From Omniverse Exchange (Recommended)

```bash
# Open Omniverse
# Window → Extensions → Search "Portalis WASM Runtime"
# Click Install → Enable
```

#### Option B: Manual Install

```bash
# Clone repository
git clone https://github.com/portalis/omniverse-integration.git
cd omniverse-integration

# Copy extension to Omniverse
cp -r extension/exts/portalis.wasm.runtime ~/.local/share/ov/pkg/create/extensions/

# Restart Omniverse and enable extension
```

### Step 2: Install Dependencies (1 minute)

```bash
# Install Wasmtime Python library
pip install wasmtime numpy psutil

# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Step 3: Run Demo (3 minutes)

```bash
cd demonstrations/projectile_physics

# Compile Python to WASM
cd rust_translation
cargo build --release --target wasm32-unknown-unknown

# Module created at: target/wasm32-unknown-unknown/release/projectile.wasm

# Create Omniverse scene
cd ../omniverse_scene
python projectile_scene.py

# Output: projectile_scene.usd created
```

### Step 4: Load in Omniverse

1. Open Omniverse Create
2. File → Open → Select `projectile_scene.usd`
3. Press **Play** ▶️
4. Watch projectile physics simulation powered by WASM!

### Step 5: Verify Performance

1. Window → Portalis WASM Runtime
2. Check loaded modules (should show "projectile_physics")
3. Click "Show Performance Stats"
4. Verify:
   - FPS: 60+
   - Latency: <5ms
   - Memory: <30MB

---

## Next Steps

### Explore Other Demos

```bash
# Robot kinematics
cd demonstrations/robot_kinematics
# ... follow similar steps

# Sensor processing
cd demonstrations/sensor_fusion

# Digital twin
cd demonstrations/digital_twin

# Fluid dynamics
cd demonstrations/fluid_dynamics
```

### Create Your Own WASM Module

1. **Write Python function**:
   ```python
   def my_function(x: float, y: float) -> float:
       return x * y + (x / y if y != 0 else 0.0)
   ```

2. **Translate to Rust** (manual or via Portalis):
   ```rust
   #[no_mangle]
   pub extern "C" fn my_function(x: f64, y: f64) -> f64 {
       if y != 0.0 {
           x * y + (x / y)
       } else {
           x * y
       }
   }
   ```

3. **Compile to WASM**:
   ```bash
   cargo build --release --target wasm32-unknown-unknown
   ```

4. **Use in USD scene**:
   ```python
   from portalis_usd import create_wasm_module_prim

   prim = create_wasm_module_prim(
       stage, "/World/MyModule",
       "./my_function.wasm",
       "my_module",
       "my_function"
   )
   ```

### Performance Benchmarking

```bash
cd benchmarks
python performance_suite.py

# Output: Comprehensive performance report
# - Timing metrics
# - Memory usage
# - Target validation
# - JSON export
```

### Read Full Documentation

- **README**: `docs/README.md` (complete guide)
- **API Reference**: Extension and schema APIs
- **Troubleshooting**: Common issues and solutions
- **Video Tutorials**: `scripts/video_storyboards/`

---

## Common Issues

### "Wasmtime not available"

**Solution**:
```bash
pip install wasmtime
```

### "WASM module not found"

**Solution**:
- Check path in USD attribute is correct
- Use absolute path or path relative to USD file
- Verify .wasm file has read permissions

### "Low FPS" (<30)

**Solution**:
- Compile with `--release` flag
- Enable LTO in Cargo.toml
- Reduce `updateRate` attribute
- Check `executionMode` is "continuous"

### "Module not loading in Omniverse"

**Solution**:
- Click "Scan Stage for WASM Modules" in extension window
- Check USD prim has `wasmPath` attribute set
- Verify extension is enabled in Extension Manager
- Check Console for error messages

---

## Support

- **Documentation**: `/workspace/portalis/omniverse-integration/docs/`
- **GitHub**: https://github.com/portalis/omniverse-integration
- **Discord**: discord.gg/portalis
- **Email**: support@portalis.dev

---

## What You've Built

After this quick start, you have:

✅ **WASM Runtime**: Extension running in Omniverse
✅ **Demo Scene**: Projectile physics simulation
✅ **Performance**: 60+ FPS real-time execution
✅ **Foundation**: Ready to build industrial applications

---

**Next**: Explore advanced scenarios, optimize performance, deploy to production!

**Version**: 1.0.0
**Last Updated**: 2025-10-03
