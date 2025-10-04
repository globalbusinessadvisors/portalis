# WASM Deployment Quick Start

**Status**: Ready to begin Phase 1
**Target**: Production WASM deployment in 15 days
**Current**: 191/219 tests passing (87.2%), no WASM infrastructure

---

## Immediate Next Steps (Day 1)

### Morning (3 hours): Toolchain Setup

```bash
# 1. Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 2. Add WASM target
rustup target add wasm32-unknown-unknown

# 3. Verify installation
wasm-pack --version
cargo --version --verbose
```

### Afternoon (4 hours): Add WASM Dependencies

Edit `agents/transpiler/Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
# Existing dependencies...
wasm-bindgen = "0.2"
serde-wasm-bindgen = "0.6"
js-sys = "0.3"
console_error_panic_hook = "0.1"

[dev-dependencies]
# Existing...
wasm-bindgen-test = "0.3"

[features]
default = []
wasm = ["wasm-bindgen"]
```

Create `agents/transpiler/src/wasm.rs`:

```rust
use wasm_bindgen::prelude::*;
use crate::feature_translator::FeatureTranslator;

#[wasm_bindgen]
pub struct WasmTranspiler {
    translator: FeatureTranslator,
}

#[wasm_bindgen]
impl WasmTranspiler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            translator: FeatureTranslator::new(),
        }
    }

    #[wasm_bindgen]
    pub fn translate(&mut self, python_code: &str) -> Result<String, JsValue> {
        self.translator
            .translate(python_code)
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))
    }

    #[wasm_bindgen]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
```

Update `agents/transpiler/src/lib.rs`:

```rust
// Add after existing modules
#[cfg(target_arch = "wasm32")]
pub mod wasm;
```

**Validation**:
```bash
cargo check --target wasm32-unknown-unknown --features wasm
```

### Evening (2 hours): First WASM Build

Create `scripts/build-wasm.sh`:

```bash
#!/bin/bash
set -e

echo "Building WASM for web target..."
cd agents/transpiler
wasm-pack build --target web --out-dir ../../wasm-pkg/web

echo "WASM build complete!"
ls -lh ../../wasm-pkg/web/
```

Run:
```bash
chmod +x scripts/build-wasm.sh
./scripts/build-wasm.sh
```

**Expected Output**:
- `wasm-pkg/web/portalis_transpiler_bg.wasm` (~200KB)
- `wasm-pkg/web/portalis_transpiler.js`
- `wasm-pkg/web/portalis_transpiler.d.ts`

---

## Day 1 Success Criteria

- ✅ wasm-pack installed and functional
- ✅ WASM dependencies added to Cargo.toml
- ✅ `wasm.rs` module created
- ✅ `cargo check --target wasm32-unknown-unknown` succeeds
- ✅ First WASM build completes successfully
- ✅ WASM artifacts generated in `wasm-pkg/web/`

---

## Quick Reference: 15-Day Plan

| Days | Phase | Focus | Outcome |
|------|-------|-------|---------|
| 1-3 | Infrastructure | WASM tooling, builds, CI/CD | WASM packages building |
| 4-5 | Examples | Browser demo, Node.js CLI | End-to-end validation |
| 6-10 | Features | Import system, tuple unpacking, with statements | 95%+ tests passing |
| 11-15 | Polish | Decorators, generators, stdlib mapping | Production release |

---

## Critical Path

```
Day 1: Toolchain ────┐
Day 2: Build Scripts │
Day 3: WASM Tests   ─┴──► Day 4: Browser Demo ───┐
                                                   │
                          Day 5: Node.js CLI ─────┴──► Days 6-10: Features ──► Days 11-15: Polish
```

---

## Testing Commands

```bash
# Run Rust tests
cargo test --package portalis-transpiler

# Run WASM tests (after Day 3)
wasm-pack test --headless --chrome

# Build all WASM targets (after Day 2)
./scripts/build-wasm.sh

# Run browser demo (after Day 4)
cd examples/browser-demo && npm run dev

# Run Node.js CLI (after Day 5)
node examples/nodejs-cli/transpile.js examples/fibonacci.py
```

---

## Rollback Plan

If issues arise on Day 1:

```bash
# Remove WASM module
rm agents/transpiler/src/wasm.rs

# Revert Cargo.toml
git restore agents/transpiler/Cargo.toml

# Clean build artifacts
cargo clean
rm -rf wasm-pkg/
```

---

## Support & Troubleshooting

### Common Issues

**Issue**: `wasm-pack: command not found`
**Solution**: Ensure installation script succeeded, check `~/.cargo/bin` in PATH

**Issue**: `cargo check` fails with WASM target
**Solution**: Check Rust version (need 1.70+), run `rustup update`

**Issue**: WASM build fails
**Solution**: Run `cargo clean`, ensure all dependencies are compatible

**Issue**: "error: crate-type 'cdylib' requires '-C prefer-dynamic'"
**Solution**: This is expected for some targets, build with wasm-pack instead

---

## Key Resources

- **Full Strategy**: `WASM_DEPLOYMENT_IMPLEMENTATION_STRATEGY.md`
- **Current Status**: `PYTHON_TO_WASM_TRANSPILER_COMPLETE.md`
- **Test Suite**: `agents/transpiler/src/*_test.rs`
- **Examples**: `examples/` directory

---

## Next Actions After Day 1

Once Day 1 is complete and validated:

1. Review `WASM_DEPLOYMENT_IMPLEMENTATION_STRATEGY.md` Day 2 section
2. Begin NPM package setup
3. Create CI/CD workflow
4. Start browser demo prototype

**Estimated Time**: Day 1 completion = 9 hours (manageable in one workday)

---

**Ready to start? Run the Day 1 Morning commands above!**
