#!/usr/bin/env node
/**
 * Node.js example for Portalis WASM transpiler
 * Translates Python files to Rust
 */

const { TranspilerWasm } = require('../../wasm-pkg/nodejs/portalis_transpiler.js');
const fs = require('fs');
const path = require('path');

function translateFile(inputPath, outputPath) {
    // Initialize transpiler
    const transpiler = new TranspilerWasm();

    // Read Python file
    const pythonCode = fs.readFileSync(inputPath, 'utf-8');

    console.log(`\n🐍 Translating ${inputPath}...`);
    const startTime = Date.now();

    try {
        // Translate
        const rustCode = transpiler.translate(pythonCode);

        // Write output
        if (outputPath) {
            fs.writeFileSync(outputPath, rustCode);
            console.log(`✅ Translation complete in ${Date.now() - startTime}ms`);
            console.log(`🦀 Output: ${outputPath}`);
        } else {
            console.log(`\n🦀 Rust Output:\n`);
            console.log(rustCode);
        }

        return true;
    } catch (error) {
        console.error(`❌ Translation failed: ${error}`);
        return false;
    }
}

function showUsage() {
    console.log(`
Portalis WASM Transpiler - Node.js CLI

Usage:
  node translate.js <input.py> [output.rs]

Examples:
  node translate.js example.py               # Print to console
  node translate.js example.py output.rs     # Save to file
  node translate.js fibonacci.py fibonacci.rs

Options:
  -h, --help    Show this help message
  -v, --version Show transpiler version
    `);
}

// CLI handling
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
        showUsage();
        process.exit(0);
    }

    if (args.includes('-v') || args.includes('--version')) {
        const { TranspilerWasm } = require('../../wasm-pkg/nodejs/portalis_transpiler.js');
        console.log(`Portalis Transpiler v${TranspilerWasm.version()}`);
        process.exit(0);
    }

    const inputPath = args[0];
    const outputPath = args[1];

    if (!fs.existsSync(inputPath)) {
        console.error(`Error: File not found: ${inputPath}`);
        process.exit(1);
    }

    const success = translateFile(inputPath, outputPath);
    process.exit(success ? 0 : 1);
}

module.exports = { translateFile };
