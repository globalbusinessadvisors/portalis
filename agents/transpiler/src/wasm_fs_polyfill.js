/**
 * WASM Filesystem Polyfill for Browser
 *
 * Provides a virtual filesystem using IndexedDB for Python code running in WASM
 * without WASI support. This allows file operations to work in the browser.
 */

class VirtualFilesystem {
    constructor() {
        this.dbName = 'portalis_wasm_fs';
        this.storeName = 'files';
        this.db = null;
    }

    /**
     * Initialize the IndexedDB database
     */
    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName, { keyPath: 'path' });
                }
            };
        });
    }

    /**
     * Read a file from the virtual filesystem
     * @param {string} path - File path
     * @returns {Promise<Uint8Array>} File contents
     */
    async readFile(path) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(path);

            request.onerror = () => reject(new Error(`File not found: ${path}`));
            request.onsuccess = () => {
                if (request.result) {
                    resolve(new Uint8Array(request.result.content));
                } else {
                    reject(new Error(`File not found: ${path}`));
                }
            };
        });
    }

    /**
     * Write a file to the virtual filesystem
     * @param {string} path - File path
     * @param {Uint8Array} content - File contents
     */
    async writeFile(path, content) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.put({
                path,
                content: Array.from(content),
                timestamp: Date.now()
            });

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve();
        });
    }

    /**
     * Check if a file exists
     * @param {string} path - File path
     * @returns {Promise<boolean>}
     */
    async exists(path) {
        if (!this.db) await this.init();

        return new Promise((resolve) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(path);

            request.onerror = () => resolve(false);
            request.onsuccess = () => resolve(!!request.result);
        });
    }

    /**
     * Delete a file
     * @param {string} path - File path
     */
    async deleteFile(path) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(path);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve();
        });
    }

    /**
     * List all files
     * @returns {Promise<string[]>} Array of file paths
     */
    async listFiles() {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAllKeys();

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
        });
    }

    /**
     * Mount a file from external source (e.g., uploaded file)
     * @param {string} path - Virtual path
     * @param {File} file - JavaScript File object
     */
    async mountFile(path, file) {
        const buffer = await file.arrayBuffer();
        const content = new Uint8Array(buffer);
        await this.writeFile(path, content);
    }

    /**
     * Download a file from virtual filesystem
     * @param {string} path - File path
     */
    async downloadFile(path) {
        const content = await this.readFile(path);
        const blob = new Blob([content]);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = path.split('/').pop();
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Global instance
const wasmFS = new VirtualFilesystem();

// Export for use in WASM module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VirtualFilesystem, wasmFS };
}

/**
 * Initialize filesystem for WASM module
 */
async function initWasmFilesystem(wasmModule) {
    await wasmFS.init();

    // Inject filesystem functions into WASM imports
    const fsImports = {
        wasi_fs_read: async (pathPtr, pathLen) => {
            const path = wasmModule.readString(pathPtr, pathLen);
            try {
                const content = await wasmFS.readFile(path);
                return wasmModule.allocateBytes(content);
            } catch (e) {
                console.error('WASI FS read error:', e);
                return 0;
            }
        },

        wasi_fs_write: async (pathPtr, pathLen, dataPtr, dataLen) => {
            const path = wasmModule.readString(pathPtr, pathLen);
            const data = wasmModule.readBytes(dataPtr, dataLen);
            try {
                await wasmFS.writeFile(path, data);
                return 1;
            } catch (e) {
                console.error('WASI FS write error:', e);
                return 0;
            }
        },

        wasi_fs_exists: async (pathPtr, pathLen) => {
            const path = wasmModule.readString(pathPtr, pathLen);
            return await wasmFS.exists(path) ? 1 : 0;
        },

        wasi_fs_delete: async (pathPtr, pathLen) => {
            const path = wasmModule.readString(pathPtr, pathLen);
            try {
                await wasmFS.deleteFile(path);
                return 1;
            } catch (e) {
                console.error('WASI FS delete error:', e);
                return 0;
            }
        }
    };

    return fsImports;
}

/**
 * Example usage:
 *
 * ```javascript
 * import init, { TranspilerWasm } from './portalis_transpiler.js';
 *
 * await init();
 * await wasmFS.init();
 *
 * // Mount a file
 * await wasmFS.writeFile('/input.py', new TextEncoder().encode(`
 *   with open('/data.txt', 'w') as f:
 *       f.write('Hello from WASM!')
 * `));
 *
 * const transpiler = new TranspilerWasm();
 * const pythonCode = await wasmFS.readFile('/input.py');
 * const rustCode = transpiler.translate(new TextDecoder().decode(pythonCode));
 *
 * console.log(rustCode);
 * ```
 */

// Browser global
if (typeof window !== 'undefined') {
    window.wasmFS = wasmFS;
    window.initWasmFilesystem = initWasmFilesystem;
}
