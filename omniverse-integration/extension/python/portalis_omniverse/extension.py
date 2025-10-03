"""
Portalis WASM Runtime Extension for NVIDIA Omniverse

Main extension entry point that integrates WASM modules with Omniverse Kit
"""

import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.usd
from pathlib import Path
import asyncio
import logging
from typing import Dict, List, Optional, Any

# Import WASM bridge
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "wasm_bridge"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "usd_schemas"))

from wasm_bridge import WasmtimeBridge, WasmModuleConfig, create_wasm_bridge, WASMTIME_AVAILABLE
from usd_schemas import (
    WasmModuleSchema,
    WasmPhysicsSchema,
    WasmRoboticsSchema,
    WasmSensorSchema,
    WasmFluidSchema,
    create_wasm_module_prim,
    USD_AVAILABLE
)


class PortalisWasmRuntimeExtension(omni.ext.IExt):
    """
    Portalis WASM Runtime Extension

    Provides WASM execution capabilities within Omniverse simulations
    """

    def on_startup(self, ext_id):
        """Extension startup"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Portalis WASM Runtime Extension starting...")

        # Extension ID
        self._ext_id = ext_id

        # WASM bridge
        self._wasm_bridge: Optional[WasmtimeBridge] = None

        # USD stage
        self._stage = None
        self._stage_event_sub = None

        # WASM module prims being updated
        self._active_modules: Dict[str, Dict[str, Any]] = {}

        # Update subscription
        self._update_sub = None

        # Performance monitoring
        self._frame_count = 0
        self._total_wasm_time_ms = 0.0

        # UI window
        self._window = None

        # Initialize WASM bridge
        self._initialize_wasm_bridge()

        # Subscribe to USD events
        self._subscribe_to_stage_events()

        # Subscribe to update events
        self._subscribe_to_updates()

        # Create UI
        self._create_ui()

        self.logger.info("Portalis WASM Runtime Extension started successfully")

    def _initialize_wasm_bridge(self):
        """Initialize WASM runtime bridge"""
        if not WASMTIME_AVAILABLE:
            self.logger.error("Wasmtime not available. Please install: pip install wasmtime")
            return

        # Get cache directory from settings
        cache_dir = Path(omni.kit.app.get_app().get_extension_manager().get_extension_path(self._ext_id))
        cache_dir = cache_dir.parent / "wasm_cache"
        cache_dir.mkdir(exist_ok=True)

        # Get max memory from settings
        settings = omni.kit.app.get_app().get_settings()
        max_memory_mb = settings.get("/exts/portalis.wasm.runtime/max_memory_mb") or 512

        # Create bridge
        config = WasmModuleConfig(
            max_memory_mb=max_memory_mb,
            cache_dir=cache_dir,
            cache_enabled=True
        )

        self._wasm_bridge = WasmtimeBridge(config)
        self.logger.info(f"WASM bridge initialized (cache: {cache_dir}, max_memory: {max_memory_mb}MB)")

    def _subscribe_to_stage_events(self):
        """Subscribe to USD stage open/close events"""
        usd_context = omni.usd.get_context()

        def on_stage_event(event):
            if event.type == int(omni.usd.StageEventType.OPENED):
                self._on_stage_opened()
            elif event.type == int(omni.usd.StageEventType.CLOSED):
                self._on_stage_closed()

        self._stage_event_sub = usd_context.get_stage_event_stream().create_subscription_to_pop(
            on_stage_event
        )

        # Check if stage already open
        if usd_context.get_stage():
            self._on_stage_opened()

    def _on_stage_opened(self):
        """Handle stage opened event"""
        self.logger.info("USD stage opened")
        self._stage = omni.usd.get_context().get_stage()

        # Scan for WASM module prims
        self._scan_wasm_modules()

    def _on_stage_closed(self):
        """Handle stage closed event"""
        self.logger.info("USD stage closed")
        self._stage = None

        # Unload all modules
        if self._wasm_bridge:
            self._wasm_bridge.unload_all()

        self._active_modules.clear()

    def _scan_wasm_modules(self):
        """Scan stage for WASM module prims and load them"""
        if not self._stage:
            return

        # Find all WASM module prims
        from pxr import Usd

        for prim in self._stage.Traverse():
            # Check if prim has WASM module attributes
            if prim.HasAttribute(WasmModuleSchema.ATTR_WASM_PATH):
                self._load_wasm_module_from_prim(prim)

    def _load_wasm_module_from_prim(self, prim):
        """Load WASM module from USD prim"""
        if not self._wasm_bridge:
            return

        try:
            # Get module configuration from prim
            wasm_path = WasmModuleSchema.GetWasmPath(prim)
            module_id = WasmModuleSchema.GetModuleId(prim)
            enabled = WasmModuleSchema.GetEnabled(prim)

            if not wasm_path or not module_id or not enabled:
                return

            # Resolve path (may be relative to USD stage)
            wasm_path = Path(wasm_path)
            if not wasm_path.is_absolute():
                stage_path = Path(self._stage.GetRootLayer().realPath).parent
                wasm_path = stage_path / wasm_path

            if not wasm_path.exists():
                self.logger.warning(f"WASM module not found: {wasm_path}")
                return

            # Load module
            actual_module_id = self._wasm_bridge.load_module(wasm_path, module_id)

            # Store active module info
            self._active_modules[str(prim.GetPath())] = {
                "prim": prim,
                "module_id": actual_module_id,
                "wasm_path": wasm_path,
                "last_update_time": 0.0
            }

            self.logger.info(f"Loaded WASM module: {module_id} from {wasm_path}")

        except Exception as e:
            self.logger.error(f"Failed to load WASM module from prim {prim.GetPath()}: {e}")

    def _subscribe_to_updates(self):
        """Subscribe to frame update events"""
        update_stream = omni.kit.app.get_app().get_update_event_stream()

        def on_update(e):
            self._on_update(e.payload['dt'])

        self._update_sub = update_stream.create_subscription_to_pop(on_update)

    def _on_update(self, dt: float):
        """Update callback - execute WASM modules"""
        if not self._wasm_bridge or not self._active_modules:
            return

        self._frame_count += 1

        # Update each active module
        for prim_path, module_info in self._active_modules.items():
            try:
                self._update_wasm_module(module_info, dt)
            except Exception as e:
                self.logger.error(f"Error updating WASM module {prim_path}: {e}")

        # Log performance stats every 60 frames
        if self._frame_count % 60 == 0:
            stats = self._wasm_bridge.get_performance_stats()
            avg_time = stats.get('avg_execution_time_ms', 0.0)
            self.logger.debug(f"WASM performance: {avg_time:.3f}ms avg")

    def _update_wasm_module(self, module_info: Dict[str, Any], dt: float):
        """Update single WASM module"""
        prim = module_info['prim']
        module_id = module_info['module_id']

        # Check execution mode
        exec_mode_attr = prim.GetAttribute(WasmModuleSchema.ATTR_EXECUTION_MODE)
        if not exec_mode_attr:
            return

        exec_mode = exec_mode_attr.Get()

        if exec_mode == "continuous":
            # Check update rate
            update_rate_attr = prim.GetAttribute(WasmModuleSchema.ATTR_UPDATE_RATE)
            update_rate = update_rate_attr.Get() if update_rate_attr else 60.0

            # Check if enough time has passed
            time_since_update = dt + module_info.get('time_accumulator', 0.0)
            update_interval = 1.0 / update_rate

            if time_since_update >= update_interval:
                # Get entry function
                entry_func_attr = prim.GetAttribute(WasmModuleSchema.ATTR_ENTRY_FUNCTION)
                entry_func = entry_func_attr.Get() if entry_func_attr else None

                if entry_func:
                    # Call WASM function
                    # For now, just call with delta time
                    try:
                        result = self._wasm_bridge.call_function(
                            module_id,
                            entry_func,
                            int(dt * 1000)  # Convert to milliseconds
                        )

                        module_info['last_result'] = result
                        module_info['time_accumulator'] = 0.0

                    except Exception as e:
                        self.logger.error(f"Error calling {entry_func}: {e}")
            else:
                module_info['time_accumulator'] = time_since_update

    def _create_ui(self):
        """Create extension UI window"""
        self._window = ui.Window("Portalis WASM Runtime", width=400, height=600)

        with self._window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Portalis WASM Runtime Control", style={"font_size": 18})

                ui.Spacer(height=10)

                # Status section
                with ui.CollapsableFrame("Status", collapsed=False):
                    with ui.VStack(spacing=5):
                        self._status_label = ui.Label("Initializing...")
                        self._modules_label = ui.Label("Loaded modules: 0")
                        self._performance_label = ui.Label("Performance: N/A")

                ui.Spacer(height=5)

                # Module list
                with ui.CollapsableFrame("Loaded Modules", collapsed=False):
                    self._module_list = ui.VStack(spacing=3)
                    with self._module_list:
                        ui.Label("No modules loaded")

                ui.Spacer(height=5)

                # Controls
                with ui.CollapsableFrame("Controls", collapsed=False):
                    with ui.VStack(spacing=5):
                        ui.Button("Scan Stage for WASM Modules", clicked_fn=self._on_scan_clicked)
                        ui.Button("Reload All Modules", clicked_fn=self._on_reload_clicked)
                        ui.Button("Unload All Modules", clicked_fn=self._on_unload_clicked)
                        ui.Button("Show Performance Stats", clicked_fn=self._on_stats_clicked)

        # Start UI update timer
        self._start_ui_updates()

    def _start_ui_updates(self):
        """Start periodic UI updates"""
        async def update_ui_loop():
            while self._window and self._window.visible:
                self._update_ui()
                await asyncio.sleep(0.5)  # Update every 500ms

        asyncio.ensure_future(update_ui_loop())

    def _update_ui(self):
        """Update UI with current status"""
        if not self._window or not self._window.visible:
            return

        # Update status
        if WASMTIME_AVAILABLE:
            self._status_label.text = "WASM Runtime: Active"
        else:
            self._status_label.text = "WASM Runtime: NOT AVAILABLE"

        # Update module count
        num_modules = len(self._active_modules)
        self._modules_label.text = f"Loaded modules: {num_modules}"

        # Update performance
        if self._wasm_bridge:
            stats = self._wasm_bridge.get_performance_stats()
            avg_time = stats.get('avg_execution_time_ms', 0.0)
            total_calls = stats.get('total_executions', 0)
            self._performance_label.text = (
                f"Performance: {avg_time:.3f}ms avg ({total_calls} calls)"
            )

        # Update module list
        self._module_list.clear()
        with self._module_list:
            if self._active_modules:
                for prim_path, info in self._active_modules.items():
                    module_id = info.get('module_id', 'unknown')
                    ui.Label(f"• {module_id} ({prim_path})")
            else:
                ui.Label("No modules loaded")

    def _on_scan_clicked(self):
        """Scan button clicked"""
        self.logger.info("Scanning stage for WASM modules...")
        self._scan_wasm_modules()

    def _on_reload_clicked(self):
        """Reload button clicked"""
        self.logger.info("Reloading all WASM modules...")
        if self._wasm_bridge:
            self._wasm_bridge.unload_all()
        self._active_modules.clear()
        self._scan_wasm_modules()

    def _on_unload_clicked(self):
        """Unload button clicked"""
        self.logger.info("Unloading all WASM modules...")
        if self._wasm_bridge:
            self._wasm_bridge.unload_all()
        self._active_modules.clear()

    def _on_stats_clicked(self):
        """Stats button clicked"""
        if not self._wasm_bridge:
            return

        stats = self._wasm_bridge.get_performance_stats()

        # Log detailed stats
        self.logger.info("=== WASM Performance Statistics ===")
        self.logger.info(f"Total executions: {stats['total_executions']}")
        self.logger.info(f"Total time: {stats['total_execution_time_ms']:.2f}ms")
        self.logger.info(f"Average time: {stats['avg_execution_time_ms']:.3f}ms")
        self.logger.info(f"Loaded modules: {stats['loaded_modules']}")

        for module_id, module_stats in stats.get('modules', {}).items():
            self.logger.info(f"  {module_id}:")
            self.logger.info(f"    Calls: {module_stats['call_count']}")
            self.logger.info(f"    Total time: {module_stats['total_execution_time_ms']:.2f}ms")
            self.logger.info(f"    Avg time: {module_stats['avg_execution_time_ms']:.3f}ms")
            self.logger.info(f"    Memory: {module_stats['memory_usage_mb']:.2f}MB")

    def on_shutdown(self):
        """Extension shutdown"""
        self.logger.info("Portalis WASM Runtime Extension shutting down...")

        # Unsubscribe from events
        if self._update_sub:
            self._update_sub.unsubscribe()
            self._update_sub = None

        if self._stage_event_sub:
            self._stage_event_sub.unsubscribe()
            self._stage_event_sub = None

        # Unload all WASM modules
        if self._wasm_bridge:
            self._wasm_bridge.unload_all()
            self._wasm_bridge = None

        # Cleanup UI
        if self._window:
            self._window.destroy()
            self._window = None

        self.logger.info("Portalis WASM Runtime Extension shut down successfully")
