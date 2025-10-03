"""
USD Schema for WASM Modules
Define USD primitives for WASM module integration in Omniverse
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    from pxr import Usd, UsdGeom, Sdf, Gf, Tf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


class WasmModuleSchema:
    """
    USD Schema for WASM Module Primitive

    Defines attributes and relationships for WASM modules in USD scenes
    """

    # Schema type name
    SCHEMA_TYPE = "WasmModule"

    # Attribute names
    ATTR_WASM_PATH = "wasmPath"
    ATTR_MODULE_ID = "moduleId"
    ATTR_ENTRY_FUNCTION = "entryFunction"
    ATTR_ENABLED = "enabled"
    ATTR_EXECUTION_MODE = "executionMode"  # "continuous", "on_demand", "event_driven"
    ATTR_UPDATE_RATE = "updateRate"  # Hz for continuous mode
    ATTR_INPUT_PARAMS = "inputParams"
    ATTR_OUTPUT_PARAMS = "outputParams"
    ATTR_PERFORMANCE_MONITORING = "performanceMonitoring"

    @staticmethod
    def Define(stage: 'Usd.Stage', path: str) -> 'Usd.Prim':
        """
        Define a WasmModule prim in the USD stage

        Args:
            stage: USD stage
            path: Prim path

        Returns:
            Created prim
        """
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        prim = stage.DefinePrim(path, "Xform")

        # Add custom attributes
        prim.CreateAttribute(
            WasmModuleSchema.ATTR_WASM_PATH,
            Sdf.ValueTypeNames.String
        )

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_MODULE_ID,
            Sdf.ValueTypeNames.String
        )

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_ENTRY_FUNCTION,
            Sdf.ValueTypeNames.String
        )

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_ENABLED,
            Sdf.ValueTypeNames.Bool
        ).Set(True)

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_EXECUTION_MODE,
            Sdf.ValueTypeNames.Token
        ).Set("continuous")

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_UPDATE_RATE,
            Sdf.ValueTypeNames.Float
        ).Set(60.0)  # 60 Hz default

        prim.CreateAttribute(
            WasmModuleSchema.ATTR_PERFORMANCE_MONITORING,
            Sdf.ValueTypeNames.Bool
        ).Set(True)

        return prim

    @staticmethod
    def Get(stage: 'Usd.Stage', path: str) -> 'Usd.Prim':
        """Get existing WasmModule prim"""
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        return stage.GetPrimAtPath(path)

    @staticmethod
    def SetWasmPath(prim: 'Usd.Prim', wasm_path: str):
        """Set WASM module path"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_WASM_PATH)
        if attr:
            attr.Set(wasm_path)

    @staticmethod
    def GetWasmPath(prim: 'Usd.Prim') -> Optional[str]:
        """Get WASM module path"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_WASM_PATH)
        if attr:
            return attr.Get()
        return None

    @staticmethod
    def SetModuleId(prim: 'Usd.Prim', module_id: str):
        """Set module identifier"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_MODULE_ID)
        if attr:
            attr.Set(module_id)

    @staticmethod
    def GetModuleId(prim: 'Usd.Prim') -> Optional[str]:
        """Get module identifier"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_MODULE_ID)
        if attr:
            return attr.Get()
        return None

    @staticmethod
    def SetEntryFunction(prim: 'Usd.Prim', function_name: str):
        """Set entry function name"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_ENTRY_FUNCTION)
        if attr:
            attr.Set(function_name)

    @staticmethod
    def GetEnabled(prim: 'Usd.Prim') -> bool:
        """Check if module is enabled"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_ENABLED)
        if attr:
            return attr.Get()
        return False

    @staticmethod
    def SetEnabled(prim: 'Usd.Prim', enabled: bool):
        """Enable/disable module"""
        attr = prim.GetAttribute(WasmModuleSchema.ATTR_ENABLED)
        if attr:
            attr.Set(enabled)


class WasmPhysicsSchema:
    """
    USD Schema for WASM-based Physics Simulation

    Integrates WASM modules with PhysX physics engine
    """

    SCHEMA_TYPE = "WasmPhysics"

    ATTR_PHYSICS_FUNCTION = "physicsFunction"
    ATTR_FORCE_MULTIPLIER = "forceMultiplier"
    ATTR_GRAVITY_OVERRIDE = "gravityOverride"
    ATTR_COLLISION_HANDLER = "collisionHandler"

    @staticmethod
    def Define(stage: 'Usd.Stage', path: str) -> 'Usd.Prim':
        """Define WasmPhysics prim"""
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        # Define as WasmModule first
        prim = WasmModuleSchema.Define(stage, path)

        # Add physics-specific attributes
        prim.CreateAttribute(
            WasmPhysicsSchema.ATTR_PHYSICS_FUNCTION,
            Sdf.ValueTypeNames.String
        ).Set("update_physics")

        prim.CreateAttribute(
            WasmPhysicsSchema.ATTR_FORCE_MULTIPLIER,
            Sdf.ValueTypeNames.Float
        ).Set(1.0)

        prim.CreateAttribute(
            WasmPhysicsSchema.ATTR_GRAVITY_OVERRIDE,
            Sdf.ValueTypeNames.Float3
        ).Set(Gf.Vec3f(0.0, -9.81, 0.0))

        return prim


class WasmRoboticsSchema:
    """
    USD Schema for WASM-based Robotics Control

    Defines robot control interface using WASM modules
    """

    SCHEMA_TYPE = "WasmRobotics"

    ATTR_KINEMATICS_FUNCTION = "kinematicsFunction"
    ATTR_JOINT_TARGETS = "jointTargets"
    ATTR_END_EFFECTOR_TARGET = "endEffectorTarget"
    ATTR_IK_SOLVER = "ikSolver"
    ATTR_CONTROL_MODE = "controlMode"  # "position", "velocity", "torque"

    @staticmethod
    def Define(stage: 'Usd.Stage', path: str, num_joints: int = 6) -> 'Usd.Prim':
        """Define WasmRobotics prim"""
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        prim = WasmModuleSchema.Define(stage, path)

        # Add robotics attributes
        prim.CreateAttribute(
            WasmRoboticsSchema.ATTR_KINEMATICS_FUNCTION,
            Sdf.ValueTypeNames.String
        ).Set("solve_ik")

        prim.CreateAttribute(
            WasmRoboticsSchema.ATTR_JOINT_TARGETS,
            Sdf.ValueTypeNames.FloatArray
        ).Set([0.0] * num_joints)

        prim.CreateAttribute(
            WasmRoboticsSchema.ATTR_END_EFFECTOR_TARGET,
            Sdf.ValueTypeNames.Float3
        ).Set(Gf.Vec3f(0, 0, 0))

        prim.CreateAttribute(
            WasmRoboticsSchema.ATTR_CONTROL_MODE,
            Sdf.ValueTypeNames.Token
        ).Set("position")

        return prim


class WasmSensorSchema:
    """
    USD Schema for WASM-based Sensor Processing

    Processes sensor data using WASM modules
    """

    SCHEMA_TYPE = "WasmSensor"

    ATTR_SENSOR_TYPE = "sensorType"  # "lidar", "camera", "imu", "gps"
    ATTR_PROCESSING_FUNCTION = "processingFunction"
    ATTR_DATA_BUFFER_SIZE = "dataBufferSize"
    ATTR_FILTER_TYPE = "filterType"

    @staticmethod
    def Define(stage: 'Usd.Stage', path: str, sensor_type: str = "generic") -> 'Usd.Prim':
        """Define WasmSensor prim"""
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        prim = WasmModuleSchema.Define(stage, path)

        # Add sensor attributes
        prim.CreateAttribute(
            WasmSensorSchema.ATTR_SENSOR_TYPE,
            Sdf.ValueTypeNames.Token
        ).Set(sensor_type)

        prim.CreateAttribute(
            WasmSensorSchema.ATTR_PROCESSING_FUNCTION,
            Sdf.ValueTypeNames.String
        ).Set("process_sensor_data")

        prim.CreateAttribute(
            WasmSensorSchema.ATTR_DATA_BUFFER_SIZE,
            Sdf.ValueTypeNames.Int
        ).Set(1024)

        return prim


class WasmFluidSchema:
    """
    USD Schema for WASM-based Fluid Dynamics

    Simulates fluid dynamics using WASM modules
    """

    SCHEMA_TYPE = "WasmFluid"

    ATTR_SIMULATION_FUNCTION = "simulationFunction"
    ATTR_GRID_RESOLUTION = "gridResolution"
    ATTR_VISCOSITY = "viscosity"
    ATTR_DENSITY = "density"
    ATTR_TIME_STEP = "timeStep"

    @staticmethod
    def Define(stage: 'Usd.Stage', path: str, grid_res: int = 64) -> 'Usd.Prim':
        """Define WasmFluid prim"""
        if not USD_AVAILABLE:
            raise RuntimeError("USD not available")

        prim = WasmModuleSchema.Define(stage, path)

        # Add fluid simulation attributes
        prim.CreateAttribute(
            WasmFluidSchema.ATTR_SIMULATION_FUNCTION,
            Sdf.ValueTypeNames.String
        ).Set("update_fluid")

        prim.CreateAttribute(
            WasmFluidSchema.ATTR_GRID_RESOLUTION,
            Sdf.ValueTypeNames.Int3
        ).Set(Gf.Vec3i(grid_res, grid_res, grid_res))

        prim.CreateAttribute(
            WasmFluidSchema.ATTR_VISCOSITY,
            Sdf.ValueTypeNames.Float
        ).Set(1.0)

        prim.CreateAttribute(
            WasmFluidSchema.ATTR_DENSITY,
            Sdf.ValueTypeNames.Float
        ).Set(1000.0)

        prim.CreateAttribute(
            WasmFluidSchema.ATTR_TIME_STEP,
            Sdf.ValueTypeNames.Float
        ).Set(0.016)  # ~60 FPS

        return prim


def create_wasm_module_prim(
    stage: 'Usd.Stage',
    path: str,
    wasm_path: str,
    module_id: str,
    entry_function: str = "main",
    schema_type: str = "base"
) -> 'Usd.Prim':
    """
    Convenience function to create WASM module prim

    Args:
        stage: USD stage
        path: Prim path
        wasm_path: Path to WASM module
        module_id: Module identifier
        entry_function: Entry function name
        schema_type: Schema type (base, physics, robotics, sensor, fluid)

    Returns:
        Created prim
    """
    if not USD_AVAILABLE:
        raise RuntimeError("USD not available")

    # Choose schema based on type
    if schema_type == "physics":
        prim = WasmPhysicsSchema.Define(stage, path)
    elif schema_type == "robotics":
        prim = WasmRoboticsSchema.Define(stage, path)
    elif schema_type == "sensor":
        prim = WasmSensorSchema.Define(stage, path)
    elif schema_type == "fluid":
        prim = WasmFluidSchema.Define(stage, path)
    else:
        prim = WasmModuleSchema.Define(stage, path)

    # Set common attributes
    WasmModuleSchema.SetWasmPath(prim, wasm_path)
    WasmModuleSchema.SetModuleId(prim, module_id)
    WasmModuleSchema.SetEntryFunction(prim, entry_function)

    return prim
