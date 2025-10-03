"""
Projectile Physics Omniverse Scene Setup
Creates USD scene with WASM projectile physics
"""

from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import sys

# Add schema path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "extension" / "usd_schemas"))

from usd_schemas import WasmPhysicsSchema, WasmModuleSchema


def create_projectile_scene(stage_path: str, wasm_module_path: str):
    """
    Create Omniverse scene with projectile physics WASM module

    Args:
        stage_path: Path to save USD stage
        wasm_module_path: Path to compiled WASM module
    """
    # Create stage
    stage = Usd.Stage.CreateNew(stage_path)

    # Set up axis and units
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root
    root = UsdGeom.Xform.Define(stage, "/World")

    # Create ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([
        Gf.Vec3f(-50, 0, -50),
        Gf.Vec3f(50, 0, -50),
        Gf.Vec3f(50, 0, 50),
        Gf.Vec3f(-50, 0, 50),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])

    # Add physics collision
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Create projectile launcher (visual only)
    launcher = UsdGeom.Cone.Define(stage, "/World/Launcher")
    launcher.AddTranslateOp().Set(Gf.Vec3d(0, 0.5, 0))
    launcher.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, -45))  # 45 degree angle
    launcher.AddScaleOp().Set(Gf.Vec3d(0.5, 1.0, 0.5))
    launcher.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.8)])

    # Create projectile sphere
    projectile = UsdGeom.Sphere.Define(stage, "/World/Projectile")
    projectile.AddTranslateOp().Set(Gf.Vec3d(0, 0.5, 0))
    projectile.CreateRadiusAttr(0.2)
    projectile.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.3, 0.3)])

    # Add rigid body physics to projectile
    UsdPhysics.RigidBodyAPI.Apply(projectile.GetPrim())
    UsdPhysics.MassAPI.Apply(projectile.GetPrim())
    mass_api = UsdPhysics.MassAPI(projectile.GetPrim())
    mass_api.CreateMassAttr(1.0)

    # Add collision
    UsdPhysics.CollisionAPI.Apply(projectile.GetPrim())
    UsdPhysics.SphereCollisionAPI.Apply(projectile.GetPrim())

    # Create WASM physics controller
    wasm_controller = WasmPhysicsSchema.Define(
        stage,
        "/World/ProjectilePhysicsController"
    )

    # Configure WASM module
    WasmModuleSchema.SetWasmPath(wasm_controller, wasm_module_path)
    WasmModuleSchema.SetModuleId(wasm_controller, "projectile_physics")
    WasmModuleSchema.SetEntryFunction(wasm_controller, "update_physics")
    WasmModuleSchema.SetEnabled(wasm_controller, True)

    # Set execution mode
    exec_mode_attr = wasm_controller.GetAttribute(WasmModuleSchema.ATTR_EXECUTION_MODE)
    exec_mode_attr.Set("continuous")

    update_rate_attr = wasm_controller.GetAttribute(WasmModuleSchema.ATTR_UPDATE_RATE)
    update_rate_attr.Set(60.0)  # 60 Hz

    # Set physics parameters
    force_mult_attr = wasm_controller.GetAttribute(WasmPhysicsSchema.ATTR_FORCE_MULTIPLIER)
    force_mult_attr.Set(1.0)

    # Create relationship to projectile
    wasm_controller.CreateRelationship("projectile").AddTarget(projectile.GetPath())

    # Create trajectory visualization (points instancer for trail)
    trail = UsdGeom.Points.Define(stage, "/World/ProjectileTrail")
    trail.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 0.0)])
    trail.CreateWidthsAttr([0.1] * 100)  # 100 trail points

    # Create camera
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(15, 8, 15))
    camera.AddRotateXYZOp().Set(Gf.Vec3d(-20, 45, 0))

    # Create light
    light = UsdGeom.DistantLight.Define(stage, "/World/SunLight")
    light.AddRotateXYZOp().Set(Gf.Vec3d(-45, 45, 0))
    light.CreateIntensityAttr(1000.0)

    # Add physics scene
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # Set default prim
    stage.SetDefaultPrim(root.GetPrim())

    # Save stage
    stage.GetRootLayer().Save()

    print(f"Created projectile scene at: {stage_path}")
    print(f"WASM module: {wasm_module_path}")
    print(f"Scene contains:")
    print(f"  - Ground plane with collision")
    print(f"  - Projectile launcher (visual)")
    print(f"  - Projectile sphere with physics")
    print(f"  - WASM physics controller")
    print(f"  - Trajectory trail visualization")
    print(f"  - Camera and lighting")


def create_projectile_scene_simple(output_path: str):
    """Create simplified scene for testing (no WASM dependencies)"""
    stage = Usd.Stage.CreateNew(output_path)

    # Set up axis
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Create root
    root = UsdGeom.Xform.Define(stage, "/World")

    # Create projectile path (arc)
    points = []
    for i in range(50):
        t = i / 50.0 * 3.0  # 3 seconds
        # Simple parabola
        x = 20.0 * t
        y = 15.0 * t - 4.905 * t * t
        if y < 0:
            break
        points.append(Gf.Vec3f(x, y, 0))

    # Create curve
    curve = UsdGeom.BasisCurves.Define(stage, "/World/TrajectoryPath")
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([0.1] * len(points))
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateDisplayColorAttr([Gf.Vec3f(1, 1, 0)])

    # Create sphere at start
    sphere = UsdGeom.Sphere.Define(stage, "/World/Projectile")
    sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    sphere.CreateRadiusAttr(0.5)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()

    print(f"Created simple projectile scene at: {output_path}")


if __name__ == "__main__":
    # Example usage
    import tempfile

    # Create temp directory for demo
    temp_dir = Path(tempfile.gettempdir()) / "portalis_demo"
    temp_dir.mkdir(exist_ok=True)

    # Create simple scene (no WASM)
    simple_scene = str(temp_dir / "projectile_simple.usd")
    create_projectile_scene_simple(simple_scene)

    print(f"\nOpen in Omniverse: {simple_scene}")
