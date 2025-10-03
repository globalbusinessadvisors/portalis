"""
Projectile Physics Simulation - Python Source
Demonstrates Python → Rust → WASM translation for physics calculations
"""

import math
from typing import Tuple


def calculate_trajectory(
    initial_velocity: float,
    angle_degrees: float,
    time: float,
    gravity: float = 9.81
) -> Tuple[float, float, float]:
    """
    Calculate projectile position at given time

    Args:
        initial_velocity: Initial velocity in m/s
        angle_degrees: Launch angle in degrees
        time: Time since launch in seconds
        gravity: Gravitational acceleration in m/s²

    Returns:
        Tuple of (x, y, z) position in meters
    """
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)

    # Calculate velocity components
    vx = initial_velocity * math.cos(angle_rad)
    vy = initial_velocity * math.sin(angle_rad)

    # Calculate position
    x = vx * time
    y = vy * time - 0.5 * gravity * time * time
    z = 0.0  # No lateral movement

    return (x, y, z)


def calculate_impact_time(
    initial_velocity: float,
    angle_degrees: float,
    initial_height: float = 0.0,
    gravity: float = 9.81
) -> float:
    """
    Calculate time until projectile hits ground

    Args:
        initial_velocity: Initial velocity in m/s
        angle_degrees: Launch angle in degrees
        initial_height: Initial height above ground in meters
        gravity: Gravitational acceleration in m/s²

    Returns:
        Impact time in seconds
    """
    angle_rad = math.radians(angle_degrees)
    vy = initial_velocity * math.sin(angle_rad)

    # Quadratic formula: -0.5*g*t² + vy*t + h = 0
    a = -0.5 * gravity
    b = vy
    c = initial_height

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return 0.0  # No real solution

    # Take positive root
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    return max(t1, t2)


def calculate_max_height(
    initial_velocity: float,
    angle_degrees: float,
    gravity: float = 9.81
) -> float:
    """
    Calculate maximum height reached by projectile

    Args:
        initial_velocity: Initial velocity in m/s
        angle_degrees: Launch angle in degrees
        gravity: Gravitational acceleration in m/s²

    Returns:
        Maximum height in meters
    """
    angle_rad = math.radians(angle_degrees)
    vy = initial_velocity * math.sin(angle_rad)

    # At max height, vertical velocity is 0
    # vy - g*t = 0 => t = vy/g
    time_to_max = vy / gravity

    # h = vy*t - 0.5*g*t²
    max_height = vy * time_to_max - 0.5 * gravity * time_to_max * time_to_max

    return max_height


def calculate_range(
    initial_velocity: float,
    angle_degrees: float,
    initial_height: float = 0.0,
    gravity: float = 9.81
) -> float:
    """
    Calculate horizontal range of projectile

    Args:
        initial_velocity: Initial velocity in m/s
        angle_degrees: Launch angle in degrees
        initial_height: Initial height above ground in meters
        gravity: Gravitational acceleration in m/s²

    Returns:
        Horizontal range in meters
    """
    impact_time = calculate_impact_time(
        initial_velocity,
        angle_degrees,
        initial_height,
        gravity
    )

    angle_rad = math.radians(angle_degrees)
    vx = initial_velocity * math.cos(angle_rad)

    return vx * impact_time


def update_physics(delta_time_ms: int) -> int:
    """
    Update physics simulation - WASM entry point

    This function is called from Omniverse at each frame

    Args:
        delta_time_ms: Time since last update in milliseconds

    Returns:
        Status code (0 = success)
    """
    # Convert to seconds
    dt = delta_time_ms / 1000.0

    # This is a simplified version for WASM
    # Real implementation would maintain state
    return 0


# Example usage
if __name__ == "__main__":
    # Test projectile calculations
    velocity = 20.0  # m/s
    angle = 45.0  # degrees

    print(f"Projectile with v={velocity} m/s at {angle}°")
    print(f"Max height: {calculate_max_height(velocity, angle):.2f} m")
    print(f"Range: {calculate_range(velocity, angle):.2f} m")
    print(f"Impact time: {calculate_impact_time(velocity, angle):.2f} s")

    # Sample trajectory
    print("\nTrajectory:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        x, y, z = calculate_trajectory(velocity, angle, t)
        print(f"  t={t:.1f}s: ({x:.2f}, {y:.2f}, {z:.2f}) m")
