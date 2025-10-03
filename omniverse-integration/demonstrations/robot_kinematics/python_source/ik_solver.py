"""
Inverse Kinematics Solver - Python Source
6-DOF robot arm IK solver for Portalis translation
"""

import math
from typing import List, Tuple, Optional


class RobotArm6DOF:
    """6 degree-of-freedom robot arm with inverse kinematics"""

    def __init__(
        self,
        link_lengths: List[float] = None,
        joint_limits: List[Tuple[float, float]] = None
    ):
        """
        Initialize robot arm

        Args:
            link_lengths: Lengths of each link (6 values)
            joint_limits: Min/max angles for each joint in radians
        """
        self.link_lengths = link_lengths or [1.0, 1.0, 1.0, 0.5, 0.5, 0.3]
        self.joint_limits = joint_limits or [
            (-math.pi, math.pi),   # Joint 1 (base)
            (-math.pi/2, math.pi/2),  # Joint 2
            (-math.pi/2, math.pi/2),  # Joint 3
            (-math.pi, math.pi),   # Joint 4
            (-math.pi/2, math.pi/2),  # Joint 5
            (-math.pi, math.pi),   # Joint 6 (end effector)
        ]

    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[float, float, float]:
        """
        Calculate end effector position from joint angles

        Args:
            joint_angles: List of 6 joint angles in radians

        Returns:
            (x, y, z) position of end effector
        """
        if len(joint_angles) != 6:
            raise ValueError("Must provide 6 joint angles")

        # Simplified FK for demonstration
        x = 0.0
        y = 0.0
        z = 0.0

        # Cumulative transformation
        angle_sum = 0.0

        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            angle_sum += angle

            if i < 2:  # Base joints affect XY plane
                x += length * math.cos(angle_sum)
                y += length * math.sin(angle_sum)
            else:  # Higher joints affect Z
                z += length * math.sin(angle)

        return (x, y, z)

    def inverse_kinematics_2d(
        self,
        target_x: float,
        target_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Solve 2D IK for first two joints (simplified)

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate

        Returns:
            (theta1, theta2) joint angles or None if unreachable
        """
        l1 = self.link_lengths[0]
        l2 = self.link_lengths[1]

        # Distance to target
        d = math.sqrt(target_x * target_x + target_y * target_y)

        # Check if reachable
        if d > (l1 + l2) or d < abs(l1 - l2):
            return None

        # Law of cosines
        cos_theta2 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)

        # Clamp to valid range
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        # Elbow-up solution
        theta2 = math.acos(cos_theta2)

        # Calculate theta1
        alpha = math.atan2(target_y, target_x)
        beta = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = alpha - beta

        return (theta1, theta2)

    def solve_ik(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        max_iterations: int = 100,
        tolerance: float = 0.01
    ) -> List[float]:
        """
        Solve full 6-DOF IK using iterative method (Jacobian-based)

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            target_z: Target Z coordinate
            max_iterations: Maximum iterations
            tolerance: Position error tolerance

        Returns:
            List of 6 joint angles
        """
        # Start with zero configuration
        joint_angles = [0.0] * 6

        # Solve 2D IK for first two joints
        result_2d = self.inverse_kinematics_2d(target_x, target_y)

        if result_2d:
            joint_angles[0], joint_angles[1] = result_2d

        # Simple iterative refinement
        for iteration in range(max_iterations):
            # Get current position
            current_x, current_y, current_z = self.forward_kinematics(joint_angles)

            # Calculate error
            error_x = target_x - current_x
            error_y = target_y - current_y
            error_z = target_z - current_z

            error = math.sqrt(error_x*error_x + error_y*error_y + error_z*error_z)

            # Check convergence
            if error < tolerance:
                break

            # Simple gradient descent (not true Jacobian)
            learning_rate = 0.1
            for i in range(6):
                # Numerical gradient
                delta = 0.001
                joint_angles[i] += delta
                pos_plus = self.forward_kinematics(joint_angles)
                joint_angles[i] -= delta

                # Gradient
                grad_x = (pos_plus[0] - current_x) / delta
                grad_y = (pos_plus[1] - current_y) / delta
                grad_z = (pos_plus[2] - current_z) / delta

                # Update
                joint_angles[i] += learning_rate * (
                    error_x * grad_x + error_y * grad_y + error_z * grad_z
                )

                # Clamp to limits
                min_angle, max_angle = self.joint_limits[i]
                joint_angles[i] = max(min_angle, min(max_angle, joint_angles[i]))

        return joint_angles

    def clamp_angles(self, joint_angles: List[float]) -> List[float]:
        """Clamp joint angles to limits"""
        clamped = []
        for angle, (min_angle, max_angle) in zip(joint_angles, self.joint_limits):
            clamped.append(max(min_angle, min(max_angle, angle)))
        return clamped


# WASM entry points
_robot_arm = RobotArm6DOF()


def solve_ik_wasm(
    target_x: float,
    target_y: float,
    target_z: float
) -> int:
    """
    WASM entry point for IK solving

    Stores result in global state (simplified for WASM)

    Returns:
        0 if successful, -1 if failed
    """
    try:
        angles = _robot_arm.solve_ik(target_x, target_y, target_z)
        # In real WASM, would write to memory
        return 0
    except Exception:
        return -1


def get_joint_angle(joint_index: int) -> float:
    """Get joint angle by index (for WASM)"""
    # In real implementation, would read from stored state
    return 0.0


# Example usage
if __name__ == "__main__":
    robot = RobotArm6DOF()

    # Test forward kinematics
    print("Forward Kinematics Test:")
    angles = [0.0, math.pi/4, -math.pi/4, 0.0, 0.0, 0.0]
    pos = robot.forward_kinematics(angles)
    print(f"  Angles: {[f'{a:.2f}' for a in angles]}")
    print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Test inverse kinematics
    print("\nInverse Kinematics Test:")
    target = (1.5, 1.0, 0.5)
    print(f"  Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

    solved_angles = robot.solve_ik(*target)
    print(f"  Solution: {[f'{a:.2f}' for a in solved_angles]}")

    # Verify
    result_pos = robot.forward_kinematics(solved_angles)
    print(f"  Reached: ({result_pos[0]:.2f}, {result_pos[1]:.2f}, {result_pos[2]:.2f})")

    error = math.sqrt(
        sum((t - r)**2 for t, r in zip(target, result_pos))
    )
    print(f"  Error: {error:.4f}")
