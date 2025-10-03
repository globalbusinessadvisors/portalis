"""
CUDA Utility Functions

Helper functions for CUDA/GPU operations.
"""

from typing import Optional, Dict, Any
import os
from loguru import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, CUDA features disabled")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def check_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        True if CUDA is available, False otherwise
    """
    if not HAS_TORCH:
        return False

    available = torch.cuda.is_available()

    if available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        logger.info(f"CUDA available: {device_count} device(s), Primary: {device_name}")
    else:
        logger.info("CUDA not available, using CPU")

    return available


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Get GPU memory information.

    Args:
        device_id: CUDA device ID

    Returns:
        Dictionary with memory information
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {
            "available": False,
            "total_mb": 0,
            "allocated_mb": 0,
            "free_mb": 0,
        }

    try:
        torch.cuda.set_device(device_id)

        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        cached = torch.cuda.memory_reserved(device_id)
        free = total - cached

        return {
            "available": True,
            "device_id": device_id,
            "device_name": torch.cuda.get_device_name(device_id),
            "total_mb": total / (1024 ** 2),
            "allocated_mb": allocated / (1024 ** 2),
            "cached_mb": cached / (1024 ** 2),
            "free_mb": free / (1024 ** 2),
            "utilization_percent": (cached / total) * 100,
        }

    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}")
        return {"available": False, "error": str(e)}


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def set_gpu_device(device_id: int) -> bool:
    """
    Set the active GPU device.

    Args:
        device_id: CUDA device ID to use

    Returns:
        True if successful, False otherwise
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return False

    try:
        torch.cuda.set_device(device_id)
        logger.info(f"Set GPU device to {device_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to set GPU device: {e}")
        return False


def optimize_gpu_memory() -> None:
    """Optimize GPU memory usage."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    # Empty cache
    torch.cuda.empty_cache()

    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.debug("Enabled memory-efficient attention")


def get_cuda_capability() -> Optional[tuple[int, int]]:
    """
    Get CUDA compute capability.

    Returns:
        Tuple of (major, minor) version or None if not available
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def is_cuda_capability_sufficient(min_major: int = 7, min_minor: int = 0) -> bool:
    """
    Check if CUDA compute capability meets minimum requirements.

    Args:
        min_major: Minimum major version
        min_minor: Minimum minor version

    Returns:
        True if capability is sufficient
    """
    capability = get_cuda_capability()

    if capability is None:
        return False

    major, minor = capability

    if major > min_major:
        return True
    elif major == min_major:
        return minor >= min_minor
    else:
        return False
