import numpy as np

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    cp = None
    _cupy_available = False

# Select backend (GPU if available, else CPU)
if _cupy_available:
    try:
        _ = cp.zeros((1,))
        DEVICE = "gpu"
        xp = cp
    except Exception:
        DEVICE = "cpu"
        xp = np
else:
    DEVICE = "cpu"
    xp = np

def to_device(array):
    """Move NumPy array to GPU if CuPy is available."""
    if DEVICE == "gpu" and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array

def to_cpu(array):
    """Move CuPy array to CPU NumPy array if needed."""
    if DEVICE == "gpu" and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def get_device():
    """Return current backend device."""
    return DEVICE
