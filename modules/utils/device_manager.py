try:
    import cupy as xp
    DEVICE = "gpu"
except ImportError:
    try:
        import jax.numpy as xp
        DEVICE = "jax_cpu"
    except ImportError:
        import numpy as xp
        DEVICE = "cpu"

def to_device(x): return xp.asarray(x)
def to_cpu(x): return xp.asnumpy(x) if DEVICE == "gpu" else x
def print_device_info(): print(f"🧠 Running on {DEVICE.upper()}")