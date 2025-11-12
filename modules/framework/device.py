import os

class Device:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Device, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._current_device = 'cpu'
        self._cpu_backend = __import__('numpy')
        self._gpu_backend = None
        
        # Check for CuPy availability
        try:
            import cupy
            self._gpu_backend = cupy
            print("CuPy found. GPU acceleration available.")
        except ImportError:
            print("CuPy not found. Running on CPU only.")

    @property
    def backend(self):
        if self._current_device == 'gpu' and self._gpu_backend:
            return self._gpu_backend
        return self._cpu_backend

    @property
    def current_device(self):
        return self._current_device

    def set_device(self, device_type):
        if device_type not in ['cpu', 'gpu']:
            raise ValueError("Device type must be 'cpu' or 'gpu'.")
        if device_type == 'gpu' and not self._gpu_backend:
            raise RuntimeError("CuPy not available. Cannot set device to GPU.")
        self._current_device = device_type
        print(f"Device set to {self._current_device.upper()}")

    def to_cpu(self, array):
        if self._gpu_backend and isinstance(array, self._gpu_backend.ndarray):
            return self._gpu_backend.asnumpy(array)
        return array

    def to_gpu(self, array):
        if self._gpu_backend:
            if isinstance(array, self._cpu_backend.ndarray):
                return self._gpu_backend.asarray(array)
            return array
        raise RuntimeError("CuPy not available. Cannot move array to GPU.")

# Global device instance
device = Device()