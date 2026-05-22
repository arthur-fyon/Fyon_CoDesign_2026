import jax
from jax import Array

def get_cpu_device():
    """
    Get CPU device for storing non-critical tensors.
    
    Returns:
    --------
    jax.Device
        CPU device
    """
    return jax.devices('cpu')[0]


def get_gpu_device():
    """
    Get default GPU device for computation.
    
    Returns:
    --------
    jax.Device
        Default GPU device (or CPU if no GPU available)
    """
    try:
        return jax.devices('gpu')[0]
    except RuntimeError:
        # No GPU available, use CPU
        return jax.devices('cpu')[0]


def to_cpu(array: Array) -> Array:
    """
    Move array to CPU device.
    
    Useful for diagnostic outputs that don't need to be on GPU,
    saving GPU memory for the forward pass.
    
    Args:
    -----
    array: Array
        JAX array to move to CPU
        
    Returns:
    --------
    Array on CPU device
    """
    cpu = get_cpu_device()
    return jax.device_put(array, cpu)


def to_gpu(array: Array) -> Array:
    """
    Move array to GPU device.
    
    Args:
    -----
    array: Array
        JAX array to move to GPU
        
    Returns:
    --------
    Array on GPU device (or CPU if no GPU)
    """
    gpu = get_gpu_device()
    return jax.device_put(array, gpu)


def maybe_to_cpu(array: Array, condition: bool) -> Array:
    """
    Conditionally move array to CPU.
    
    Args:
    -----
    array: Array
        JAX array
    condition: bool
        If True, move to CPU; otherwise keep on current device
        
    Returns:
    --------
    Array (on CPU if condition is True)
    """
    if condition:
        return to_cpu(array)
    return array
