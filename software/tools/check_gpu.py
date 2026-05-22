import jax

def main():
    devices = jax.devices()
    has_gpu = any(device.platform == 'gpu' for device in devices)
    print("JAX devices:", devices)
    print("GPU support:", "Available" if has_gpu else "Not available")

if __name__ == "__main__":
    main()
