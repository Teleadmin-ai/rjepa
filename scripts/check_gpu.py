#!/usr/bin/env python3
"""
Check GPU/CUDA availability for R-JEPA.
"""
import sys
import subprocess


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print("✅ PyTorch installed")
        print(f"   Version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✅ CUDA available")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n   GPU {i}: {props.name}")
                print(f"   - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"   - Compute Capability: {props.major}.{props.minor}")

            # Test allocation
            try:
                x = torch.zeros(100, 100).cuda()
                print("\n✅ GPU allocation test: SUCCESS")
            except Exception as e:
                print(f"\n❌ GPU allocation test: FAILED ({e})")
        else:
            print("❌ CUDA not available")
            print("   PyTorch is installed but cannot detect CUDA.")
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Run: python scripts/install_pytorch_cuda.py")


def check_nvidia_docker():
    """Check if nvidia-docker is available."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("\n✅ nvidia-docker available")
            print("   Docker can access NVIDIA GPUs")
        else:
            print("\n❌ nvidia-docker NOT available")
            print("   Install NVIDIA Container Toolkit:")
            print("   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
    except FileNotFoundError:
        print("\n⚠️  Docker not found")
    except subprocess.TimeoutExpired:
        print("\n⚠️  nvidia-docker test timeout")
    except Exception as e:
        print(f"\n⚠️  Error checking nvidia-docker: {e}")


def main():
    print("=" * 80)
    print("R-JEPA GPU/CUDA Check")
    print("=" * 80 + "\n")

    check_cuda()
    check_nvidia_docker()

    print("\n" + "=" * 80)
    print("Check complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
