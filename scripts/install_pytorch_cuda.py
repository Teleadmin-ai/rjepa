#!/usr/bin/env python3
"""
Auto-detect CUDA version and install compatible PyTorch.
"""
import sys
import subprocess
import re


def detect_cuda_version():
    """Detect installed CUDA version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse CUDA version from nvidia-smi output
        match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        if match:
            cuda_version = match.group(1)
            return cuda_version
        else:
            print("⚠️  Could not parse CUDA version from nvidia-smi")
            return None

    except FileNotFoundError:
        print("❌ nvidia-smi not found. Is NVIDIA driver installed?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return None


def get_pytorch_install_command(cuda_version):
    """Get pip install command for PyTorch based on CUDA version."""
    if cuda_version is None:
        # CPU-only fallback
        return "pip install torch torchvision torchaudio"

    # Map CUDA version to PyTorch CUDA version
    major, minor = cuda_version.split(".")
    cuda_major = int(major)
    cuda_minor = int(minor)

    if cuda_major == 12 and cuda_minor >= 1:
        # CUDA 12.1+
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_major == 11 and cuda_minor >= 8:
        # CUDA 11.8
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print(f"⚠️  CUDA {cuda_version} might not be fully supported.")
        print("   Trying CUDA 12.1 PyTorch...")
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"


def main():
    print("=" * 80)
    print("PyTorch CUDA Auto-Installer for R-JEPA")
    print("=" * 80 + "\n")

    # Detect CUDA
    print("Detecting CUDA version...")
    cuda_version = detect_cuda_version()

    if cuda_version:
        print(f"✅ CUDA {cuda_version} detected")
    else:
        print("⚠️  CUDA not detected, will install CPU-only PyTorch")

    # Get install command
    install_cmd = get_pytorch_install_command(cuda_version)
    print(f"\nInstall command: {install_cmd}")

    # Confirm
    response = input("\nProceed with installation? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Installation cancelled.")
        sys.exit(0)

    # Install
    print("\nInstalling PyTorch...")
    try:
        subprocess.run(install_cmd, shell=True, check=True)
        print("\n✅ PyTorch installed successfully!")

        # Verify
        print("\nVerifying installation...")
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Installation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
