import torch

def check_pytorch_support():
    print(f"PyTorch version: {torch.__version__}")
    
    # 1. Проверка NVIDIA CUDA (Windows/Linux)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (NVIDIA) available: {cuda_available}")
    if cuda_available:
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")

    # 2. Проверка Apple Silicon MPS (macOS)
    mps_available = torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {mps_available}")

    # 3. Проверка Intel XPU (Intel Arc / iGPU)
    # Доступно в новых версиях PyTorch для Intel
    try:
        xpu_available = torch.xpu.is_available()
        print(f"XPU (Intel) available: {xpu_available}")
    except AttributeError:
        print("XPU: Support not detected in this PyTorch build")

    # 4. Определение рабочего устройства (современный метод)
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"\nRecommended device for training: {device}")

if __name__ == "__main__":
    check_pytorch_support()