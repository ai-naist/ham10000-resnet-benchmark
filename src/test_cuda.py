import torch

if torch.cuda.is_available():
    print("PyTorch: CUDA is available!")
    print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}") # 0は最初のGPU
else:
    print("PyTorch: CUDA is NOT available.")