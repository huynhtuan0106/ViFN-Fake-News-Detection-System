import torch

print("=== GPU Status ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor creation on GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Test tensor on GPU: {x}")
    print("GPU is working!")
else:
    print("CUDA not available")
    print("GPU detection failed") 