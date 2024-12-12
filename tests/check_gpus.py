import torch

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs detected: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Try creating a small tensor on each GPU
    x = torch.randn((2,2), device=f"cuda:{i}")
    print(f"Successfully created a tensor on GPU {i}")
