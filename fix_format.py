import torch
from safetensors.torch import load_file
import os

# Paths
folder = "fine_tuning/adapters/dilu-llama3_1-8b-v1"
safe_file = os.path.join(folder, "adapter_model.safetensors")
bin_file = os.path.join(folder, "adapter_model.bin")

# Check if safetensors exists
if os.path.exists(safe_file):
    print(f"Found {safe_file}. Converting to .bin format...")
    # Load the safetensors
    tensors = load_file(safe_file)
    # Save as standard PyTorch .bin
    torch.save(tensors, bin_file)
    print("Success! Created adapter_model.bin")
else:
    print(f"Error: Could not find {safe_file}")