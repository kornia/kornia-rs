import torch
import numpy as np
from safetensors.torch import save_file
from pathlib import Path


def generate_layernorm_test_data(num_tests=10,
                                batch_sizes=[1, 4, 8], dtype=torch.float32, seed=42, eps=1e-6):
    """
    Generate test data for layer normalization validation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    test_data = {}
    
    for i in range(num_tests):
        dim = 1152
        batch_size = np.random.choice(batch_sizes)
        
        # Generate input tensor [batch_size, dim]
        x = torch.randn(batch_size, dim, dtype=dtype, device=torch.device("cpu"))
        
        # Generate learnable parameters (weight and bias)
        weight = torch.ones(dim, dtype=dtype, device=torch.device("cpu"))  # gamma
        bias = torch.zeros(dim, dtype=dtype, device=torch.device("cpu"))   # beta
        
        # Apply layer normalization manually
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        y = x_normalized * weight + bias
        
        # Store test data
        test_data[f"test_{i}_input"] = x
        test_data[f"test_{i}_weigh  t"] = weight
        test_data[f"test_{i}_bias"] = bias
        test_data[f"test_{i}_output"] = y
        test_data[f"test_{i}_dim"] = torch.tensor(dim, dtype=torch.int32)
        test_data[f"test_{i}_batch_size"] = torch.tensor(batch_size, dtype=torch.int32)
    
    # Store global parameters
    test_data["num_tests"] = torch.tensor(num_tests, dtype=torch.int32)
    test_data["seed"] = torch.tensor(seed, dtype=torch.int32)
    test_data["eps"] = torch.tensor(eps, dtype=torch.float32)
    
    return test_data


def main():
    output_dir = Path("../validation_data")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate test data for float32
    test_data = generate_layernorm_test_data(num_tests=20, dtype=torch.float32, seed=42)
    output_file = output_dir / "layernorm_test_data_float32.safetensors"
    save_file(test_data, output_file)
    print(f"Saved layer norm test data to {output_file}")


if __name__ == "__main__":
    main()
