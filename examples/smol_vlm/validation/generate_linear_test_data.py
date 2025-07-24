import torch
import numpy as np
from safetensors.torch import save_file
from pathlib import Path


def generate_linear_test_data(num_tests=10, input_dims=[512, 1024, 2048], output_dims=[256, 512, 1024], 
                             batch_sizes=[1, 4, 8], dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    test_data = {}
    for i in range(num_tests):
        input_dim = np.random.choice(input_dims)
        output_dim = np.random.choice(output_dims)
        batch_size = np.random.choice(batch_sizes)
        
        x = torch.randn(batch_size, input_dim, dtype=dtype, device=torch.device("cpu"))
        weight = torch.randn(output_dim, input_dim, dtype=dtype, device=torch.device("cpu"))
        # y = torch.nn.functional.linear(x, weight, bias=None)
        y = torch.matmul(x, weight.t())
        # y = torch.ones_like(y)
        
        test_data[f"test_{i}_input"] = x
        test_data[f"test_{i}_weight"] = weight
        test_data[f"test_{i}_output"] = y
        test_data[f"test_{i}_input_dim"] = torch.tensor(input_dim, dtype=torch.int32)
        test_data[f"test_{i}_output_dim"] = torch.tensor(output_dim, dtype=torch.int32)
        test_data[f"test_{i}_batch_size"] = torch.tensor(batch_size, dtype=torch.int32)
    
    test_data["num_tests"] = torch.tensor(num_tests, dtype=torch.int32)
    test_data["seed"] = torch.tensor(seed, dtype=torch.int32)
    return test_data


def main(num_tests=50, output_dir="./tests/data", seed=42, dtype="float32"):
    dtype_map = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    test_data = generate_linear_test_data(num_tests=num_tests, dtype=torch_dtype, seed=seed)
    
    output_file = output_dir / f"linear_test_data_{dtype}.safetensors"
    save_file(test_data, output_file)
    print(f"Saved {num_tests} test cases to {output_file}")

if __name__ == "__main__":
    main()
