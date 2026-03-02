import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

def run_gradient_persistence_demo(checkpoint_name: str = "tensor_checkpoint.pt") -> None:
    # 1. Setup synthetic data and computational graph
    # Using larger fonts for any potential plots/logs is implied in display logic
    checkpoint_path = Path(checkpoint_name)
    
    # Initialize a tensor with requires_grad=True
    x = torch.randn(5, 5, requires_grad=True)
    target = torch.randn(5, 5)
    
    # Perform a dummy forward and backward pass
    loss = (x - target).pow(2).sum()
    loss.backward()
    
    # Verify gradient exists
    assert x.grad is not None, "Gradient was not calculated correctly."
    original_grad = x.grad.clone()
    
    print(f"Original Gradient Mean: {original_grad.mean().item():.6f}")

    # 2. Demonstrate standard torch.save (Fails to keep .grad)
    torch.save(x, checkpoint_path)
    loaded_x_standard = torch.load(checkpoint_path, weights_only=False)
    
    print(f"Standard load .grad is None: {loaded_x_standard.grad is None}")

    # 3. Correct Approach: Explicit Checkpointing
    # We bundle the data and grad into a dictionary
    payload: Dict[str, Any] = {
        'data': x.data,
        'grad': x.grad,
        'requires_grad': x.requires_grad
    }
    
    torch.save(payload, checkpoint_path)
    
    # 4. Restoration
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Reconstruct the tensor
    y = checkpoint['data'].clone()
    y.requires_grad = checkpoint['requires_grad']
    y.grad = checkpoint['grad']
    
    # 5. Validation
    # Ensure numerical parity between original and restored gradients
    assert torch.equal(y.grad, original_grad), "Restored gradient does not match original!"
    print("Verification Successful: Gradients restored and matched.")
    
    # Clean up
    if checkpoint_path.exists():
        checkpoint_path.unlink()

if __name__ == "__main__":
    run_gradient_persistence_demo()