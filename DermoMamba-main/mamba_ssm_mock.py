# Mock mamba_ssm implementation for testing purposes
# This is a simplified implementation that doesn't require CUDA compilation

import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Mock implementation of selective scan function
    This is a simplified version for testing purposes
    """
    batch_size, seq_len, d_model = u.shape
    
    # Simple linear transformation as a placeholder
    # In the real implementation, this would be the selective scan operation
    x = u
    if D is not None:
        x = x + D.unsqueeze(0) * u
    
    # Apply some non-linear transformation
    x = torch.sigmoid(x)
    
    if z is not None:
        x = x * z
    
    if return_last_state:
        return x, x[:, -1:]
    return x

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """Mock reference implementation"""
    return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
