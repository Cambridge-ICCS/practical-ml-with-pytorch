"""
PyTorch import for XPUs (Intel GPUs).

Performing the import before any other imports can avoid conflicts
among shared-object libraries.
"""
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
