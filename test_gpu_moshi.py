#!/usr/bin/env python3
"""
Test Moshi GPU configuration and memory usage.
"""

import asyncio
import torch
from src.coda.components.voice.models import MoshiConfig
from src.coda.components.voice.moshi_integration import MoshiClient

async def test_moshi_gpu():
    """Test Moshi GPU configuration."""
    print("🚀 Testing Moshi GPU Configuration...")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

    # Test configuration with GPU
    config = MoshiConfig(
        model_path=None,
        device='cuda',
        vram_allocation='8GB'
    )

    print("Creating MoshiClient with GPU...")
    try:
        client = MoshiClient(config)
        print("✅ MoshiClient created")
        print(f"GPU memory after client creation: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        print("Initializing Moshi model...")
        await client.initialize()
        print(f"✅ Moshi initialized on device: {client.device}")
        print(f"GPU memory after initialization: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # Check if model is on GPU
        if hasattr(client, 'lm_model') and client.lm_model is not None:
            model_device = next(client.lm_model.parameters()).device
            print(f"LM Model device: {model_device}")
        
        if hasattr(client, 'compression_model') and client.compression_model is not None:
            comp_device = next(client.compression_model.parameters()).device
            print(f"Compression Model device: {comp_device}")
            
        # Test a simple forward pass to ensure GPU usage
        print("Testing GPU inference...")
        if hasattr(client, 'lm_model'):
            # Create a dummy input tensor on GPU
            dummy_input = torch.randint(0, 1000, (1, 10), device='cuda')
            print(f"Dummy input device: {dummy_input.device}")
            
            # This would normally require proper Moshi input format
            # but we're just testing device placement
            print("✅ GPU inference test setup complete")
        
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # Cleanup
        await client.cleanup()
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_moshi_gpu())
