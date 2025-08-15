#!/usr/bin/env python3
"""
Test script for Dual-Stream PhysFormer model.
This script creates a simple test to verify the model works correctly.
"""

import torch
import numpy as np
from neural_methods.model.DualStreamPhysFormer import DualStreamPhysFormer

def test_dual_stream_physformer():
    """Test the dual-stream PhysFormer model with dummy data."""
    
    print("Testing Dual-Stream PhysFormer Model...")
    
    # Model parameters
    batch_size = 2
    frames = 160
    height = 128  # Must be large enough so that after 3 MaxPool3d layers, we get 16x16
    width = 128   # Then 16x16 with 8x8 patches gives 2x2 spatial patches
    dim = 96
    
    # Create dummy data
    rgb_data = torch.randn(batch_size, 3, frames, height, width)
    depth_data = torch.randn(batch_size, 1, frames, height, width)
    gra_sharp = 2.0
    
    print(f"RGB data shape: {rgb_data.shape}")
    print(f"Depth data shape: {depth_data.shape}")
    
    # Initialize model
    model = DualStreamPhysFormer(
        image_size=(frames, height, width),
        patches=(4, 8, 8),  # Temporal: 4, Spatial: 8x8 (adjusted for smaller input)
        dim=dim,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.2,
        theta=0.7
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            rPPG, score1, score2, score3 = model(rgb_data, depth_data, gra_sharp)
            print(f"✓ Forward pass successful!")
            print(f"rPPG output shape: {rPPG.shape}")
            print(f"Expected rPPG shape: ({batch_size}, {frames})")
            print(f"Score shapes: {score1.shape}, {score2.shape}, {score3.shape}")
            
            # Verify output dimensions
            assert rPPG.shape == (batch_size, frames), f"Expected rPPG shape ({batch_size}, {frames}), got {rPPG.shape}"
            print("✓ Output dimensions correct!")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    # Test with different batch sizes
    try:
        single_batch_rgb = torch.randn(1, 3, frames, height, width)
        single_batch_depth = torch.randn(1, 1, frames, height, width)
        
        rPPG_single, _, _, _ = model(single_batch_rgb, single_batch_depth, gra_sharp)
        assert rPPG_single.shape == (1, frames), f"Single batch failed: expected (1, {frames}), got {rPPG_single.shape}"
        print("✓ Single batch processing successful!")
        
    except Exception as e:
        print(f"✗ Single batch test failed: {e}")
        return False
    
    # Test model parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Verify reasonable parameter count (should be similar to original PhysFormer)
        assert total_params > 1000000, "Model seems too small"
        assert total_params < 10000000, "Model seems too large"
        print("✓ Parameter count reasonable!")
        
    except Exception as e:
        print(f"✗ Parameter check failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Dual-Stream PhysFormer is working correctly.")
    return True

def test_model_configurations():
    """Test different model configurations."""
    
    print("\nTesting different model configurations...")
    
    frames, height, width = 160, 128, 128  # Must be large enough so that after 3 MaxPool3d layers, we get 16x16
    
    # Test with custom stem channels
    try:
        model_custom = DualStreamPhysFormer(
            image_size=(frames, height, width),
            patches=(4, 8, 8),  # Temporal: 4, Spatial: 8x8
            dim=128,
            ff_dim=256,
            num_heads=8,
            num_layers=6,
            dropout_rate=0.1,
            theta=0.5,
            rgb_stem_channels=[32, 64, 128],
            depth_stem_channels=[16, 32, 64]
        )
        
        rgb_data = torch.randn(1, 3, frames, height, width)
        depth_data = torch.randn(1, 1, frames, height, width)
        
        rPPG, _, _, _ = model_custom(rgb_data, depth_data, 2.0)
        assert rPPG.shape == (1, frames), "Custom configuration failed"
        print("✓ Custom stem channels configuration successful!")
        
    except Exception as e:
        print(f"✗ Custom configuration failed: {e}")
        return False
    
    # Test with minimal configuration
    try:
        model_minimal = DualStreamPhysFormer(
            image_size=(frames, height, width),
            patches=(4, 4, 4),
            dim=64,
            ff_dim=128,
            num_heads=2,
            num_layers=6,
            dropout_rate=0.1,
            theta=0.5
        )
        
        rgb_data = torch.randn(1, 3, frames, height, width)
        depth_data = torch.randn(1, 1, frames, height, width)
        
        rPPG, _, _, _ = model_minimal(rgb_data, depth_data, 2.0)
        assert rPPG.shape == (1, frames), "Minimal configuration failed"
        print("✓ Minimal configuration successful!")
        
    except Exception as e:
        print(f"✗ Minimal configuration failed: {e}")
        return False
    
    print("✓ All configuration tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("DUAL-STREAM PHYSFORMER TEST SUITE")
    print("=" * 60)
    
    # Run basic tests
    basic_tests_passed = test_dual_stream_physformer()
    
    # Run configuration tests
    config_tests_passed = test_model_configurations()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if basic_tests_passed and config_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The Dual-Stream PhysFormer model is ready to use.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    print("=" * 60)
