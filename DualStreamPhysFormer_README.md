# Dual-Stream PhysFormer

This repository contains an implementation of a dual-stream PhysFormer model that processes both RGB and depth video streams for improved rPPG (remote photoplethysmography) performance.

## Overview

The dual-stream approach involves processing RGB and depth videos in parallel initially, then fusing them before the tokenization step. This architecture allows the model to leverage both color/texture information from RGB data and geometric/motion information from depth data, leading to better motion compensation and more robust rPPG signal extraction.

## Architecture

### Step 1: Parallel Stems 🧠
- **RGB Stem**: Processes 3-channel RGB video to extract color and texture features
- **Depth Stem**: Processes 1-channel depth video to extract geometry, shape, and motion features

### Step 2: Feature Fusion ✨
- Concatenates RGB and depth feature maps along the channel dimension
- Applies a fusion layer to create a unified representation
- Results in a single, richer feature map containing both modalities

### Step 3: Tube Tokenization
- Standard PhysFormer transformer processing on the fused features
- Each tube token inherently contains both RGB and depth information
- Enables learning of complex cross-modal relationships

## Files

- `neural_methods/model/DualStreamPhysFormer.py` - The dual-stream model implementation
- `neural_methods/trainer/DualStreamPhysFormerTrainer.py` - Trainer for the dual-stream model
- `configs/train_configs/DualStreamPhysFormer_example.yaml` - Example configuration file

## Usage

### 1. Configuration

Create a configuration file based on the example:

```yaml
MODEL:
  NAME: 'DualStreamPhysFormer'
  DUALSTREAMPHYSFORMER:
    PATCH_SIZE: 4
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    THETA: 0.7
    RGB_STEM_CHANNELS: [24, 48, 96]    # Optional: customize RGB stem
    DEPTH_STEM_CHANNELS: [12, 24, 48]  # Optional: customize depth stem

TRAIN:
  DATA:
    PREPROCESS:
      DATA_TYPE: ['RGB', 'Depth']  # Specify both data types
```

### 2. Data Format

The model expects dual-stream input:
- **RGB Data**: `[B, 3, T, H, W]` - 3-channel RGB video
- **Depth Data**: `[B, 1, T, H, W]` - 1-channel depth video
- **Labels**: `[B, T]` - Ground truth PPG signals

### 3. Training

```bash
python main.py --config_file configs/train_configs/DualStreamPhysFormer_example.yaml
```

### 4. Inference

```bash
python main.py --config_file configs/train_configs/DualStreamPhysFormer_example.yaml --toolbox_mode only_test
```

## Key Features

### Flexible Stem Configuration
- Customizable channel configurations for both RGB and depth stems
- Default configurations provided for common use cases
- Easy to adapt for different input resolutions and computational budgets

### Backward Compatibility
- Falls back to duplicating single-stream data if only RGB is provided
- Maintains compatibility with existing single-stream datasets
- Gradual migration path from single to dual-stream

### Enhanced Motion Compensation
- Depth information provides geometric context for motion
- Better separation of motion artifacts from physiological signals
- Improved robustness in challenging scenarios

## Model Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `PATCH_SIZE` | Size of patches for tokenization | 4 |
| `DIM` | Feature dimension | 96 |
| `FF_DIM` | Feed-forward dimension | 144 |
| `NUM_HEADS` | Number of attention heads | 4 |
| `NUM_LAYERS` | Number of transformer layers | 12 |
| `THETA` | CDC convolution parameter | 0.7 |
| `RGB_STEM_CHANNELS` | RGB stem channel progression | [24, 48, 96] |
| `DEPTH_STEM_CHANNELS` | Depth stem channel progression | [12, 24, 48] |

## Performance Benefits

1. **Motion Robustness**: Depth information helps distinguish between head movements and physiological signals
2. **Geometric Context**: Better understanding of facial structure and movements
3. **Multi-modal Learning**: Leverages complementary information from both streams
4. **Improved Accuracy**: Enhanced rPPG signal quality in challenging conditions

## Requirements

- PyTorch >= 1.8.0
- Same dependencies as the original PhysFormer implementation
- Compatible with the rPPG-Toolbox framework

## Citation

If you use this implementation, please cite the original PhysFormer paper and mention the dual-stream extension:

```bibtex
@article{yu2022physformer,
  title={PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
  author={Yu, Zitong and Li, Xiaobai and Niu, Xuesong and Zhao, Guoying and Zhao, Gang},
  journal={arXiv preprint arXiv:2203.14518},
  year={2022}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the dual-stream implementation.

## License

This implementation follows the same license as the original PhysFormer and rPPG-Toolbox.
