# CVSMLoader Implementation Guide

## Overview

The CVSMLoader is a specialized data loader that processes iPad RGBD data through MediaPipe face detection to extract 1D signals (average green and depth intensities) for unsupervised CVSM heart rate estimation, while maintaining compatibility with the neural network preprocessing pipeline.

## Key Features

### 1. **MediaPipe Face Detection**
- Uses MediaPipe FaceMesh for robust face landmark detection
- Extracts face ROI (Region of Interest) for signal extraction
- Configurable ROI selection (cheek_n_nose, forehead, etc.)

### 2. **1D Signal Extraction**
- **Green Channel**: Primary PPG signal from RGB data
- **Depth Channel**: Additional depth information from RGBD data
- Both signals averaged within detected face ROI

### 3. **Neural Network Compatibility**
- Saves data in same format as other neural network loaders (.npy files)
- Maintains preprocessing pipeline structure
- Compatible with existing evaluation metrics

### 4. **Unsupervised Method Integration**
- CVSM method automatically detects preprocessed vs raw data
- Processes 1D signals directly without additional face detection
- Maintains backward compatibility with raw frame processing

## Implementation Structure

### Files Created/Modified

1. **`dataset/data_loader/CVSMLoader.py`** - Main CVSMLoader class
2. **`unsupervised_methods/methods/CVSM.py`** - Updated to handle 1D signals
3. **`configs/infer_configs/iPadData_CVSM_PREPROCESSED.yaml`** - Configuration file
4. **`main.py`** - Added CVSMLoader support for unsupervised methods
5. **`dataset/data_loader/__init__.py`** - Added CVSMLoader import

### Data Flow

```
Raw RGBD Frames (T, H, W, 4)
         ↓
    MediaPipe Face Detection
         ↓
    ROI Mask Creation
         ↓
    Signal Extraction
         ↓
    1D Signals (T, 2) [green, depth]
         ↓
    Save as .npy files
         ↓
    CVSM Unsupervised Method
         ↓
    Heart Rate Estimation
```

## Usage Instructions

### Step 1: Configuration

Use the provided configuration file:
```yaml
# configs/infer_configs/iPadData_CVSM_PREPROCESSED.yaml
TOOLBOX_MODE: "unsupervised_method"
UNSUPERVISED:
  METHOD: "CVSM"
  DATA:
    DATASET: iPadData_CVSM  # Key: Use CVSMLoader
    DO_PREPROCESS: True     # Extract 1D signals on first run
    DATA_PATH: "/nfs/turbo/coe-mni/iPadData/test"
    CACHED_PATH: "/nfs/turbo/coe-mni/iPadData_CVSM_preprocessed"
```

### Step 2: First Run (Preprocessing)

Extract 1D signals from raw RGBD data:
```bash
python main.py --config_file configs/infer_configs/iPadData_CVSM_PREPROCESSED.yaml
```

This will:
- Load raw RGBD frames from iPadData
- Run MediaPipe face detection on each frame
- Extract average green and depth intensities in face ROI
- Save as .npy files in CACHED_PATH

### Step 3: Subsequent Runs

Set `DO_PREPROCESS: False` and rerun for faster processing using cached signals.

## Technical Details

### CVSMLoader Class Methods

#### Core Methods
- `extract_cvsm_signal(frames)`: Main signal extraction with MediaPipe
- `get_pixels_in_ROI(b_pixels, h, w)`: Create ROI mask from landmarks
- `get_bounding_box(roi_name, landmarks)`: Get face region pixels

#### Compatibility Methods
- `get_raw_data(data_path)`: Same as iPadDataLoader
- `split_raw_data(data_dirs, begin, end)`: Same as iPadDataLoader
- `read_video(video_file)`: Same as iPadDataLoader (RGBD frames)
- `read_wave(bvp_file)`: Same as iPadDataLoader (PPG labels)

### CVSM Method Updates

The CVSM method now handles two input types:

#### Preprocessed Signals (T, 2)
```python
# Input: (T, 2) where channels are [green, depth]
BVP = frames[:, 0]  # Use green channel
BVP = BVP - np.mean(BVP)  # Remove DC component
# Apply bandpass filtering (0.75-2.5 Hz)
```

#### Raw Frames (T, H, W, 4)
```python
# Falls back to original face detection processing
return _process_raw_frames(frames, fs, roi_name)
```

### Face ROI Options

Available ROI regions for signal extraction:
- `'cheek_n_nose'` (default) - Best for PPG signal quality
- `'forehead'` - Alternative region
- `'left_cheek'`, `'right_cheek'` - Specific cheek regions
- `'whole_face'` - Entire face region

## Advantages

### 1. **Efficiency**
✅ **Preprocessing Once**: MediaPipe face detection runs only during preprocessing
✅ **Fast Inference**: Unsupervised method processes 1D signals directly
✅ **Cached Results**: Preprocessed signals reused for multiple experiments

### 2. **Compatibility**
✅ **Neural Network Format**: Uses same .npy file structure as other loaders
✅ **Existing Pipeline**: Integrates with current evaluation framework
✅ **Backward Compatible**: Original CVSM still works with raw frames

### 3. **Flexibility**
✅ **Configurable ROI**: Easy to experiment with different face regions
✅ **Dual Signal**: Both green (RGB) and depth information preserved
✅ **Standard Preprocessing**: Maintains detrending and filtering options

### 4. **Research Value**
✅ **RGBD Utilization**: Leverages both RGB and depth modalities
✅ **Consistent Evaluation**: Same preprocessing across all test videos
✅ **Diagnostic Capability**: Face detection quality can be monitored

## Expected Output Structure

### Preprocessed Data Directory
```
iPadData_CVSM_preprocessed/
├── DataFileLists/
│   └── [dataset_split_files.csv]
├── [video_id]_input[clip].npy    # Shape: (T, 2) [green, depth]
├── [video_id]_label[clip].npy    # Shape: (T,) PPG labels
└── ...
```

### Evaluation Results
- Standard metrics: MAE, RMSE, MAPE, Pearson, SNR
- Bland-Altman plots for method comparison
- Heart rate estimation accuracy
- Signal quality assessment

## Performance Considerations

### Memory Usage
- 1D signals: ~2.4 KB per second (2 channels × 30 fps × 4 bytes)
- Much smaller than raw frames: ~9.2 MB per second (640×480×4×30)
- **~3800x reduction** in storage requirements

### Processing Speed
- Preprocessing: MediaPipe face detection (~0.1-0.5s per frame)
- Inference: Very fast 1D signal processing (~ms per frame)
- Overall: Front-loaded computation with fast repeated usage

## Troubleshooting

### Common Issues

1. **MediaPipe Installation**
   ```bash
   pip install mediapipe
   ```

2. **Face Detection Failures**
   - Check lighting conditions in videos
   - Monitor face detection success rate in logs
   - Consider different ROI if detection is poor

3. **GPU Conflicts**
   - CVSMLoader forces CPU-only MediaPipe: `os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'`
   - Avoids conflicts with PyTorch GPU usage

4. **Memory Issues**
   - Process videos one at a time (batch_size=1)
   - Use chunking if individual videos are very long

### Configuration Tips

- **First Run**: Set `DO_PREPROCESS: True`
- **Debugging**: Monitor console output for face detection success rates
- **ROI Selection**: Try different `CVSM_ROI` values if signal quality is poor
- **Evaluation**: Use standard unsupervised evaluation metrics

## Future Enhancements

1. **Multi-ROI Fusion**: Combine signals from multiple face regions
2. **Depth-RGB Fusion**: More sophisticated combination of depth and RGB
3. **Quality Assessment**: Automatic ROI selection based on signal quality
4. **Real-time Processing**: Optimize for live video processing
5. **Alternative Landmarks**: Support for different face landmark models

This implementation provides a robust foundation for RGBD-based unsupervised heart rate estimation while maintaining full compatibility with the existing rPPG toolbox infrastructure.
