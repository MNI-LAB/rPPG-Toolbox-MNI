# MY_FFT Implementation Summary

## Overview
The MY_FFT method is a custom FFT-based heart rate estimation approach that replicates the original CVSM program's FFT implementation. It provides both heart rate estimation and diagnostic visualization capabilities.

## Key Features

### 1. FFT Methodology
- **Algorithm**: Uses `np.fft.rfft()` for real-valued FFT
- **Frequency Range**: 50-150 BPM (physiological heart rate range)
- **Peak Detection**: Finds maximum magnitude in frequency spectrum
- **No Zero-Padding**: Unlike standard FFT, doesn't pad to next power of 2

### 2. Comparison with Standard Methods

| Feature | Standard FFT | Label FFT | MY_FFT |
|---------|-------------|-----------|---------|
| FFT Method | `scipy.signal.periodogram` | `np.fft.rfft` | `np.fft.rfft` |
| Frequency Range | 0.6-3.3 Hz (36-198 BPM) | 50-150 BPM | 50-150 BPM |
| Zero-padding | Yes (to power of 2) | No | No |
| Preprocessing | Bandpass + Savgol | None | **Savgol only** |
| Diagnostic Plots | No | No | **Yes** |

*Note: MY_FFT intentionally uses only Savgol filtering to compare different preprocessing approaches

### 3. Diagnostic Capabilities
MY_FFT generates two types of diagnostic plots:

#### HR Frequency Spectrum
- Shows FFT magnitude vs frequency (BPM)
- Highlights detected peak frequency
- Saved as `{name}_HR_diagram.pdf`

#### Time Domain Waveform  
- Shows original PPG signal vs time
- Includes estimated HR in title
- Saved as `{name}_waveform.pdf`

## Implementation Details

### Function Signature
```python
def find_HR(intensity, config, name, fps):
    """
    Calculate heart rate using MY_FFT method (based on original CVSM approach).
    
    Args:
        intensity: PPG signal
        config: Configuration object  
        name: Name for saving plots ('pred' or 'label')
        fps: Sampling frequency
        
    Returns:
        HR: Estimated heart rate in BPM
    """
```

### Algorithm Steps
1. **FFT Computation**: `np.fft.rfft(intensity)`
2. **Magnitude Spectrum**: `np.abs(fft_result)`
3. **Frequency Axis**: `np.fft.rfftfreq(len(intensity), 1.0/fps) * 60.0`
4. **Filtering**: Keep only 50-150 BPM range
5. **Peak Detection**: `freq_filtered[np.argmax(magnitude)]`
6. **Plot Generation**: Create diagnostic visualizations

## Configuration

### YAML Settings
```yaml
INFERENCE:
  EVALUATION_METHOD: "MY_FFT"  # Enable MY_FFT
```

### Directory Structure
MY_FFT creates the following directory structure:
```
LOG_PATH/
├── EXP_DATA_NAME/
│   ├── HR_diagram/
│   │   ├── pred/
│   │   │   └── pred_HR_diagram.pdf
│   │   └── label/
│   │       └── label_HR_diagram.pdf
│   └── waveform/
│       ├── pred/
│       │   └── pred_waveform.pdf
│       └── label/
│           └── label_waveform.pdf
```

## Improvements Made

### 1. Enhanced Error Handling
- Validates frequency range availability
- Provides fallback HR value (75 BPM) if no valid frequencies found
- Graceful handling of plot generation failures

### 2. Better Visualizations
- Improved plot aesthetics with proper legends and labels
- Enhanced HR spectrum plot with peak highlighting
- Time domain plot includes estimated HR in title

### 3. Savgol-Only Preprocessing Approach
- Intentionally uses `use_bandpass=False` with `use_savgol=True`
- Allows comparison between bandpass vs Savgol-only preprocessing methods
- Savgol filter provides smoothing while preserving signal characteristics

### 4. Documentation
- Added comprehensive docstrings
- Clear parameter descriptions
- Return value specification

## When to Use MY_FFT

### Advantages
✅ **Visual Inspection**: Diagnostic plots allow manual verification of results
✅ **CVSM Compatibility**: Matches original CVSM approach exactly  
✅ **Simplicity**: Straightforward FFT without complex spectral estimation
✅ **Debugging**: Easy to identify issues in heart rate estimation
✅ **Savgol-Only Preprocessing**: Tests effectiveness of smoothing without frequency domain filtering
✅ **Methodological Comparison**: Allows direct comparison of preprocessing approaches

### Considerations
⚠️ **Performance**: Generates many plots for large datasets
⚠️ **Storage**: Diagnostic plots require significant disk space
⚠️ **Speed**: Slower than standard FFT due to plot generation

### Best Use Cases
- Research and development
- Method validation and comparison
- Debugging heart rate estimation issues
- Visual inspection of PPG signal quality
- Replicating original CVSM results

## Example Usage

```python
# In your config file
INFERENCE:
  EVALUATION_METHOD: "MY_FFT"

# The method will be automatically called during evaluation
# and generate diagnostic plots in the specified directories
```

## Future Enhancements

1. **Configurable Plot Generation**: Add flag to enable/disable plots
2. **Custom Frequency Ranges**: Allow user-defined frequency bounds
3. **Additional Metrics**: Include confidence measures for peak detection
4. **Batch Processing**: Optimize for large-scale evaluations
5. **Interactive Plots**: Web-based visualization capabilities

## Troubleshooting

### Common Issues
1. **Missing Directories**: Ensure LOG_PATH and EXP_DATA_NAME are properly set
2. **Plot Generation Errors**: Check matplotlib backend and file permissions
3. **Empty Frequency Range**: Verify signal quality and sampling rate
4. **Memory Issues**: Consider disabling plots for very large datasets

### Debug Tips
- Check generated plots for signal quality assessment
- Compare with standard FFT results for validation
- Monitor console output for warnings and errors
- Verify directory permissions for plot saving
