# rPPG Inference Pipeline - Mermaid Diagram

```mermaid
graph LR
    %% Input Data
    A[Video Clips<br/>20 fps, 400 frames<br/>20 seconds] --> B[Intensity Frames<br/>72×72×1]
    A --> C[Depth Frames<br/>72×72×1]
    
    %% Ground Truth Input
    D[GT PPG JSON<br/>1200 samples<br/>60 fps, 20 seconds] --> E[Resample GT PPG<br/>400 samples @ 20 Hz<br/>Accounting for 3:1 ratio]
    
    %% Preprocessing
    B --> F[Preprocessing<br/>Resize & Standardize<br/>Chunk into batches of 10]
    C --> F
    
    %% Model Inference
    F --> G[OMNI-CAN<br/>Neural Network<br/>1-ch intensity + 1-ch depth]
    G --> H[Predicted PPG Signal<br/>400 samples @ 20 Hz]
    
    %% HR Calculations
    H --> I[Calculate Predicted HR<br/>FFT Method<br/>fs = 30 Hz]
    E --> J[Calculate GT HR<br/>find_HR Method<br/>fs = 20 Hz - FIXED!]
    
    %% Evaluation
    I --> K[Error Metrics<br/>MAE, MAPE, Correlation]
    J --> K
    
    %% Output
    I --> L[Diagnostic Plots<br/>Spectra & Waveforms]
    J --> L
    K --> M[Final Results<br/>HR Comparison]
    L --> M
    
    %% Styling
    classDef inputData fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef model fill:#fff3e0
    classDef calculation fill:#e8f5e8
    classDef output fill:#fce4ec
    classDef fix fill:#ffebee,stroke:#d32f2f,stroke-width:3px
    
    class A,B,C,D inputData
    class F,E processing
    class G model
    class I,J calculation
    class K,L,M output
    class J fix
```

## Key Features:

### 🔄 **Data Flow**
- **Input**: Video clips with intensity/depth frames
- **Processing**: Preprocessing and neural network inference
- **Output**: Heart rate predictions and evaluation metrics

### ⚡ **Sampling Rate Handling**
- **Original GT**: 1200 samples @ 60 Hz
- **Resampled GT**: 400 samples @ 20 Hz (3:1 ratio)
- **Video**: 400 frames @ 20 Hz
- **Predicted**: 400 samples @ 20 Hz

### 🛠️ **Key Fix Applied**
- **Ground Truth HR**: Now uses `fs = 20 Hz` (was incorrectly `fs = 30 Hz`)
- **Predicted HR**: Uses `fs = 30 Hz` (unchanged)
- **Result**: Realistic GT HR values for proper evaluation

### 📊 **Outputs**
- **Error Metrics**: MAE, MAPE, Pearson correlation
- **Diagnostic Plots**: Frequency spectra and time-domain waveforms
- **HR Comparison**: Predicted vs Ground Truth heart rates

### 🎯 **Model Performance**
- **OMNI-CAN**: Processes 1-channel intensity + 1-channel depth
- **Input Size**: 72×72×1 for both intensity and depth
- **Batch Processing**: Chunks of 10 frames for temporal modeling
