"""CVSM (Classical Video-based Signal Measurement)
Custom method for extracting PPG signals from RGBD video data using face mesh detection and ROI analysis.
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
from classical_methods.face_mesh_module import FaceMeshDetector


# Face ROI definitions for CVSM
face_roi_definitions = {
    'nose': np.array([196, 419, 455, 235]),
    'forehead': np.array([109, 338, 9]),
    'cheek_n_nose': np.array([117, 346, 411, 187]),
    'left_cheek': np.array([131, 165, 214, 50]),
    'right_cheek': np.array([372, 433, 358]),
    'low_forehead': np.array([108, 337, 8]),
    'left_eye': np.array([33, 160, 159, 158, 133, 153, 145, 144]),
    'right_eye': np.array([263, 387, 386, 385, 362, 380, 374, 373]),
    'whole_face': np.array([109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])
}


def get_pixels_in_ROI(b_pixels: np.ndarray, h: int, w: int) -> np.ndarray:
    """Create mask for pixels within ROI polygon."""
    mask_canvas = Image.new('L', (w, h), 0)
    pixels_passed_in = list(map(tuple, b_pixels.tolist()))
    ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1)
    pixels_in_ROI = np.array(mask_canvas)
    return pixels_in_ROI


def get_bounding_box(roi_name: str, landmarks_pixels: np.ndarray) -> np.ndarray:
    """Get bounding box pixels for specified ROI."""
    landmark_indices = face_roi_definitions[roi_name]
    bounding_box_pixels = landmarks_pixels[landmark_indices]
    return bounding_box_pixels


def CVSM(frames, fs=30, roi_name='cheek_n_nose'):
    """
    Extract PPG signal from RGBD video frames using CVSM method.
    
    Args:
        frames: Input video frames of shape (T, H, W, C) where C=4 for RGBD
        fs: Frame rate (default 30 fps)
        roi_name: Face ROI to use for signal extraction
        
    Returns:
        BVP: Extracted blood volume pulse signal
    """
    # Initialize face mesh detector
    face_detector = FaceMeshDetector()
    
    ppg_signal = []
    
    for frame in frames:
        # Handle both RGB (3 channels) and RGBD (4 channels) input
        if frame.shape[2] == 4:
            # RGBD input - split into RGB and depth
            rgb_frame = frame[:, :, :3]
            depth_frame = frame[:, :, 3]
        else:
            # RGB only input
            rgb_frame = frame[:, :, :3]
            depth_frame = None
        
        # Ensure RGB frame is in correct format (uint8)
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        
        # Detect face landmarks
        landmarks = face_detector.get_landmarks(rgb_frame)
        
        if landmarks is not None:
            # Get ROI pixels
            roi_pixels = get_bounding_box(roi_name, landmarks)
            
            # Create ROI mask
            h, w = rgb_frame.shape[:2]
            roi_mask = get_pixels_in_ROI(roi_pixels, h, w)
            
            # Extract average RGB values in ROI
            masked_rgb = rgb_frame * roi_mask[:, :, np.newaxis]
            valid_pixels = np.sum(roi_mask)
            
            if valid_pixels > 0:
                avg_rgb = np.sum(masked_rgb, axis=(0, 1)) / valid_pixels
                # Use green channel as primary PPG signal (standard approach)
                ppg_value = avg_rgb[1]  # Green channel
            else:
                ppg_value = 0.0
        else:
            # No face detected, use fallback (could be improved)
            ppg_value = 0.0
            
        ppg_signal.append(ppg_value)
    
    # Convert to numpy array and normalize
    BVP = np.array(ppg_signal)
    
    # Basic preprocessing - remove DC component
    BVP = BVP - np.mean(BVP)
    
    # Optional: Apply basic bandpass filtering for HR range
    from scipy.signal import butter, filtfilt
    # Filter to heart rate range: 0.75-2.5 Hz (45-150 BPM)
    nyquist = fs / 2
    low_freq = 0.75 / nyquist
    high_freq = 2.5 / nyquist
    
    if high_freq < 1.0 and low_freq > 0:
        b, a = butter(4, [low_freq, high_freq], btype='band')
        BVP = filtfilt(b, a, BVP)
    
    return BVP
