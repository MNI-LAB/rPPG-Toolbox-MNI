"""
CVSMLoader for iPad RGBD data with MediaPipe face detection.
Extracts 1D signals (green + depth) for unsupervised CVSM method while maintaining
compatibility with the neural network preprocessing pipeline.
"""

import glob
import json
import os
import numpy as np
import cv2
from dataset.data_loader.BaseLoader import BaseLoader
from classical_methods.face_mesh_module import FaceMeshDetector
from PIL import Image, ImageDraw
from tqdm import tqdm

# Force MediaPipe to use CPU only to avoid GPU context conflicts
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# Face ROI definitions (using cheek_n_nose as default for PPG extraction)
face_roi_definitions = {
    'nose': np.array([196, 419, 455, 235]),
    'forehead': np.array([109, 338, 9]),
    'cheek_n_nose': np.array([117, 346, 411, 187]),
    'left_cheek': np.array([131, 165, 214, 50]),
    'right_cheek': np.array([372, 433, 358]),
    'low_forehead': np.array([108, 337, 8]),
    'whole_face': np.array([109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])
}


class CVSMLoader(BaseLoader):
    """
    CVSM data loader for iPad RGBD data.
    
    This loader processes RGBD frames through MediaPipe face detection to extract
    1D signals (average green and depth intensities in face ROI) suitable for
    unsupervised PPG estimation while maintaining compatibility with the neural
    network preprocessing pipeline.
    """

    def __init__(self, name, data_path, config_data, device=None):
        """Initialize CVSMLoader."""
        self.roi_name = getattr(config_data, 'CVSM_ROI', 'cheek_n_nose')
        print(f"CVSMLoader initialized with ROI: {self.roi_name}")
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (same as iPadDataLoader)."""
        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            index = int(subject_trail_val)
            subject_id = os.path.split(data_dir)[-1].split('_')[0]
            dirs.append({"index": index, "path": data_dir, "subject": subject_id})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs (same as iPadDataLoader)."""
        if begin == 0 and end == 1:
            return data_dirs
        
        data_dirs = sorted(data_dirs, key=lambda x: x['index'])
        begin_index = int(begin * len(data_dirs))    
        end_index = int(end * len(data_dirs)) 
        
        data_dirs_new = []
        for i in range(begin_index, end_index):
            data_dirs_new.append(data_dirs[i])
        
        data_dirs_new = sorted(data_dirs_new, key=lambda x: x['index'])
        return data_dirs_new

    def get_pixels_in_ROI(self, b_pixels, h, w):
        """Create mask for pixels within ROI polygon."""
        mask_canvas = Image.new('L', (w, h), 0)
        pixels_passed_in = list(map(tuple, b_pixels.tolist()))
        ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1)
        pixels_in_ROI = np.array(mask_canvas)
        return pixels_in_ROI

    def get_bounding_box(self, roi_name, landmarks_pixels):
        """Get bounding box pixels for specified ROI."""
        landmark_indices = face_roi_definitions[roi_name]
        bounding_box_pixels = landmarks_pixels[landmark_indices]
        return bounding_box_pixels

    def extract_cvsm_signal(self, frames):
        """
        Extract 1D CVSM signal from RGBD frames using MediaPipe face detection.
        
        Args:
            frames: RGBD frames of shape (T, H, W, 4)
            
        Returns:
            tuple: (green_signal, depth_signal) both of shape (T,)
        """
        # Initialize face mesh detector
        face_detector = FaceMeshDetector(static_image_mode=True)
        
        green_signal = []
        depth_signal = []
        failed_detections = 0
        
        try:
            for i, frame in enumerate(frames):
                try:
                    # Split RGBD frame
                    rgb_frame = frame[:, :, :3]  # RGB channels
                    depth_frame = frame[:, :, 3]  # Depth channel
                    
                    # Ensure RGB frame is in correct format (uint8)
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    
                    # Detect face landmarks
                    face_detected, landmarks = face_detector.find_face_mesh(rgb_frame, draw=False)
                    
                    if face_detected:
                        # Get ROI pixels
                        roi_pixels = self.get_bounding_box(self.roi_name, landmarks)
                        
                        # Create ROI mask
                        h, w = rgb_frame.shape[:2]
                        roi_mask = self.get_pixels_in_ROI(roi_pixels, h, w)
                        
                        # Extract average values in ROI
                        valid_pixels = np.sum(roi_mask)
                        
                        if valid_pixels > 0:
                            # Green channel average (primary PPG signal)
                            masked_green = rgb_frame[:, :, 1] * roi_mask  # Green channel
                            avg_green = np.sum(masked_green) / valid_pixels
                            
                            # Depth channel average (additional depth info)
                            masked_depth = depth_frame * roi_mask
                            avg_depth = np.sum(masked_depth) / valid_pixels
                        else:
                            avg_green = 0.0
                            avg_depth = 0.0
                            failed_detections += 1
                    else:
                        # No face detected, use fallback values
                        avg_green = 0.0
                        avg_depth = 0.0
                        failed_detections += 1
                        
                    green_signal.append(avg_green)
                    depth_signal.append(avg_depth)
                    
                except Exception as e:
                    print(f"Warning: Frame {i} processing failed: {e}")
                    green_signal.append(0.0)
                    depth_signal.append(0.0)
                    failed_detections += 1
                    continue
            
            # Print warning if too many frames failed
            if failed_detections > len(frames) * 0.5:
                print(f"Warning: {failed_detections}/{len(frames)} frames failed face detection.")
            
        finally:
            # Cleanup face detector resources
            try:
                if hasattr(face_detector, '__del__'):
                    face_detector.__del__()
            except:
                pass
        
        return np.array(green_signal), np.array(depth_signal)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Process single video: extract CVSM signals and save in neural network format."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # print(f"Processing {filename} for CVSM...")

        # Read RGBD frames (same as iPadDataLoader)
        if 'None' in config_preprocess.DATA_AUG:
            frames = self.read_video(os.path.join(data_dirs[i]['path'], ""))
            if frames is None or len(frames) == 0:
                print(f"Failed to load frames for {filename}")
                return 
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG for {self.dataset_name}! Received {config_preprocess.DATA_AUG}.')

        # Extract 1D CVSM signals
        green_signal, depth_signal = self.extract_cvsm_signal(frames)
        
        # Combine signals: shape (T, 2) where channels are [green, depth]
        cvsm_signals = np.stack([green_signal, depth_signal], axis=1)
        
        # Read PPG labels (same as iPadDataLoader)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'], "{0}.json".format(filename)))
        
        # Ensure signals and labels have same length
        target_length = len(green_signal)
        if len(bvps) != target_length:
            print(f"Resampling labels from {len(bvps)} to {target_length} samples")
            bvps = BaseLoader.resample_ppg(bvps, target_length)
        
        # Save as single clips (no chunking for CVSM)
        if config_preprocess.DO_CHUNK:
            print("Warning: Chunking disabled for CVSM - using full sequences")
        
        signals_clips = np.array([cvsm_signals])  # Shape: (1, T, 2) 
        bvps_clips = np.array([bvps])  # Shape: (1, T)
        
        # Save using the same format as neural network loaders
        input_name_list, label_name_list = self.save_multi_process(signals_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Read RGBD video frames (same as iPadDataLoader.read_video)."""
        frames = list()
        all_png = sorted(glob.glob(video_file + "video/" + '*.png'))
        all_depth = sorted(glob.glob(video_file + "depth/" + '*.png'))
        i = 0
        num_frames = min(len(all_png), len(all_depth))
        
        while i < num_frames:
            img = cv2.imread(all_png[i])
            depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
            depth = depth[:, :, 0]
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
            
            # Resize img to depth's shape
            img = cv2.resize(img, (depth.shape[1], depth.shape[0]))
            
            # Stack RGB and depth
            rgbd = np.concatenate((img, depth), axis=2)
            frames.append(rgbd)
            i += 1
            
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Read PPG labels from JSON file (same as iPadDataLoader.read_wave)."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["waveform"] for label in labels]
        return np.asarray(waves)