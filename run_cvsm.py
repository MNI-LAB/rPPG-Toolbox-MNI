import os
import argparse
from config import get_config
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import json
import torch
import scipy.signal
from tqdm import tqdm
from PIL import Image, ImageDraw
from typing import Tuple
from evaluation.metrics import calculate_metrics # Import the metrics function

# Assume face_mesh_module.py is in the same directory or available in the Python path
from classical_methods.face_mesh_module import FaceMeshDetector

# load iPadDataLoader
from dataset.data_loader.iPadDataLoader import iPadDataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser

# --- Helper Functions (From previous code snippets) ---

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
    mask_canvas = Image.new('L', (w, h), 0)
    pixels_passed_in = list(map(tuple, b_pixels.tolist()))
    ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1)
    pixels_in_ROI = np.array(mask_canvas)
    return pixels_in_ROI

def get_bounding_box(roi_name: str, landmarks_pixels: np.ndarray) -> np.ndarray:
    landmark_indices = face_roi_definitions[roi_name]
    bounding_box_pixels = landmarks_pixels[landmark_indices]
    return bounding_box_pixels

def phase3(intensity: np.ndarray, fps: int, order: int = 2, i_window_size: int = 10) -> Tuple[float, float]:
    def find_HR(intensity: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        if len(intensity) == 0:
            return 0.0, np.array([]), np.array([]), 0.0

        Intensity_freq = np.fft.rfft(intensity)
        X_final = np.abs(Intensity_freq)
        freq = np.fft.rfftfreq(len(intensity), 1.0 / fps) * 60.0
        
        mask = (freq >= 50) & (freq <= 150)
        freq_filtered = freq[mask]
        hr_arr = X_final[mask]

        if hr_arr.size == 0:
            return 0.0, np.array([]), np.array([]), 0.0

        HR = freq_filtered[np.argmax(hr_arr)]
        
        if np.sum(hr_arr) == 0:
            confidence = 0.0
        else:
            hr_intensity = hr_arr[np.argmax(hr_arr)]
            confidence = hr_intensity / np.mean(hr_arr)
        return HR, freq_filtered, hr_arr, confidence

    if len(intensity) >= i_window_size and i_window_size % 2 == 1:
        intensity = scipy.signal.savgol_filter(intensity, i_window_size, order, mode='nearest')
    
    HR_raw, _, _, confidence = find_HR(intensity)
    return HR_raw, confidence

def interval_process(intensity: np.ndarray, fps: int, interval_size: int = 5, overlap: float = 2.5) -> Tuple[list, list]:
    hrs = []
    confs = []
    
    step = int((interval_size - overlap) * fps)
    if step <= 0:
        step = 1

    interval_frames = interval_size * fps
    
    for i in range(0, len(intensity) - interval_frames + 1, step):
        interval_intensity = intensity[i : i + interval_frames]
        HR_raw, confidence = phase3(interval_intensity, fps)
        hrs.append(HR_raw)
        confs.append(confidence)
    
    original_length = len(confs)
    if original_length > 0:
        mean_conf = np.mean(confs)
        std_conf = np.std(confs)
        filtered = [(hr, confs[i]) for i, hr in enumerate(hrs) if abs(confs[i] - mean_conf) <= 1 * std_conf]
        if filtered:
            hrs, confs = zip(*filtered)
            hrs = list(hrs)
            confs = list(confs)
        else:
            hrs, confs = [], []

    return hrs, confs

# --- Data Loading Functions ---

def read_video(video_file: str) -> np.ndarray:
    """Reads a rgb video and depth video, returns combined frames (T, H, W, 4)"""
    frames = list()
    all_png = sorted(glob.glob(os.path.join(video_file, "video", '*.png')))
    all_depth = sorted(glob.glob(os.path.join(video_file, "depth", '*.png')))
    
    num_frames = min(len(all_png), len(all_depth))
    if num_frames == 0:
        print(f"Warning: No video or depth frames found in {video_file}")
        return np.asarray([])

    i = 0
    for i in tqdm(range(num_frames), desc=f"Loading frames from {os.path.basename(video_file)}"):
        img = cv2.imread(all_png[i])
        depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
        
        if img is None or depth is None:
            i += 1
            continue

        if depth.ndim > 2:
            depth = depth[:, :, 0]

        img = cv2.resize(img, (depth.shape[1], depth.shape[0]))
        
        depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
        
        rgbd = np.concatenate((img, depth), axis=2)
        frames.append(rgbd)
        i += 1
    return np.asarray(frames)

def read_wave(bvp_file: str) -> np.ndarray:
    with open(bvp_file, "r") as f:
        labels = json.load(f)
        waves = [label["waveform"] for label in labels]
    return np.asarray(waves)

def chunk_signals(signal: np.ndarray, chunk_len: int, device: str = 'cpu') -> dict:
    """
    Chunks a full-length signal into a dictionary of tensors,
    compatible with rPPG-Toolbox's metric calculation.
    """
    chunked_signals = {}
    total_len = len(signal)
    
    # Discard the last incomplete chunk to match framework's behavior
    num_chunks = total_len // chunk_len
    
    for i in range(num_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunk = signal[start:end]
        
        # Reshape to (chunk_len, 1) and convert to a PyTorch tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device).view(-1, 1)
        chunked_signals[i] = chunk_tensor
    
    # graph signal using plt
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Original Signal', color='blue')
    for i, chunk in chunked_signals.items():
        plt.plot(range(i * chunk_len, (i + 1) * chunk_len), chunk.cpu().numpy(), label=f'Chunk {i}', linestyle='--')
    plt.title('Signal Chunking Visualization')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid()
    plt.savefig('signal_chunking_visualization.png', dpi=300)
    plt.close()
    
    return chunked_signals

# --- FaceProcessing Class (Adapted for this script) ---

class FaceProcessing:
    def __init__(self, fps: int):
        self.fps = fps
        self.face_roi_definitions = face_roi_definitions
        self.face_mesh_detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def Depth_compensation(self, I_raw, Depth, timeWindow, Fs):
        I_comp = np.ones_like(I_raw, dtype=np.float32)
        best = 1
        best_rem = 1
        
        for ROI in range(1):
            I_comp_ROI = np.ones(len(I_raw), dtype=np.float32)
            i = 1
            while (i * (timeWindow * Fs)) <= len(I_raw):
                cor = 2
                start_idx = (i - 1) * (timeWindow * Fs)
                end_idx = i * (timeWindow * Fs)
                I_seg = I_raw[start_idx:end_idx]
                D_seg = Depth[start_idx:end_idx]
                
                if np.std(D_seg) == 0:
                    bI_comp = (I_seg - np.mean(I_seg)) / (np.std(I_seg) + 1e-7) if np.std(I_seg) > 0 else np.zeros_like(I_seg)
                    I_comp_ROI[start_idx:end_idx] = bI_comp
                    i += 1
                    continue
                
                for bi in np.arange(0.2, 5.01, 0.01):
                    bI_comp_raw = I_seg / (D_seg ** (-bi))
                    corr_v = np.corrcoef(bI_comp_raw, D_seg)
                    corr_ = abs(corr_v[1, 0])
                    if corr_ < cor:
                        cor = corr_
                        best = bI_comp_raw
                I_comp_ROI[start_idx:end_idx] = (best - np.mean(best)) / (np.std(best) + 1e-7) if np.std(best) > 0 else np.zeros_like(best)
                i += 1
            
            start_idx = ((i - 1) * (timeWindow * Fs))
            if start_idx < len(I_raw):
                cor = 2
                I_rem = I_raw[start_idx:]
                D_rem = Depth[start_idx:]

                if np.std(D_rem) == 0:
                    bI_comp = (I_rem - np.mean(I_rem)) / (np.std(I_rem) + 1e-7) if np.std(I_rem) > 0 else np.zeros_like(I_rem)
                    I_comp_ROI[start_idx:len(I_raw)] = bI_comp
                else:
                    for bii in np.arange(0.2, 5.1, 0.1):
                        bI_comp_raw = I_rem / (D_rem ** (-bii))
                        corr_v = np.corrcoef(bI_comp_raw, D_rem)
                        corr_ = abs(corr_v[1, 0])
                        if corr_ < cor:
                            cor = corr_
                            best_rem = bI_comp_raw
                    I_comp_ROI[start_idx:len(I_raw)] = (best_rem - np.mean(best_rem)) / (np.std(best_rem) + 1e-7) if np.std(best_rem) > 0 else np.zeros_like(best_rem)
            I_comp = I_comp_ROI
        return I_comp


    def get_pixels_in_ROI(self, b_pixels,h,w):
        return get_pixels_in_ROI(b_pixels, h, w)
    
    def get_bounding_box(self, roi_name, landmarks_pixels):
        return get_bounding_box(roi_name, landmarks_pixels)

    # def predict(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    #     ppg_signal_g = []
    #     depth_signal = []

    #     if rgb.size == 0 or depth.size == 0:
    #         return np.array([])
            
    #     num_frames = rgb.shape[0]

    #     for i in tqdm(range(num_frames), desc="Processing frames for PPG signal"):
    #         rgb_f = rgb[i]
    #         depth_f = depth[i]
            
    #         face_detected, landmarks_pixels = self.face_mesh_detector.find_face_mesh(image=rgb_f, draw=False)

    #         if face_detected:
    #             bounding_box_pixels = self.get_bounding_box('cheek_n_nose', landmarks_pixels)
    #             h, w = rgb_f.shape[:2]
    #             pixels_in_ROI = self.get_pixels_in_ROI(bounding_box_pixels, h, w)
                
    #             g = rgb_f[:, :, 1]
    #             mean_intensity = np.mean(g[pixels_in_ROI > 0])
    #             ppg_signal_g.append(mean_intensity)
                
    #             mean_depth = np.mean(depth_f[pixels_in_ROI > 0])
    #             depth_signal.append(mean_depth)
        
    #     ppg_signal_g = np.array(ppg_signal_g)
    #     depth_signal = np.array(depth_signal)

    #     if ppg_signal_g.size == 0:
    #         print("No valid PPG signal detected for this clip.")
    #         return np.array([])

    #     time_window_sec = 5
    #     compensated_ppg_signal = self.Depth_compensation(ppg_signal_g, depth_signal, time_window_sec, self.fps)
        
    #     return compensated_ppg_signal
    def predict(self, video_path: str) -> np.ndarray:
        """Process video frames and extract PPG signal in a single loop"""
        ppg_signal_g = []
        depth_signal = []
        
        # Get file lists
        all_png = sorted(glob.glob(os.path.join(video_path, "video", '*.png')))
        all_depth = sorted(glob.glob(os.path.join(video_path, "depth", '*.png')))
        
        num_frames = min(len(all_png), len(all_depth))
        if num_frames == 0:
            print(f"Warning: No video or depth frames found in {video_path}")
            return np.array([])
        
        print(f"Processing {num_frames} frames from {os.path.basename(video_path)}...")

        for i in tqdm(range(num_frames), desc="Processing frames"):
            # Load images directly from disk
            img = cv2.imread(all_png[i])
            depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
            
            if img is None or depth is None:
                continue

            # Process depth image
            if depth.ndim > 2:
                depth = depth[:, :, 0]

            # Resize RGB to match depth dimensions
            img = cv2.resize(img, (depth.shape[1], depth.shape[0]))
            
            # Face detection and PPG extraction
            face_detected, landmarks_pixels = self.face_mesh_detector.find_face_mesh(image=img, draw=False)

            if face_detected:
                bounding_box_pixels = self.get_bounding_box('cheek_n_nose', landmarks_pixels)
                h, w = img.shape[:2]
                pixels_in_ROI = self.get_pixels_in_ROI(bounding_box_pixels, h, w)
                
                # Extract green channel intensity
                g = img[:, :, 1]
                mean_intensity = np.mean(g[pixels_in_ROI > 0])
                ppg_signal_g.append(mean_intensity)
                
                # Extract depth information
                mean_depth = np.mean(depth[pixels_in_ROI > 0])
                depth_signal.append(mean_depth)
        
        ppg_signal_g = np.array(ppg_signal_g)
        depth_signal = np.array(depth_signal)

        if ppg_signal_g.size == 0:
            print("No valid PPG signal detected for this clip.")
            return np.array([])

        # Apply depth compensation
        time_window_sec = 5
        compensated_ppg_signal = self.Depth_compensation(ppg_signal_g, depth_signal, time_window_sec, self.fps)
        
        return compensated_ppg_signal

# --- Main Script Execution ---

if __name__ == "__main__":
    # class Config:
    #     def __init__(self):
    #         self.BASE = ['']
    #         self.TOOLBOX_MODE = "only_test"
    #         self.TEST = type('TEST', (), {
    #             'METRICS': ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA'],  # All metrics from YAML
    #             'USE_LAST_EPOCH': False,
    #             'DATA': type('DATA', (), {
    #                 'FS': 30,
    #                 'DATASET': "iPadData",
    #                 'DO_PREPROCESS': True,
    #                 'DATA_FORMAT': "NDCHW",
    #                 'DATA_PATH': "/nfs/turbo/coe-mni/iPadData/training",
    #                 'CACHED_PATH': "/nfs/turbo/coe-mni/iPadData_preprocess",
    #                 'EXP_DATA_NAME': "",
    #                 'BEGIN': 0.8,
    #                 'END': 1.0,
    #                 'PREPROCESS': type('PREPROCESS', (), {
    #                     'DATA_TYPE': [ 'DiffNormalized','Standardized' ],
    #                     'LABEL_TYPE': "Raw",  # <-- Adjusted to match your raw data
    #                     'DO_CHUNK': True,
    #                     'CHUNK_LENGTH': 180,  # <-- Adjusted to match YAML
    #                     'CROP_FACE': type('CROP_FACE', (), {
    #                         'DO_CROP_FACE': True,
    #                         'BACKEND': 'HC',
    #                         'USE_LARGE_FACE_BOX': True,
    #                         'LARGE_BOX_COEF': 1.5,
    #                         'DETECTION': type('DETECTION', (), {
    #                             'DO_DYNAMIC_DETECTION': False,
    #                             'DYNAMIC_DETECTION_FREQUENCY': 30,
    #                             'USE_MEDIAN_FACE_BOX': False,
    #                         })
    #                     }),
    #                     'RESIZE': type('RESIZE', (), {
    #                         'H': 560,
    #                         'W': 560
    #                     })
    #                 })
    #             })
    #         })
    #         self.DEVICE = "cpu"  # Assuming you're running on CPU for classical methods
    #         self.NUM_OF_GPU_TRAIN = 1
    #         self.LOG = type('LOG', (), {
    #             'PATH': "/nfs/turbo/coe-mni/toolbox_runs/ipad_cvsm_exp"
    #         })
    #         self.MODEL = type('MODEL', (), {
    #             'DROP_RATE': 0.2,
    #             'NAME': "CVSM",
    #         })
    #         self.INFERENCE = type('INFERENCE', (), {
    #             'BATCH_SIZE': 4,
    #             'EVALUATION_METHOD': "FFT",
    #             'MODEL_PATH': "cvsm_classical_pipeline",
    #             'EVALUATION_WINDOW': type('EVALUATION_WINDOW', (), {
    #                 'USE_SMALLER_WINDOW': False,
    #                 'WINDOW_SIZE': 30
    #             })
    #         })

    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    fps = config.TEST.DATA.FS
    chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

    # Get the full list of data directories
    # ipadDataLoader = iPadDataLoader(name="test", data_path=config.TEST.DATA.DATA_PATH, config_data=config.TEST.DATA, device=config.DEVICE)
    
    full_data_path = "/nfs/turbo/coe-mni/iPadData/test"
    all_dirs_with_index = []
    for i, dir_name in enumerate(sorted(os.listdir(full_data_path))):
        full_path = os.path.join(full_data_path, dir_name)
        if os.path.isdir(full_path):
            all_dirs_with_index.append({'index': i, 'path': full_path})

    # data_dir_full = ipadDataLoader.split_raw_data(all_dirs_with_index, begin=0.8, end=1.0)
    data_dir = [d['path'] for d in all_dirs_with_index]
    data_dir = data_dir[:2]
    data_dir.sort()
    print(f'All data directories found: {data_dir}')
    all_predictions = {}
    all_labels = {}
    face_processor = FaceProcessing(fps=fps)

    for clip_path in tqdm(data_dir, desc="Processing clips"):
        clip_name = os.path.basename(clip_path)
        subject_id = clip_name
        
        # Load ground truth BVP
        bvp_file_path = os.path.join(clip_path, f"{clip_name}.json")
        try:
            gt_bvp_wave = read_wave(bvp_file_path)
        except FileNotFoundError:
            print(f"Warning: BVP file not found for {clip_name}. Skipping...")
            continue
        
        # Process video and extract PPG signal in single loop
        compensated_ppg_signal = face_processor.predict(clip_path)

        if compensated_ppg_signal.size > 0:
            # Ensure both signals have the same length
            min_length = min(len(compensated_ppg_signal), len(gt_bvp_wave))
            compensated_ppg_signal = compensated_ppg_signal[:min_length]
            gt_bvp_wave = gt_bvp_wave[:min_length]
            
            # Convert to chunked format for metrics calculation
            predictions_chunks = chunk_signals(compensated_ppg_signal, chunk_len, config.DEVICE)
            labels_chunks = chunk_signals(gt_bvp_wave, chunk_len, config.DEVICE)
            
            # Only add if chunks were created (signal was long enough)
            if predictions_chunks:
                all_predictions[subject_id] = predictions_chunks
                all_labels[subject_id] = labels_chunks
            else:
                print(f"Warning: Signal for {clip_name} too short for chunking. Skipping...")
        else:
            print(f"Warning: No valid signal detected for {clip_name}. Skipping...")

    print("\nProcessing complete. All predictions and labels have been collected.")

    # Filter out subjects with no valid predictions before calculating metrics
    all_predictions_filtered = {k: v for k, v in all_predictions.items() if v}  # v is now a dict, so this works
    all_labels_filtered = {k: v for k, v in all_labels.items() if k in all_predictions_filtered}

    if not all_predictions_filtered:
        print("No valid predictions were generated. Cannot calculate metrics.")
    else:
        print(f"\nCalculating metrics for all {len(all_predictions_filtered)} clips...")
        calculate_metrics(all_predictions_filtered, all_labels_filtered, config)