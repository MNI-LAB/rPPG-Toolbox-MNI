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
import random
from evaluation.metrics import calculate_metrics # Import the metrics function
from evaluation.post_process import calculate_metric_per_video, _detrend # Import standard toolbox functions
from scipy.signal import butter
from omnican_gt_hr_values import get_omnican_gt_hr_values # Import hardcoded GT values

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
    CONFIG_FILE = "configs/infer_configs/iPadData_CVSM_GREATLAKES.yaml"
    parser.add_argument('--config_file', required=False,
                        default=CONFIG_FILE, type=str, help="The name of the model.")
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

# Removed phase3 function - will use standard toolbox HR calculation methods instead

# Removed interval_process function - will use standard toolbox evaluation windowing instead

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

def chunk_signals_for_standard_evaluation(pred_signal: np.ndarray, gt_signal: np.ndarray, config) -> Tuple[dict, dict]:
    """
    Prepare signals for standard toolbox evaluation by creating single-chunk dictionaries.
    The standard toolbox evaluation will handle windowing internally.
    """
    # Create single chunk containing the entire signal for each subject
    # The standard evaluation function will apply windowing based on config.INFERENCE.EVALUATION_WINDOW
    pred_dict = {0: torch.tensor(pred_signal, dtype=torch.float32, device=config.DEVICE).view(-1, 1)}
    gt_dict = {0: torch.tensor(gt_signal, dtype=torch.float32, device=config.DEVICE).view(-1, 1)}
    
    return pred_dict, gt_dict

def calculate_metrics_with_hardcoded_gt(predictions, labels, config):
    """Calculate metrics using hardcoded GT HR values from OMNICAN for consistency"""
    from evaluation.metrics import _reform_data_from_dict
    from evaluation.post_process import calculate_metric_per_video
    import numpy as np
    from datetime import datetime
    import os
    import sys
    
    # Get the hardcoded GT values
    hardcoded_gt_hrs = get_omnican_gt_hr_values()
    print(f"Using {len(hardcoded_gt_hrs)} hardcoded GT HR values from OMNICAN")
    
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    
    print("Calculating metrics with hardcoded GT values...")
    
    gt_index = 0  # Index for hardcoded GT values
    
    for index in sorted(predictions.keys()):
        try:
            prediction = _reform_data_from_dict(predictions[index])
            label = _reform_data_from_dict(labels[index])  # We still need this for SNR calculation
            
            video_frame_size = prediction.shape[0]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(prediction), window_frame_size):
                pred_window = prediction[i:i+window_frame_size]
                label_window = label[i:i+window_frame_size]

                if len(pred_window) < 9:
                    print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                        config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                    diff_flag_test = False
                elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                    diff_flag_test = True
                else:
                    raise ValueError("Unsupported label type in testing!")
                
                if config.INFERENCE.EVALUATION_METHOD == "FFT":
                    # Calculate predicted HR using standard method
                    _, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                        pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                    
                    # Use hardcoded GT HR instead of calculated one
                    if gt_index < len(hardcoded_gt_hrs):
                        gt_hr_fft = hardcoded_gt_hrs[gt_index]
                        gt_index += 1
                    else:
                        print(f"Warning: Not enough hardcoded GT values. Using calculated value.")
                        gt_hr_fft, _, _, _ = calculate_metric_per_video(
                            pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                    
                    gt_hr_fft_all.append(gt_hr_fft)
                    predict_hr_fft_all.append(pred_hr_fft)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                else:
                    raise ValueError("Only FFT evaluation method is supported in this version!")
                    
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue
    
    # Convert to numpy arrays
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    SNR_all = np.array(SNR_all)
    MACC_all = np.array(MACC_all)
    
    print(f'Predicted HR FFT: {predict_hr_fft_all}')
    print(f'Ground    HR FFT: {gt_hr_fft_all}')
    
    num_test_samples = len(predict_hr_fft_all)
    
    # Calculate all the metrics
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
            print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
        elif metric == "RMSE":
            squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
            RMSE_FFT = np.sqrt(np.mean(squared_errors))
            standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
            print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
            print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            correlation_coefficient = Pearson_FFT[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
        elif metric == "SNR":
            SNR_FFT = np.mean(SNR_all)
            standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
            print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
        elif metric == "MACC":
            MACC_avg = np.mean(MACC_all)
            standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
            print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
        elif "BA" in metric:
            from evaluation.BlandAltmanPy import BlandAltman
            compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
            # Generate file name
            filename_id = f"CVSM_vs_OMNICAN_GT"
            compare.scatter_plot(
                x_label='GT PPG HR [bpm]',
                y_label='rPPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
            compare.difference_plot(
                x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                y_label='Average of rPPG HR and GT PPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
        else:
            pass  # Skip unknown metrics
    
    print(f"\\nUsed {gt_index} out of {len(hardcoded_gt_hrs)} hardcoded GT values")

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
        
        # print(f"Processing {num_frames} frames from {os.path.basename(video_path)}...")

        # for i in tqdm(range(num_frames), desc="Processing frames"):
        for i in range(num_frames):
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
    
    
# Removed debug_chunks function - no longer needed with standard evaluation

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
    
    # Ensure we're using evaluation windows like the standard toolbox
    print(f"Using evaluation window size: {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE} seconds")
    print(f"Evaluation method: {config.INFERENCE.EVALUATION_METHOD}")
    print(f"Label type: {config.TEST.DATA.PREPROCESS.LABEL_TYPE}")

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
    # data_dir = data_dir[:2]
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
            
            # Check minimum signal length for evaluation
            if min_length < 9:  # Same minimum as standard toolbox
                print(f"Warning: Signal for {clip_name} too short ({min_length} frames). Skipping...")
                continue
            
            # Prepare signals for standard evaluation (single chunk per subject)
            pred_chunks, gt_chunks = chunk_signals_for_standard_evaluation(
                compensated_ppg_signal, gt_bvp_wave, config)
            
            all_predictions[subject_id] = pred_chunks
            all_labels[subject_id] = gt_chunks
        else:
            print(f"Warning: No valid signal detected for {clip_name}. Skipping...")

    print("\nProcessing complete. All predictions and labels have been collected.")

    # Filter out subjects with no valid predictions before calculating metrics
    all_predictions_filtered = {k: v for k, v in all_predictions.items() if v}
    all_labels_filtered = {k: v for k, v in all_labels.items() if k in all_predictions_filtered}

    if not all_predictions_filtered:
        print("No valid predictions were generated. Cannot calculate metrics.")
    else:
        print(f"\nCalculating metrics for all {len(all_predictions_filtered)} clips using hardcoded OMNICAN GT values...")
        # Use the custom metrics calculation with hardcoded GT values
        calculate_metrics_with_hardcoded_gt(all_predictions_filtered, all_labels_filtered, config)