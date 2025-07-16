"""The dataloader for MR-NIRP datasets (NIR, RGB, PulseOx)."""

import glob
import os
import pickle
import scipy.io
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from PIL import ImageStat, Image



def load_pulseox_mat(mat_path):
    """Loads pulseOx.mat and returns the one non-meta array inside."""
    mat = scipy.io.loadmat(mat_path)
    # filter out MATLAB meta-keys
    keys = [k for k in mat.keys() if not k.startswith('__')]
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one data variable in {mat_path}, got {keys}")
    arr = mat[keys[0]].squeeze()
    return arr


class MRNIRPLoader(BaseLoader):
    """Data loader for the MR-NIRP dataset (NIR, RGB, PulseOx)."""

    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Find all clip folders:
        
        MR-NIRP/
            Subject1/
                clip_name/
                    NIR/
                    PulseOX/
                    RGB/
                clip_name2/
                    ...
        """
        pattern = os.path.join(data_path, '*')
        dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
        if not dirs:
            raise ValueError(f"{self.dataset_name}: no data found under {data_path}")
        out = []
        for clip in dirs:
            clip_name = os.path.basename(clip)
            subj = os.path.basename(os.path.dirname(clip))
            nir_dir = os.path.join(clip, 'NIR', 'NIR')
            pulseox_dir = os.path.join(clip, 'PulseOX', 'PulseOX')
            out.append({
                'index': clip_name,
                'path': clip,
                'subject': subj,
                'datapath': data_path,
                'nir_dir': nir_dir,
                'pulseox_dir': pulseox_dir,
            })
        return out

    def split_raw_data(self, data_dirs, begin, end):
        """Split by subject so no overlap between train/val/test."""
        if begin == 0 and end == 1:
            return data_dirs
        info = {}
        for d in data_dirs:
            info.setdefault(d['index'], []).append(d)
        subjects = sorted(info.keys())
        n = len(subjects)
        sel = subjects[int(begin * n): int(end * n)]
        out = []
        for subj in sel:
            out.extend(info[subj])
        return out
    
    def correct_irregular_sampling(self, ppg, timestamps, target_fs=30):
        """Resampling functionality borrowed from: https://github.com/ToyotaResearchInstitute/RemotePPG"""
        resampled_ppg = []
        for curr_time in np.arange(0.0, timestamps[-1], 1.0/target_fs):
            time_diff = timestamps - curr_time
            stop_idx = np.argmax(time_diff > 0)
            start_idx = stop_idx - 1 if stop_idx > 0 else stop_idx
            
            time_span = time_diff[stop_idx] - time_diff[start_idx]
            weight = - time_diff[start_idx] / time_span if time_span != 0 else 0
            
            interpolated_ppg = ppg[start_idx] * (1 - weight) + ppg[stop_idx] * weight
            resampled_ppg.append(interpolated_ppg)
        
        return np.array(resampled_ppg)
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        clip = data_dirs[i]
        clip_name = clip['index']
        nir_path = clip['nir_dir']
        pulseox_path = clip['pulseox_dir']
        
        # NOTE: need to put this into .yaml file!!!!!
        clip_length = 60 # 1 min clip
        fps = 30
                
        # --- load full PPG log and values --- 
        # Look for either 'pulseOx.mat' or 'pulseOX.mat'
        mat_file_lower = os.path.join(pulseox_path, 'pulseOx.mat')
        mat_file_upper = os.path.join(pulseox_path, 'pulseOX.mat')
        if os.path.isfile(mat_file_lower):
            mat_file = mat_file_lower
        elif os.path.isfile(mat_file_upper):
            mat_file = mat_file_upper
        else:
            raise FileNotFoundError(f"Neither 'pulseOx.mat' nor 'pulseOX.mat' found in {pulseox_path}")
        mat = scipy.io.loadmat(mat_file)
        ppg = mat['pulseOxRecord'][0]
        timestamps = (mat['pulseOxTime'][0] - mat['pulseOxTime'][0][0])
        # print(f'ppg shape: {ppg.shape}')
        # print(f'timestamp shape: {timestamps.shape}')
        ppg = self.correct_irregular_sampling(ppg, timestamps, target_fs=fps)
        ppg = ppg[:fps*clip_length]
        
        # --- read NIR frames ---
        nir_files = sorted(glob.glob(os.path.join(nir_path, 'Frame*.pgm')))
        nir_files = nir_files[:fps*clip_length*2] # x2 to filter out the black frames
        
        # check if to skip first or 2nd frame
        im = Image.open(nir_files[0]).convert('L')
        brightness = ImageStat.Stat(im)
        if brightness.mean[0] > 20:
            nir_files = nir_files[::2]
        else:
            nir_files = nir_files[1::2]
        
        if len(nir_files) != len(ppg):
            common_length = min(len(nir_files), len(ppg))
            nir_files = nir_files[:common_length]
            ppg = ppg[:common_length]
        
        nir_frames = []
        for f in nir_files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[Warning] Failed to load image: {f}")
                continue
            img = img[..., None] if img.ndim == 2 else img  # ensure shape (H, W, 1)
            nir_frames.append(img)
        
        nir_frames = np.stack(nir_frames, axis=0).astype(np.float32)
        # print(f'nir_frames shape: {nir_frames.shape}')
        # Save PPG waveform as a matplotlib plot
        # ppg_plot_path = f"debug_frames/{clip_name}_ppg.png"
        # plt.figure(figsize=(10, 3))
        # plt.plot(ppg)
        # plt.title(f"PPG waveform: {clip_name}")
        # plt.xlabel("Frame")
        # plt.ylabel("Amplitude")
        # plt.tight_layout()
        # plt.savefig(ppg_plot_path)
        # plt.close()
        
        # if i == 0:
        #     # Save PPG waveform as a matplotlib plot
        #     ppg_plot_path = f"{clip_name}_ppg.png"
        #     plt.figure(figsize=(10, 3))
        #     plt.plot(ppg)
        #     plt.title(f"PPG waveform: {clip_name}")
        #     plt.xlabel("Frame")
        #     plt.ylabel("Amplitude")
        #     plt.tight_layout()
        #     plt.savefig(ppg_plot_path)
        #     plt.close()

        #     # Save NIR frames as an AVI video (more robust)
        #     video_path = f"{clip_name}_nir.avi"
        #     height, width = nir_frames.shape[1:3]
        #     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height), isColor=True)
        #     j = 0 
        #     for frame in nir_frames:
        #         frame = frame.squeeze()  # (H, W)

        #         # Normalize if needed
        #         if frame.dtype == np.float32 and (frame.max() > 255 or frame.min() < 0):
        #             frame = 255 * (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

        #         frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)

        #         # Enforce 3-channel RGB format
        #         if frame_uint8.ndim == 2:
        #             frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        #         else:
        #             frame_rgb = frame_uint8
        #         out.write(frame_rgb)
        #         # cv2.imwrite(f"debug_frames/debug_frame_{j}.png", frame_rgb)
        #         j += 1
                

        #     out.release()
        #     print(f"Saved AVI video: {video_path}")

        # Normalize if needed
        frame = nir_frames[0]
        if frame.dtype == np.float32 and (frame.max() > 255 or frame.min() < 0):
            frame = 255 * (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)

        # Enforce 3-channel RGB format
        if frame_uint8.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        else:
            frame_rgb = frame_uint8
        cv2.imwrite(f"debug_frames/file_{clip_name}_frame0.png", frame_rgb)
        
        # --- preprocess and save ---
        # print(f'Frame shape: {nir_frames.shape}, Labels shape: {ppg.shape}')
        clips, labs = self.preprocess(nir_frames, ppg, config_preprocess)
        inputs, _ = self.save_multi_process(clips, labs, clip_name)
        file_list_dict[i] = inputs