"""The dataloader for MR-NIRP datasets (NIR, RGB, PulseOx)."""

import glob
import os
import pickle
import scipy.io
import cv2
import numpy as np
import scipy.io
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


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

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        clip = data_dirs[i]
        clip_name = clip['index']
        nir_path = clip['nir_dir']
        pulseox_path = clip['pulseox_dir']
                
        # --- load full PPG log and values --- 
        log_txt = os.path.join(pulseox_path, 'cam0_full_log.txt')
        with open(log_txt, 'r') as f:
            ts_str = f.read().strip().lstrip('[').rstrip(']')
        timestamps = np.array(list(map(float, ts_str.split(','))))
        rel_ts = timestamps - timestamps[0]

        # ppg values
        pkl_file = os.path.join(pulseox_path, 'cam0_full.pkl')
        with open(pkl_file, 'rb') as f:
            pkl = pickle.load(f, encoding='latin1')
        pulseOxRecord = pkl['pulseOxRecord']
        pulseOxRecord = np.array(pulseOxRecord, dtype=np.float32)
        
        pulseOxTime = pkl['pulseOxTime']
        pulseOxTime = np.array(pulseOxTime, dtype=np.float32)
        
        numPulseSample = pkl['numPulseSample']
        print(f'Loaded {clip_name} with {len(pulseOxRecord)} PPG samples')
        # mat file (not needed)
        # mat_file = os.path.join(pulseox_path, 'pulseOx.mat')
        # mat = scipy.io.loadmat(mat_file)
        # # print(mat)

        # --- read NIR frames ---
        nir_files = sorted(glob.glob(os.path.join(nir_path, 'Frame*.pgm')))
        # print(f'Found {len(nir_files)} NIR frames in {nir_path}')

        nir_frames = []
        
        # save first frame as a reference
        f = nir_files[0]
        ref_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join('/home/mperform/ondemand/rPPG-Toolbox-MNI', 'ref_frame.png'), ref_img)
        
        for f in tqdm(nir_files, desc="Loading NIR .pgm frames"):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[Warning] Failed to load image: {f}")
                continue
            img = img[..., None] if img.ndim == 2 else img  # ensure shape (H, W, 1)
            nir_frames.append(img)
            
        if len(nir_frames) == 0:
            raise RuntimeError("No NIR frames loaded. Check file paths or image format.")
        
        # nir_frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED)[..., None] for f in nir_files]
        frames = np.stack(nir_frames, axis=0).astype(np.float32)
        print("Frames shape:", frames.shape)  # should be (N, H, W)
        # --- read RGB frames ---
        # rgb_dir = os.path.join(clip_path, 'RGB')
        # rgb_files = sorted(glob.glob(os.path.join(rgb_dir, 'Frame*.pgm')))
        
        # print(f'Found {len(rgb_files)} RGB frames in {rgb_dir}')
        
        # rgb_frames = []
        # for f in tqdm(rgb_files, desc="Loading RGB .pgm frames"):
        #     img = cv2.imread(f)
        #     rgb_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # rgb = np.stack(rgb_frames, axis=0).astype(np.float32)

        # frames = np.concatenate([nir, rgb], axis=-1)

        # --- align PPG to video frames @30 fps ---
        N = frames.shape[0]
        fps = getattr(config_preprocess, 'FS_VIDEO', 30)
        frame_dt = 1.0 / fps

        # Convert pulseOxTime to relative time
        rel_ts = pulseOxTime - pulseOxTime[0]
        edges = np.arange(0, (N + 1) * frame_dt, frame_dt)

        ppg_per_frame = np.zeros(N, dtype=np.float32)
        sq_per_frame  = np.ones(N, dtype=np.float32)
        for k in range(N):
            idx = np.where((rel_ts >= edges[k]) & (rel_ts < edges[k+1]))[0]
            if idx.size:
                ppg_per_frame[k] = pulseOxRecord[idx].mean()
            else:
                ppg_per_frame[k] = np.nan
                sq_per_frame[k] = 0.0

        # filter out frames without PPG
        valid = sq_per_frame > 0.0
        frames = frames[valid]
        labels = ppg_per_frame[valid]
        # # --- align PPG to video frames @30 fps ---
        # N = frames.shape[0]
        # fps = getattr(config_preprocess, 'FS_VIDEO', 30)
        # frame_dt = 1.0 / fps
        # edges = np.arange(0, (N + 1) * frame_dt, frame_dt)

        # ppg_per_frame = np.zeros(N, dtype=np.float32)
        # sq_per_frame  = np.ones(N, dtype=np.float32)
        # for k in range(N):
        #     idx = np.where((rel_ts >= edges[k]) & (rel_ts < edges[k+1]))[0]
        #     if idx.size:
        #         ppg_per_frame[k] = ppg_values[idx].mean()
        #     else:
        #         ppg_per_frame[k] = np.nan
        #         sq_per_frame[k] = 0.0

        # # filter out frames without PPG
        # valid = sq_per_frame > 0.0
        # frames = frames[valid]
        # labels = ppg_per_frame[valid]

        # --- preprocess and save ---
        print(f'Frame shape: {frames.shape}, Labels shape: {labels.shape}')
        clips, labs = self.preprocess(frames, labels, config_preprocess)
        inputs, _ = self.save_multi_process(clips, labs, clip_name)
        file_list_dict[i] = inputs
