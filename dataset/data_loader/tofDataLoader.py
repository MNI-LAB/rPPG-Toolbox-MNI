"""The dataloader for tof data: RGBND
"""
import glob
import glob
import json
import os
import re
import random
from math import ceil

import cv2
import numpy as np
import torch
from dataset.data_loader.BaseLoader import BaseLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class tofDataLoader(BaseLoader):
    """The data loader for the tof dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """
        All data are under the Data folder in data-collector/
        Data structure:
        |-- year-month-day-hour-minute-second_subject_name/
            |-- Depth/
                |-- *.png # (640x480)
            |-- RGB/
                |-- *.png # (1920x1080)
            |-- Intensity/ 
                |-- *.png # (640x480)
            |-- year-month-day-hour-minute-second_subject_name.json (ground truth label)
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For tof dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        subject_id_map = {} # string, int pairs 
        subject_idx = 0
        clip_idx = 0
        for data_dir in data_dirs:
            # get the subject name
            subject_name = os.path.split(data_dir)[-1].split('_')[-1]
            if subject_name not in subject_id_map:
                subject_id_map[subject_name] = subject_idx
                subject_idx += 1
                clip_idx = 0
            else:
                clip_idx += 1
            subject_id = subject_id_map[subject_name]
            dirs.append({"index": clip_idx, "path": data_dir, "subject": subject_id})
            
        # randomize the dirs
        random.shuffle(dirs)
        # print out how many clips per subject
        # Count clips per subject
        clips_per_subject = {}
        for dir_info in dirs:
            subject_id = dir_info["subject"]
            for name, id in subject_id_map.items():
                if id == subject_id:
                    if name not in clips_per_subject:
                        clips_per_subject[name] = 1
                    else:
                        clips_per_subject[name] += 1
                    break
        
        print(f'Number of clips per subject:')
        for subject_name, num_clips in clips_per_subject.items():
            print(f'Subject: {subject_name}, Number of clips: {num_clips}')
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs
        
        # find start and end of the dataset
        # get the subject number from the data_dirs
        data_dirs = sorted(data_dirs, key=lambda x: x['index'])  # sort by index
        # find the starting index 
        begin_index = int(begin * len(data_dirs))    
        end_index = int(end * len(data_dirs)) 
        
        # build new data_dirs list
        data_dirs_new = []
        for i in range(begin_index, end_index):
            data_dirs_new.append(data_dirs[i])
        
        # sort the data_dirs_new by index
        data_dirs_new = sorted(data_dirs_new, key=lambda x: x['index'])  # sort by index
        
        
        # print out all clip names
        # print("Data directories for split:")
        # for data_dir in data_dirs_new:
        #     print(f"Index: {data_dir['index']}, Path: {data_dir['path']}")
        
        return data_dirs_new
    
    def load_rgb_depth_pair(self, rgb_path, depth_path):
        """
        Loads RGB and Depth images, then stacks them into a 4D array.
        """
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # Shape: (H, W, 3)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Shape: (H, W)

        if rgb is None or depth is None:
            print(f"Failed to load {rgb_path} or {depth_path}")
            return None

        # resize both to 480 width, 640 height if not already
        if rgb.shape[0] != 640 or rgb.shape[1] != 480:
            rgb = cv2.resize(rgb, (480, 640))
        if depth.shape[0] != 640 or depth.shape[1] != 480:
            depth = cv2.resize(depth, (480, 640))   
        
        # Normalize depth to 0-255 range
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Expand Depth to 1 channel
        depth = np.expand_dims(depth, axis=2)  # Shape: (H, W, 1)
        # Stack RGB and Depth channels -> (H, W, 4)
        rgbd = np.concatenate((rgb, depth), axis=2)  # Shape: (H, W, 4)

        
        return rgbd
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'], ""))
            if frames is None or len(frames) == 0:
                return 
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Ensure all sequences have exactly 400 frames
        target_length = 400
        if frames.shape[0] < target_length:
            # Pad shorter sequences by repeating the last frame
            padding_frames = np.tile(frames[-1:], (target_length - frames.shape[0], 1, 1, 1))
            frames = np.concatenate([frames, padding_frames], axis=0)
        elif frames.shape[0] > target_length:
            # Truncate longer sequences to 400 frames
            frames = frames[:target_length]
        
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            # bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            print(f'This option is NOT allowed for tofDataLoader. Please use the default option.')
            exit()
        else:
            # There's only 1 json file for each clip so just find the first one
            bvps = self.read_wave(
                glob.glob(os.path.join(data_dirs[i]['path'], "*.json"))[0])
        
        # Ensure labels also have exactly 400 samples
        if bvps.shape[0] > target_length:
            bvps = self.resample_gt_ppg(bvps, target_length, gt_fps=60, video_fps=20)
        elif bvps.shape[0] < target_length:
            # Pad shorter sequences by repeating the last frame
            bvps = self.resample_gt_ppg(bvps, target_length, gt_fps=60, video_fps=20)
        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        # save the two chunks of bvps for debugging
        # print(f'bvps_clips shape: {bvps_clips.shape}')
        # plt.plot(bvps_clips[0], label='bvps_clips[0]')
        # plt.plot(bvps_clips[1], label='bvps_clips[1]')
        # plt.legend()
        # plt.savefig('bvps_clips_chunked.png')
        # plt.close()
        # exit()
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a rgb video file, NIR video file, and depth video file, returns frames(T, H, W, 5) """
        frames = list()
        all_png = sorted(glob.glob(video_file + "RGB/" + '*.png'))
        all_depth = sorted(glob.glob(video_file + "Depth/" + '*.png'))
        all_nir = sorted(glob.glob(video_file + "Intensity/" + '*.png'))
        i = 0
        num_frames = min(len(all_png), len(all_depth), len(all_nir))
        # print(num_frames)
        # pbar = tqdm(range(num_frames), desc='Reading frames')
        for i in range(num_frames):
            img = cv2.imread(all_png[i])
            depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
            nir = cv2.imread(all_nir[i], cv2.IMREAD_COLOR)
            if nir.ndim == 3:
                nir = nir[:, :, :1]
            if depth.ndim == 3:
                depth = depth[:, :, :1]
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
            
            # resize RGB and NIR to depth's shape
            if img.shape[:2] != depth.shape[:2]:
                # crop image width from 1920 to 1436.4
                margins = int((1920 - 1436.4) / 2)
                img = img[:, margins:1920-margins, :]
                img = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            if nir.shape[:2] != depth.shape[:2]:
                nir = cv2.resize(nir, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

            # stack rgb and depth
            rgbd = np.concatenate((img, nir, depth), axis=2)
            frames.append(rgbd)
        return np.asarray(frames)
    @staticmethod
    def read_wave(bvp_file):
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["waveform"] for label in labels]
        return np.asarray(waves)
    
    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection, 
                         detection_freq, use_median_box, width, height):
        """Crop face and resize frames for tof dataset. Since we have RGBNIR and depth, we need to do face detection on RGB then resize the entire stacked frames.

        Args:
            frames(np.array): Video frames with shape (T, H, W, 5) where channels are [R, G, B, NIR, Depth].
            use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                         and resizing.
                                         If True, it performs face detection every "detection_freq" frames.
            detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
            use_face_detection(bool):  Whether crop the face.
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
        Returns:
            resized_frames(np.array): Resized and cropped frames with shape (T, height, width, 5)
        """
        # Face Cropping
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        
        # Perform face detection by num_dynamic_det times.
        for idx in range(num_dynamic_det):
            if use_face_detection:
                # Use only RGB channels (first 3) for face detection
                rgb_frame = frames[detection_freq * idx][:, :, :3]
                face_region_all.append(self.face_detection(rgb_frame, backend, use_larger_box, larger_box_coef))
            else:
                # Use full frame if no face detection
                face_region_all.append([0, 0, frames.shape[2], frames.shape[1]])  # [x, y, width, height]
        
        face_region_all = np.asarray(face_region_all, dtype='int')
        
        if use_median_box:
            # Generate a median bounding box based on all detected face regions
            face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Frame Resizing
        total_frames, frame_height, frame_width, channels = frames.shape
        resized_frames = np.zeros((total_frames, height, width, channels), dtype=frames.dtype)
        
        for i in range(total_frames):
            frame = frames[i]  # Shape: (H, W, 5)
            
            # Determine which face region to use
            if use_dynamic_detection:
                reference_index = i // detection_freq
            else:
                reference_index = 0
            
            # Apply face cropping if enabled
            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]
                
                # Extract face region from entire frame (all 5 channels)
                x, y, w, h = face_region
                y_start = max(y, 0)
                y_end = min(y + h, frame_height)
                x_start = max(x, 0)
                x_end = min(x + w, frame_width)
                
                frame = frame[y_start:y_end, x_start:x_end, :]  # Crop all channels
            
            # Resize each channel separately (OpenCV has issues with >4 channels)
            if frame.shape[2] > 4:  # More than 4 channels
                resized_channels = []
                for c in range(frame.shape[2]):
                    channel = frame[:, :, c]
                    if channel.ndim == 2:  # Single channel
                        resized_channel = cv2.resize(channel, (width, height), interpolation=cv2.INTER_AREA)
                        resized_channels.append(resized_channel)
                    else:  # Multi-channel (shouldn't happen, but just in case)
                        resized_channel = cv2.resize(channel, (width, height), interpolation=cv2.INTER_AREA)
                        resized_channels.append(resized_channel)
                resized = np.stack(resized_channels, axis=2)
            else:
                # Standard resize for <=4 channels
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Ensure proper shape for single channel data
            if resized.ndim == 2:  # grayscale image
                resized = resized[..., np.newaxis]  # shape (H, W, 1)
            
            # save the first frame of the resized frame for debugging
            if i == 0:
                cv2.imwrite(f'resized_frame_{i}_rgb.png', resized[:, :, :3])
                cv2.imwrite(f'resized_frame_{i}_nir.png', resized[:, :, 3])
                cv2.imwrite(f'resized_frame_{i}_depth.png', resized[:, :, 4])
            
            resized_frames[i] = resized
            
        return resized_frames

    def resample_gt_ppg(self, gt_ppg: np.ndarray, target_length: int, gt_fps: int = 60, video_fps: int = 20) -> np.ndarray:
        """Resample ground truth PPG to match video frame length accounting for different sampling rates."""
        if gt_ppg.size == 0:
            return np.empty((0,))
        
        # Simple resampling: just use the standard resample_ppg function
        # The key is that we want the GT PPG to have the same number of samples as video frames
        # regardless of the original sampling rates
        resampled = BaseLoader.resample_ppg(gt_ppg, target_length)
        # save the first frame of the resampled ppg for debugging
        # if len(resampled) > 0:
        #     plt.plot(resampled[0], label='resampled_ppg')
        #     plt.legend()
        #     plt.savefig('resampled_ppg.png')
        #     plt.close()
        #     exit()
        print(f"  Resampling GT: {len(gt_ppg)} samples -> {len(resampled)} samples")
        return resampled

    def __getitem__(self, index):
        """Returns a clip of video(5,T,W,H) and it's corresponding signals(T) for RGBNIR+Depth data."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        
        # DEBUG: Print what we're actually loading
        # print(f"Raw data shape: {data.shape}, Raw label shape: {label.shape}")

        # exit()
        # Handle different data formats for RGBNIR+Depth (5 channels)
        if self.data_format == 'NDCHW':
            # Convert from (T, H, W, C) to (T, C, H, W) for RGBNIR+Depth
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            # Convert from (C, T, H, W) to (T, C, H, W) for RGBNIR+Depth
            data = np.transpose(data, (1, 0, 2, 3))
        elif self.data_format == 'NDHWC':
            # Keep as (T, H, W, C) - no transpose needed
            pass
        else:
            raise ValueError(f'Unsupported Data Format for tofData: {self.data_format}!')
        
        data = np.float32(data)
        label = np.float32(label)
        
        # print(f"RETURNING: data.shape={data.shape}, label.shape={label.shape}")
        # plt.plot(label, label='label')
        # plt.legend()
        # plt.savefig('label.png')
        # plt.close()
        # Extract metadata
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.rindex('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        
        return data, label, filename, chunk_id

    @staticmethod
    def custom_collate_fn(batch):
        """Custom collate function to properly handle RGBNIR+Depth data and labels."""
        # batch is a list of tuples: [(data, label, filename, chunk_id), ...]
        data_list = []
        label_list = []
        filename_list = []
        chunk_id_list = []
        
        for data, label, filename, chunk_id in batch:
            data_list.append(torch.from_numpy(data))
            label_list.append(torch.from_numpy(label))
            filename_list.append(filename)
            chunk_id_list.append(chunk_id)
        
        # Stack data and labels properly
        data_batch = torch.stack(data_list, dim=0)  # Shape: (batch_size, T, C, H, W)
        label_batch = torch.stack(label_list, dim=0)  # Shape: (batch_size, T)
        
        return data_batch, label_batch, filename_list, chunk_id_list