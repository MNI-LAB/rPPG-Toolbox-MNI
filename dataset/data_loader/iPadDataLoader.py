"""The dataloader for iPad data: RGBD
"""
import glob
import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class iPadDataLoader(BaseLoader):
    """The data loader for the iPad dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """
        
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            print(subject_trail_val)
            index = int(subject_trail_val)
            # subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": 0}) # hard coded subject
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

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'], "{0}.json".format(filename)))
        if bvps.shape[0] == frames.shape[0]:
            print(f"Warning: {filename} has different length of frames and labels.")
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a rgb video file, extracts green channel only, returns frames(T, H, W, 1) """
        frames = list()
        all_png = sorted(glob.glob(video_file + "video/" + '*.png'))
        all_depth = sorted(glob.glob(video_file + "depth/" + '*.png'))
        num_frames = len(all_png)
        if num_frames < 600:
            print(f'Warning: current clip has insufficient # of frames: {num_frames}')
            return None
        i = 0
        while i < num_frames:
            img = cv2.imread(all_png[i])
            depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
            depth = depth[:, :, 0]
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
            if img is None or depth is None:
                print(f"Warning: failed to read {all_png[i]}")
                i += 1
                continue
            # === Extract green channel only ===
            # green = img[:, :, 1]  # Extract green channel
            # green = green[:, :, np.newaxis]  # Expand to (H, W, 1) for consistency
            # frames.append(green)
            # === RGBD ===
            rgbd = np.concatenate((img, depth), axis=2)
            frames.append(rgbd)
            i += 1
        frames = np.asarray(frames)
        return frames
    @staticmethod
    def read_video(video_file):
        """Reads a rgb video file and depth video file, returns frames(T, H, W, 4) """
        frames = list()
        all_png = sorted(glob.glob(video_file + "video/" + '*.png'))
        all_depth = sorted(glob.glob(video_file + "depth/" + '*.png'))
        i = 0
        num_frames = min(len(all_png), len(all_depth))
        # print(num_frames)
        while i < num_frames:
            img = cv2.imread(all_png[i])
            depth = cv2.imread(all_depth[i], cv2.IMREAD_UNCHANGED)
            depth = depth[:, :, 0]
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
            
            #resize img to depth's shape
            img = cv2.resize(img, (depth.shape[1], depth.shape[0]))
            # if i == 0:
            #     print(f'img shape: {img.shape}, depth shape: {depth.shape}')
            #     cv2.imshow('img', img)
            #     cv2.imshow('depth', depth)
            #     cv2.waitKey(1)
            # stack rgb and depth
            
            rgbd = np.concatenate((img, depth), axis=2)
            frames.append(rgbd)
            # print(f'frame {i}')
            # cv2.imshow('frame', img)
            # cv2.imshow('depth', depth)
            # cv2.waitKey(1)
            i += 1
        # print('finish 1 clip')
        return np.asarray(frames)
    @staticmethod
    def read_wave(bvp_file):
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["waveform"] for label in labels]
        return np.asarray(waves)