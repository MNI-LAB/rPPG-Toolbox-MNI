"""Test suite for the CVSM non ML pipeline
"""
    
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

# for face processing
import cv2
from PIL import Image, ImageDraw
from .face_mesh_module import FaceMeshDetector


class CVSMTrainer(BaseTrainer):
    
    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config 
        self.min_valid_loss = None
        self.fps = config.TEST.DATA.FS
        self.frame_depth = 10
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        if config.TOOLBOX_MODE == "train_and_test":
            # throw error bc cvsm does not require training
            raise ValueError("CVSM does not support training, only testing is allowed!")
        elif config.TOOLBOX_MODE == "only_test":
            self.cvsm = FaceProcessing(config, self.fps)
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")
        
    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        print("Predicting on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0], test_batch[1]
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                rgb = data_test[:, :3, :, :]
                depth = data_test[:, 3:4, :, :]
                print(f"Batch size: {batch_size}, RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
                pred_ppg_test = self.cvsm.predict(rgb, depth)  # Predict PPG signal

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        print(predictions)
        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)
    
class FaceProcessing:
    
    def __init__(self, config, fps):
        self.fps = fps
        self.config = config
        self.face_roi_definitions = {
            'nose': np.array([196, 419, 455, 235]),
            'forehead': np.array([109, 338, 9]),
            'cheek_n_nose': np.array([117, 346, 411, 187]),  # !!CHANGED ROIs!!
            'left_cheek': np.array([131, 165, 214, 50]), 
            'right_cheek': np.array([372, 433, 358]),
            'low_forehead': np.array([108, 337, 8]),
            'left_eye': np.array([33, 160, 159, 158, 133, 153, 145, 144]),
            'right_eye': np.array([263, 387, 386, 385, 362, 380, 374, 373]),
            'whole_face': np.array([109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])
        }
        self.face_mesh_detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    
    def Depth_compensation(self, I_raw, Depth, timeWindow, Fs):
        I_comp = np.ones_like(I_raw)
        best = 1
        best_rem = 1

        for ROI in range(1):
            I_comp_ROI = np.ones(len(I_raw))
            i = 1
            while (i * (timeWindow * Fs)) <= len(I_raw):
                cor = 2
                for bi in np.arange(0.2, 5.01, 0.01):
                    bI_comp = I_raw[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))] / (
                        (Depth[ ((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))]) ** (-bi))
                    corr_v = np.corrcoef(bI_comp, Depth[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))])
                    corr_ = abs(corr_v[1, 0])
                    if corr_ < cor:
                        cor = corr_
                        best = bI_comp
                I_comp_ROI[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))] = (best - np.mean(best)) / (np.std(best)+1e-7)
                i += 1
            cor = 2
            for bii in np.arange(0.2, 5.1, 0.1):
                bI_comp = I_raw[(((i - 1) * (timeWindow * Fs))):] / (Depth[ (((i - 1) * (timeWindow * Fs))):]) ** (-bii)
                corr_v = np.corrcoef(bI_comp, Depth[(((i - 1) * (timeWindow * Fs))):])
                corr_ = abs(corr_v[1, 0])
                if corr_ < cor:
                    cor = corr_
                    best_rem = bI_comp
            I_comp_ROI[((i - 1) * (timeWindow * Fs)): ((i * (timeWindow * Fs)))] = (best_rem - np.mean(best_rem)) / (np.std(best_rem)+1e-7)
            I_comp = I_comp_ROI
        return I_comp
        
        
    def get_pixels_in_ROI(self, b_pixels,h,w):
        mask_canvas = Image.new('L', (w, h), 0)
        pixels_passed_in = list(map(tuple, b_pixels.tolist()))
        # ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1, width=1)
        ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1)
        pixels_in_ROI = np.array(mask_canvas)
        return pixels_in_ROI
    
    
    def get_bounding_box(self, roi_name, landmarks_pixels):
        landmark_indices = self.face_roi_definitions[roi_name]
        bounding_box_pixels = landmarks_pixels[landmark_indices]
        return bounding_box_pixels


    def predict(self, rgb, depth):
        """Predicts PPG signal from RGB and depth images.
        Args:
            rgb (torch.Tensor): RGB images of shape (N, C, H, W).
            depth (torch.Tensor): Depth images of shape (N, 1, H, W).
        
        """

        ppg_signal_g = []
        depth_signal = []
        
        # convert rgb and depth shape from (N, C, H, W) to (N, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1).numpy()
        depth = depth.permute(0, 2, 3, 1).numpy()
        print(f"Input RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        num_frames = rgb.shape[0]

        # for i in tqdm(range(num_frames), desc="Processing frames for PPG signal"):
        for i in range(num_frames):
            # Get the current frame
            rgb_f_float = rgb[i]
            depth_f_float = depth[i].squeeze()

            # The FIX: Convert float32 image to uint8
            # Assuming the float values are in the range [0.0, 1.0].
            # If not, you might need to clamp them first.
            rgb_f = (rgb_f_float * 255).astype(np.uint8)
            # Depth images might have different scaling, so we won't cast them for cv2.
            # Your depth_compensation function uses them as floats, which is fine.

            face_detected, landmarks_pixels = self.face_mesh_detector.find_face_mesh(image=rgb_f, draw=False)

            if face_detected:
                # Use the green channel of the BGR image `rgb_f`
                g = rgb_f[:, :, 1]
                
                bounding_box_pixels = self.get_bounding_box('cheek_n_nose', landmarks_pixels)
                h, w = g.shape
                pixels_in_ROI = self.get_pixels_in_ROI(bounding_box_pixels, h, w)
                
                mean_intensity = np.mean(g[pixels_in_ROI > 0])
                ppg_signal_g.append(mean_intensity)
                
                mean_depth = np.mean(depth_f_float[pixels_in_ROI > 0])
                depth_signal.append(mean_depth)
            else:
                # save frame for debugging
                cv2.imwrite(f'no_frame_{i}.jpg', rgb_f)
                raise ValueError("Face not detected in the frame. Ensure the input RGB image contains a visible face.")
        
        # Convert lists to numpy arrays
        ppg_signal_g = np.array(ppg_signal_g)
        depth_signal = np.array(depth_signal)

        # Apply Depth Compensation
        time_window_sec = 5
        compensated_ppg_signal = self.Depth_compensation(ppg_signal_g, depth_signal, time_window_sec, self.fps)
        
        # Final step: convert the result back to a PyTorch tensor
        # The output of the classical pipeline is a time-series signal,
        # so it's a 1D array.
        # The trainer seems to expect a 2D tensor for the predictions,
        # with a shape like (batch_size, sequence_length).
        # We can simulate this with unsqueeze(0).
        return torch.tensor(compensated_ppg_signal, dtype=torch.float32).to(self.config.DEVICE).unsqueeze(0)

        