"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.OMNI_CAN import OMNICAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


class OmnicanTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = OMNICAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            # if model_path is specified, load the model
            # if config.INFERENCE.MODEL_PATH != "":
            #     self.model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
            #     print(f"Continue training using pretrained model!: {}")

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            # using Pearson loss
            # self.criterion = Neg_Pearson()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = OMNICAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        # Raw Channels (0-4):
        # Channel 0: Red (Raw)
        # Channel 1: Green (Raw)
        # Channel 2: Blue (Raw)
        # Channel 3: NIR/Intensity (Raw)
        # Channel 4: Depth (Raw)
        # Standardized Channels (5-9):
        # Channel 5: Red (standardized)
        # Channel 6: Green (standardized)
        # Channel 7: Blue (standardized)
        # Channel 8: NIR/Intensity (standardized)
        # Channel 9: Depth (standardized)
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len] # HR ground truth every fps
                # print(f"labels shape: {labels.shape}")
                # print(f"data shape: {data.shape}")
                # write every frame of data as a image in debug folder
                # if not os.path.exists('tof_debug'):
                #     os.makedirs('tof_debug')
                # if not os.path.exists('tof_debug/rgb_0_3'):
                #     os.makedirs('tof_debug/rgb_0_3')
                # if not os.path.exists('tof_debug/rgb_5_8'):
                #     os.makedirs('tof_debug/rgb_5_8')
                # if not os.path.exists('tof_debug/rgb_10_13'):
                #     os.makedirs('tof_debug/rgb_10_13')
                # for i in range(data.shape[0]):
                #     cur = data[i]
                #     rgb_raw = cur[0:3, :, :]
                #     rgb_raw = rgb_raw.detach().cpu().numpy()
                #     rgb_raw = np.transpose(rgb_raw, (1, 2, 0))
                #     rgb_raw = (rgb_raw - rgb_raw.min()) / (rgb_raw.max() - rgb_raw.min())
                #     rgb_raw = (rgb_raw * 255).astype(np.uint8)
                #     rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(f'tof_debug/rgb_0_3/rgb_for_epoch_{epoch}_idx_{idx}_frame_{i}.png', rgb_raw)
                #     rgb_std = cur[5:8, :, :]
                #     rgb_std = rgb_std.detach().cpu().numpy()
                #     rgb_std = np.transpose(rgb_std, (1, 2, 0))
                #     rgb_std = (rgb_std - rgb_std.min()) / (rgb_std.max() - rgb_std.min())
                #     rgb_std = (rgb_std * 255).astype(np.uint8)
                #     rgb_std = cv2.cvtColor(rgb_std, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(f'tof_debug/rgb_5_8/rgb_for_epoch_{epoch}_idx_{idx}_frame_{i}.png', rgb_std)

                #     rgb_norm = cur[10:13, :, :]
                #     rgb_norm = rgb_norm.detach().cpu().numpy()
                #     rgb_norm = np.transpose(rgb_norm, (1, 2, 0))
                #     rgb_norm = (rgb_norm - rgb_norm.min()) / (rgb_norm.max() - rgb_norm.min())
                #     rgb_norm = (rgb_norm * 255).astype(np.uint8)
                #     rgb_norm = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(f'tof_debug/rgb_10_13/rgb_for_epoch_{epoch}_idx_{idx}_frame_{i}.png', rgb_norm)
                # exit()
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                # save the RAW RGB as a image 
                # rgb_raw = data[:, 0:3, :, :]
                # rgb_raw = rgb_raw.detach().cpu().numpy()
                # rgb_raw = rgb_raw[0]
                # rgb_raw = np.transpose(rgb_raw, (1, 2, 0))
                # rgb_raw = (rgb_raw - rgb_raw.min()) / (rgb_raw.max() - rgb_raw.min())
                # rgb_raw = (rgb_raw * 255).astype(np.uint8)
                # rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'rgb_raw_for_epoch_{epoch}_idx_{idx}.png', rgb_raw)
                # exit()
                
                # display the predicted ppg and label ppg
                # plt.plot(pred_ppg.detach().cpu().numpy(), label='pred_ppg')
                # plt.plot(labels.detach().cpu().numpy(), label='label_ppg')
                # plt.legend()
                # # plt.show()
                # plt.savefig(f'pred_ppg_and_label_ppg_{idx}.png')
                # plt.close()
                # exit()
                # Use combined MSE + FFT loss for better rPPG training
                # loss, mse_loss, fft_loss_val = self.combined_loss(pred_ppg, labels, mse_weight=0.3, fft_weight=0.7)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] Combined Loss: {running_loss / 100:.3f}')
                    # print(f'  MSE Loss: {mse_loss.item():.3f}, FFT Loss: {fft_loss_val.item():.3f}')
                    plt.plot(pred_ppg.detach().cpu().numpy(), label='pred_ppg')
                    plt.plot(labels.detach().cpu().numpy(), label='label_ppg')
                    plt.legend()
                    plt.savefig(f'pred_ppg_and_label_ppg_for_epoch_{epoch}_idx_{idx}.png')
                    plt.close()
                    
                    # # display the rgb_raw as a image and save it
                    # print(f"rgb_raw shape: {data.shape}")
                    # print(f"Available channels: {data.shape[1]} (0-{data.shape[1]-1})")
                    
                    # DiffNormalized RGB: channels 0-3
                    # rgb_norm_data = data[:, 0:3, :, :]  # Shape: [200, 3, 72, 72]
                    # rgb_norm_data = rgb_norm_data.detach().cpu().numpy()
                    # for i in range(rgb_norm_data.shape[0]):
                    #     # Select the first frame and reshape to [H, W, C] for OpenCV
                    #     rgb_norm = rgb_norm_data[i]  # Take first frame: [3, 72, 72]
                    #     rgb_norm = np.transpose(rgb_norm, (1, 2, 0))  # Convert to [72, 72, 3]
                    #     # save the rgb_norm as a image in debug folder
                    #     # make the debug folder if not exists
                    #     if not os.path.exists('debug'):
                    #         os.makedirs('debug')
                    #     cv2.imwrite(f'./debug/raw_rgb_norm_for_epoch_{epoch}_idx_{idx}_frame_{i}.png', rgb_norm)
                        
                        # # Normalize to 0-255 range for visualization
                        # rgb_norm = (rgb_norm - rgb_norm.min()) / (rgb_norm.max() - rgb_norm.min())
                        # rgb_norm = (rgb_norm * 255).astype(np.uint8)
                        
                        # # print max and min of rgb_norm
                        # # print(f"rgb_norm max: {rgb_norm.max()}, rgb_norm min: {rgb_norm.min()}")
                        
                        # rgb_norm = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(f'./debug/rgb_norm_for_epoch_{epoch}_idx_{idx}_frame_{i}.png', rgb_norm)
                    
                    # # display the standardized rgb: channels 5-8
                    # rgb_std = data[:, 5:8, :, :]  # Shape: [200, 3, 72, 72]
                    # rgb_std = rgb_std.detach().cpu().numpy()
                    # rgb_std = rgb_std[0]  # Take first frame: [3, 72, 72]
                    # rgb_std = np.transpose(rgb_std, (1, 2, 0))  # Convert to [72, 72, 3]
                    # rgb_std = (rgb_std - rgb_std.min()) / (rgb_std.max() - rgb_std.min())
                    # rgb_std = (rgb_std * 255).astype(np.uint8)
                    # rgb_std = cv2.cvtColor(rgb_std, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(f'rgb_std_for_epoch_{epoch}_idx_{idx}.png', rgb_std)
                    
                    # exit()
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                # Use combined loss for validation too
                # loss, _, _ = self.combined_loss(pred_ppg_valid, labels_valid, mse_weight=0.3, fft_weight=0.7)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)

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

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
        
        
    def fft_loss(self, pred_ppg, labels, fs=20, hr_range=(0.5, 3.0)):
        """
        FFT-based loss function for rPPG training.
        Computes the difference between frequency domain representations using PyTorch.
        Optionally focuses on heart rate frequency range.
        
        Args:
            pred_ppg: Predicted PPG signal (batch_size, seq_len)
            labels: Ground truth PPG signal (batch_size, seq_len)
            fs: Sampling frequency (default: 20 Hz for your data)
            hr_range: Heart rate range in Hz (min_hr, max_hr)
        
        Returns:
            FFT-based loss value (PyTorch tensor with gradients)
        """
        # Ensure inputs are 2D (batch_size, seq_len)
        if pred_ppg.dim() == 1:
            pred_ppg = pred_ppg.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
            
        # Compute real FFT (only positive frequencies)
        pred_fft = torch.fft.rfft(pred_ppg, dim=-1)
        labels_fft = torch.fft.rfft(labels, dim=-1)
        
        # Compute magnitude spectra
        pred_magnitude = torch.abs(pred_fft)
        labels_magnitude = torch.abs(labels_fft)
        
        # Optionally create frequency mask for heart rate range
        if hr_range is not None:
            seq_len = pred_ppg.shape[-1]
            freqs = torch.fft.rfftfreq(seq_len, 1/fs, device=pred_ppg.device)
            
            # Create mask for heart rate range (0.5-3.0 Hz = 30-180 BPM)
            hr_mask = (freqs >= hr_range[0]) & (freqs <= hr_range[1])
            
            # Apply mask to focus on heart rate frequencies
            pred_magnitude = pred_magnitude * hr_mask
            labels_magnitude = labels_magnitude * hr_mask
        
        # Compute loss as mean absolute difference of magnitude spectra
        loss = torch.mean(torch.abs(pred_magnitude - labels_magnitude))
        
        return loss
    
    def combined_loss(self, pred_ppg, labels, mse_weight=0.5, fft_weight=0.5):
        """
        Combined MSE and FFT loss for rPPG training.
        
        Args:
            pred_ppg: Predicted PPG signal
            labels: Ground truth PPG signal
            mse_weight: Weight for MSE loss
            fft_weight: Weight for FFT loss
        
        Returns:
            Combined loss value
        """
        mse_loss = torch.nn.functional.mse_loss(pred_ppg, labels)
        fft_loss_val = self.fft_loss(pred_ppg, labels)
        
        combined_loss = mse_weight * mse_loss + fft_weight * fft_loss_val
        
        return combined_loss, mse_loss, fft_loss_val