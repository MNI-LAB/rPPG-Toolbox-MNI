"""Trainer for RCNNiBVP."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.RCNNiBVP import RCNNiBVP
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm


class RCNNiBVPTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        frames = self.config.MODEL.RCNNiBVP.FRAME_NUM
        in_channels = self.config.MODEL.RCNNiBVP.CHANNELS  # green/NIR channel count
        depth_channels = self.config.MODEL.RCNNiBVP.DEPTH_CHANNELS  # depth channel count

        self.model = RCNNiBVP(frames=frames, in_channels=in_channels, depth_channels=depth_channels).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("RCNNiBVP trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
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
                
                # Extract green/NIR and depth channels
                # Assuming data format: [batch, frames, channels, height, width]
                # where channels = [RGB channels, depth channel] (e.g., [R, G, B, D] or [G, D])
                N, D, C, H, W = data.shape
                
                # Extract green channel (assuming it's channel 1 for RGB or channel 0 for green-only)
                if C >= 4:  # RGBD format
                    green_nir = data[:, :, 1:2, :, :]  # Green channel
                    depth = data[:, :, 3:4, :, :]      # Depth channel
                elif C == 2:  # Green + Depth format
                    green_nir = data[:, :, 0:1, :, :]  # Green/NIR channel
                    depth = data[:, :, 1:2, :, :]      # Depth channel
                else:
                    raise ValueError(f"Unsupported channel format: {C} channels. Expected 2 (Green+Depth) or 4+ (RGBD)")
                
                # Reshape from [N, D, 1, H, W] to [N, 1, D, H, W] for model input
                green_nir = green_nir.permute(0, 2, 1, 3, 4)
                depth = depth.permute(0, 2, 1, 3, 4)
                
                self.optimizer.zero_grad()
                pred_ppg = self.model(green_nir, depth)  # Shape: [batch, frames-1]
                
                # Labels should match the prediction shape
                # pred_ppg is [batch, model_frames-1], labels is [batch, data_frames]
                # We need to match the prediction length to the label length
                pred_frames = pred_ppg.shape[1]
                labels_matched = labels[:, :pred_frames].contiguous()  # Take first pred_frames from labels
                
                # Flatten both to match loss function expectations
                pred_ppg_flat = pred_ppg.view(-1, 1)
                labels_flat = labels_matched.view(-1, 1)
                
                loss = self.loss_model(pred_ppg_flat, labels_flat)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
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
                
                # Extract green/NIR and depth channels
                N, D, C, H, W = data_valid.shape
                
                if C >= 4:  # RGBD format
                    green_nir = data_valid[:, :, 1:2, :, :]  # Green channel
                    depth = data_valid[:, :, 3:4, :, :]      # Depth channel
                elif C == 2:  # Green + Depth format
                    green_nir = data_valid[:, :, 0:1, :, :]  # Green/NIR channel
                    depth = data_valid[:, :, 1:2, :, :]      # Depth channel
                else:
                    raise ValueError(f"Unsupported channel format: {C} channels. Expected 2 (Green+Depth) or 4+ (RGBD)")
                
                # Reshape from [N, D, 1, H, W] to [N, 1, D, H, W] for model input
                green_nir = green_nir.permute(0, 2, 1, 3, 4)
                depth = depth.permute(0, 2, 1, 3, 4)
                
                pred_ppg_valid = self.model(green_nir, depth)  # Shape: [batch, frames-1]
                
                # Labels should match the prediction shape
                pred_frames = pred_ppg_valid.shape[1] 
                labels_matched = labels_valid[:, :pred_frames].contiguous()  # Take first pred_frames from labels
                
                # Flatten both to match loss function expectations
                pred_ppg_flat = pred_ppg_valid.view(-1, 1)
                labels_flat = labels_matched.view(-1, 1)
                
                loss = self.loss_model(pred_ppg_flat, labels_flat)
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
                
                # Extract green/NIR and depth channels
                N, D, C, H, W = data_test.shape
                
                if C >= 4:  # RGBD format
                    green_nir = data_test[:, :, 1:2, :, :]  # Green channel
                    depth = data_test[:, :, 3:4, :, :]      # Depth channel
                elif C == 2:  # Green + Depth format
                    green_nir = data_test[:, :, 0:1, :, :]  # Green/NIR channel
                    depth = data_test[:, :, 1:2, :, :]      # Depth channel
                else:
                    raise ValueError(f"Unsupported channel format: {C} channels. Expected 2 (Green+Depth) or 4+ (RGBD)")
                
                # Reshape from [N, D, 1, H, W] to [N, 1, D, H, W] for model input
                green_nir = green_nir.permute(0, 2, 1, 3, 4)
                depth = depth.permute(0, 2, 1, 3, 4)
                
                # Labels should match the output shape [batch_size, frames-1]
                labels_test = labels_test[:, :-1].contiguous().view(-1, 1)
                
                pred_ppg_test = self.model(green_nir, depth)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    # For iBVP-style output (continuous signal)
                    pred_frames = pred_ppg_test.shape[1]  # Should be frames-1 due to diff operation
                    predictions[subj_index][sort_index] = pred_ppg_test[idx, :]
                    labels[subj_index][sort_index] = labels_test[idx * pred_frames:(idx + 1) * pred_frames]

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
