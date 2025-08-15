"""Trainer for Dual-Stream PhysFormer.

This trainer extends the original PhysFormer trainer to handle dual-stream input:
- RGB video stream
- Depth video stream

The trainer processes both streams in parallel and fuses them before feeding to the model.
"""

import os
import numpy as np
import math
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.DualStreamPhysFormer import DualStreamPhysFormer
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch

class DualStreamPhysFormerTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.DUALSTREAMPHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.DUALSTREAMPHYSFORMER.DIM
        self.ff_dim = config.MODEL.DUALSTREAMPHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.DUALSTREAMPHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.DUALSTREAMPHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.DUALSTREAMPHYSFORMER.THETA
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.frame_rate = config.TRAIN.DATA.FS
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0

        # Dual-stream specific parameters
        self.rgb_stem_channels = getattr(config.MODEL.DUALSTREAMPHYSFORMER, 'RGB_STEM_CHANNELS', None)
        self.depth_stem_channels = getattr(config.MODEL.DUALSTREAMPHYSFORMER, 'DEPTH_STEM_CHANNELS', None)

        if config.TOOLBOX_MODE == "train_and_test":
            self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = DualStreamPhysFormer(
                image_size=(self.chunk_len, config.TRAIN.DATA.PREPROCESS.RESIZE.H, config.TRAIN.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, 
                dim=self.dim, 
                ff_dim=self.ff_dim, 
                num_heads=self.num_heads, 
                num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, 
                theta=self.theta,
                rgb_stem_channels=self.rgb_stem_channels,
                depth_stem_channels=self.depth_stem_channels
            ).to(self.device)
            
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
            
        elif config.TOOLBOX_MODE == "only_test":
            self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = DualStreamPhysFormer(
                image_size=(self.chunk_len, config.TRAIN.DATA.PREPROCESS.RESIZE.H, config.TRAIN.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, 
                dim=self.dim, 
                ff_dim=self.ff_dim, 
                num_heads=self.num_heads, 
                num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, 
                theta=self.theta,
                rgb_stem_channels=self.rgb_stem_channels,
                depth_stem_channels=self.depth_stem_channels
            ).to(self.device)
            
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DualStreamPhysFormer trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for dual-stream model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # a --> Pearson loss; b --> frequency loss
        a_start = 1.0
        b_start = 1.0
        exp_a = 0.5     # Unused
        exp_b = 1.0

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            loss_rPPG_avg = []
            loss_peak_avg = []
            loss_kl_avg_test = []
            loss_hr_mae = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                hr = torch.tensor([self.get_hr(i) for i in batch[1]]).float().to(self.device)
                
                # batch[0] = rgb_data, batch[2] = depth_data, batch[1] = label
                data, label = batch[0].to(
                    self.device), batch[1].to(self.device)
                data = data[:, :580, :, :, :]
                # label NEEDS format: [batch size, # frames]
                if label.shape[0] > label.shape[1]: # [600, 1]
                    label = label[:580]
                    label = label.view(-1, 1) # now it's [1, 580]
                else:
                    label = label[:, :580] 
                    
                # Split data into rgb and depth based on channel structure
                rgb_data = data[:, :, :3, :, :]  # First 3 channels for RGB [B, 3, 600(or any other frame length), 128, 128]
                depth_data = data[:, :, 3:4, :, :]  # 4th channel for depth [B, 1, 600(or any other frame length), 128, 128]
                
                # Reshape rgb and depth data to [B, 3, 580, 128, 128] and [B, 1, 580, 128, 128]
                rgb_data = rgb_data.reshape(rgb_data.shape[0], 3, 580, 128, 128)
                depth_data = depth_data.reshape(depth_data.shape[0], 1, 580, 128, 128)

                self.optimizer.zero_grad()

                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(rgb_data, depth_data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG, axis=-1).view(-1, 1)    # normalize
                
                loss_rPPG = self.criterion_Pearson(rPPG, label)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(rgb_data.shape[0]):
                    loss_distribution_kl, \
                    fre_loss_temp, \
                    train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb],
                        hr[bb],
                        self.frame_rate,
                        std=1.0
                    )
                    fre_loss = fre_loss+fre_loss_temp
                    kl_loss = kl_loss+loss_distribution_kl
                    train_mae = train_mae+train_mae_temp
                fre_loss /= rgb_data.shape[0]
                kl_loss /= rgb_data.shape[0]
                train_mae /= rgb_data.shape[0]

                if epoch>10:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start
                    # exp ascend
                    b = b_start*math.pow(exp_b, epoch/10.0)

                loss = a*loss_rPPG + b*(fre_loss+kl_loss)
                loss.backward()
                self.optimizer.step()

                n = rgb_data.size(0)
                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'\nepoch:{epoch}, batch:{idx + 1}, total:{len(data_loader["train"]) // self.batch_size}, '
                        f'lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, '
                        f'\nb:{b:.3f}, kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, '
                        f'hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}')
                    
            # Append the current learning rate to the list
            lrs.append(self.scheduler.get_last_lr())
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(loss_rPPG_avg))
            self.save_model(epoch)
            self.scheduler.step()
            self.model.eval()

            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx+1}')
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                    
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        self.optimizer.zero_grad()
        with torch.no_grad():
            hrs = []
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                # Handle dual-stream validation data
                data, label = val_batch[0].to(self.device), val_batch[1].to(self.device)
                
                data = data[:, :580, :, :, :]
                
                # label NEEDS format: [batch size, # frames]
                if label.shape[0] > label.shape[1]: # [600, 1]
                    label = label[:580]
                    label = label.view(-1, 1) # now it's [1, 580]
                else:
                    label = label[:, :580] 
                    

                # Split data into rgb and depth based on channel structure
                rgb_data = data[:, :, :3, :, :]  # First 3 channels for RGB [B, 3, 600(or any other frame length), 128, 128]
                depth_data = data[:, :, 3:4, :, :]  # 4th channel for depth [B, 1, 600(or any other frame length), 128, 128]

                # Reshape rgb and depth data to [B, 3, 580, 128, 128] and [B, 1, 580, 128, 128]
                rgb_data = rgb_data.reshape(rgb_data.shape[0], 3, 580, 128, 128)
                depth_data = depth_data.reshape(depth_data.shape[0], 1, 580, 128, 128)
                    
                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(rgb_data, depth_data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG).view(-1, 1) # output shape = [1, 580]
                
                for _1, _2 in zip(rPPG, label):
                    hrs.append((self.get_hr(_1.cpu().detach().numpy()), self.get_hr(_2.cpu().detach().numpy())))
            RMSE = np.mean([(i-j)**2 for i, j in hrs])**0.5
            print(f'HRs (pred, GT): {hrs}')
        return RMSE

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
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
                
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                data = data[:, :580, :, :, :]
                # label NEEDS format: [batch size, # frames]
                if label.shape[0] > label.shape[1]: # [600, 1]
                    label = label[:580]
                    label = label.view(-1, 1) # now it's [1, 580]
                else:
                    label = label[:, :580] 
                # Split data into rgb and depth based on channel structure
                rgb_data = data[:, :, :3, :, :]  # First 3 channels for RGB [B, 3, 600(or any other frame length), 128, 128]
                depth_data = data[:, :, 3:4, :, :]  # 4th channel for depth [B, 1, 600(or any other frame length), 128, 128]
                rgb_data = rgb_data.reshape(rgb_data.shape[0], 3, 580, 128, 128)
                depth_data = depth_data.reshape(depth_data.shape[0], 1, 580, 128, 128)
                    
                gra_sharp = 2.0
                pred_ppg_test, _, _, _ = self.model(rgb_data, depth_data, gra_sharp)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

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

    # HR calculation based on ground truth label
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
