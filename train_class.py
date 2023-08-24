import os.path
import csv
import matplotlib.pyplot as plt
import torch
#torch.autograd.set_detect_anomaly(True)
import datetime
from torch import nn
from prep_data import choose_cuda
import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')
import model_v4
import model_v6
import model_v7
from dataset import create_dataset
import wandb

class TrainClass:
    def __init__(self, num_epochs, lr, loss_weights, batch_size, unet_depth, Ns, arch_name, num_workers, snrs, output_path, cuda_num, short_run, check_points, activation):
        self.start_time = datetime.datetime.now()
        self.num_epochs = num_epochs
        self.lr = lr
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unet_depth = unet_depth
        self.arch_name = arch_name
        self.Ns = Ns
        self.snrs = snrs
        self.data_loader()
        self.output_path = output_path
        self.loss_fn = torch.nn.MSELoss()
        self.device = choose_cuda(cuda_num)
        if '2_level_unet_nc' in self.arch_name:
            self.model = model_v6.Model(self.unet_depth, self.Ns, self.activation).to(self.device)
        elif 'one_level_unet' in self.arch_name:
            self.model = model_v4.Model(self.unet_depth, self.Ns, self.activation).to(self.device)
        elif ('2_level_unet_nn' in self.arch_name) or ('2_level_unet_2n2c' in self.arch_name) or ('2_level_unet_cc' in self.arch_name):
            self.model = model_v7.Model(self.unet_depth, self.Ns, self.activation).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-05)
        self.short_run = short_run
        self.check_points = check_points
        self.create_output_dir()
        self.loss_weights = loss_weights

    def init_wnb(self):
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project="final_project",

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.lr,
                "architecture": self.arch_name,
                "batch_size": self.batch_size,
                "epochs": self.num_epochs,
                "unet_depth": self.unet_depth,
                "max_channels": self.Ns[self.unet_depth],
                "name": self.dir_name,
            }
        )

    def data_loader(self):
        # create train dataloader
        train_dataset = create_dataset('train')
        train_dataset.filter_by_snrs(self.snrs)
        print(f'{len(train_dataset)=}')
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers)

        # create val dataloader
        self.val_dataloader_dict = {}
        for snr in self.snrs:
            val_dataset = create_dataset('val')
            val_dataset.filter_by_snrs([snr])
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers)
            self.val_dataloader_dict[snr] = val_dataloader

        # create test dataloader
        self.test_dataloader_dict = {}
        for snr in self.snrs:
            test_dataset = create_dataset('test')
            test_dataset.filter_by_snrs([snr])
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                          shuffle=True, num_workers=self.num_workers)
            self.test_dataloader_dict[snr] = test_dataloader

    def run(self):
        self.epoch_train_losses = []
        self.epoch_val_losses = {}
        for epoch in range(self.num_epochs):
            epoch_train_loss = self.train_epoch(epoch)
            self.epoch_train_losses.append(epoch_train_loss)
            print(f'epoch num {str(epoch)}:\t{epoch_train_loss=}')
            for val_dataloader in self.val_dataloader_dict.keys():
                if epoch == 0:
                    self.epoch_val_losses[val_dataloader] = []
                epoch_val_loss = self.val_epoch(val_dataloader, epoch)
                self.epoch_val_losses[val_dataloader].append(epoch_val_loss)
            if (((epoch + 1) % check_points) == 0) and check_points and ((epoch + 1) != self.num_epochs):
                path = f'{self.full_path}/{epoch + 1}_epochs_checkpoint'
                self.save_model(path)

        self.test_losses = {}
        for test_dataloader in self.test_dataloader_dict.keys():
            self.test_losses[test_dataloader] = []
            test_loss = self.test(test_dataloader)
            self.test_losses[test_dataloader].append(test_loss)
        self.prep_output_files()

    def train_epoch(self, epoch):
        # Set train mode for both the encoder and the decoder
        batch_loss_list = []
        sum_loss_y1 = 0
        sum_loss_y2 = 0
        sum_loss_n1 = 0
        sum_loss_n2 = 0
        self.model.train()
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for batch_idx, all_data in enumerate(self.train_dataloader):
            clean_seg, noise, dataset_snr = all_data

            # Move tensor to the proper device
            clean_seg = clean_seg.to(self.device)
            noise = noise.to(self.device)

            # generate noised signal
            clean_seg_clone = clean_seg.clone()
            noised_signal = clean_seg_clone + noise
            noised_signal = torch.view_as_real(noised_signal)
            noised_signal = torch.permute(noised_signal, (0, 3, 1, 2))
            noised_signal = noised_signal.to(self.device)
            noised_signal_clone = noised_signal.clone()

            self.optim.zero_grad()
            clean_seg_abs = torch.abs(clean_seg)
            noise_abs = torch.abs(noise)

            if '2_level_unet_nc' in self.arch_name:
                loss_y1, loss_y2, batch_loss = self.loss_2level_unet_nc(noised_signal_clone, clean_seg_abs)
                sum_loss_y1 += loss_y1.item()
                sum_loss_y2 += loss_y2.item()
            elif '2_level_unet_cc' in self.arch_name:
                loss_y1, loss_y2, batch_loss = self.loss_2level_unet_cc(noised_signal_clone, clean_seg_abs)
                sum_loss_y1 += loss_y1.item()
                sum_loss_y2 += loss_y2.item()
            elif '2_level_unet_nn' in self.arch_name:
                loss_n1, loss_n2, batch_loss = self.loss_2level_unet_nn(noised_signal_clone, noise_abs)
                sum_loss_n1 += loss_n1.item()
                sum_loss_n2 += loss_n2.item()
            elif '2_level_unet_2n2c':
                loss_y1, loss_y2, loss_n1, loss_n2, batch_loss = self.loss_2level_unet_2n2c(noised_signal_clone, clean_seg_abs, noise_abs)
                sum_loss_y1 += loss_y1.item()
                sum_loss_y2 += loss_y2.item()
                sum_loss_n1 += loss_n1.item()
                sum_loss_n2 += loss_n2.item()
            else:
                filtered_signal = self.model(noised_signal_clone)
                filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
                filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
                filtered_signal = torch.view_as_complex(filtered_signal)
                recon_clean_abs = torch.abs(filtered_signal)
                batch_loss = self.loss_fn(recon_clean_abs, clean_seg_abs)


            batch_loss.backward()
            self.optim.step()
            batch_loss_list.append(batch_loss.item())
            if (batch_idx % 1000 == 0):
                print(f'********************* train: {epoch=}___{batch_idx=}*********************')
            # short run
            if short_run and (batch_idx == short_run):
                print(f'stopping at index {batch_idx}')
                break
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        wandb.log({"train_epoch_loss": epoch_loss, "epoch": epoch})
        if(sum_loss_y1):
            epoch_loss_y1 = sum_loss_y1 / len(batch_loss_list)
            wandb.log({"train_epoch_loss_y1": epoch_loss_y1, "epoch": epoch})
        if(sum_loss_y2):
            epoch_loss_y2 = sum_loss_y2 / len(batch_loss_list)
            wandb.log({"train_epoch_loss_y2": epoch_loss_y2, "epoch": epoch})
        if(sum_loss_n1):
            epoch_loss_n1 = sum_loss_n1 / len(batch_loss_list)
            wandb.log({"train_epoch_loss_n1": epoch_loss_n1, "epoch": epoch})
        if(sum_loss_n2):
            epoch_loss_n2 = sum_loss_n2 / len(batch_loss_list)
            wandb.log({"train_epoch_loss_n2": epoch_loss_n2, "epoch": epoch})
        return epoch_loss

    def loss_2level_unet_nc(self, noised_signal, clean_seg_abs):
        y1, y2, _ = self.model(noised_signal)
        y1 = torch.permute(y1, (0, 2, 3, 1))
        y2 = torch.permute(y2, (0, 2, 3, 1))
        y1 = y1.contiguous()
        y2 = y2.contiguous()
        y1 = torch.view_as_complex(y1)
        y2 = torch.view_as_complex(y2)
        y1_abs = torch.abs(y1)
        y2_abs = torch.abs(y2)
        loss_y1 = self.loss_fn(y1_abs, clean_seg_abs)
        loss_y2 = self.loss_fn(y2_abs, clean_seg_abs)
        batch_loss = torch.add((self.loss_weights['y1'] * loss_y1), (self.loss_weights['y2'] * loss_y2))
        return loss_y1, loss_y2, batch_loss

    def loss_2level_unet_cc(self, noised_signal, clean_seg_abs):
        y1, y2, _, _ = self.model(noised_signal)
        y1 = torch.permute(y1, (0, 2, 3, 1))
        y2 = torch.permute(y2, (0, 2, 3, 1))
        y1 = y1.contiguous()
        y2 = y2.contiguous()
        y1 = torch.view_as_complex(y1)
        y2 = torch.view_as_complex(y2)
        y1_abs = torch.abs(y1)
        y2_abs = torch.abs(y2)
        loss_y1 = self.loss_fn(y1_abs, clean_seg_abs)
        loss_y2 = self.loss_fn(y2_abs, clean_seg_abs)
        batch_loss = torch.add((self.loss_weights['y1'] * loss_y1), (self.loss_weights['y2'] * loss_y2))
        return loss_y1, loss_y2, batch_loss

    def loss_2level_unet_nn(self, noised_signal, noise_abs):
        _, _, noise1, noise2 = self.model(noised_signal)
        noise1 = torch.permute(noise1, (0, 2, 3, 1))
        noise2 = torch.permute(noise2, (0, 2, 3, 1))
        noise1 = noise1.contiguous()
        noise2 = noise2.contiguous()
        noise1 = torch.view_as_complex(noise1)
        noise2 = torch.view_as_complex(noise2)
        noise1_abs = torch.abs(noise1)
        noise2_abs = torch.abs(noise2)
        loss_noise1 = self.loss_fn(noise1_abs, noise_abs)
        loss_noise2 = self.loss_fn(noise2_abs, noise_abs)
        batch_loss = torch.add((self.loss_weights['n1'] * loss_noise1), (self.loss_weights['n2'] * loss_noise2))
        return loss_noise1, loss_noise2, batch_loss

    def loss_2level_unet_2n2c(self, noised_signal, clean_abs, noise_abs):
        y1, y2, noise1, noise2 = self.model(noised_signal)

        noise1 = torch.permute(noise1, (0, 2, 3, 1))
        noise2 = torch.permute(noise2, (0, 2, 3, 1))
        noise1 = noise1.contiguous()
        noise2 = noise2.contiguous()
        noise1 = torch.view_as_complex(noise1)
        noise2 = torch.view_as_complex(noise2)
        noise1_abs = torch.abs(noise1)
        noise2_abs = torch.abs(noise2)
        loss_noise1 = self.loss_fn(noise1_abs, noise_abs)
        loss_noise2 = self.loss_fn(noise2_abs, noise_abs)

        y1 = torch.permute(y1, (0, 2, 3, 1))
        y2 = torch.permute(y2, (0, 2, 3, 1))
        y1 = y1.contiguous()
        y2 = y2.contiguous()
        y1 = torch.view_as_complex(y1)
        y2 = torch.view_as_complex(y2)
        y1_abs = torch.abs(y1)
        y2_abs = torch.abs(y2)
        loss_y1 = self.loss_fn(y1_abs, clean_abs)
        loss_y2 = self.loss_fn(y2_abs, clean_abs)

        noise_loss = torch.add((self.loss_weights['n1'] * loss_noise1), (self.loss_weights['n2'] * loss_noise2))
        clean_loss = torch.add((self.loss_weights['y1'] * loss_y1), (self.loss_weights['y2'] * loss_y2))
        batch_loss = torch.add(noise_loss,clean_loss)

        return loss_y1, loss_y2, loss_noise1, loss_noise2, batch_loss

    def val_epoch(self, snr, epoch):
        batch_loss_list = []
        sum_loss_y1 = 0
        sum_loss_y2 = 0
        sum_loss_n1 = 0
        sum_loss_n2 = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, all_data in enumerate(self.val_dataloader_dict[snr]):
                clean_seg, noise, dataset_snr = all_data

                # Move tensor to the proper device
                clean_seg = clean_seg.to(self.device)
                noise = noise.to(self.device)

                # generate noised signal
                clean_seg_clone = clean_seg.clone()
                noised_signal = clean_seg_clone + noise
                noised_signal = torch.view_as_real(noised_signal)
                noised_signal = torch.permute(noised_signal, (0, 3, 1, 2))
                noised_signal = noised_signal.to(self.device)
                noised_signal_clone = noised_signal.clone()

                clean_seg_abs = torch.abs(clean_seg)
                noise_abs = torch.abs(noise)

                if '2_level_unet_nc' in self.arch_name:
                    loss_y1, loss_y2, batch_loss = self.loss_2level_unet_nc(noised_signal_clone, clean_seg_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                elif '2_level_unet_cc' in self.arch_name:
                    loss_y1, loss_y2, batch_loss = self.loss_2level_unet_cc(noised_signal_clone, clean_seg_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                elif '2_level_unet_nn' in self.arch_name:
                    loss_n1, loss_n2, batch_loss = self.loss_2level_unet_nn(noised_signal_clone, noise_abs)
                    sum_loss_n1 += loss_n1.item()
                    sum_loss_n2 += loss_n2.item()
                elif '2_level_unet_2n2c':
                    loss_y1, loss_y2, loss_n1, loss_n2, batch_loss = self.loss_2level_unet_2n2c(noised_signal_clone,
                                                                                                clean_seg_abs,
                                                                                                noise_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                    sum_loss_n1 += loss_n1.item()
                    sum_loss_n2 += loss_n2.item()
                else:
                    filtered_signal = self.model(noised_signal_clone)
                    filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
                    filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
                    filtered_signal = torch.view_as_complex(filtered_signal)
                    recon_clean_abs = torch.abs(filtered_signal)
                    batch_loss = self.loss_fn(recon_clean_abs, clean_seg_abs)

                batch_loss_list.append(batch_loss.item())
                if (batch_idx % 1000 == 0):
                    print(f'********************* val_{snr=}: {epoch=} *********************')
                # short run
                if short_run and (batch_idx == short_run):
                    print(f'stopping at index {batch_idx}')
                    break
            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            wandb.log({f"val_{snr=}_epoch_loss": epoch_loss, "epoch": epoch})
            if (sum_loss_y1):
                epoch_loss_y1 = sum_loss_y1 / len(batch_loss_list)
                wandb.log({f"val_{snr=}_epoch_loss_y1": epoch_loss_y1, "epoch": epoch})
            if (sum_loss_y2):
                epoch_loss_y2 = sum_loss_y2 / len(batch_loss_list)
                wandb.log({f"val_{snr=}_epoch_loss_y2": epoch_loss_y2, "epoch": epoch})
            if (sum_loss_n1):
                epoch_loss_n1 = sum_loss_n1 / len(batch_loss_list)
                wandb.log({f"val_{snr=}_epoch_loss_n1": epoch_loss_n1, "epoch": epoch})
            if (sum_loss_n2):
                epoch_loss_n2 = sum_loss_n2 / len(batch_loss_list)
                wandb.log({f"val_{snr=}_epoch_loss_n2": epoch_loss_n2, "epoch": epoch})
            return epoch_loss

    def test(self, snr):
        batch_loss_list = []
        sum_loss_y1 = 0
        sum_loss_y2 = 0
        sum_loss_n1 = 0
        sum_loss_n2 = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, all_data in enumerate(self.test_dataloader_dict[snr]):
                clean_seg, noise, dataset_snr = all_data

                # Move tensor to the proper device
                clean_seg = clean_seg.to(self.device)
                noise = noise.to(self.device)

                # generate noised signal
                clean_seg_clone = clean_seg.clone()
                noised_signal = clean_seg_clone + noise
                noised_signal = torch.view_as_real(noised_signal)
                noised_signal = torch.permute(noised_signal, (0, 3, 1, 2))
                noised_signal = noised_signal.to(self.device)
                noised_signal_clone = noised_signal.clone()

                clean_seg_abs = torch.abs(clean_seg)
                noise_abs = torch.abs(noise)

                if '2_level_unet_nc' in self.arch_name:
                    loss_y1, loss_y2, batch_loss = self.loss_2level_unet_nc(noised_signal_clone, clean_seg_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                elif '2_level_unet_cc' in self.arch_name:
                    loss_y1, loss_y2, batch_loss = self.loss_2level_unet_cc(noised_signal_clone, clean_seg_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                elif '2_level_unet_nn' in self.arch_name:
                    loss_n1, loss_n2, batch_loss = self.loss_2level_unet_nn(noised_signal_clone, noise_abs)
                    sum_loss_n1 += loss_n1.item()
                    sum_loss_n2 += loss_n2.item()
                elif '2_level_unet_2n2c':
                    loss_y1, loss_y2, loss_n1, loss_n2, batch_loss = self.loss_2level_unet_2n2c(noised_signal_clone,
                                                                                                clean_seg_abs,
                                                                                                noise_abs)
                    sum_loss_y1 += loss_y1.item()
                    sum_loss_y2 += loss_y2.item()
                    sum_loss_n1 += loss_n1.item()
                    sum_loss_n2 += loss_n2.item()
                else:
                    filtered_signal = self.model(noised_signal_clone)
                    filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
                    filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
                    filtered_signal = torch.view_as_complex(filtered_signal)
                    recon_clean_abs = torch.abs(filtered_signal)
                    batch_loss = self.loss_fn(recon_clean_abs, clean_seg_abs)

                batch_loss_list.append(batch_loss.item())
                if (batch_idx % 1000 == 0):
                    print(f'********************* test: {snr=}___{batch_idx=}*********************')
                # short run
                if short_run and (batch_idx == short_run):
                    print(f'stopping at index {batch_idx}')
                    break
            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            wandb.log({f"test_{snr=}_loss": epoch_loss})
            if (sum_loss_y1):
                epoch_loss_y1 = sum_loss_y1 / len(batch_loss_list)
                wandb.log({f"test_{snr=}_loss_y1": epoch_loss_y1})
            if (sum_loss_y2):
                epoch_loss_y2 = sum_loss_y2 / len(batch_loss_list)
                wandb.log({f"test_{snr=}_loss_y2": epoch_loss_y2})
            if (sum_loss_n1):
                epoch_loss_n1 = sum_loss_n1 / len(batch_loss_list)
                wandb.log({f"test_{snr=}_loss_n1": epoch_loss_n1})
            if (sum_loss_n2):
                epoch_loss_n2 = sum_loss_n2 / len(batch_loss_list)
                wandb.log({f"test_{snr=}_loss_n2": epoch_loss_n2})
            return epoch_loss

    def create_output_dir(self):
        run_name = f'{self.arch_name}_model_{self.num_epochs}epochs_depth_{self.Ns[self.unet_depth] * 2}channels_batch{self.batch_size}'
        self.dir_name = f'{self.start_time}_{run_name}'
        self.full_path = os.path.join(self.output_path, self.dir_name)
        os.mkdir(self.full_path)

    def prep_output_files(self):
        csv_file = 'loss_values.csv'
        with open(os.path.join(self.full_path, csv_file), 'w', newline='') as file:
            writer = csv.writer(file)
            epoch_col = ['epoch']
            train_loss_col = ['train_loss']
            val_loss_snr_m3 = ['val_loss (snr -3 dB)']
            val_loss_snr_0 = ['val_loss (snr 0 dB)']
            val_loss_snr_3 = ['val_loss (snr 3 dB)']
            val_loss_snr_6 = ['val_loss (snr 6 dB)']
            val_loss_snr_9 = ['val_loss (snr 9 dB)']
            val_loss_snr_12 = ['val_loss (snr 12 dB)']
            val_loss_snr_15 = ['val_loss (snr 15 dB)']
            for i in range(self.num_epochs):
                epoch_col.append(str(i))
                train_loss_col.append(self.epoch_train_losses[i])
                for snr in self.epoch_val_losses.keys():
                    if snr == '-3':
                        val_loss_snr_m3.append(self.epoch_val_losses[snr][i])
                    elif snr == '0':
                        val_loss_snr_0.append(self.epoch_val_losses[snr][i])
                    elif snr == '3':
                        val_loss_snr_3.append(self.epoch_val_losses[snr][i])
                    elif snr == '6':
                        val_loss_snr_6.append(self.epoch_val_losses[snr][i])
                    elif snr == '9':
                        val_loss_snr_9.append(self.epoch_val_losses[snr][i])
                    elif snr == '12':
                        val_loss_snr_12.append(self.epoch_val_losses[snr][i])
                    else:
                        val_loss_snr_15.append(self.epoch_val_losses[snr][i])
            epoch_col.append('test')
            train_loss_col.append('')
            for snr in self.test_losses.keys():
                if snr == '-3':
                    val_loss_snr_m3.append(self.test_losses[snr][0])
                elif snr == '0':
                    val_loss_snr_0.append(self.test_losses[snr][0])
                elif snr == '3':
                    val_loss_snr_3.append(self.test_losses[snr][0])
                elif snr == '6':
                    val_loss_snr_6.append(self.test_losses[snr][0])
                elif snr == '9':
                    val_loss_snr_9.append(self.test_losses[snr][0])
                elif snr == '12':
                    val_loss_snr_12.append(self.test_losses[snr][0])
                else:
                    val_loss_snr_15.append(self.test_losses[snr][0])
            loss_png_name = 'loss_graphs.png'
            png_full_path = os.path.join(self.full_path, loss_png_name)
            data = []
            data.append(epoch_col)
            fig, ax = plt.subplots()
            ax.plot(epoch_col[1:-1], train_loss_col[1:-1], color='red', label='train loss')
            data.append(train_loss_col)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_m3[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_m3[1:-1], color='blue', label='val loss snr -3')
                data.append(val_loss_snr_m3)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_0[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_0[1:-1], color='green', label='val loss snr 0')
                data.append(val_loss_snr_0)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_3[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_3[1:-1], color='yellow', label='val loss snr 3')
                data.append(val_loss_snr_3)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_6[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_6[1:-1], color='grey', label='val loss snr 6')
                data.append(val_loss_snr_6)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_9[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_9[1:-1], color='orange', label='val loss snr 9')
                data.append(val_loss_snr_9)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_12[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_12[1:-1], color='purple', label='val loss snr 12')
                data.append(val_loss_snr_12)
            if (len(epoch_col[1:-1]) == len(val_loss_snr_15[1:-1])):
                ax.plot(epoch_col[1:-1], val_loss_snr_15[1:-1], color='brown', label='val loss snr 15')
                data.append(val_loss_snr_15)
            ax.legend()
            fig.savefig(png_full_path)
            # data = [epoch_col, train_loss_col, val_loss_snr_m3, val_loss_snr_0, val_loss_snr_3,
            #         val_loss_snr_6, val_loss_snr_9, val_loss_snr_12, val_loss_snr_15]
            data_transposed = zip(*data)
            for row in data_transposed:
                writer.writerow(row)
        filename = f"FinalModel"
        model_tar_path = f'{self.full_path}/{filename}'
        self.save_model(model_tar_path)

    def save_model(self, full_path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'num_epoch': self.num_epochs,
            'batch_train': self.batch_size},  # end of parameters-to-be-saved list
            f"{full_path}.tar")

if __name__ == "__main__":
    num_epochs = 30
    print(num_epochs)
    lr = 0.001
    #torch.manual_seed(0)
    cuda_num = 1
    batch_size = 16
    num_workers = 9
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    # Ss = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    snrs = ['-3','0','3','6','9','12','15']
    # snrs = ['6']
    print(f'{snrs=}')
    arch_name = "2_level_unet_cc"
    print(f'{arch_name=}')
    output_path = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new'
    loss_weights = {'y1': 1, 'y2': 20, 'n1': 1, 'n2': 1}

    short_run = 0     # 0 - full run, else stop after {short_run} batches
    check_points = 5  # 0 - no checkpoint, else save the model each {check_points} epochs

    train_class = TrainClass(num_epochs, lr, loss_weights, batch_size, unet_depth, Ns, arch_name, num_workers, snrs, output_path, cuda_num, short_run, check_points, activation)
    train_class.init_wnb()
    train_class.run()

