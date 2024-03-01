import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from prep_data import choose_cuda

import sys
from pesq import pesq
import librosa
sys.path.append('/home/dsi/hazbanb/project/git/models')
import model_v6
import model_v4
import model_v7
from torch import nn
from reconstruction import Reconstruct
import os
import pickle
import wandb

EPS = sys.float_info.epsilon
# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#
#     # track hyperparameters and run metadata
#     config={
#     }
# )

class LoadRecon:
    def __init__(self, cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num, pesq_flag):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.recon_dataloader = recon_dataloader
        self.arch_name = arch_name
        self.gen_model(unet_depth, Ns, activation, cuda_num)
        self.trans_num = trans_num
        self.pesq_flag = pesq_flag
        self.all_snr = []
    def get_snr(self):
        print(self.all_snr)
        print(len(self.all_snr))

    def gen_model(self, unet_depth, Ns, activation, cuda_num):
        if "2_level_unet_nc" in self.arch_name:
            self.model = model_v6.Model(unet_depth, Ns, activation)
        elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or (
                "2_level_unet_cc" in self.arch_name):
            self.model = model_v7.Model(unet_depth, Ns, activation)
        else:
            self.model = model_v4.Model(unet_depth, Ns, activation)
        checkpoint = torch.load(os.path.join(self.run_dir, self.tar_name))
        self.model.load_state_dict(checkpoint['model'])
        # connect to GPU
        self.device = choose_cuda(cuda_num)
        # move model to device
        self.model.to(self.device)

    def back_to_wav(self):
        test_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split'
        self.reconstruct = Reconstruct(self.device)
        self.model.eval()
        self.segment_snr_list = []
        self.total_snr_vec = torch.zeros(self.trans_num+1)
        self.segment_pesq_list = []
        self.total_frq_snr = torch.empty(1,1025).to(self.device)
        self.hist = torch.zeros(1,1025).to(self.device)
        self.arr= []
        with torch.no_grad():
            for example in self.recon_dataloader:
                print (example)
                self.segment_snr_list=[]
                dir_num = example.split('_')[0]
                clean_path = os.path.join(test_dir, dir_num, example)
                noise_name = example.replace('stft', 'noise_stft').replace('clean', 'SNR9_db')
                noise_path = os.path.join(test_dir, dir_num, noise_name)
                with open(clean_path, 'rb') as handle:
                    clean_stft = pickle.load(handle).unsqueeze(0)
                with open(noise_path, 'rb') as handle:
                    noise_stft = pickle.load(handle).unsqueeze(0)
                clean_stft = clean_stft.to(self.device)
                noise_stft = noise_stft.to(self.device)
                noisy_stft = clean_stft + noise_stft
                noisy_signal = torch.view_as_real(noisy_stft)
                noisy_signal = torch.permute(noisy_signal, (0, 3, 1, 2))
                noisy_signal = noisy_signal.to(self.device)

                # save noise, clean and noisy wav files
                clean_name = example.split('.pickle')[0] + '.wav'
                clean_wav = self.stft_to_wav(clean_stft)

                noisy_name = example.split('clean')[0] + 'noisy.wav'
                self.stft_to_wav(noisy_stft)

                noise_name = example.split('clean')[0] + 'noise.wav'
                noise_wav = self.stft_to_wav(noise_stft)

                # send noisy to model
                if "2_level_unet_nc" in self.arch_name:
                    y1, y2, current_snr, current_pesq = self.recon_model_v6(noisy_signal, example, clean_wav, noise_wav)
                    self.segment_snr_list.append(current_snr)
                    self.segment_pesq_list.append(current_pesq)

                elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or (
                        "2_level_unet_cc" in self.arch_name):
                    y1, y2, current_snr, current_pesq = self.recon_model_v7(noisy_signal, example, clean_wav, noise_wav)
                    self.segment_snr_list.append(current_snr)
                    self.segment_pesq_list.append(current_pesq)
                else:
                    filtered_signal = self.model(noisy_signal)

                # for histograma
                y1_complex = torch.permute(y1, (0, 2, 3, 1))
                y1_complex = torch.view_as_complex(y1_complex)
                current_frq_snr = self.frq_snr(clean_stft, (clean_stft-y1_complex))
                self.total_frq_snr = torch.add(current_frq_snr, self.total_frq_snr)
                self.hist[0,torch.argmin(current_frq_snr)] += 1
                self.arr.append(torch.argmin(current_frq_snr))

                idx = 0
                while self.trans_num > 1:
                    idx += 1
                    if "2_level_unet_nc" in self.arch_name:
                        if self.pesq_flag:
                            if self.best_pesq_flag == 'y1':
                                y1, y2, current_snr, current_pesq = self.recon_model_v6(y1, f'rep{idx}_{example}',clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                            else:
                                y1, y2, current_snr, current_pesq = self.recon_model_v6(y2, f'rep{idx}_{example}',clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                        else:
                            if self.less_noisy == 'y1':
                                y1, y2, current_snr, current_pesq = self.recon_model_v6(y1, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                            else:
                                y1, y2, current_snr, current_pesq = self.recon_model_v6(y2, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)

                    elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or (
                            "2_level_unet_cc" in self.arch_name):
                        if self.pesq_flag:
                            if self.best_pesq_flag == 'y1':
                                y1, y2, current_snr = self.recon_model_v7(y1, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                            else:
                                y1, y2, current_snr = self.recon_model_v7(y2, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                        else:
                            if self.less_noisy == 'y1':
                                y1, y2, current_snr = self.recon_model_v7(y1, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                            else:
                                y1, y2, current_snr = self.recon_model_v7(y2, f'rep{idx}_{example}', clean_wav, noise_wav)
                                self.segment_snr_list.append(current_snr)
                                self.segment_pesq_list.append(current_pesq)
                    self.trans_num -= 1

            segment_snr_tensor = torch.tensor(self.segment_snr_list)
            self.total_snr_vec = torch.add(self.total_snr_vec,segment_snr_tensor)

        bars = self.hist.cpu()
        bars_np = bars.numpy()
        bars_np = bars_np.squeeze()
        x = []
        for i in range(0, 1025):
            x.append(i)
        plt.bar(x, bars_np)
        plt.show()
        hist_array = []

    def recon_model_v6(self, noisy_signal, example, clean_wav, noise_wav):
        y1_stft, y2_stft, est_noise_stft = self.model(noisy_signal)

        recon_y1 = self.stft_from_model_to_wav(y1_stft)
        pesq_y1 = self.pesq_calc(clean_wav, recon_y1)

        recon_y2 = self.stft_from_model_to_wav(y2_stft)
        pesq_y2 = self.pesq_calc(clean_wav, recon_y2)

        best_pesq = max(pesq_y1, pesq_y2)
        # recon_name_est_noise = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise.wav"
        # self.stft_from_model_to_wav(recon_name_est_noise, est_noise_stft)

        # calc snr
        self.less_noisy, current_snr = self.snr(clean_wav, noise_wav, recon_y1, recon_y2)

        return y1_stft, y2_stft, current_snr, best_pesq

    def recon_model_v7(self, noisy_signal, example, clean_wav, noise_wav):
        y1_stft, y2_stft, est_noise1_stft, est_noise2_stft = self.model(noisy_signal)

        recon_y1 = self.stft_from_model_to_wav(y1_stft)
        pesq_y1 = self.pesq_calc(clean_wav, recon_y1)

        recon_y2 = self.stft_from_model_to_wav(y2_stft)
        pesq_y2 = self.pesq_calc(clean_wav, recon_y2)
        best_pesq = max(pesq_y1, pesq_y2)
        if pesq_y1 > pesq_y2:
            self.best_pesq_flag = 'y1'
        else:
            self.best_pesq_flag = 'y2'



        self.stft_from_model_to_wav(est_noise1_stft)
        self.stft_from_model_to_wav(est_noise2_stft)

        # calc snr
        self.less_noisy, current_snr = self.snr(clean_wav, noise_wav, recon_y1, recon_y2)

        return y1_stft, y2_stft, current_snr, best_pesq

    def stft_to_wav(self, stft):
        wav = self.reconstruct.istft_recon(stft)
        return wav

    def stft_from_model_to_wav(self, stft):
        stft = torch.permute(stft, (0, 2, 3, 1))
        stft = stft.contiguous()
        stft = torch.view_as_complex(stft)
        wav = self.reconstruct.istft_recon(stft)
        return wav

    def snr(self, clean, noise, recon_y1, recon_y2):
        org_snr = self.snr_calc(clean, noise)
        y1_snr = self.snr_calc(clean, (recon_y1 - clean))
        y2_snr = self.snr_calc(clean, (recon_y2 - clean))
        # print(f'{name=}:\n{org_snr=}\n{y1_snr=}\n{y2_snr=}\n***********************')
        if y1_snr > y2_snr:
            return 'y1', y1_snr
        else:
            return 'y2', y2_snr

    def snr_calc(self, clean, noise):
        Ps = torch.mean(clean ** 2)
        Pn = torch.mean(noise ** 2)
        snr = 10 * torch.log10(Ps / (Pn + EPS))
        return snr

    def frq_snr(self, clean, noise):
        Ps = torch.mean(torch.abs(clean)**2, 2)
        Pn = torch.mean(torch.abs(noise)**2, 2)
        snr = 10 * torch.log10(Ps / (Pn + EPS))
        return snr

    def pesq_calc(self, ref, deg):
        ref = ref.cpu().squeeze()
        ref = ref.numpy()
        print(ref)
        print(ref.shape)
        deg = deg.cpu().squeeze()
        deg = deg.numpy()
        ref_resampled = librosa.resample(ref, orig_sr=44100, target_sr=16000)
        deg_resampled = librosa.resample(deg, orig_sr=44100, target_sr=16000)
        pesq_result = pesq(16000, ref_resampled, deg_resampled, mode='wb')
        return pesq_result


    # def spectogram_snr(self,clean, noise, recon_y1, recon_y2, name):


#############################################################################################################

if __name__ == "__main__":
    cuda_num = 1
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    arch_name = "2_level_unet_2n2c"
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new/2023-08-17 02:17:44.150340_2_level_unet_2n2c_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    pesq_flag = 0
    recon_dataloader = []
    recon_dataloader = ['2629_stft_sec0_clean.pickle', '2114_stft_sec51_clean.pickle', '2486_stft_sec34_clean.pickle',
                        '2550_stft_sec53_clean.pickle', '2629_stft_sec102_clean.pickle']
    # for root, _, files in os.walk('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split'):
    #     for file in files:
    #         if file.endswith('clean.pickle'):
    #             recon_dataloader.append(file)

    # transfer in the model again
    trans_num = 5

    check_recon = LoadRecon(cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num, pesq_flag)
    check_recon.back_to_wav()
    # check_recon.get_snr()
