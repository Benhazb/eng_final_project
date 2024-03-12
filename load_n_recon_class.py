import torch
import torchaudio
from prep_data import choose_cuda

import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')
import model_v6
import model_v4
import model_v7
from torch import nn
from reconstruction import Reconstruct
import os
import pickle
EPS =  sys.float_info.epsilon

class LoadRecon:
    def __init__(self, cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num, snr_list):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.recon_dataloader = recon_dataloader
        self.arch_name = arch_name
        self.gen_model(unet_depth, Ns, activation, cuda_num)
        self.trans_num = trans_num
        self.snr_list = snr_list
        self.wav_files_dir = self.create_wav_dirs()

    def gen_model(self, unet_depth, Ns, activation, cuda_num):
        if "2_level_unet_nc" in self.arch_name:
            self.model = model_v6.Model(unet_depth, Ns, activation)
        elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or ("2_level_unet_cc" in self.arch_name):
            self.model = model_v7.Model(unet_depth, Ns, activation)
        else:
            self.model = model_v4.Model(unet_depth, Ns, activation)
        checkpoint = torch.load(os.path.join(self.run_dir, self.tar_name))
        self.model.load_state_dict(checkpoint['model'])
        # connect to GPU
        self.device = choose_cuda(cuda_num)
        # move model to device
        self.model.to(self.device)

    def create_wav_dirs(self):
        wav_files_dir = os.path.join(self.run_dir,'wav_files')
        wav_num = []
        if not os.path.isdir(wav_files_dir):
            os.mkdir(wav_files_dir)
        for seg in self.recon_dataloader:
            seg_num = seg.split('_')[0]
            if seg_num not in wav_num:
                wav_num.append(seg_num)
        for snr in self.snr_list:
            snr_dir = os.path.join(wav_files_dir, f'snr{snr}')
            if not os.path.isdir(snr_dir):
                os.mkdir(snr_dir)
            for wav in wav_num:
                wav_dir = os.path.join(snr_dir, wav)
                if not os.path.isdir(wav_dir):
                    os.mkdir(wav_dir)
        return wav_files_dir

    def back_to_wav(self, snr):
        test_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split'
        self.reconstruct = Reconstruct(self.device)
        self.model.eval()
        with torch.no_grad():
            for example in self.recon_dataloader:
                dir_num = example.split('_')[0]
                seg_num = example.split('sec')[-1].split('_')[0]
                clean_path = os.path.join(test_dir, dir_num, example)
                noise_name = example.replace('stft', 'noise_stft').replace('clean', f'SNR{snr}_db')
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
                self.seg_path = os.path.join(self.run_dir, self.wav_files_dir, f'snr{snr}', dir_num, f'seg{seg_num}')
                if not os.path.isdir(self.seg_path):
                    os.mkdir(self.seg_path)

                clean_wav = self.stft_to_wav(os.path.join(self.seg_path, 'clean.wav'), clean_stft)

                self.stft_to_wav(os.path.join(self.seg_path, 'noisy.wav'), noisy_stft)

                noise_wav = self.stft_to_wav(os.path.join(self.seg_path, 'noise.wav'), noise_stft)

                idx = 1
                # send noisy to model
                if "2_level_unet_nc" in self.arch_name:
                    y1, y2 = self.recon_model_v6(noisy_signal, f'rep{idx}_{example}', clean_wav, noise_wav)
                elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or ("2_level_unet_cc" in self.arch_name):
                    y1, y2 = self.recon_model_v7(noisy_signal, f'rep{idx}_{example}', clean_wav, noise_wav)
                else:
                    filtered_signal = self.model(noisy_signal)

                    filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
                    filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
                    filtered_signal = torch.view_as_complex(filtered_signal)

                    recon_wav = self.reconstruct.istft_recon(filtered_signal)
                    recon_wav = recon_wav.detach().cpu()

                    recon_name = f"{example.split('clean')[0]}_reconstruct_{tar_name}.wav"
                    recon_wav_path = os.path.join(self.run_dir, recon_name)
                    torchaudio.save(recon_wav_path, recon_wav, 44100)

                repeat = self.trans_num
                while(repeat>1):
                    idx += 1
                    if "2_level_unet_nc" in self.arch_name:
                        if self.less_noisy == 'y1':
                            y1, y2 = self.recon_model_v6(y1, f'rep{idx}_{example}', clean_wav, noise_wav)
                        else:
                            y1, y2 = self.recon_model_v6(y2, f'rep{idx}_{example}', clean_wav, noise_wav)
                    elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or ("2_level_unet_cc" in self.arch_name):
                        if self.less_noisy == 'y1':
                            y1, y2 = self.recon_model_v7(y1, f'rep{idx}_{example}', clean_wav, noise_wav)
                        else:
                            y1, y2 = self.recon_model_v7(y2, f'rep{idx}_{example}', clean_wav, noise_wav)
                    repeat -= 1


    def recon_model_v6(self, noisy_signal, example, clean_wav, noise_wav):
        y1_stft, y2_stft, est_noise_stft = self.model(noisy_signal)

        # save wav files
        recon_y1 = self.stft_from_model_to_wav(y1_stft)
        recon_y2 = self.stft_from_model_to_wav(y2_stft)

        # est_noise = self.stft_from_model_to_wav(est_noise_stft)
        # est_noise_name = f"{self.tar_name.split('.tar')[0]}_est_noise_{example.split('_')[0]}.wav"
        # self.save_wav_file(os.path.join(self.seg_path, est_noise_name), est_noise)

        # calc snr
        self.less_noisy = self.snr(clean_wav, noise_wav, recon_y1, recon_y2, example.split('_clean')[0])

        y_name = f"{self.tar_name.split('.tar')[0]}_y_recon_{example.split('_')[0]}.wav"
        if self.less_noisy == 'y1':
            self.save_wav_file(os.path.join(self.seg_path, y_name), recon_y1)
        else:
            self.save_wav_file(os.path.join(self.seg_path, y_name), recon_y2)

        return y1_stft, y2_stft

    def recon_model_v7(self, noisy_signal, example, clean_wav, noise_wav):
        y1_stft, y2_stft, est_noise1_stft, est_noise2_stft = self.model(noisy_signal)

        recon_y1 = self.stft_from_model_to_wav(y1_stft)
        recon_y2 = self.stft_from_model_to_wav(y2_stft)

        # est_noise_1 = self.stft_from_model_to_wav(est_noise1_stft)
        # est_noise_2 = self.stft_from_model_to_wav(est_noise2_stft)

        # calc snr
        self.less_noisy = self.snr(clean_wav, noise_wav, recon_y1, recon_y2, example.split('_clean')[0])
        # self.more_similar_noise = self.closer_est_noise_check(noise_wav, est_noise_1, est_noise_2)

        y_name = f"{self.tar_name.split('.tar')[0]}_y_recon_{example.split('_')[0]}.wav"
        if self.less_noisy == 'y1':
            self.save_wav_file(os.path.join(self.seg_path, y_name), recon_y1)
        else:
            self.save_wav_file(os.path.join(self.seg_path, y_name), recon_y2)

        # est_noise_name = f"{self.tar_name.split('.tar')[0]}_est_noise_{example.split('_')[0]}.wav"
        # if self.more_similar_noise == 'est1':
        #     self.save_wav_file(os.path.join(self.seg_path, est_noise_name), est_noise_1)
        # else:
        #     self.save_wav_file(os.path.join(self.seg_path, est_noise_name), est_noise_2)

        return y1_stft, y2_stft

    def stft_to_wav(self, path, stft):
        wav = self.reconstruct.istft_recon(stft)
        wav_cpu = wav.detach().cpu()
        torchaudio.save(path, wav_cpu, 44100)
        os.chmod(path, 0o777)
        return wav

    def stft_from_model_to_wav(self, stft):
        stft = torch.permute(stft, (0, 2, 3, 1))
        stft = stft.contiguous()
        stft = torch.view_as_complex(stft)
        wav = self.reconstruct.istft_recon(stft)
        # wav = wav.detach().cpu()
        # wav_path = os.path.join(self.run_dir, name)
        # torchaudio.save(wav_path, wav, 44100)
        # os.chmod(wav_path,0o777)
        return wav

    def snr(self, clean, noise, recon_y1, recon_y2, name):
        org_snr = self.snr_calc(clean, noise)
        y1_snr = self.snr_calc(clean, (recon_y1 - clean))
        y2_snr = self.snr_calc(clean, (recon_y2 - clean))
        print(f'{name=}:\n{org_snr=}\n{y1_snr=}\n{y2_snr=}\n***********************')
        if y1_snr > y2_snr:
            return 'y1'
        else:
            return 'y2'

    def snr_calc(self, clean, noise):
        Ps = torch.mean(clean ** 2)
        Pn = torch.mean(noise ** 2)
        snr = 10 * torch.log10(Ps/(Pn+EPS))
        return snr

    def save_wav_file(self, path, wav):
        wav = wav.detach().cpu()
        torchaudio.save(path, wav, 44100)
        os.chmod(path, 0o777)

    def closer_est_noise_check(self, real_noise, est_1, est_2):
        mse1 = torch.sqrt(torch.mean(torch.pow(torch.sub(real_noise, est_1), 2)))
        mse2 = torch.sqrt(torch.mean(torch.pow(torch.sub(real_noise, est_2), 2)))
        if mse1 <= mse2:
            return 'est1'
        else:
            return 'est2'



#############################################################################################################

if __name__ == "__main__":
    cuda_num = 1
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    arch_name = "2_level_unet_nn"
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new/2023-08-16 17:18:08.884059_2_level_unet_nn_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    recon_dataloader = []
    for root, _, files in os.walk('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split'):
        for file in files:
            if file.endswith('clean.pickle'):
                recon_dataloader.append(file)

    # transfer in the model again
    trans_num = 20

    snr_list = ['0', '3', '6', '9', '12']
    check_recon = LoadRecon(cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num, snr_list)
    for snr in snr_list:
        check_recon.back_to_wav(snr)