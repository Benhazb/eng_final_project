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
    def __init__(self, cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.recon_dataloader = recon_dataloader
        self.arch_name = arch_name
        self.gen_model(unet_depth, Ns, activation, cuda_num)
        self.trans_num = trans_num

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

    def back_to_wav(self):
        test_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split'
        self.reconstruct = Reconstruct(self.device)
        self.model.eval()
        with torch.no_grad():
            for example in self.recon_dataloader:
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
                clean_wav = self.stft_to_wav(clean_name, clean_stft)

                noisy_name = example.split('clean')[0] + 'noisy.wav'
                self.stft_to_wav(noisy_name, noisy_stft)

                noise_name = example.split('clean')[0] + 'noise.wav'
                noise_wav = self.stft_to_wav(noise_name, noise_stft)


                # send noisy to model
                if "2_level_unet_nc" in self.arch_name:
                    y1, y2 = self.recon_model_v6(noisy_signal, example, clean_wav, noise_wav)
                elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name) or ("2_level_unet_cc" in self.arch_name):
                    y1, y2 = self.recon_model_v7(noisy_signal, example, clean_wav, noise_wav)
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
                idx = 0
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

        recon_name_y1 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y1.wav"
        recon_y1 = self.stft_from_model_to_wav(recon_name_y1, y1_stft)

        recon_name_y2 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y2.wav"
        recon_y2 = self.stft_from_model_to_wav(recon_name_y2, y2_stft)

        recon_name_est_noise = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise.wav"
        self.stft_from_model_to_wav(recon_name_est_noise, est_noise_stft)

        # calc snr
        self.less_noisy = self.snr(clean_wav, noise_wav, recon_y1, recon_y2, example.split('_clean')[0])

        return y1_stft, y2_stft

    def recon_model_v7(self, noisy_signal, example, clean_wav, noise_wav):
        y1_stft, y2_stft, est_noise1_stft, est_noise2_stft = self.model(noisy_signal)

        recon_name_y1 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y1.wav"
        recon_y1 = self.stft_from_model_to_wav(recon_name_y1, y1_stft)

        recon_name_y2 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y2.wav"
        recon_y2 = self.stft_from_model_to_wav(recon_name_y2, y2_stft)

        recon_name_est_noise1 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise1.wav"
        self.stft_from_model_to_wav(recon_name_est_noise1, est_noise1_stft)

        recon_name_est_noise2 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise2.wav"
        self.stft_from_model_to_wav(recon_name_est_noise2, est_noise2_stft)

        # calc snr
        self.less_noisy = self.snr(clean_wav, noise_wav, recon_y1, recon_y2, example.split('_clean')[0])

        return y1_stft, y2_stft

    def stft_to_wav(self, name, stft):
        wav = self.reconstruct.istft_recon(stft)
        wav = wav.detach().cpu()
        wav_path = os.path.join(self.run_dir, name)
        torchaudio.save(wav_path, wav, 44100)
        return wav

    def stft_from_model_to_wav(self, name, stft):
        stft = torch.permute(stft, (0, 2, 3, 1))
        stft = stft.contiguous()
        stft = torch.view_as_complex(stft)
        wav = self.reconstruct.istft_recon(stft)
        wav = wav.detach().cpu()
        wav_path = os.path.join(self.run_dir, name)
        torchaudio.save(wav_path, wav, 44100)
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



#############################################################################################################

if __name__ == "__main__":
    cuda_num = 1
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    arch_name = "2_level_unet_2n2c"
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new/2023-08-17 02:17:44.150340_2_level_unet_2n2c_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    recon_dataloader = ['2114_stft_sec51_clean.pickle', '2486_stft_sec34_clean.pickle', '2550_stft_sec53_clean.pickle', '2629_stft_sec102_clean.pickle']

    # transfer in the model again
    trans_num = 20

    check_recon = LoadRecon(cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader, trans_num)
    check_recon.back_to_wav()