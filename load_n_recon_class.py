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
    def __init__(self, cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.recon_dataloader = recon_dataloader
        self.arch_name = arch_name
        self.gen_model(unet_depth, Ns, activation, cuda_num)

    def gen_model(self, unet_depth, Ns, activation, cuda_num):
        if "2_level_unet_nc" in self.arch_name:
            self.model = model_v6.Model(unet_depth, Ns, activation)
        elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name):
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
        reconstruct = Reconstruct(self.device)
        self.model.eval()
        with torch.no_grad():
            for example in self.recon_dataloader:
                dir_num = example.split('_')[0]
                clean_path = os.path.join(test_dir, dir_num, example)
                noise_name = example.replace('stft', 'noise_stft').replace('clean', 'SNR6_db')
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

                if "2_level_unet_nc" in self.arch_name:
                    y1, y2, est_noise = self.model(noisy_signal)
                    # y1_tag, _, est_noise_tag = self.model(y1)

                    y1        = torch.permute(y1, (0, 2, 3, 1))
                    y2        = torch.permute(y2, (0, 2, 3, 1))
                    est_noise = torch.permute(est_noise, (0, 2, 3, 1))
                    y1        = y1.contiguous()
                    y2        = y2.contiguous()
                    est_noise = est_noise.contiguous()
                    y1        = torch.view_as_complex(y1)
                    y2        = torch.view_as_complex(y2)
                    est_noise = torch.view_as_complex(est_noise)

                    recon_wav_y1        = reconstruct.istft_recon(y1)
                    recon_wav_y2        = reconstruct.istft_recon(y2)
                    recon_wav_est_noise = reconstruct.istft_recon(est_noise)
                    recon_wav_y1        = recon_wav_y1.detach().cpu()
                    recon_wav_y2        = recon_wav_y2.detach().cpu()
                    recon_wav_est_noise = recon_wav_est_noise.detach().cpu()

                    recon_name_y1        = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y1.wav"
                    recon_name_y2        = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y2.wav"
                    recon_name_est_noise = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise.wav"

                    recon_wav_path_y1        = os.path.join(self.run_dir, recon_name_y1)
                    recon_wav_path_y2        = os.path.join(self.run_dir, recon_name_y2)
                    recon_wav_path_est_noise = os.path.join(self.run_dir, recon_name_est_noise)
                    torchaudio.save(recon_wav_path_y1, recon_wav_y1, 44100)
                    torchaudio.save(recon_wav_path_y2, recon_wav_y2, 44100)
                    torchaudio.save(recon_wav_path_est_noise, recon_wav_est_noise, 44100)

                elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name):
                    y1, y2, est_noise1, est_noise2 = self.model(noisy_signal)

                    y1        = torch.permute(y1, (0, 2, 3, 1))
                    y2        = torch.permute(y2, (0, 2, 3, 1))
                    est_noise1 = torch.permute(est_noise1, (0, 2, 3, 1))
                    est_noise2 = torch.permute(est_noise2, (0, 2, 3, 1))
                    y1        = y1.contiguous()
                    y2        = y2.contiguous()
                    est_noise1 = est_noise1.contiguous()
                    est_noise2 = est_noise2.contiguous()
                    y1        = torch.view_as_complex(y1)
                    y2        = torch.view_as_complex(y2)
                    est_noise1 = torch.view_as_complex(est_noise1)
                    est_noise2 = torch.view_as_complex(est_noise2)

                    recon_wav_y1        = reconstruct.istft_recon(y1)
                    recon_wav_y2        = reconstruct.istft_recon(y2)
                    recon_wav_est_noise1 = reconstruct.istft_recon(est_noise1)
                    recon_wav_est_noise2 = reconstruct.istft_recon(est_noise2)
                    recon_wav_y1        = recon_wav_y1.detach().cpu()
                    recon_wav_y2        = recon_wav_y2.detach().cpu()
                    recon_wav_est_noise1 = recon_wav_est_noise1.detach().cpu()
                    recon_wav_est_noise2 = recon_wav_est_noise2.detach().cpu()

                    recon_name_y1        = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y1.wav"
                    recon_name_y2        = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_y2.wav"
                    recon_name_est_noise1 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise1.wav"
                    recon_name_est_noise2 = f"{example.split('clean')[0]}_reconstruct_{self.tar_name.split('.tar')[0]}_est_noise2.wav"

                    recon_wav_path_y1        = os.path.join(self.run_dir, recon_name_y1)
                    recon_wav_path_y2        = os.path.join(self.run_dir, recon_name_y2)
                    recon_wav_path_est_noise1 = os.path.join(self.run_dir, recon_name_est_noise1)
                    recon_wav_path_est_noise2 = os.path.join(self.run_dir, recon_name_est_noise2)
                    torchaudio.save(recon_wav_path_y1, recon_wav_y1, 44100)
                    torchaudio.save(recon_wav_path_y2, recon_wav_y2, 44100)
                    torchaudio.save(recon_wav_path_est_noise1, recon_wav_est_noise1, 44100)
                    torchaudio.save(recon_wav_path_est_noise2, recon_wav_est_noise2, 44100)
                else:
                    filtered_signal = self.model(noisy_signal)

                    filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
                    filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
                    filtered_signal = torch.view_as_complex(filtered_signal)

                    recon_wav = reconstruct.istft_recon(filtered_signal)
                    recon_wav = recon_wav.detach().cpu()

                    recon_name = f"{example.split('clean')[0]}_reconstruct_{tar_name}.wav"
                    recon_wav_path = os.path.join(self.run_dir, recon_name)
                    torchaudio.save(recon_wav_path, recon_wav, 44100)

                clean_wav  = reconstruct.istft_recon(clean_stft)
                noisy_wav  = reconstruct.istft_recon(noisy_stft)
                noise_wav  = reconstruct.istft_recon(noise_stft)
                clean_wav  = clean_wav.detach().cpu()
                noisy_wav  = noisy_wav.detach().cpu()
                noise_wav  = noise_wav.detach().cpu()

                clean_name = example.split('.pickle')[0] + '.wav'
                clean_wav_path = os.path.join(self.run_dir, clean_name)
                torchaudio.save(clean_wav_path, clean_wav, 44100)

                noisy_name = example.split('clean')[0] + 'noisy.wav'
                noisy_wav_path = os.path.join(self.run_dir, noisy_name)
                torchaudio.save(noisy_wav_path, noisy_wav, 44100)

                noise_name = example.split('clean')[0] + 'noise.wav'
                noise_wav_path = os.path.join(self.run_dir, noise_name)
                torchaudio.save(noise_wav_path, noise_wav, 44100)

                org_snr = self.snr_calc(clean_wav, noise_wav)
                est_snr1 = self.snr_calc(clean_wav, (recon_wav_y1-clean_wav))
                est_snr2 = self.snr_calc(clean_wav, (recon_wav_y2-clean_wav))
                print(f'{noise_name=}\t:\n{org_snr=}\n{est_snr1=}\n*************************')
                print(f'{noise_name=}\t:\n{org_snr=}\n{est_snr2=}\n*************************')


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
    arch_name = "2_level_unet"
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/2023-08-04 16:27:56.556137_2_level_unet_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    recon_dataloader = ['2114_stft_sec51_clean.pickle', '2486_stft_sec34_clean.pickle', '2550_stft_sec53_clean.pickle', '2629_stft_sec102_clean.pickle']

    check_recon = LoadRecon(cuda_num, unet_depth, activation, Ns, arch_name, run_dir, tar_name, recon_dataloader)
    check_recon.back_to_wav()