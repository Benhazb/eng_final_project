import torch
from prep_data import choose_cuda
import torchaudio
import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')
import model_v6
import model_v4
import model_v7
from torch import nn
from reconstruction import Reconstruct
import os

class wrapper():
    def __init__(self, unet_depth, activation, Ns, run_dir, tar_name, input_file,  wind_size, hop_size, samples_per_seg, overlap, device, output_dir):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.activation = activation
        self.input = input_file
        self.device = device
        self.samples_per_seg = samples_per_seg
        self.hop_size = hop_size
        self.overlap = overlap
        self.wind_size = wind_size
        self.wind = torch.hann_window(self.wind_size, device=self.device)
        self.seg_list = []
        self.gen_model(unet_depth, Ns, self.activation)
        self.output_dir = output_dir
        self.creat_output_dir(self.input.split('/')[-1].split('.wav')[0])

    def creat_output_dir(self, name):
        self.path = os.path.join(self.output_dir, name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
    def split_n_stft(self):
        self.x, fs = torchaudio.load(self.input)
        x = self.x.to(self.device)
        if x.dim() > 1:
            x = torch.mean(x, dim=0)
        num_of_sec = (x.shape[0]-self.samples_per_seg)/(self.samples_per_seg*(1-self.overlap)) + 1
        num_of_sec = int(num_of_sec)
        for i in range(num_of_sec):
            start_idx = i * (self.samples_per_seg * (1 - self.overlap))
            end_idx = start_idx + self.samples_per_seg
            wav_sec = x[int(start_idx):int(end_idx)]
            norm_wav_sec = self.normalize(wav_sec)
            stft_sec = torch.stft(input=norm_wav_sec,
                                  n_fft=self.wind_size,
                                  hop_length=self.hop_size,
                                  win_length=self.wind_size,
                                  window=self.wind,
                                  center=True,
                                  pad_mode='reflect',
                                  normalized=False,
                                  onesided=None,
                                  return_complex=True)
            self.seg_list.append(stft_sec)

    def normalize(self, signal):
        signal_max = abs(signal.max())
        centered_signal = signal - signal.mean()
        norm_signal = centered_signal/signal_max
        norm_signal = 0.98 * norm_signal
        return norm_signal

    def gen_model(self, unet_depth, Ns, activation):
        if "2_level_unet_nc" in self.run_dir:
            self.model = model_v6.Model(unet_depth, Ns, activation)
        elif ("2_level_unet_nn" in self.run_dir) or ("2_level_unet_2n2c" in self.run_dir) or ("2_level_unet_cc" in self.run_dir):
            self.model = model_v7.Model(unet_depth, Ns, activation)
        else:
            self.model = model_v4.Model(unet_depth, Ns, activation)
        checkpoint = torch.load(os.path.join(self.run_dir, self.tar_name))
        self.model.load_state_dict(checkpoint['model'])
        # move model to device
        self.model.to(self.device)

    def through_the_model(self):
        self.reconstruct = Reconstruct(self.device)
        with torch.no_grad():
            for idx, noise_segment in enumerate(self.seg_list):
                noise_segment = noise_segment.to(self.device)
                noise_segment = noise_segment.unsqueeze(0)
                noise_segment = torch.view_as_real(noise_segment)
                noise_segment = torch.permute(noise_segment, (0, 3, 1, 2))
                if "2_level_unet_nc" in self.run_dir:
                    y1, y2, est_noise = self.model(noise_segment)
                    recon_wav_y1 = self.stft_from_model_to_wav(y1)
                    recon_wav_y2 = self.stft_from_model_to_wav(y2)
                    recon_wav_est_noise = self.stft_from_model_to_wav(est_noise)
                    if idx == 0:
                        y1_final = recon_wav_y1
                        y2_final = recon_wav_y2
                        n1_final = recon_wav_est_noise
                    else:
                        y1_final = torch.cat((y1_final, recon_wav_y1[109568:]), dim=0)
                        y2_final = torch.cat((y2_final, recon_wav_y2[109568:]), dim=0)
                        n1_final = torch.cat((n1_final, recon_wav_est_noise[109568:]), dim=0)
                    if idx == (len(self.seg_list)-1):
                        self.save_final_wav(self.x, 'clean.wav')
                        self.save_final_wav(y1_final, 'y1.wav')
                        self.save_final_wav(y2_final, 'y2.wav')
                        self.save_final_wav(n1_final, 'n1.wav')
                elif ("2_level_unet_nn" in self.run_dir) or ("2_level_unet_2n2c" in self.run_dir) or ("2_level_unet_cc" in self.run_dir):
                    y1, y2, est_noise1, est_noise2 = self.model(noise_segment)
                    recon_wav_y1 = self.stft_from_model_to_wav(y1)
                    recon_wav_y2 = self.stft_from_model_to_wav(y2)
                    recon_wav_est_noise1 = self.stft_from_model_to_wav(est_noise1)
                    recon_wav_est_noise2 = self.stft_from_model_to_wav(est_noise2)
                    if idx == 0:
                        y1_final = recon_wav_y1
                        y2_final = recon_wav_y2
                        n1_final = recon_wav_est_noise1
                        n2_final = recon_wav_est_noise2
                    else:
                        y1_final = torch.cat((y1_final, recon_wav_y1[109568:]), dim=0)
                        y2_final = torch.cat((y2_final, recon_wav_y2[109568:]), dim=0)
                        n1_final = torch.cat((n1_final, recon_wav_est_noise1[109568:]), dim=0)
                        n2_final = torch.cat((n2_final, recon_wav_est_noise2[109568:]), dim=0)
                    if idx == (len(self.seg_list) - 1):
                        torchaudio.save(os.path.join(self.path, 'org.wav'), self.x, 44100)
                        self.save_final_wav(y1_final, 'y1.wav')
                        self.save_final_wav(y2_final, 'y2.wav')
                        self.save_final_wav(n1_final, 'n1.wav')
                        self.save_final_wav(n2_final, 'n2.wav')
        return y2_final

    def stft_from_model_to_wav(self, stft):
        stft = torch.permute(stft, (0, 2, 3, 1))
        stft = stft.contiguous() # Ensure contiguous memory layout
        stft = torch.view_as_complex(stft)
        recon_wav = self.reconstruct.istft_recon(stft).squeeze(0)
        return recon_wav

    def save_final_wav(self, wav, name):
        wav = wav.detach().cpu()
        wav = wav.unsqueeze(0)
        full_path = os.path.join(self.path,name)
        torchaudio.save(full_path, wav, 44100)



if __name__ == "__main__":
    cuda_num = 1
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    device = choose_cuda(cuda_num)
    n_window = 2048
    hop_size = 1024
    samples_per_sec = 219136  # ~5 sec
    overlap = 0.5  # 50% overlap for more data
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new/2023-08-17 02:17:44.150340_2_level_unet_2n2c_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    input_file = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data/1819.wav'
    output_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs_new/test'
    wrap = wrapper(unet_depth, activation, Ns, run_dir, tar_name, input_file, n_window, hop_size, samples_per_sec, overlap, device, output_dir)
    wrap.split_n_stft()
    recon = wrap.through_the_model()
    full_segmented_song = wrap.seg_list
