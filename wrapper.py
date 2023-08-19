import torch
from prep_data import choose_cuda
import torchaudio
import sys
sys.path.append('/home/dsi/peryyuv/proj/git/models')
import model_v6
import model_v4
import model_v7
from torch import nn
from reconstruction import Reconstruct
import os

class wrapper():
    def __init__(self, unet_depth, activation, Ns, run_dir, tar_name, input_file,  wind_size, hop_size, samples_per_seg, overlap, device):
        self.run_dir = run_dir
        self.tar_name = tar_name
        self.arch_name = arch_name
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

    # def prep_n_trans_to_net(self):
    #     self.split_n_stft(self.input)

    def split_n_stft(self):
        x, fs = torchaudio.load(self.input)
        x = x.to(self.device)
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
            # seg_phase = torch.angle(stft_sec)
            # post_net_seg = self.send_to_net(abs(stft_sec))
            # rec_seg = post_net_seg + seg_phase



    def normalize(self, signal):
        signal_max = abs(signal.max())
        centered_signal = signal - signal.mean()
        norm_signal = centered_signal/signal_max
        norm_signal = 0.98 * norm_signal
        return norm_signal

    def gen_model(self, unet_depth, Ns, activation):
        if "2_level_unet_nc" in self.arch_name:
            self.model = model_v6.Model(unet_depth, Ns, activation)
        elif ("2_level_unet_nn" in self.arch_name) or ("2_level_unet_2n2c" in self.arch_name):
            self.model = model_v7.Model(unet_depth, Ns, activation)
        else:
            self.model = model_v4.Model(unet_depth, Ns, activation)
        checkpoint = torch.load(os.path.join(self.run_dir, self.tar_name))
        self.model.load_state_dict(checkpoint['model'])
        # move model to device
        self.model.to(self.device)

    def throw_the_model(self):
        reconstruct = Reconstruct(self.device)
        with torch.no_grad():
            for idx, noise_segment in enumerate(self.seg_list):
                noise_segment = noise_segment.unsqueeze(0).to(self.device)
                noise_segment = torch.view_as_real(noise_segment)
                noise_segment = torch.permute(noise_segment, (0, 3, 1, 2))
                if "2_level_unet_nc" in self.arch_name:
                    y1, y2, est_noise = self.model(noise_segment)

                    y1 = torch.permute(y1, (0, 2, 3, 1))
                    y2 = torch.permute(y2, (0, 2, 3, 1))
                    est_noise = torch.permute(est_noise, (0, 2, 3, 1))
                    y1 = y1.contiguous()
                    y2 = y2.contiguous()
                    est_noise = est_noise.contiguous()
                    y1 = torch.view_as_complex(y1)
                    y2 = torch.view_as_complex(y2)
                    est_noise = torch.view_as_complex(est_noise)
                    recon_wav_y1 = reconstruct.istft_recon(y1)
                    recon_wav_y2 = reconstruct.istft_recon(y2)
                    recon_wav_est_noise = reconstruct.istft_recon(est_noise)
                    if idx == 0:
                        y1_final = recon_wav_y1
                        y2_final = recon_wav_y2
                        n1_final = recon_wav_est_noise
                    else:
                        y1_final = torch.cat((y1_final, recon_wav_y1[109568:]), dim=0)
                        y2_final = torch.cat((y2_final, recon_wav_y2[109568:]), dim=0)
                        n1_final = torch.cat((n1_final, recon_wav_est_noise[109568:]), dim=0)
                    if idx == len(self.seg_list):

                        recon_y1_final = recon_y1_final.detach().cpu()
                        recon_y2_final = recon_y2_final.detach().cpu()
                        recon_n1_final = recon_n1_final.detach().cpu()

                        torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/test/y1.wav', recon_y1_final, 44100)
                        torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/test/y2.wav', recon_y2_final, 44100)
                        torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/test/n1.wav', recon_n1_final, 44100)



if __name__ == "__main__":

    cuda_num = 2
    unet_depth = 6
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    arch_name = "2_level_unet_nc"
    device = choose_cuda(cuda_num)
    n_window = 2048
    hop_size = 1024
    samples_per_sec = 219136  # ~5 sec
    overlap = 0.5  # 50% overlap for more data
    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/2023-08-04 16:27:56.556137_2_level_unet_model_30epochs_depth_512channels_batch16'
    tar_name = 'FinalModel.tar'
    input_file = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data/1759.wav'
    wrap = wrapper(unet_depth, activation, Ns, run_dir, tar_name, input_file, n_window, hop_size, samples_per_sec, overlap, device)
    wrap.split_n_stft()
    wrap.throw_the_model()
    full_segmented_song = wrap.seg_list
    print(full_segmented_song[1].size())
