import torch
import torchaudio
from prep_data import choose_cuda
from models import AutoEncoder
from dataset import create_dataset
import os

class Reconstruct():
    def __init__(self, device):
        self.device = device
        self.wind_len = 2048
        self.hop = 1024
        self.n_iter = 64
        self.rand_init = True
        self.power = 1
        self.griffin = torchaudio.transforms.GriffinLim(n_fft=self.wind_len, n_iter=self.n_iter, win_length=self.wind_len,
                                                          hop_length=self.hop, power=self.power, rand_init=self.rand_init).to(self.device)
        self.wind = torch.hann_window(self.wind_len, device=self.device)

    def griffin_recon(self, audio):
        return self.griffin(audio)

    def istft_recon(self, audio):
        rec_signal = torch.istft(input=audio,
                                 n_fft=self.wind_len,
                                 hop_length=self.hop,
                                 win_length=self.wind_len,
                                 window=self.wind,
                                 center=True,
                                 normalized=False,
                                 onesided=None,
                                 return_complex=False)
        return rec_signal





#############################################################################################################

if __name__ == "__main__":
    n_fft = 2048  # Default: 400
    n_iter = 32  # Default: 32
    win_length = n_fft  # Default: n_fft
    hop_length = win_length // 2  # Default: win_length // 2
    rand_init = True  # Default: True (Initializes phase randomly if True and to zero otherwise)
    power = 1  # Default: 2
    strans_griffin = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=n_iter, win_length=win_length,
                                                      hop_length=hop_length, power=power, rand_init=rand_init)
