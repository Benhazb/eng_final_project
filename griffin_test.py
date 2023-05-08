import torch
import torchaudio
from prep_data import choose_cuda
from models import AutoEncoder
from dataset import create_dataset
import os

class griffin_lin():
    def __init__(self, audio, wind_len, hop, device):
        self.audio = audio
        self.device = device
        self.wind_len = wind_len
        self.hop = hop
        self.wind = torch.hann_window(self.wind_len, device=self.device)
        self.griffin = torchaudio.transforms.GriffinLim(win_length=self.wind_len, hop_length=self.hop, n_fft=self.wind_len, n_iter=50).to(device)

    def griffin_test(self):
        x, fs = torchaudio.load(self.audio)
        x = x.to(device)
        spec = self.create_stft(x)
        recon = self.griffin(spec)
        recon = recon.detach().cpu()
        torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/recon_1788.wav', recon, fs)


    def create_stft(self, audio):
        stft_sec = torch.stft(input=audio,
                              n_fft=self.wind_len,
                              hop_length=self.hop,
                              win_length=self.wind_len,
                              window=self.wind,
                              center=False,
                              pad_mode='reflect',
                              normalized=False,
                              onesided=None,
                              return_complex=True)
        return torch.abs(stft_sec)







#############################################################################################################

if __name__ == "__main__":
    cuda_num = 0
    device = choose_cuda(cuda_num)
    wav = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data/1788.wav'
    wind_len = 2048
    hop = 1024

    griffin = griffin_lin(wav, wind_len, hop, device)
    griffin.griffin_test()