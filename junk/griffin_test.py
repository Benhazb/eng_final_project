import torch
import torchaudio
from prep_data import choose_cuda
from models import AutoEncoder
from dataset import create_dataset
import os

class griffin_lim():
    def __init__(self, audio, wind_len, hop, device):
        self.audio = audio
        self.device = device
        self.wind_len = wind_len
        self.hop = hop
        self.wind = torch.hann_window(self.wind_len, device=self.device)
        #self.griffin = torchaudio.transforms.GriffinLim(win_length=self.wind_len, hop_length=self.hop,
        #                                                n_fft=self.wind_len, n_iter=32, power=1).to(device)
        self.griffin = torchaudio.transforms.GriffinLim(win_length=400, hop_length=200,
                                                        n_fft=400, n_iter=32, power=1).to(device)
    def griffin_test(self):
        x, fs = torchaudio.load(self.audio)
        x = x.to(device)
        x = x[:,:500000]
        print(x.shape)
        spec = self.create_stft(x)
        recon = self.griffin(spec)
        recon = recon.detach().cpu()
        torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/recon_1788.wav', recon, fs)


    def create_stft(self, audio):
        #stft_sec = torch.stft(input=audio,
        #                      n_fft=self.wind_len,
        #                      hop_length=self.hop,
        #                      win_length=self.wind_len,
        #                      window=self.wind,
        #                      center=False,
        #                      pad_mode='reflect',
        #                      normalized=False,
        #                      onesided=True,
        #                      return_complex=True)
        stft_sec = torch.stft(input=audio,
                              n_fft=400,
                              hop_length=200,
                              win_length=400,
                              window=torch.hann_window(window_length=400, device=self.device),
                              center=False,
                              pad_mode='reflect',
                              normalized=False,
                              onesided=True,
                              return_complex=True)
        return torch.abs(stft_sec)



#############################################################################################################

if __name__ == "__main__":
    cuda_num = 0
    device = choose_cuda(cuda_num)
    wav = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data/1788.wav'
    wind_len = 2048
    hop = 1024

    #griffin = griffin_lim(wav, wind_len, hop, device)
    #griffin.griffin_test()

    # =======================================================
    # ---- Now Test on a real song ----------
    # =======================================================
    load_path = "/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data/1788.wav"

    waveform, sample_rate = torchaudio.load(load_path)
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    waveform = waveform[:,:500000]
    print(f"waveform:{waveform.shape}\n")

    print(f"waveform.device:{waveform.device}")
    waveform = waveform.to(device)
    print(f"waveform.device:{waveform.device}\n")

    # ---- calc spectrogram -> abs(stft) ------
    # https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.Spectrogram
    n_fft = 2048  # Default: 400
    win_length = n_fft  # Default: n_fft
    hop_length = win_length // 2  # Default: win_length//2
    power = 1  # Default: 2
    # trans_stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
    #                                                hop_length=hop_length, power=power)
    # spect = trans_stft(waveform).to(device)
    #------------
    spect = torch.stft(input=waveform,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=torch.hann_window(window_length=win_length, device=device),
                          center=False,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)
    spect = torch.abs(spect)
    spect = spect.to(device)

    print(f"spect:{spect.shape}")  # [batch, freq, time]

    n_fft = 2048  # Default: 400
    n_iter = 32  # Default: 32
    win_length = n_fft  # Default: n_fft
    hop_length = win_length // 2  # Default: win_length // 2
    rand_init = True  # Default: True (Initializes phase randomly if True and to zero otherwise)
    power = 1  # Default: 2
    strans_griffin = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=n_iter, win_length=win_length,
                                                      hop_length=hop_length, power=power, rand_init=rand_init)
    strans_griffin = strans_griffin.to(device)

    #start = time.time()
    reconst_wave = strans_griffin(spect)
    #end = time.time()
    #print(f"Griffing-Lim took:{end - start}[s]\n")

    print(f"wave:{waveform.shape}")
    print(f"wave:{reconst_wave.shape}")

    recon = reconst_wave.detach().cpu()
    torchaudio.save('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/recon_1788.wav', recon, sample_rate)
    #path = "adele_all_i_ask_griffin.wav"
    #reconst_wave = reconst_wave.to("cpu")
    #torchaudio.save(path, reconst_wave, sample_rate)
    # --------------------------------