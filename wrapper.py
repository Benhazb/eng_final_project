import torch
from prep_data import choose_cuda
import torchaudio


class wrapper():
    def __init__(self, input_file, device, samples_per_seg, overlap):
        self.input = input_file
        self.device = device
        self.samples_per_seg = samples_per_seg
        self.overlap = overlap
        self.wind_size = 2048
        self.wind = torch.hann_window(self.wind_size, device=self.device)


    def prep_n_trans_to_net(self):
        self.split_n_stft(self.input)

    def split_n_stft(self, audio):
        x, fs = torchaudio.load(audio)
        x = x.to(self.device)
        if x.dim() > 1:
            x = torch.mean(x, dim=0)
        num_of_sec = (x.shape[0]-self.samples_per_seg)/(self.samples_per_seg*(1-self.overlap)) + 1
        num_of_sec = int(num_of_sec)
        for i in range(num_of_sec):
            start_idx = i * (self.samples_per_sect * (1 - self.overlap_rate))
            end_idx = start_idx + self.samples_per_sect
            wav_sec = x[int(start_idx):int(end_idx)]
            norm_wav_sec = self.normalize(wav_sec)
            stft_sec = torch.stft(input=norm_wav_sec,
                                  n_fft=self.wind_size,
                                  hop_length=self.overlap,
                                  win_length=self.wind_size,
                                  window=self.wind,
                                  center=True,
                                  pad_mode='reflect',
                                  normalized=False,
                                  onesided=None,
                                  return_complex=True)
            seg_phase = torch.angle(stft_sec)
            post_net_seg = self.send_to_net(abs(stft_sec))
            rec_seg = post_net_seg + seg_phase



    def normalize(self, signal):
        signal_max = abs(signal.max())
        centered_signal = signal - signal.mean()
        norm_signal = centered_signal/signal_max
        norm_signal = 0.98 * norm_signal
        return norm_signal





if __name__ == "__main__":
    device = choose_cuda(1)
    input_file = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data/1788.wav'
    samples_per_sec = 220160 #~5 sec
    overlap = 0.5 #50% overlap for more data
    wrap = wrapper(input_file, device, samples_per_sec, overlap)
    wrap.prep_n_trans_to_net()