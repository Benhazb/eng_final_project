import torchaudio
import torch
import os
#from util import choose_cuda
import sys
EPS = sys.float_info.epsilon
import pickle
import random


def choose_cuda(cuda_num):
    if cuda_num=="cpu" or cuda_num==-1:
        device = "cpu"
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if 0 <= cuda_num <= device_count - 1:  # devieces starts from '0'
            device = torch.device(f"cuda:{cuda_num}")
        else:
            print(f"Cuda Num:{cuda_num} NOT found, choosing cuda:0")
            device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")

    print("*******************************************")
    print(f" ****** running on device: {device} ******")
    print("*******************************************")
    return device

class prep_data():
    def __init__(self, data_dir, wind_size, hop, samples_per_sect, overlap, device):
        self.data_dir = data_dir
        self.wind_size = wind_size
        self.hop = hop
        self.device = device
        self.wind = torch.hann_window(self.wind_size, device=self.device)
        self.samples_per_sect = samples_per_sect
        self.overlap_rate = overlap


    def prep_audio(self):
        self.clean_data_list = self.extract_clean_data()
        self.noise_list = self.extract_noise('gramophone noise')
        self.noise_index = list(range(0,len(self.noise_list)))
        for wav_file in self.clean_data_list:
            self.split_n_stft(wav_file)

    def extract_noise(self, noise_dir_name):
        noise_paths_list = []
        noise_path = os.path.join(self.data_dir, noise_dir_name)
        for root, _, files in os.walk(noise_path):
            for file in files:
                if file.endswith('.wav'):
                    noise_paths_list.append(os.path.join(root, file))
        return noise_paths_list

    def extract_clean_data(self):
        clean_dir_path = f"{self.data_dir}/musicnet/train_data"
        wav_files = []
        train_data_split_path = f"{self.data_dir}/musicnet/train_data_split"
        if not os.path.exists(train_data_split_path):
            os.mkdir(train_data_split_path)
        if not os.path.exists(f"{train_data_split_path}/clean_data"):
            os.mkdir(f"{train_data_split_path}/clean_data")
        for root, dirs, files in os.walk(clean_dir_path):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.append(os.path.join(root, file))
                    file_dir = f"{train_data_split_path}/clean_data/" + file.replace(".wav", "")
                    if not os.path.exists(file_dir):
                        os.mkdir(file_dir)
        return wav_files

    def split_n_stft(self, audio):
        x, fs = torchaudio.load(audio)
        x = x.to(self.device)
        audio_split_dir_name = audio.split('/')[-1].replace(".wav", "")
        audio_split_dir_path = f"{self.data_dir}/musicnet/train_data_split/clean_data/{audio_split_dir_name}"
        if x.dim() > 1:
            x = torch.mean(x, dim=0)
        num_of_sec = (x.shape[0]-self.samples_per_sect*self.overlap_rate)/(self.samples_per_sect*self.overlap_rate)
        num_of_sec = int(num_of_sec)
        for i in range(num_of_sec):
            wav_sec = x[int(i * (self.samples_per_sect * self.overlap_rate)):int((i / 2 + 1) * self.samples_per_sect)]
            noise = self.rand_n_prep_noise()
            norm_wav_sec = self.normalize(wav_sec)
            norm_noise = self.normalize(noise)
            self.save_clean_stft(norm_wav_sec, str(i), audio_split_dir_name, audio_split_dir_path)
            SNR = [2, 5, 10]
            for snr in SNR:
                self.gen_n_save_SNR(norm_wav_sec, norm_noise, snr, str(i), audio_split_dir_name, audio_split_dir_path)
            #stft_sec = torch.stft(input=norm_wav_sec,
            #                      n_fft=self.wind_size,
            #                      hop_length=self.hop,
            #                      win_length=self.wind_size,
            #                      window=self.wind,
            #                      center=True,
            #                      pad_mode='reflect',
            #                      normalized=False,
            #                      onesided=None,
            #                      return_complex=True)
            #stft_sec_file_name = f"{audio_split_dir_name}_stft_sec{str(i)}_clean.pickle" #TO DO: check number of windows
            #stft_sec_path = f"{audio_split_dir_path}/{stft_sec_file_name}"
            #with open(stft_sec_path, 'wb') as file:
            #    stft_tnzr_sec = stft_sec.detach().cpu()
            #    pickle.dump(stft_tnzr_sec, file)

    def save_clean_stft(self, norm_wav_sec, index ,dir_name, dir_path):
        stft_sec = torch.stft(input=norm_wav_sec,
                              n_fft=self.wind_size,
                              hop_length=self.hop,
                              win_length=self.wind_size,
                              window=self.wind,
                              center=True,
                              pad_mode='reflect',
                              normalized=False,
                              onesided=None,
                              return_complex=True)
        stft_sec_file_name = f"{dir_name}_stft_sec{index}_clean.pickle" #TO DO: check number of windows
        stft_sec_path = f"{dir_path}/{stft_sec_file_name}"
        with open(stft_sec_path, 'wb') as file:
            stft_tnzr_sec = stft_sec.detach().cpu()
            pickle.dump(stft_tnzr_sec, file)

    def gen_n_save_SNR(self, norm_wav, norm_noise, snr, index, dir_name, dir_path):
        Ps = torch.mean(norm_wav ** 2)
        Pn = torch.mean(norm_noise ** 2)
        SNR = 10 * torch.log10(Ps/Pn)
        ratio = 10 ** (SNR/(10*snr))
        noised_signal = norm_wav + ratio*norm_noise
        stft_sec = torch.stft(input=noised_signal,
                              n_fft=self.wind_size,
                              hop_length=self.hop,
                              win_length=self.wind_size,
                              window=self.wind,
                              center=True,
                              pad_mode='reflect',
                              normalized=False,
                              onesided=None,
                              return_complex=True)
        stft_sec_file_name = f"{dir_name}_stft_sec{index}_SNR{str(snr)}_db.pickle"
        stft_sec_path = f"{dir_path}/{stft_sec_file_name}"
        with open(stft_sec_path, 'wb') as file:
            stft_tnzr_sec = stft_sec.detach().cpu()
            pickle.dump(stft_tnzr_sec, file)

    def rand_n_prep_noise(self):
        if(len(self.noise_index) == 0):
            self.noise_index = list(range(0, len(self.noise_list)))
        rand_index = self.noise_index.pop(random.randrange(len(self.noise_index)))
        noise_path = self.noise_list[rand_index]
        noise, fs = torchaudio.load(noise_path)
        noise = noise.to(self.device)
        if noise.dim() > 1:
            noise = torch.mean(noise, dim=0)
        if(len(noise) < self.samples_per_sect):
            noise = self.ext_to_sect(noise)
        else:
            noise = noise[0:self.samples_per_sect]
        return noise

    def ext_to_sect(self, noise):
        while(len(noise) < self.samples_per_sect):
            noise = torch.cat((noise, noise), dim=0)
        noise = noise[0:self.samples_per_sect]
        return noise

    def normalize(self, signal):
        signal_max = signal.max()
        signal_min = signal.min()
        centered_signal = signal - signal_min
        scaled_signal = centered_signal/(signal_max - signal_min)
        norm_signal = 2 * scaled_signal - 1
        norm_signal = 0.98 * norm_signal
        return norm_signal


############################################
##
############################################
if __name__ == "__main__":
    #prep_data_params
    cuda_num = 1
    device = choose_cuda(cuda_num)
    data_base_dir = f"/dsi/scratch/from_netapp/users/hazbanb/dataset"
    n_window = 2048
    hop_size = 1024
    samples_per_sec = 220160 #~5 sec
    overlap = 0.5 #50% overlap for more data
    data = prep_data(data_base_dir, n_window, hop_size, samples_per_sec, overlap, device)
    data.prep_audio()

