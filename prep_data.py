import torchaudio
import torch
import os
import sys
EPS = sys.float_info.epsilon
import pickle
import random
import scipy.io.wavfile as wavfile


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
        clean_data_list = []
        with open('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data/clean_data.txt', 'r') as file:
            for line in file:
                clean_data_list.append(line.strip())
        clean_dir_path = f"{self.data_dir}/musicnet/train_data"
        wav_files = []
        train_data_split_path = f"{self.data_dir}/musicnet/train_data_split"
        if not os.path.exists(train_data_split_path):
            os.mkdir(train_data_split_path)
        for root, dirs, files in os.walk(clean_dir_path):
            for file in files:
                if file.endswith(".wav"):
                    if(file.split('.')[0] not in clean_data_list):
                        continue
                    wav_files.append(os.path.join(root, file))
                    file_dir = f"{train_data_split_path}/" + file.replace(".wav", "")
                    if not os.path.exists(file_dir):
                        os.mkdir(file_dir)
        return wav_files

    def split_n_stft(self, audio):
        x, fs = torchaudio.load(audio)
        x = x.to(self.device)
        audio_split_dir_name = audio.split('/')[-1].replace(".wav", "")
        audio_split_dir_path = f"{self.data_dir}/musicnet/train_data_split/{audio_split_dir_name}"
        if x.dim() > 1:
            x = torch.mean(x, dim=0)
        num_of_sec = (x.shape[0]-self.samples_per_sect)/(self.samples_per_sect*(1-self.overlap_rate)) + 1
        num_of_sec = int(num_of_sec)
        for i in range(num_of_sec):
            start_idx = i * (self.samples_per_sect * (1-self.overlap_rate))
            end_idx = start_idx + self.samples_per_sect
            wav_sec = x[int(start_idx):int(end_idx)]
            noise = self.rand_n_prep_noise()
            norm_wav_sec = self.normalize(wav_sec)
            norm_noise = self.normalize(noise)
            self.save_clean_stft(norm_wav_sec, str(i), audio_split_dir_name, audio_split_dir_path)
            SNR = [-3, 0, 3, 6, 9, 12, 15]
            for snr in SNR:
                self.gen_n_save_SNR(norm_wav_sec, norm_noise, snr, str(i), audio_split_dir_name, audio_split_dir_path)


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
        stft_sec_file_name = f"{dir_name}_stft_sec{index}_clean.pickle"
        stft_sec_path = f"{dir_path}/{stft_sec_file_name}"
        with open(stft_sec_path, 'wb') as file:
            stft_tnzr_sec = stft_sec.detach().cpu()
            pickle.dump(stft_tnzr_sec, file)

    def gen_n_save_SNR(self, norm_wav, norm_noise, snr, index, dir_name, dir_path):
        Ps = torch.mean(norm_wav ** 2)
        Pn = torch.mean(norm_noise ** 2)
        SNR = 10 * torch.log10(Ps/Pn)
        a = 10 ** ((SNR-snr)/20)
        noise_snr = a * norm_noise
        stft_sec = torch.stft(input=noise_snr,
                              n_fft=self.wind_size,
                              hop_length=self.hop,
                              win_length=self.wind_size,
                              window=self.wind,
                              center=True,
                              pad_mode='reflect',
                              normalized=False,
                              onesided=None,
                              return_complex=True)
        stft_sec_file_name = f"{dir_name}_noise_stft_sec{index}_SNR{str(snr)}_db.pickle"
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
        fade_in = torch.cat((torch.linspace(0, 1, 22050), torch.ones(len(noise)-22050)), dim=0)
        fade_in = fade_in.to(self.device)
        fade_in_noise = noise * fade_in
        while(len(noise) < self.samples_per_sect):
            fade_out = torch.cat((torch.ones(len(noise) - 22050), torch.linspace(1, 0, 22050)), dim=0)
            fade_out = fade_out.to(self.device)
            noise = noise * fade_out
            noise = torch.cat((noise[:-22050], noise[-22050:] + fade_in_noise[:22050], fade_in_noise[22050:]), dim=0)
        noise = noise[0:self.samples_per_sect]
        return noise

    def normalize(self, signal):
        signal_max = torch.max(torch.abs(signal))
        centered_signal = signal - torch.mean(signal)
        norm_signal = centered_signal/signal_max
        norm_signal = 0.98 * norm_signal
        return norm_signal

    def reconstuct(self, file_num, seg, snr):
        with open(f"/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split/{file_num}/{file_num}_noise_stft_sec{seg}_{snr}.pickle", 'rb') as file:
            pickle_file = pickle.load(file)
            self.save_as_wav(pickle_file, f"{file_num}_sec{seg}_{snr}.wav")


    def save_as_wav(self, signal, file_name):
        signal = signal.to(self.device)
        print(type(signal))
        rec_signal = torch.istft(input=signal,
                                 n_fft=self.wind_size,
                                 hop_length=self.hop,
                                 win_length=self.wind_size,
                                 window=self.wind,
                                 center=True,
                                 normalized=False,
                                 onesided=None,
                                 return_complex=False)
        #torchaudio.play(rec_signal, 44100)
        rec_signal = rec_signal.detach().cpu()
        rec_signal = rec_signal.numpy()
        rec_signal = (rec_signal*32767).astype('int16')
        wavfile.write(f'/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/{file_name}', 44100, rec_signal)

    def add_snrs(self, exmp_list):
        self.noise_list = self.extract_noise('gramophone noise')
        self.noise_index = list(range(0,len(self.noise_list)))
        clean_data = self.extract_clean_data()
        SNR = ['SNR3_db', 'SNR6_db', 'SNR9_db', 'SNR0_db', 'SNR-3_db', 'SNR12_db']
        for wav_file in clean_data:
            file_num = wav_file.split('/')[-1].split('.wav')[0]
            if file_num in exmp_list:
                self.split_n_stft(wav_file)
        for example in exmp_list:
            for snr in SNR:
                self.reconstuct(example, 15, snr)

    def rand_val_n_test(self, dest_dir):
        source_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split/'
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        dirs_list = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        random_dirs = random.sample(dirs_list, 4)
        print(random_dirs)
        for dir_name in random_dirs:
            src = os.path.join(source_dir, dir_name)
            dst = os.path.join(dest_dir, dir_name)
            os.rename(src, dst)


############################################
##
############################################
if __name__ == "__main__":
    #prep_data_params
    cuda_num = 0
    device = choose_cuda(cuda_num)
    data_base_dir = f"/dsi/scratch/from_netapp/users/hazbanb/dataset"
    n_window = 2048
    hop_size = 1024
    samples_per_sec = 219136 #~5 sec
    overlap = 0.5 #50% overlap for more data
    data = prep_data(data_base_dir, n_window, hop_size, samples_per_sec, overlap, device)
    data.prep_audio()
    data.rand_val_n_test('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/val_data_split/')
    data.rand_val_n_test('/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split/')

    #with open(f"/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split/1733/1733_stft_sec15_clean.pickle", 'rb') as file:
    #    pickle_file = pickle.load(file)
    #    data.save_as_wav(pickle_file, f"1733_sec15_clean.wav")


