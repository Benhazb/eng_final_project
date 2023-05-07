import pandas as pd
import pickle
import torch
import multiprocessing as mp
import datetime as time
import timeit
import matplotlib.pyplot as plt
#from torchinfo import summary

import models

class create_dataset():
    def __init__(self, mode):
        self.root = f'/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/{mode}_data_split'
        self.csv_name = 'dataset.csv'
        self.df = pd.DataFrame()
        self.data_len = 0

    def filter_by_snrs(self, snrs):
        df = pd.read_csv(f"{self.root}/{self.csv_name}")
        self.df = df[df['SNR'].isin(snrs)]
        self.data_len = self.df.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        noise = self.df.iloc[idx, 0]
        noise_fpath = f"{self.root}{noise}"
        snr = self.df.iloc[idx, 1]
        audio_num = noise.split('/')[-1].split('_')[0]
        seg_num = noise.split('/')[-1].split('sec')[1].split('_')[0]
        clean_fpath = f"{self.root}/{audio_num}/{audio_num}_stft_sec{seg_num}_clean.pickle"
        with open(clean_fpath, 'rb') as handle:
            clean_data = pickle.load(handle)
        with open(noise_fpath, 'rb') as handle:
            noise_data = pickle.load(handle)
        return clean_data.unsqueeze(0), noise_data.unsqueeze(0), snr

def test_num_workers(dataset_mode):
    print(f"mp.cpu_count():{mp.cpu_count()}")
    all_times = []
    for num_workers in range(0, 48):
        print(f"--- start num_workers={num_workers} ----")
        train_loader = torch.utils.data.DataLoader(dataset_mode, batch_size=128, shuffle=True, num_workers=num_workers)
        start = timeit.default_timer()
        print(start)
        for i, data in enumerate(train_loader):
            if (i * 128) >= 1000:
                break
        end = timeit.default_timer()
        print(end)
        worker_time = end - start
        all_times.append(worker_time)
        print(f"Finish with:{worker_time:4f} second, num_workers={num_workers}")
    plt.plot(all_times)
    plt.show()


if __name__ == "__main__":
    snrs = ['3']
    train_dataset = create_dataset('train')
    train_dataset.filter_by_snrs(snrs)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_num_workers(train_dataset)








