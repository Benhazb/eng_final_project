import pandas as pd
import pickle

class create_dataset():
    def __init__(self, root, csv_name):
        self.root = root
        self.csv_name = csv_name
        self.df = pd.DataFrame()
        self.clean_df = pd.DataFrame()
        self.data_len = 0

    def filter_by_snrs(self, snrs):
        df = pd.read_csv(f"{self.root}/{self.csv_name}")
        self.df = df[df['SNR'].isin(snrs)]
        self.clean_df = df[df['SNR'] == 'clean']
        self.data_len = self.df.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        noise = self.df.iloc[idx, 0]
        noise_fpath = f"{self.root}{noise}"
        snr = self.df.iloc[idx, 1]
        audio_num = noise.split('/')[-1].split('_')[0]
        seg_num = noise.split('/')[-1].split('sec')[1].split('_')[0]
        clean_signal = f"{audio_num}_stft_sec{seg_num}"
        clean_fpath = ""
        for clean in self.clean_df.iloc[:, 0]:
            if clean_signal in clean:
                clean_fpath = f"{self.root}{clean}"
        if(not clean_fpath):
            raise Exception("clean signal doesn't found")
        with open(clean_fpath, 'rb') as handle:
            clean_data = pickle.load(handle)
        with open(noise_fpath, 'rb') as handle:
            noise_data = pickle.load(handle)
        return clean_data, noise_data, snr



if __name__ == "__main__":
    dir_root = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split'
    csv_file_name = 'dataset.csv'
    snrs = ['3', '9']
    dataset = create_dataset(dir_root, csv_file_name)
    dataset.filter_by_snrs(snrs)


