import pandas as pd



class create_dataset():
    def __init__(self, root, csv_name, snrs):
        self.root = root
        self.csv_name = csv_name

    def filter_by_snrs(self, snrs):
        for snr in snrs:
            print(snr)



if __name__ == "__main__":
    dir_root = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split'
    csv_file_name = 'dataset.csv'
    snrs = [3, 9]

    dataset = create_dataset(dir_root, csv_file_name, snrs)
    dataset.filter_by_snrs(snrs)
