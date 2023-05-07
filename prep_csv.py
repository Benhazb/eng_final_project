import os
import re
import csv

class gen_csv():
    def __init__(self, root):
        self.root = root

    def prep_csv(self):
        csv_file = f'{self.root}/dataset.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            first_row = ['relative path', 'SNR']
            writer.writerow(first_row)
            for root, _, files in os.walk(self.root):
                for file in files:
                    if file.endswith('.pickle'):
                        dir = file.split('_')[0]
                        if(re.search(r'SNR(\d+)_db', file)):
                            SNR = re.search(r'SNR(\d+)_db', file).group(1)
                        elif('SNR-' in file):
                            SNR = '-' + file.split('-')[1].split('_')[0]
                        else:
                            SNR = 'clean'
                        relative_path = f"/{dir}/{file}"
                        curr_row = [relative_path, SNR]
                        writer.writerow(curr_row)

if __name__ == "__main__":
    train_root = f"/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split"
    csv_ins = gen_csv(train_root)
    csv_ins.prep_csv()
    csv_ins.root = f"/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/val_data_split"
    csv_ins.prep_csv()
    csv_ins.root = f"/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/test_data_split"
    csv_ins.prep_csv()