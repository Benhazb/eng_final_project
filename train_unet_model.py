import os.path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
from torch import nn
from prep_data import choose_cuda
import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')

from model_v2 import AutoEncoder
from dataset import create_dataset

### Training function
def train_epoch(model, device, train_dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    batch_loss_list = []
    model.train()
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, all_data in enumerate(train_dataloader):
        clean_seg, noise, dataset_snr = all_data
        optimizer.zero_grad()
        # Move tensor to the proper device
        clean_seg = clean_seg.to(device)
        noise = noise.to(device)
        noised_signal = clean_seg + noise
        noised_signal = torch.view_as_real(noised_signal)
        noised_signal = torch.permute(noised_signal, (0,3,1,2))
        noised_signal = noised_signal.to(device)
        noise_pred_1 = model(noised_signal)
        recon_clean = noised_signal - noise_pred_1
        recon_clean = torch.permute(recon_clean, (0,2,3,1))
        recon_clean = torch.view_as_complex(recon_clean)
        recon_clean_abs = torch.abs(recon_clean)
        clean_seg_abs = torch.abs(clean_seg)
        loss = loss_fn(recon_clean_abs, clean_seg_abs)
        loss.backward()
        optimizer.step()
        batch_loss_list.append(loss.item())
    epoch_loss = sum(batch_loss_list)/len(batch_loss_list)
    return epoch_loss

def outputs(path, loss_list, batch_size, name, model, optim):
    dir_name = f'{datetime.datetime.now()}_{name}_b{batch_size}'
    full_path = os.path.join(path, dir_name)
    os.mkdir(full_path)
    csv_file = 'loss_values.csv'
    with open(os.path.join(full_path, csv_file), 'w', newline='') as file:
        writer = csv.writer(file)
        epoch_num = len(loss_list)
        epoch_col = ['epoch']
        train_loss_col = ['train_loss']
        for i in range(epoch_num):
            epoch_col.append(str(i))
            train_loss_col.append(loss_list[i])
        loss_png_name = 'loss_graphs.png'
        png_full_path = os.path.join(full_path, loss_png_name)
        fig, ax = plt.subplots()
        ax.plot(epoch_col[1:], train_loss_col[1:], color='red', label='train loss')
        ax.legend()
        fig.savefig(png_full_path)
        data = [epoch_col, train_loss_col]
        data_transposed = zip(*data)
        for row in data_transposed:
            writer.writerow(row)
    filename = f"FinalModel"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'num_epoch': epoch_num,
        'batch_train': batch_size},  # end of parameters-to-be-saved list
        f"{full_path}/{filename}.tar")





#############################################################################################################


if __name__ == "__main__":
    num_epochs = 30
    lr = 0.001
    #torch.manual_seed(0)
    cuda_num = 0
    batch_size = 8
    val_batch_size = 32
    num_workers = 9
    unet_depth = 4
    activation = nn.ELU
    Ns = [4, 8, 16, 16, 32, 64, 64, 128]
    Ss = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    num_tfc = 3
    snrs = ['6']

    # create train dataloader
    train_dataset = create_dataset('train')
    train_dataset.filter_by_snrs(snrs)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = AutoEncoder(unet_depth, activation, Ns, Ss, num_tfc)  #

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model = model.to(device)
    epoch_loss_list = []
    for epoch in range(num_epochs):
        epoch_train_loss = train_epoch(model, device, train_dataloader, loss_fn, optim)
        epoch_loss_list.append(epoch_train_loss)
        print(f'epoch num {str(epoch)}:\t{epoch_train_loss=}')


    path = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs'
    name = 'model_v2_sanity_check'
    outputs(path, epoch_loss_list, batch_size, name, model, optim)




