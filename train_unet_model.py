import os.path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
from torch import nn
from prep_data import choose_cuda

from models import AutoEncoderUnet
from dataset import create_dataset

### Training function
def train_epoch(model, device, train_dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, all_data in enumerate(train_dataloader):
        clean_seg, noise, dataset_snr = all_data
        optimizer.zero_grad()
        # Move tensor to the proper device
        clean_seg = clean_seg.to(device)
        noise = noise.to(device)
        noised_signal = clean_seg + noise
        real_mtx = torch.real(noised_signal)
        img_mtx = torch.imag(noised_signal)
        noised_signal = torch.concat([real_mtx,img_mtx], dim=(1))
        noised_signal = noised_signal.to(device)
        model(noised_signal)

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
    snrs = ['-3', '0', '3', '6', '9', '12', '15']

    # create train dataloader
    train_dataset = create_dataset('train')
    train_dataset.filter_by_snrs(snrs)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = AutoEncoderUnet(unet_depth, activation, Ns, Ss, num_tfc)  #

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model = model.to(device)
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, train_dataloader, loss_fn, optim)
