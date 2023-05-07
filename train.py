import os.path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
from torch import nn
from prep_data import choose_cuda
#from torchinfo import summary

from models import AutoEncoder
from dataset import create_dataset

### Training function
def train_epoch(model, device, train_dataloader, val_dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = {}
    train_loss['loss_rec'] = []
    train_loss['loss_cod'] = []
    train_loss['loss_tot'] = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, all_data in enumerate(train_dataloader):
        clean_seg, noise, _ = all_data
        # Move tensor to the proper device
        optimizer.zero_grad()
        clean_seg = clean_seg.to(device)
        noise = noise.to(device)
        noised_signal_abs = torch.abs(clean_seg + noise)
        coded_noisy,recon_data = model(noised_signal_abs)
        clean_abs = torch.abs(clean_seg).to(device)
        coded_clean = model.encoder.forward(clean_abs)
        # Evaluate loss
        loss_recon = loss_fn(recon_data, clean_abs)
        loss_codes = loss_fn(coded_clean, coded_noisy)
        # Backward pass
        loss_total = loss_recon + loss_codes
        loss_total.backward()
        optimizer.step()
        train_loss['loss_tot'].append(loss_total.item())
        train_loss['loss_rec'].append(loss_recon.item())
        train_loss['loss_cod'].append(loss_codes.item())
        # Print batch loss
        #if batch_idx % 1 == 0:
        #    print('partial train loss (single batch): %f' % loss.data)
    for loss in train_loss.keys():
        train_loss[loss] = sum(train_loss[loss])/len(train_loss[loss])
    # validation
    val_loss = {}
    val_loss['loss_rec'] = []
    val_loss['loss_cod'] = []
    val_loss['loss_tot'] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, all_data in enumerate(val_dataloader):
            clean_seg, noise, _ = all_data
            clean_abs = torch.abs(clean_seg).to(device)
            noise = noise.to(device)
            noised_signal_abs = torch.abs(clean_abs + noise)
            coded_noisy,recon_data = model(noised_signal_abs)
            coded_clean = model.encoder.forward(clean_abs)
            # Evaluate loss
            loss_recon = loss_fn(recon_data, clean_abs)
            loss_codes = loss_fn(coded_clean, coded_noisy)
            loss_total = loss_recon + loss_codes
            val_loss['loss_rec'].append(loss_recon.item())
            val_loss['loss_cod'].append(loss_codes.item())
            val_loss['loss_tot'].append(loss_total.item())
        for loss in val_loss.keys():
            val_loss[loss] = sum(val_loss[loss]) / len(val_dataloader)
    return train_loss, val_loss

def outputs(path, loss_values, batch_size, name):
    dir_name = f'{datetime.datetime.now()}_{name}_b{batch_size}'
    full_path = os.path.join(path, dir_name)
    os.mkdir(full_path)
    csv_file = 'loss_values.csv'
    with open(os.path.join(full_path, csv_file), 'w', newline='') as file:
        writer = csv.writer(file)
        epoch_num = len(loss_values['epoch'])
        epoch_col = ['epoch']
        train_loss_tot_col = ['train_loss_tot']
        train_loss_rec_col = ['train_loss_rec']
        train_loss_cod_col = ['train_loss_cod']
        val_loss_tot_col = ['val_loss_tot']
        val_loss_rec_col = ['val_loss_rec']
        val_loss_cod_col = ['val_loss_cod']
        for i in range(epoch_num):
            for key in loss_values.keys():
                if key == 'train':
                    for loss in loss_values[key][i].keys():
                        if loss == 'loss_tot':
                            train_loss_tot_col.append(loss_values[key][i]['loss_tot'])
                        elif loss == 'loss_rec':
                            train_loss_rec_col.append(loss_values[key][i]['loss_rec'])
                        elif loss == 'loss_cod':
                            train_loss_cod_col.append(loss_values[key][i]['loss_cod'])
                elif key == 'val':
                    for loss in loss_values[key][i].keys():
                        if loss == 'loss_tot':
                            val_loss_tot_col.append(loss_values[key][i]['loss_tot'])
                        elif loss == 'loss_rec':
                            val_loss_rec_col.append(loss_values[key][i]['loss_rec'])
                        elif loss == 'loss_cod':
                            val_loss_cod_col.append(loss_values[key][i]['loss_cod'])
                else:
                    epoch_col.append(loss_values[key][i])

        loss_png_name = 'loss_graphs.png'
        png_full_path = os.path.join(full_path, loss_png_name)
        fig, ax = plt.subplots()
        ax.plot(loss_values['epoch'], train_loss_tot_col[1:], color='red', label='train loss')
        ax.plot(loss_values['epoch'], val_loss_tot_col[1:], color='blue', label='val loss')
        ax.legend()
        fig.savefig(png_full_path)

        data = [epoch_col, train_loss_tot_col, train_loss_rec_col, train_loss_cod_col, val_loss_tot_col, val_loss_rec_col, val_loss_cod_col]
        data_transposed = zip(*data)
        for row in data_transposed:
            writer.writerow(row)
    return full_path

#############################################################################################################


if __name__ == "__main__":
    num_epochs = 30
    lr = 0.001
    #torch.manual_seed(0)
    d = 1024
    cuda_num = 0
    batch_size = 128
    num_workers = 9
    #add lambda

    # create train dataloader
    snrs = ['6', '9', '12']
    train_dataset = create_dataset('train')
    train_dataset.filter_by_snrs(snrs)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # create validation dataloader
    val_dataset = create_dataset('val')
    val_dataset.filter_by_snrs(snrs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # create test dataloader
    test_dataset = create_dataset('test')
    test_dataset.filter_by_snrs(snrs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = AutoEncoder(d=d)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model.to(device)
    loss_dict = {}
    loss_dict['epoch'] = []
    loss_dict['train'] = []
    loss_dict['val'] = []
    for epoch in range(num_epochs):
        train_loss, val_loss = train_epoch(model, device, train_dataloader, val_dataloader, loss_fn, optim)
        loss_dict['epoch'].append(epoch)
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        print(f'epoch num {str(epoch)}:\t{train_loss["loss_tot"]=}\t{val_loss["loss_tot"]=}')

    path = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs'
    name = 'sanity_check'
    dir_full_path = outputs(path, loss_dict, batch_size, name)


    filename = f"FinalModel"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'num_epoch': num_epochs,
        'batch_train': batch_size},  # end of parameters-to-be-saved list
        f"{dir_full_path}/{filename}.tar")
