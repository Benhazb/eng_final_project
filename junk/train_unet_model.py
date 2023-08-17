import os.path
import csv
import matplotlib.pyplot as plt
import torch
#torch.autograd.set_detect_anomaly(True)
import datetime
from torch import nn
from prep_data import choose_cuda
import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')
import model_v5
import model_v4
from dataset import create_dataset
import wandb


### Training function
def train_epoch(model, device, train_dataloader, loss_fn, optimizer, short_run):
    # Set train mode for both the encoder and the decoder
    batch_loss_list = []
    model.train()
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, all_data in enumerate(train_dataloader):
        clean_seg, noise, dataset_snr = all_data

        # Move tensor to the proper device
        clean_seg = clean_seg.to(device)
        noise     = noise.to(device)

        # generate noised signal
        clean_seg_clone = clean_seg.clone()
        noised_signal   = clean_seg_clone + noise
        noised_signal   = torch.view_as_real(noised_signal)
        noised_signal   = torch.permute(noised_signal, (0,3,1,2))
        noised_signal   = noised_signal.to(device)
        noised_signal_clone = noised_signal.clone()

        optimizer.zero_grad()

        filtered_signal = model(noised_signal_clone)
        filtered_signal = torch.permute(filtered_signal, (0,2,3,1))
        filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
        filtered_signal = torch.view_as_complex(filtered_signal)

        recon_clean_abs = torch.abs(filtered_signal)
        clean_seg_abs = torch.abs(clean_seg)

        loss_batch = loss_fn(recon_clean_abs, clean_seg_abs)

        loss_batch.backward()
        optimizer.step()
        batch_loss_list.append(loss_batch.item())
        # short run
        if (batch_idx%1000 == 0):
            print(f'********************* train: {epoch=}___{batch_idx=}*********************')
        if short_run and batch_idx == 4:
            print(f'stopping at index {batch_idx}')
            break
    epoch_loss = sum(batch_loss_list)/len(batch_loss_list)
    return epoch_loss

def val_epoch(model, device, val_dataloader, loss_fn, short_run):
    batch_loss_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, all_data in enumerate(val_dataloader):
            clean_seg, noise, dataset_snr = all_data

            # Move tensor to the proper device
            clean_seg = clean_seg.to(device)
            noise = noise.to(device)

            # generate noised signal
            clean_seg_clone = clean_seg.clone()
            noised_signal = clean_seg_clone + noise
            noised_signal = torch.view_as_real(noised_signal)
            noised_signal = torch.permute(noised_signal, (0, 3, 1, 2))
            noised_signal = noised_signal.to(device)
            noised_signal_clone = noised_signal.clone()

            filtered_signal = model(noised_signal_clone)
            filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
            filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
            filtered_signal = torch.view_as_complex(filtered_signal)

            recon_clean_abs = torch.abs(filtered_signal)
            clean_seg_abs = torch.abs(clean_seg)

            loss_batch = loss_fn(recon_clean_abs, clean_seg_abs)
            batch_loss_list.append(loss_batch.item())
            if (batch_idx % 1000 == 0):
                print(f'********************* val: {epoch=}___{batch_idx=}*********************')
            # short run
            if short_run and batch_idx == 1:
                print(f'stopping at index {batch_idx}')
                break
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        return epoch_loss

def outputs(full_path, train_losses, val_losses, test_losses, batch_size, model, optim):
    csv_file = 'loss_values.csv'
    with open(os.path.join(full_path, csv_file), 'w', newline='') as file:
        writer = csv.writer(file)
        epoch_num = len(train_losses)
        epoch_col = ['epoch']
        train_loss_col = ['train_loss']
        val_loss_snr_m3 = ['val_loss (snr -3 dB)']
        val_loss_snr_0 = ['val_loss (snr 0 dB)']
        val_loss_snr_3 = ['val_loss (snr 3 dB)']
        val_loss_snr_6 = ['val_loss (snr 6 dB)']
        val_loss_snr_9 = ['val_loss (snr 9 dB)']
        val_loss_snr_12 = ['val_loss (snr 12 dB)']
        val_loss_snr_15 = ['val_loss (snr 15 dB)']
        for i in range(epoch_num):
            epoch_col.append(str(i))
            train_loss_col.append(train_losses[i])
            for snr in val_losses.keys():
                if snr == '-3':
                    val_loss_snr_m3.append(val_losses[snr][i])
                elif snr == '0':
                    val_loss_snr_0.append(val_losses[snr][i])
                elif snr == '3':
                    val_loss_snr_3.append(val_losses[snr][i])
                elif snr == '6':
                    val_loss_snr_6.append(val_losses[snr][i])
                elif snr == '9':
                    val_loss_snr_9.append(val_losses[snr][i])
                elif snr == '12':
                    val_loss_snr_12.append(val_losses[snr][i])
                else:
                    val_loss_snr_15.append(val_losses[snr][i])
        epoch_col.append('test')
        train_loss_col.append('')
        for snr in test_losses:
            if snr == '-3':
                val_loss_snr_m3.append(test_losses[snr][0])
            elif snr == '0':
                val_loss_snr_0.append(test_losses[snr][0])
            elif snr == '3':
                val_loss_snr_3.append(test_losses[snr][0])
            elif snr == '6':
                val_loss_snr_6.append(test_losses[snr][0])
            elif snr == '9':
                val_loss_snr_9.append(test_losses[snr][0])
            elif snr == '12':
                val_loss_snr_12.append(test_losses[snr][0])
            else:
                val_loss_snr_15.append(test_losses[snr][0])
        loss_png_name = 'loss_graphs.png'
        png_full_path = os.path.join(full_path, loss_png_name)
        fig, ax = plt.subplots()
        ax.plot(epoch_col[1:-1], train_loss_col[1:-1], color='red', label='train loss')
        ax.plot(epoch_col[1:-1], val_loss_snr_m3[1:-1], color='blue', label='val loss snr -3')
        ax.plot(epoch_col[1:-1], val_loss_snr_0[1:-1], color='green', label='val loss snr 0')
        ax.plot(epoch_col[1:-1], val_loss_snr_3[1:-1], color='yellow', label='val loss snr 3')
        ax.plot(epoch_col[1:-1], val_loss_snr_6[1:-1], color='grey', label='val loss snr 6')
        ax.plot(epoch_col[1:-1], val_loss_snr_9[1:-1], color='orange', label='val loss snr 9')
        ax.plot(epoch_col[1:-1], val_loss_snr_12[1:-1], color='purple', label='val loss snr 12')
        ax.plot(epoch_col[1:-1], val_loss_snr_15[1:-1], color='brown', label='val loss snr 15')
        ax.legend()
        fig.savefig(png_full_path)
        data = [epoch_col,train_loss_col,val_loss_snr_m3,val_loss_snr_0,val_loss_snr_3,
                val_loss_snr_6,val_loss_snr_9,val_loss_snr_12,val_loss_snr_15]
        data_transposed = zip(*data)
        for row in data_transposed:
            writer.writerow(row)
    filename = f"FinalModel"
    model_tar_path = f'{full_path}/{filename}'
    save_model(model, optim, epoch_num, batch_size, model_tar_path)

def save_model(model, optim, epoch_num, batch_size, full_path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'num_epoch': epoch_num,
        'batch_train': batch_size},  # end of parameters-to-be-saved list
        f"{full_path}.tar")


#############################################################################################################


if __name__ == "__main__":
    num_epochs = 30
    print(num_epochs)
    lr = 0.001
    #torch.manual_seed(0)
    cuda_num = 3
    batch_size = 16
    num_workers = 9
    unet_depth = 1
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    Ss = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    snrs = ['-3','0','3','6','9','12','15']

    # wandb login
    #wandb.login(key="797cdb53ac9ef72208838ecdbe0f153671b6c48a")
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="final_project",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "dense_net",
            "batch_size": batch_size,
            "epochs": num_epochs,
            "unet_depth": unet_depth,
            "max_channels": Ns[unet_depth],
        }
    )
    ############


    output_path = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs'
    run_name = f'densenet_model_{num_epochs}epochs_depth_{Ns[unet_depth]*2}channels_batch{batch_size}'
    dir_name = f'{datetime.datetime.now()}_{run_name}'
    full_path = os.path.join(output_path, dir_name)
    os.mkdir(full_path)


    # flags
    short_run = 1
    check_points = 0

    # create train dataloader
    train_dataset = create_dataset('train')
    train_dataset.filter_by_snrs(snrs)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

    val_dataloader_dict = {}
    for snr in snrs:
        val_dataset = create_dataset('val')
        val_dataset.filter_by_snrs([snr])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers)
        val_dataloader_dict[snr] = val_dataloader


    test_dataloader_dict = {}
    for snr in snrs:
        test_dataset = create_dataset('test')
        test_dataset.filter_by_snrs([snr])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                     shuffle=True, num_workers=num_workers)
        test_dataloader_dict[snr] = test_dataloader

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = model_v5.Model(unet_depth, Ns, activation)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model = model.to(device)
    epoch_train_losses = []
    epoch_val_losses = {}
    test_losses = {}

    for epoch in range(num_epochs):
        epoch_train_loss = train_epoch(model, device, train_dataloader, loss_fn, optim, short_run)
        epoch_train_losses.append(epoch_train_loss)
        print(f'epoch num {str(epoch)}:\t{epoch_train_loss=}')
        for val_dataloader in val_dataloader_dict.keys():
            if epoch == 0:
                epoch_val_losses[val_dataloader] = []
            epoch_val_loss = val_epoch(model, device, val_dataloader_dict[val_dataloader], loss_fn, short_run)
            epoch_val_losses[val_dataloader].append(epoch_val_loss)
        if ((epoch+1)%5) == 0 and check_points:
            path = f'{full_path}/{epoch+1}_epochs_checkpoint'
            save_model(model, optim, num_epochs, batch_size, path)


    for test_dataloader in test_dataloader_dict.keys():
        test_losses[test_dataloader] = []
        test_loss = val_epoch(model, device, test_dataloader_dict[test_dataloader], loss_fn, short_run)
        test_losses[test_dataloader].append(test_loss)

    outputs(full_path, epoch_train_losses, epoch_val_losses, test_losses, batch_size, model, optim)




