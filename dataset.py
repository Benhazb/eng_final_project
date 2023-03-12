import pandas as pd
import pickle
import torch
from torch import nn
from prep_data import choose_cuda

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

#############################################################################################################
# encoder decoder
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 10, stride=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 10, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 10, stride=3, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 4 * 34, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32 * 4 * 34),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 4, 34))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 10,
                               stride=3, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 10, stride=3,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 10, stride=3,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

#############################################################################################################
### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, [clean_seg, noise,_] in enumerate(dataloader):
        # Move tensor to the proper device
        noised_signal_abs = abs(clean_seg + noise)
        clean_seg_phase = torch.angle(clean_seg)   ## phase need to be saved outside the loop?
        # send to device
        noised_signal_abs = noised_signal_abs.to(device)
        # Encode data
        encoded_data = encoder(noised_signal_abs.unsqueeze(1))
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, abs(clean_seg))   ## do we need the phase in this part?
        train_loss.append(loss)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        if batch_idx % 100 == 0:
            print('partial train loss (single batch): %f' % loss.data)
    batches_amount = len(dataloader)/256
    epoch_loss = sum(train_loss)/batches_amount
    return epoch_loss

#############################################################################################################


if __name__ == "__main__":
    dir_root = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split'
    csv_file_name = 'dataset.csv'
    snrs = ['3']
    train_dataset = create_dataset(dir_root, csv_file_name)
    train_dataset.filter_by_snrs(snrs)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()
    ### Define an optimizer (both for the encoder and the decoder!)
    lr = 0.001
    ### Set the random seed for reproducible results
    torch.manual_seed(0)
    ### Initialize the two networks
    d = 4
    encoder = Encoder(encoded_space_dim=d)
    decoder = Decoder(encoded_space_dim=d)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # connect to GPU
    device = choose_cuda(1)
    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(encoder, decoder, device, dataloader, loss_fn, optim)
        print('********************************************************************')
        print(f'epoch num {epoch} loss :  {epoch_loss}')




