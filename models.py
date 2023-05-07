from torch import nn
from prep_data import choose_cuda
import time

class AutoEncoder(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim=d)
        self.decoder = Decoder(encoded_space_dim=d)

    def forward(self, x):
        coded = self.encoder.forward(x)
        recon = self.decoder.forward(coded)
        return coded, recon


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, (25,12), stride=(5,2), padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, (11,5), stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, (8,6), stride=4, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 23 * 12, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, encoded_space_dim)
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
            nn.Linear(encoded_space_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 32 * 23 * 12),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 23, 12))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (8,6),
                               stride=4, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, (11,5), stride=2,
                               padding=0, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, (25,12), stride=(5,2),
                               padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # dataloader
    # add overfit funtion
    print("models")
#    auto_enco = AutoEncoder(d=1024)
#    device = choose_cuda(3)
#    time.sleep(20)
#    print("check the size")
#    auto_enco.decoder.to(device)
#    auto_enco.encoder.to(device)
#    time.sleep(20)


