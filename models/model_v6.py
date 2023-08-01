from torchinfo import summary
from prep_data import choose_cuda
import torchvision.transforms.functional as func
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, depth, Ns, activation):
        super().__init__()
        print('model_v6')
        self.activation = activation
        self.depth = depth
        self.Ns = Ns

        ksize=(7,7)
        self.paddings_1=get_paddings(ksize)
        self.conv2d_1 = nn.Sequential(
                            nn.Conv2d(in_channels=2,
                                out_channels=Ns[0],
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_1,
                                padding_mode='reflect'),
                            self.activation)

        self.unet_1 = Unet(self.depth, self.Ns)

        ksize=(3,3)
        self.paddings_2 = get_paddings(ksize)
        self.conv2d_2 = nn.Sequential(
                            nn.Conv2d(in_channels=Ns[0],
                                out_channels=2,
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_2,
                                padding_mode='reflect'),
                            self.activation
        )

        ksize = (1, 1)
        self.paddings_3 = get_paddings(ksize)
        self.conv2d_3 = nn.Sequential(
                            nn.Conv2d(in_channels=2,
                                out_channels=Ns[0],
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_3,
                                padding_mode='reflect'),
                            self.activation
        )

        self.conv2d_4 = nn.Sequential(
                            nn.Conv2d(in_channels=Ns[0],
                                out_channels=Ns[0],
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_3,
                                padding_mode='reflect'),
                            self.activation
        )

        self.unet_2 = Unet((self.depth-1), self.Ns[1:])

        ksize=(3,3)
        self.conv2d_5 = nn.Sequential(
                            nn.Conv2d(in_channels=Ns[1],
                                out_channels=2,
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_2,
                                padding_mode='reflect'),
                            self.activation
        )


    def forward(self, x):
        F_in_1 = self.conv2d_1(x)
        F_out_1 = self.unet_1(F_in_1)
        est_noise = self.conv2d_2(F_out_1)
        Y1 = torch.sub(x, est_noise)
        M0 = self.conv2d_4(F_out_1)
        M1 = self.conv2d_3(Y1)
        M1 = torch.sigmoid(M1)
        M = torch.mul(M0, M1)
        F_sam = torch.add(M, F_out_1)
        F_in_2 = torch.concat([F_sam,F_in_1], dim=1)
        F_out_2 = self.unet_2(F_in_2)
        Y2 = self.conv2d_5(F_out_2)
        return Y1, Y2

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, depth, Ns):
        super(Unet, self).__init__()
        self.depth = depth
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for i in range(self.depth):
            self.downs.append(DoubleConv(Ns[i], Ns[i+1]))

        # Up part of UNET
        for i in range(self.depth, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(
                    Ns[i]*2, Ns[i], kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(Ns[i]*2, Ns[i]))

        self.bottleneck = DoubleConv(Ns[self.depth], Ns[self.depth]*2)
        self.final_conv = nn.Conv2d(Ns[1], Ns[0], kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = func.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def get_paddings(K):
    #return (K[0]//2, K[0]//2 -(1- K[0]%2), K[1]//2, K[1]//2 -(1- K[1]%2), 0 ,0, 0, 0)
    return (K[0]//2, K[1]//2)  #only for odd symmetric kernel size

if __name__ == "__main__":
    #net_depth = 7
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    net_depth = 4

    activation = nn.ELU()
    cuda_num = 0
    device = choose_cuda(cuda_num)
    model = Model(net_depth, Ns, activation).to(device)  #

    # # --- Print model summary ---
    print(f"\n")

    col_names = ["input_size", "output_size", "num_params"]
    with torch.cuda.device(device):
        summary(model, input_size=[16,2,1025,215], col_names=col_names)  # set input_size to tuple for [audio, video]
    print("model_v2")
