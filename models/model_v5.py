import math
from torchinfo import summary
from prep_data import choose_cuda
import torchvision.transforms.functional as func
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, depth, Ns, activation):
        super().__init__()
        print('model_v5')
        self.activation = activation
        self.depth = depth
        self.Ns = Ns
        self.densenet_depth = 3

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

        self.unet = Unet(self.depth, self.Ns, self.densenet_depth)

        ksize=(3,3)
        self.paddings_2=get_paddings(ksize)
        self.conv2d_2 = nn.Sequential(
                            nn.Conv2d(in_channels=Ns[0],
                                out_channels=2,
                                kernel_size=ksize,
                                stride=1,
                                padding=self.paddings_2,
                                padding_mode='reflect'),
                            self.activation
        )

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.unet(x)
        pred_feats_s1 = self.conv2d_2(x)
        return pred_feats_s1

class DenseBlock(nn.Module):
    def __init__(self, N_in, N_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.N_in = N_in
        self.N_out = N_out
        self.H = nn.ModuleList()
        self.activation = nn.ELU()
        self.paddings_1 = get_paddings((3,3))

        for i in range(num_layers):
            self.H.append(nn.Sequential(
                            nn.Conv2d(in_channels=i*N_out+N_in,
                                out_channels=N_out,
                                kernel_size=(3,3),
                                stride=1,
                                padding=self.paddings_1,
                                padding_mode='reflect'),
                            self.activation)
            )

    def forward(self, x):
        x_ = self.H[0](x)
        if self.num_layers>1:
            for h in self.H[1:]:
                x = torch.concat([x_, x], dim=1)
                x_ = h(x)
        return x_

class Unet(nn.Module):
    def __init__(self, depth, Ns, densenet_depth):
        super(Unet, self).__init__()
        self.depth = depth
        self.densenet_depth = densenet_depth
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for i in range(self.depth):
            self.downs.append(DenseBlock(Ns[i], Ns[i+1], self.densenet_depth))

        # Up part of UNET
        for i in range(self.depth, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(
                    Ns[i]*2, Ns[i], kernel_size=2, stride=2,
                )
            )
            self.ups.append(DenseBlock(Ns[i]*2, Ns[i], self.densenet_depth))

        self.bottleneck = DenseBlock(Ns[self.depth], Ns[self.depth]*2, self.densenet_depth)
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
    net_depth = 4
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    activation = nn.ELU()
    cuda_num = 2
    device = choose_cuda(cuda_num)
    model = Model(net_depth, Ns, activation).to(device)  #
    loss_fn = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    model.train()
    # for i in range(3):
    #     clean = torch.randn([5, 1025, 215], dtype=torch.cfloat).to(device)
    #     noise = torch.randn([5, 1025, 215], dtype=torch.cfloat).to(device)
    #     clean_clone = clean.clone().requires_grad_()
    #
    #     print(f'********index  {i}*******')
    #     noised_signal = clean_clone + noise
    #     noised_signal   = torch.view_as_real(noised_signal)
    #     noised_signal   = torch.permute(noised_signal, (0,3,1,2))
    #     noised_signal   = noised_signal.to(device)
    #     noised_signal_clone = noised_signal.clone().requires_grad_()
    #
    #     optim.zero_grad()
    #
    #     filtered_signal = model(noised_signal_clone)
    #     filtered_signal = torch.permute(filtered_signal, (0,2,3,1))
    #     filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
    #     filtered_signal = torch.view_as_complex(filtered_signal)
    #
    #     filtered_signal = filtered_signal.abs()
    #     clean_abs = clean.abs()
    #     loss_batch = loss_fn(filtered_signal, clean_abs)
    #
    #     loss_batch.backward()
    #
    #     optim.step()

    # # --- Print model summary ---
    print(f"\n")

    col_names = ["input_size", "output_size", "num_params"]
    with torch.cuda.device(device):
        summary(model, input_size=[32,2,1025,215], col_names=col_names)  # set input_size to tuple for [audio, video]
    print("model_v2")
