import math
from torchinfo import summary
from prep_data import choose_cuda
import torchvision.transforms.functional as func
import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, depth, activation, Ns, Ss, num_tfc):
        super().__init__()
        self.depth = depth
        self.activation = activation
        self.Ns = Ns
        self.Ss = Ss
        self.num_tfc = num_tfc

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

        self.encoder_s1=Encoder(self.depth, self.activation, self.Ns, self.Ss, self.num_tfc)
        self.decoder_s1=Decoder(self.depth, self.activation, self.Ns, self.Ss, self.num_tfc)

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
        x, contracting_layers_s1 = self.encoder_s1(x)
        feats_s1 = self.decoder_s1(x, contracting_layers_s1)
        pred_feats_s1 = self.conv2d_2(feats_s1)
        return pred_feats_s1


class Encoder(nn.Module):
        def __init__(self, depth, activation, Ns, Ss, num_tfc):
            super().__init__()
            self.depth = depth
            self.activation = activation
            self.Ns = Ns
            self.Ss = Ss
            self.num_tfc = num_tfc
            self.contracting_layers = []
            self.eblocks = nn.ModuleList()

            for i in range(self.depth):
                self.eblocks.append(E_Block(N0=self.Ns[i],
                                            N=self.Ns[i+1],
                                            S=self.Ss[i],
                                            activation=self.activation,
                                            num_tfc=self.num_tfc))

            self.i_block = I_Block(self.Ns[self.depth], self.Ns[self.depth], self.activation, self.num_tfc)
        def forward(self, x):
            for i in range(self.depth):
                x, x_contract = self.eblocks[i](x)
                self.contracting_layers.append(x_contract)
            x = self.i_block(x)
            return x, self.contracting_layers

class Decoder(nn.Module):
    def __init__(self, depth, activation, Ns, Ss, num_tfc):
        super().__init__()
        self.depth = depth
        self.activation = activation
        self.Ns = Ns
        self.Ss = Ss
        self.num_tfc = num_tfc
        self.dblocks = nn.ModuleList()

        for i in range(self.depth,0,-1):
            self.dblocks.append(D_Block(N0=self.Ns[i], N=self.Ns[i-1], S=self.Ss[i],
                                        activation=self.activation, num_tfc=self.num_tfc))

    def forward(self, x, contracting_layers):
        for i in range(self.depth):
            x = self.dblocks[i](x, contracting_layers[self.depth-1-i])
        return x



class E_Block(nn.Module):
    def __init__(self, N0, N, S, activation, num_tfc):
        super().__init__()
        self.N0 = N0
        self.N = N
        self.S = S
        self.activation = activation
        self.num_tfc = num_tfc
        self.i_block = I_Block(N0, N0, activation, num_tfc)

        # ksize = (S[0]+3, S[1]+3)
        # self.paddings_2 = get_paddings(ksize)
        # self.conv2d_2 = nn.Sequential(
        #                     nn.Conv2d(in_channels=N0,
        #                         out_channels=N,
        #                         kernel_size=ksize,
        #                         stride=S,
        #                         padding=self.paddings_2,
        #                         padding_mode='reflect'),
        #                     self.activation)

        # ksize = (S[0] + 2, S[1] + 2)
        # self.conv2d_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=N0,
        #               out_channels=N,
        #               kernel_size=ksize,
        #               stride=S),
        #     self.activation)

        ksize = (S[0] + 2, S[1] + 2)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=ksize, stride=S),
            nn.Conv2d(in_channels=N0, out_channels=N, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.i_block(x)
        # x_down = self.conv2d_2(x)
        # x_pad = nn.functional.pad(x, self.paddings_2, mode='constant')
        x_down1 = self.pool(x)
        return x_down1, x

class D_Block(nn.Module):
    def __init__(self, N0, N, S, activation, num_tfc):
        super().__init__()
        self.N0 = N0
        self.N = N
        self.S = S
        self.activation = activation
        self.num_tfc = num_tfc

        # ksize = (S[0] + 3, S[1] + 3)
        # self.paddings_1 = get_paddings(ksize)
        # self.tconv_1 = nn.Sequential(
        #                 nn.ConvTranspose2d(in_channels=N0,
        #                                 out_channels=N,
        #                                 kernel_size=ksize,
        #                                 stride=S,
        #                                 padding=self.paddings_1,
        #                                 padding_mode='zeros'),
        #                 self.activation)

        ksize = (S[0]+2, S[1]+2)
        self.tconv_1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=N0,
                                        out_channels=N,
                                        kernel_size=ksize,
                                        stride=S),
                        self.activation)

        self.projection = nn.Sequential(
                            nn.Conv2d(in_channels=N0,
                                out_channels=N,
                                kernel_size=(1,1),
                                stride=1),
                            self.activation)
        self.i_block = I_Block(2*N, N, activation, num_tfc)
    def upsampling(self, input):
        return nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x, bridge):
        x1 = self.tconv_1(x)
        x2 = self.upsampling(x)
        if x2.shape[1]!=x1.shape[1]:
            x2= self.projection(x2)
        # if((x1.shape[2] >= x2.shape[2]) and (x1.shape[3] >= x2.shape[3])):
        #     x2 = func.resize(x2, size=x1.shape[2:])
        # else:
        #     x1 = func.resize(x1, size=x2.shape[2:])
        # x = torch.add(x1, x2)
        if(x2.shape != bridge.shape):
            x = func.resize(x2, size=bridge.shape[2:])
        x = torch.concat([x, bridge], dim=1)
        x = self.i_block(x)
        return x

class I_Block(nn.Module):
    def __init__(self, N_in, N_out, activation, num_tfc):
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.activation = activation
        self.num_tfc = num_tfc

        ksize = (3, 3)
        self.tfc = DenseBlock(num_tfc,self.N_in, self.N_out, ksize, activation)
        self.conv2d_res= nn.Conv2d(in_channels=self.N_in,
                                out_channels=self.N_out,
                                kernel_size=(1,1),
                                stride=1)

    def forward(self, x):
        x_tfc = self.tfc(x)
        inputs_proj = self.conv2d_res(x)
        return torch.add(x_tfc,inputs_proj)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, N_in, N_out, ksize, activation):
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.N_in = N_in
        self.N_out = N_out
        self.ksize = ksize
        self.H = nn.ModuleList()
        self.paddings_1 = get_paddings(ksize)

        for i in range(num_layers):
            self.H.append(nn.Sequential(
                            nn.Conv2d(in_channels=i*N_out+N_in,
                                out_channels=N_out,
                                kernel_size=ksize,
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

def get_paddings(K):
    #return (K[0]//2, K[0]//2 -(1- K[0]%2), K[1]//2, K[1]//2 -(1- K[1]%2), 0 ,0, 0, 0)
    return (K[0]//2, K[1]//2)  #only for odd symmetric kernel size

if __name__ == "__main__":
    unet_depth = 2
    activation = nn.ELU()
    Ns = [4, 8, 16, 16, 32, 64, 64, 128]
    Ss = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    num_tfc = 3

    cuda_num = 1
    device = choose_cuda(cuda_num)
    model = AutoEncoder(unet_depth, activation, Ns, Ss, num_tfc).to(device)  #
    loss_fn = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)

    model.train()
    for i in range(3):

        clean = torch.randn([5, 1025, 215], dtype=torch.cfloat).to(device)
        noise = torch.randn([5, 1025, 215], dtype=torch.cfloat).to(device)
        clean_clone = clean.clone().requires_grad_()

        print(f'********index  {i}*******')
        noised_signal = clean_clone + noise
        noised_signal   = torch.view_as_real(noised_signal)
        noised_signal   = torch.permute(noised_signal, (0,3,1,2))
        noised_signal   = noised_signal.to(device)
        noised_signal_clone = noised_signal.clone().requires_grad_()

        optim.zero_grad()

        filtered_signal = model(noised_signal_clone)
        filtered_signal = torch.permute(filtered_signal, (0,2,3,1))
        filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
        filtered_signal = torch.view_as_complex(filtered_signal)

        filtered_signal = filtered_signal.abs()
        clean_abs = clean.abs()
        loss_batch = loss_fn(filtered_signal, clean_abs)

        loss_batch.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}')
                print(f'Gradient: {param.grad}')
                print(f'Value: {param.data}')
                print('------------------------')

        optim.step()




