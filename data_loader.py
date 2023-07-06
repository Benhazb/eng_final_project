import argparse
import torchaudio
import torch

class pack_prep():
    def __init__(self, path, window_size, hop):
        self.path = path
        self.window_size = window_size
        self.hop = hop

    def pad_zero(self):
        self.path
def main():
    audio = get_audio()
    x, fs = torchaudio.load(audio)
    if x.dim() > 1:
        x = torch.mean(x, dim=0)
    x_p = pad_signal(x, 2048, 10, 4, 0.5)



def pad_signal(signal, window_size, window_amount, block_amount, overlap=0.5):
    hop_size = 512
    net_size = window_size + hop_size * (window_amount-1)
    mod = len(signal) % net_size
    num_of_zeros = net_size - mod
    padded_signal = torch.nn.functional.pad(signal, (0,num_of_zeros), mode='constant', value=0)
    padded_signal = torch.nn.functional.pad(padded_signal, (net_size,0), mode='constant', value=0)
    blocks_num = int(len(padded_signal)/net_size)
    if blocks_num % 2 == 0:
        padded_signal = torch.nn.functional.pad(padded_signal, (0, net_size), mode='constant', value=0)
    else:
        padded_signal = torch.nn.functional.pad(padded_signal, (0, 2*net_size), mode='constant', value=0)
    print(padded_signal)



############################################
##
############################################
if __name__ == "__main__":
    main()
    first_auodio_file = pack_prep(path, size, hop)
