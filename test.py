import torch
import torchaudio
from prep_data import choose_cuda

import sys
sys.path.append('/home/dsi/hazbanb/project/git/models')
from model_v5 import Model

from torch import nn
from reconstruction import Reconstruct
import os
import pickle


### Training function
def test(model, device, test_dataloader, loss_fn):
    test_loss = {}
    test_loss['loss_rec'] = []
    test_loss['loss_cod'] = []
    test_loss['loss_tot'] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, all_data in enumerate(test_dataloader):
            clean_seg, noise, _ = all_data
            clean_seg = clean_seg.to(device)
            noise = noise.to(device)
            noised_signal_abs = torch.abs(clean_seg + noise)
            clean_abs = torch.abs(clean_seg).to(device)
            coded_noisy,recon_data = model(noised_signal_abs)
            coded_clean = model.encoder.forward(clean_abs)
            # Evaluate loss
            loss_recon = loss_fn(recon_data, clean_abs)
            loss_codes = loss_fn(coded_clean, coded_noisy)
            loss_total = loss_recon + loss_codes
            test_loss['loss_rec'].append(loss_recon.item())
            test_loss['loss_cod'].append(loss_codes.item())
            test_loss['loss_tot'].append(loss_total.item())
        for loss in test_loss.keys():
            test_loss[loss] = sum(test_loss[loss]) / len(test_loss[loss])
    return test_loss

def back_to_wav(model, device, recon_dataloader, run_dir, tar_name):
    train_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/train_data_split'
    reconstruct = Reconstruct(device)
    model.eval()
    with torch.no_grad():
        for example in recon_dataloader:
            dir_num = example.split('_')[0]
            clean_path = os.path.join(train_dir, dir_num, example)
            noise_name = example.replace('stft', 'noise_stft').replace('clean', 'SNR6_db')
            noise_path = os.path.join(train_dir, dir_num, noise_name)
            with open(clean_path, 'rb') as handle:
                clean_stft = pickle.load(handle).unsqueeze(0)
            with open(noise_path, 'rb') as handle:
                noise_stft = pickle.load(handle).unsqueeze(0)
            clean_stft = clean_stft.to(device)
            noise_stft = noise_stft.to(device)
            noised_stft = clean_stft + noise_stft
            noised_signal = torch.view_as_real(noised_stft)
            noised_signal = torch.permute(noised_signal, (0, 3, 1, 2))
            noised_signal = noised_signal.to(device)

            filtered_signal = model(noised_signal)

            # if(griffin_lim):
            #     recon_wav = reconstruct.griffin_recon(filtered_signal).squeeze(0)
            #     clean_wav = reconstruct.griffin_recon(clean_abs).squeeze(0)
            #     recon_wav = recon_wav.detach().cpu()
            #     clean_wav = clean_wav.detach().cpu()
            #     recon_name = example.split('clean')[0] + 'recon_griffin.wav'
            #     recon_wav_path = os.path.join(run_dir, recon_name)
            #     torchaudio.save(recon_wav_path, recon_wav, 44100)
            #     clean_name = example.split('.pickle')[0] + '_griffin.wav'
            #     clean_wav_path = os.path.join(run_dir, clean_name)
            #     torchaudio.save(clean_wav_path, clean_wav, 44100)
            # else:
            #     noised_signal_phase = torch.angle(noised_signal).squeeze(0)
            #     recon_data = recon_data.squeeze(0)
            #     clean_abs = clean_abs.squeeze(0)
            #     reconstructed_stft = recon_data * torch.exp(1j * noised_signal_phase)
            #     clean_recon_stft = clean_abs * torch.exp(1j * noised_signal_phase)
            #     recon_wav = reconstruct.istft_recon(reconstructed_stft)
            #     clean_wav = reconstruct.istft_recon(clean_recon_stft)
            #     recon_wav = recon_wav.detach().cpu()
            #     clean_wav = clean_wav.detach().cpu()
            #     recon_name = example.split('clean')[0] + 'recon_noisy_phase.wav'
            #     recon_wav_path = os.path.join(run_dir, recon_name)
            #     torchaudio.save(recon_wav_path, recon_wav, 44100)
            #     clean_name = example.split('.pickle')[0] + '_noisy_phase.wav'
            #     clean_wav_path = os.path.join(run_dir, clean_name)
            #     torchaudio.save(clean_wav_path, clean_wav, 44100)

            filtered_signal = torch.permute(filtered_signal, (0, 2, 3, 1))
            filtered_signal = filtered_signal.contiguous()  # Ensure contiguous memory layout
            filtered_signal = torch.view_as_complex(filtered_signal)

            recon_wav = reconstruct.istft_recon(filtered_signal)
            clean_wav = reconstruct.istft_recon(clean_stft)
            noised_wav = reconstruct.istft_recon(noised_stft)
            recon_wav = recon_wav.detach().cpu()
            clean_wav = clean_wav.detach().cpu()
            noised_wav = noised_wav.detach().cpu()
            recon_name = f"{example.split('clean')[0]}_reconstruct_{tar_name}.wav"
            recon_wav_path = os.path.join(run_dir, recon_name)
            torchaudio.save(recon_wav_path, recon_wav, 44100)
            print(recon_wav_path)
            clean_name = example.split('.pickle')[0] + '.wav'
            clean_wav_path = os.path.join(run_dir, clean_name)
            torchaudio.save(clean_wav_path, clean_wav, 44100)
            noised_name = example.split('clean')[0] + 'noised.wav'
            noised_wav_path = os.path.join(run_dir, noised_name)
            torchaudio.save(noised_wav_path, noised_wav, 44100)

            # ### reconstruct original clean stft
            # org_clean_wav = reconstruct.istft_recon(clean_stft.squeeze(0))
            # org_clean_wav = org_clean_wav.detach().cpu()
            # org_clean_name = example.replace('pickle', 'wav')
            # org_clean_path = os.path.join(run_dir, org_clean_name)
            # torchaudio.save(org_clean_path, org_clean_wav, 44100)

            # ### reconstruct original noise stft
            # org_noise_wav = reconstruct.istft_recon(noise_stft.squeeze(0))
            # Ps = torch.mean(org_clean_wav ** 2)
            # Pn = torch.mean(org_noise_wav ** 2)
            # SNR = 10 * torch.log10(Ps / Pn)
            # print(SNR)

#############################################################################################################

if __name__ == "__main__":
    d = 1024
    cuda_num = 1
    batch_size = 128
    num_workers = 9

    #model_v4
    unet_depth = 1
    activation = nn.ELU()
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]

    model = Model(unet_depth, Ns, activation)

    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/2023-06-29 17:36:49.759766_densenet_model_30epochs_depth_16channels_batch8'
    tar_name = 'FinalModel.tar'
    checkpoint = torch.load(os.path.join(run_dir, tar_name))
    model.load_state_dict(checkpoint['model'])
    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model.to(device)

#    test_loss = test(model, device, test_dataloader, loss_fn)
#    print(f'{test_loss=}')

    recon_dataloader = ['2116_stft_sec51_clean.pickle', '2304_stft_sec34_clean.pickle', '2560_stft_sec53_clean.pickle', '1758_stft_sec102_clean.pickle']
    back_to_wav(model, device, recon_dataloader, run_dir, tar_name)