import torch
import torchaudio
from prep_data import choose_cuda
from models import AutoEncoder
from dataset import create_dataset
import os


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
            clean_abs = torch.abs(clean_seg).to(device)
            noise = noise.to(device)
            noised_signal_abs = torch.abs(clean_abs + noise)
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
            test_loss[loss] = sum(test_loss[loss]) / len(test_dataloader)
    return test_loss

def back_to_wav(model, device, recon_dataloader, run_dir):
    window_length = 2048
    hop_length = 1024
    window = torch.hann_window(window_length=window_length, device=device)
    #griffin = torchaudio.transforms.GriffinLim(n_fft=window_length, window_length=window_length, hop_length=hop_length)
    griffin = torchaudio.transforms.GriffinLim(n_fft=window_length)
    griffin = griffin.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, all_data in enumerate(recon_dataloader):
            if batch_idx <= 5:
                clean_seg, noise, _ = all_data
                clean_abs = torch.abs(clean_seg).to(device)
                noise = noise.to(device)
                noised_signal_abs = torch.abs(clean_abs + noise)
                _ , recon_data = model(noised_signal_abs)
                wav_audio = griffin(recon_data)
                wav_clean = griffin(clean_abs)
                print(wav_audio.shape)
                print(wav_clean.shape)
                wav_audio = wav_audio.detach().cpu().squeeze(0)
                wav_clean = wav_clean.detach().cpu().squeeze(0)
                torchaudio.save(os.path.join(run_dir, f'exmple_{batch_idx}_recon.wav'), wav_audio, 44100)
                torchaudio.save(os.path.join(run_dir, f'exmple_{batch_idx}_origin.wav'), wav_clean, 44100)
            else:
                break


#############################################################################################################

if __name__ == "__main__":
    d = 1024
    cuda_num = 0
    batch_size = 128
    num_workers = 9

    # create test dataloader
    snrs = ['6', '9', '12']
    test_dataset = create_dataset('val')
    test_dataset.filter_by_snrs(snrs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = AutoEncoder(d=d)

    run_dir = '/dsi/scratch/from_netapp/users/hazbanb/dataset/musicnet/outputs/2023-05-02 03:58:01.301571_sanity_check_b128'
    tar_name = 'FinalModel.tar'
    checkpoint = torch.load(os.path.join(run_dir, tar_name))
    model.load_state_dict(checkpoint['model'])
    # connect to GPU
    device = choose_cuda(cuda_num)
    # Move both the encoder and the decoder to the selected device
    model.to(device)

    test_loss = test(model, device, test_dataloader, loss_fn)
    print(f'{test_loss=}')

    batch_size = 1
    snrs = ['6', '9', '12']
    recon_dataset = create_dataset('train')
    recon_dataset.filter_by_snrs(snrs)
    recon_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
    back_to_wav(model, device, recon_dataloader, run_dir)