import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class KWaveUltrasoundDataset(Dataset):
    def __init__(self, h5_filepath, split='train', lim=None, test_fraction=0.1, val_fraction=0.1, split_seed=0):
        super().__init__()
        self.split = split
        # bound time
        self.time_start = 0
        self.time_end = 200

        with h5py.File(h5_filepath, 'r') as f:
            source_split = split

            group = f[f'/{source_split}']

            total_samples = group['rx_rf'].shape[-1]

            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(total_samples)
            num_val = int(total_samples * val_fraction)
            num_test = int(total_samples * test_fraction)
            num_train = total_samples - num_val - num_test

            selected_indices = np.sort(perm)
            if split == "train":
                selected_indices = np.sort(perm[:num_train])
            elif split == "test":
                selected_indices = np.sort(perm[num_train+1:num_train+num_test])
            elif split == "val":
                selected_indices = np.sort(perm[num_train+num_test+1:])


            # create a limit to trunkcate database [:lim]
            if lim is None:
                pass
            else:
                selected_indices = selected_indices[:lim]

            num_samples = int(selected_indices.shape[0])

            self.selected_indices = selected_indices
            self.source_split = source_split

            nt_total = int(group['pulse_rf'].shape[0])

            # MATLAB gave us [N, Nx, Nz] -> we read (Nz, Nx, N)
            self.c_map = torch.tensor(group['c_map'][:, :, selected_indices], dtype=torch.float32).permute(2, 1, 0).contiguous()
            self.rho_map = torch.tensor(group['rho_map'][:, :, selected_indices], dtype=torch.float32).permute(2, 1, 0).contiguous()
            self.inc_mask = torch.tensor(group['inc_mask'][:, :, selected_indices], dtype=torch.float32).permute(2, 1, 0).contiguous()

            # MATLAB gave us [N, Nt], we read (Nt, N), and get (N, 1, Nt) for conv later
            self.pulse_rf = torch.tensor(
                group['pulse_rf'][self.time_start:self.time_end, selected_indices],
                dtype=torch.float32,
            ).permute(1, 0).unsqueeze(1).contiguous()

            # (N, 6)
            self.cond = torch.tensor(group['cond'][:, selected_indices], dtype=torch.float32).permute(1, 0).contiguous()

            # [N, Nt, Ne] -> (N, Ne, Nt)
            self.rx_rf = torch.tensor(
                group['rx_rf'][:, self.time_start:self.time_end, selected_indices],
                dtype=torch.float32,
            ).permute(2, 0, 1).contiguous()


            self.num_samples = self.rx_rf.shape[0]


        print(
            f"Loaded {self.num_samples} samples into RAM "
            f"with Nt slice [{self.time_start}:{self.time_end}] (Nt={self.time_end - self.time_start})."
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # slice to batch it (batch/n); memory optimzatid.
        return {
                'c_map': self.c_map[idx],
                'rho_map': self.rho_map[idx],
                'inc_mask': self.inc_mask[idx],
                'pulse_rf': self.pulse_rf[idx], 
                'cond': self.cond[idx],
                'rx_rf': self.rx_rf[idx]
                }
