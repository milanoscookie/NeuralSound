import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class UltrasoundIQDataset(Dataset):
    def __init__(self, h5_filepath, tx_key='tx_data', rx_key='rx_data', seq_len=256, 
                 split='train', lim=None, test_fraction=0.1, val_fraction=0.1, split_seed=0):

        super().__init__()
        self.filepath = h5_filepath
        self.tx_key = tx_key
        self.rx_key = rx_key
        self.seq_len = seq_len
        self.split = split
        
        # open file to read shapes
        with h5py.File(self.filepath, 'r') as f:
            tx_len = f[self.tx_key].shape[0]
            rx_len = f[self.rx_key].shape[0]

        # how many samples
        self.max_samples = min(tx_len, rx_len)
        total_batches = self.max_samples // self.seq_len

        # create splits based on available chunks
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(total_batches)
        num_val = int(total_batches * val_fraction)
        num_test = int(total_batches * test_fraction)
        num_train = total_batches - num_val - num_test

        selected_indices = np.sort(perm)
        if split == "train":
            selected_indices = np.sort(perm[:num_train])
        elif split == "test":
            selected_indices = np.sort(perm[num_train:num_train+num_test])
        elif split == "val":
            selected_indices = np.sort(perm[num_train+num_test:])

        # create a limit to truncate database [:lim]
        if lim is None:
            pass
        else:
            selected_indices = selected_indices[:lim]

        self.selected_indices = selected_indices
        self.num_samples = int(selected_indices.shape[0])

        print("loaded")


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # lwk for this part can we decimate a bit?
        actual_batch_idx = self.selected_indices[idx]
        
        start_idx = actual_batch_idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # loead in data on demand
        with h5py.File(self.filepath, 'r') as f:
            tx_slice = f[self.tx_key][start_idx:end_idx]
            rx_slice = f[self.rx_key][start_idx:end_idx]
            
        # Re, Im to complex tensor
        tx_tensor = self._to_complex_tensor(tx_slice)
        rx_tensor = self._to_complex_tensor(rx_slice)
        
        # slice to batch it
        return {
            self.tx_key: tx_tensor,
            self.rx_key: rx_tensor
        }

    def _to_complex_tensor(self, data_slice):
        complex_data = data_slice[..., 0] + 1j * data_slice[..., 1]
        return torch.tensor(complex_data, dtype=torch.complex64)