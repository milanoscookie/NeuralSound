import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt

class WaveletBlock1d(nn.Module):
    """
    Upgraded Wavelet Neural Operator block with Instance Normalization 
    and a local-phase skip connection.
    """
    def __init__(self, channels, level=4, wavelet='sym8', mlp_expansion=2):
        super().__init__()
        # self.in_channels = channels
        # self.out_channels = channels
        # self.modes = 64
        self.channels = channels
        self.level = level
        self.wavelet = wavelet
        
        # skip connections
        self.w = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        # wavelet linear transform
        self.weight_cA = nn.Conv1d(channels, channels, kernel_size=1)
        self.weight_cD = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1) for _ in range(level)
        ])
        
        # norm to handle large dynamic range in RF pulse values
        self.norm1 = nn.InstanceNorm1d(channels)
        self.norm2 = nn.InstanceNorm1d(channels)
        
        # 4. Point-wise Feedforward Network
        hidden_mlp_channels = channels * mlp_expansion
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_mlp_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_mlp_channels, channels, kernel_size=1)
        )

    def forward(self, x):
        # batch_size, _, nt = x.shape
        # spectrum = torch.fft.fft(x, dim=-1, norm="ortho")
        # pre norm -> better gradients
        x_norm = self.norm1(x)
        
        x_skip = self.w(x_norm)
        
        coeffs = ptwt.wavedec(x_norm, pywt.Wavelet(self.wavelet), level=self.level, mode='symmetric')
        cA, cDs = coeffs[0], coeffs[1:]
        
        cA_out = self.weight_cA(cA)
        cDs_out = [weight_layer(cD) for weight_layer, cD in zip(self.weight_cD, cDs)]
        
        coeffs_out = [cA_out] + cDs_out
        x_wavelet = ptwt.waverec(coeffs_out, pywt.Wavelet(self.wavelet))
        # out = torch.zeros(batch_size, self.channels, nt, dtype=torch.cfloat, device=x.device)
        

        # truncate
        x_wavelet = x_wavelet[..., :x_skip.shape[-1]]
            
        # res connection over operator
        x_operator = x + F.gelu(x_wavelet + x_skip)
        
        # feed forward 
        x_operator_norm = self.norm2(x_operator)
        x_out = x_operator + self.mlp(x_operator_norm)
        
        return x_out


class UltrasoundWNO1d(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, hidden_channels=64, num_blocks=4, level=4, wavelet='sym8', mlp_expansion=2):
        super().__init__()
        # self.fc0 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        # self.spec_convs = nn.ModuleList([...])
        # self.w_convs = nn.ModuleList([...])
        
        self.lift = nn.Conv1d(in_channels + 1, hidden_channels, kernel_size=1)
        
        # add the hidden channels
        self.blocks = nn.ModuleList([
            WaveletBlock1d(hidden_channels, level, wavelet, mlp_expansion) 
            for _ in range(num_blocks)
        ])
        
        self.project_1 = nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.project_2 = nn.Conv1d(hidden_channels // 2, out_channels, kernel_size=1)

    def get_grid(self, shape, device):
        # gen temporal grid [-1,1]
        batchsize, _, seq_len = shape
        grid = torch.linspace(-1, 1, seq_len, device=device)
        grid = grid.view(1, 1, seq_len).repeat(batchsize, 1, 1)
        return grid

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # x = self.fc0(x)
            
        # add temporal coords
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) #add time as an input dimention
            
        # lift
        x = self.lift(x)
        
        # wavelet block
        for block in self.blocks:
            x = block(x)
            # x = self.spec_convs[k](x) + self.w_convs[k](x)
            # x = F.gelu(x)
            
            # project
        x = F.gelu(self.project_1(x))
        x = self.project_2(x)
        
        return x
