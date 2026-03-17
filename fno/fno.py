import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_gelu(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(F.gelu(z.real), F.gelu(z.imag))


class ComplexDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return z
        keep = 1.0 - self.p
        mask = (torch.rand_like(z.real) < keep).to(z.real.dtype) / keep
        return torch.complex(z.real * mask, z.imag * mask)


class ComplexConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, bias: bool = True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, bias=bias)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, bias=bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a, b = z.real, z.imag
        real = self.conv_r(a) - self.conv_i(b)
        imag = self.conv_i(a) + self.conv_r(b)
        return torch.complex(real, imag)


class SpectralConv1dComplex(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weight_real = nn.Parameter(torch.empty(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(torch.empty(in_channels, out_channels, modes))
        self.weight_real_neg = nn.Parameter(torch.empty(in_channels, out_channels, modes - 1))
        self.weight_imag_neg = nn.Parameter(torch.empty(in_channels, out_channels, modes - 1))
        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / (self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight_real, -scale, scale)
        nn.init.uniform_(self.weight_imag, -scale, scale)
        nn.init.uniform_(self.weight_real_neg, -scale, scale)
        nn.init.uniform_(self.weight_imag_neg, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, nt = x.shape
        spectrum = torch.fft.fft(x, dim=-1, norm="ortho")

        modes = min(self.modes, nt // 2)
        out = torch.zeros(batch_size, self.out_channels, nt, dtype=torch.cfloat, device=x.device)

        # only save k modes
        weight_pos = torch.complex(self.weight_real[..., :modes], self.weight_imag[..., :modes])
        out[:, :, :modes] = torch.einsum("bim,iom->bom", spectrum[:, :, :modes], weight_pos)

        modes_neg = min(modes - 1, nt // 2 - 1)
        if modes_neg > 0:
            weight_neg = torch.complex(self.weight_real_neg[..., :modes_neg], self.weight_imag_neg[..., :modes_neg])
            out[:, :, -modes_neg:] = torch.einsum("bim,iom->bom", spectrum[:, :, -modes_neg:], weight_neg)

        return torch.fft.ifft(out, dim=-1, norm="ortho")


class FNO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 128,
        modes: int = 64,
        depth: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.fc0 = ComplexConv1d(in_channels, width, kernel_size=1)

        self.spec_convs = nn.ModuleList([SpectralConv1dComplex(width, width, modes) for _ in range(depth)])
        self.w_convs = nn.ModuleList([ComplexConv1d(width, width, kernel_size=1) for _ in range(depth)])
        self.drop = ComplexDropout(dropout) if dropout > 0 else nn.Identity()

        self.fc1 = ComplexConv1d(width, width, kernel_size=1)
        self.fc2 = ComplexConv1d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        for index in range(len(self.spec_convs)):
            x = self.spec_convs[index](x) + self.w_convs[index](x)
            x = complex_gelu(x)
            x = self.drop(x)
        x = complex_gelu(self.fc1(x))
        return self.fc2(x)
