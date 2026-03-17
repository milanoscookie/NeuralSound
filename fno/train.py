#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset_iq import UltrasoundIQDataset
from fno import FNO1dComplex



def complex_time_mse(pred, targ):
    return torch.mean(torch.abs(pred - targ) ** 2)


def complex_spec_mse(pred, targ):
    pred_fft = torch.fft.fft(pred, dim=-1, norm="ortho")
    targ_fft = torch.fft.fft(targ, dim=-1, norm="ortho")
    return torch.mean(torch.abs(pred_fft - targ_fft) ** 2)


def wphase_loss(pred, targ, eps=1e-8):
    pred_u = pred / (pred.abs() + eps)
    targ_u = targ / (targ.abs() + eps)
    weight = targ.abs()
    weight = weight / (weight.mean(dim=-1, keepdim=True) + eps)
    weight = weight.clamp(0.0, 5.0)

    return torch.mean(pred_u.abs() * weight * (1.0 - (pred_u * targ_u.conj()).real))


def beamforming_loss(pred, targ, w_time=0.10, w_spec=0.10, w_phase=0.80):
    return w_time * complex_time_mse(pred, targ) + w_spec * complex_spec_mse(pred, targ) + w_phase * wphase_loss(pred, targ)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    time_mse_vals = []
    rel_rmse_vals = []
    phase_deg_vals = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model(x)

        mse = torch.mean(torch.abs(yhat - y) ** 2)
        time_mse_vals.append(mse.item())

        y_pow = torch.mean(torch.abs(y) ** 2)
        rel = torch.sqrt(mse / (y_pow + 1e-12))
        rel_rmse_vals.append(rel.item())

        # # only compute phase error for non high magnitude freqs
        # threshold = 0.1 * y.abs().amax(dim=-1, keepdim=True)
        # mask = y.abs() > threshold
        # dphi = torch.angle(yhat) - torch.angle(y)
        # dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))

        ph = torch.sqrt(torch.mean((dphi[mask]) ** 2)) * (180.0 / np.pi)
        phase_deg_vals.append(ph.item())

    return {
        "time_mse": float(np.mean(time_mse_vals)),
        "rel_rmse": float(np.mean(rel_rmse_vals)),
        "phase_deg": float(np.mean(phase_deg_vals))
    }


def train(
    h5_path,
    output_path="fno_model.pt",
    batch_size=32,
    epochs=120,
    learning_rate=1e-3,
    weight_decay=3e-5,
    num_workers=4,
    device_name="cuda",
    modes=64,
    width=192,
    depth=4,
    dropout=0.05,
    grad_clip=1.0,
):
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    train_dataset = UltrasondIQDataset(h5_path, split="train", normalize=True)
    val_dataset = UltrasondIQDataset(h5_path, split="val", normalize=True)
    test_dataset = UltrasondIQDataset(h5_path, split="test", normalize=True)

    in_channels = 1 + train_dataset.c_cond
    model = FNO1dComplex(
        in_channels=in_channels,
        out_channels=1,
        width=width,
        modes=modes,
        depth=depth,
        dropout=dropout,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * 2,
        eta_min=learning_rate * 0.05,
    )

    # best_time_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = beamforming_loss(pred, y)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            running_loss += loss.item()
            steps += 1

        scheduler.step()

        train_loss = running_loss / max(steps, 1)
        metrics = evaluate(model, test_loader, device)
        # test_loss = metrics["time_mse"]

      

        print(
            f"epoch {epoch} | train {train_loss:} | "
            f"test_mse {metrics['time_mse']} | rel {metrics['rel_rmse']} | "
            f"phase {metrics['phase_deg']} deg | lr {scheduler.get_last_lr()[0]}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "Cin": in_channels,
                "Ccond": train_dataset.c_cond,
                "Nt": train_dataset.nt,
                "width": width,
                "modes": modes,
                "depth": depth,
                "dropout": dropout,
            },
            output_path,
        )


if __name__ == "__main__":
    train(
        h5_path="train6000.h5",
        # output_path="fno_model.pt",
        # batch_size=64,
        # epochs=120,
        # learning_rate=1e-3,
    )
