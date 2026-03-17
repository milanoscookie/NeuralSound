#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_rf import KWaveUltrasoundDataset
from wno2 import UltrasoundWNO1d


def train(
    h5_path,
    output_path="wno_model.pt",
    batch_size=12,
    epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_workers=4,
    device_name="cuda",
    hidden_channels=64,
    num_blocks=4,
    level=4,
    wavelet="sym8",
    grad_clip=1.0,
):
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    train_dataset = KWaveUltrasoundDataset(h5_path, split="train")
    test_dataset = KWaveUltrasoundDataset(h5_path, split="test")

    sample = train_dataset[0]
    in_channels = int(sample["pulse_rf"].shape[0])
    out_channels = int(sample["rx_rf"].shape[0])

    model = UltrasoundWNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        level=level,
        wavelet=wavelet,
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

    best_test_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        train_steps = 0

        for batch in train_loader:
            x = batch["pulse_rf"].to(device, non_blocking=True).float()
            y = batch["rx_rf"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            running_train += loss.item()
            train_steps += 1

        scheduler.step()

        model.eval()
        running_test = 0.0
        test_steps = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["pulse_rf"].to(device, non_blocking=True).float()
                y = batch["rx_rf"].to(device, non_blocking=True).float()
                pred = model(x)
                running_test += torch.mean((pred - y) ** 2).item()
                test_steps += 1

        train_mse = running_train / max(train_steps, 1)
        test_mse = running_test / max(test_steps, 1)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "hidden_channels": hidden_channels,
                    "num_blocks": num_blocks,
                    "level": level,
                    "wavelet": wavelet,
                },
                output_path,
            )

        print(
            f"epoch {epoch:03d} | train_mse {train_mse:.6e} | "
            f"test_mse {test_mse:.6e} | best {best_test_mse:.6e} | lr {scheduler.get_last_lr()[0]:.3e}"
        )

    print(f"Saved best checkpoint to {output_path}")


if __name__ == "__main__":
    train(
        h5_path="out2000.h5",
        output_path="wno_model.pt",
    )
