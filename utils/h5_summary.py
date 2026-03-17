# h5_summary.py
import h5py
import numpy as np
import argparse
from pathlib import Path


def format_bytes(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def dataset_stats(dset, max_samples=1000):
    try:
        if dset.ndim >= 1:
            n = min(max_samples, dset.shape[0])
            data = dset[:n]
        else:
            data = dset[()]
        data = np.asarray(data)
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
        }
    except Exception as e:
        return {"error": str(e)}


def print_dataset_info(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"\n📦 Dataset: {name}")
        print(f"  Shape      : {obj.shape}")
        print(f"  Dtype      : {obj.dtype}")
        print(f"  Chunks     : {obj.chunks}")
        print(f"  Compression: {obj.compression}")
        print(f"  Size (raw) : {format_bytes(obj.size * obj.dtype.itemsize)}")

        stats = dataset_stats(obj)
        if "error" not in stats:
            print(f"  Sample Min : {stats['min']:.4e}")
            print(f"  Sample Max : {stats['max']:.4e}")
            print(f"  Sample Mean: {stats['mean']:.4e}")
        else:
            print(f"  Stats error: {stats['error']}")


def summarize_h5(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    print("=" * 60)
    print(f"HDF5 SUMMARY: {path}")
    print("=" * 60)

    with h5py.File(path, "r") as f:
        print("\n🔹 File-level info:")
        print(f"  SWMR mode supported : {f.swmr_mode}")
        print(f"  User block size     : {f.userblock_size}")

        total_bytes = 0

        def visitor(name, obj):
            nonlocal total_bytes
            if isinstance(obj, h5py.Dataset):
                total_bytes += obj.size * obj.dtype.itemsize
                print_dataset_info(name, obj)

        f.visititems(visitor)

        print("\n" + "-" * 60)
        print(f"Estimated total raw dataset size: {format_bytes(total_bytes)}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True, help="Path to HDF5 file")
    args = parser.parse_args()

    summarize_h5(args.h5)
