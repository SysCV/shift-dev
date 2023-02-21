
import os
import sys

root_dir = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir)
)
sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend


def main():
    dataset = SHIFTDataset(
        data_root='../../SHIFT_dataset/v2/public',
        split="train",
        keys_to_load=[
            Keys.images,
            Keys.intrinsics,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
            Keys.segmentation_masks,
        ],
        views_to_load=["front"],
        backend=ZipBackend(),  # also supports HDF5Backend(), FileBackend()
        verbose=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    print(f"Number of samples:", len(dataset))

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        for k, data in batch["front"].items():
            if isinstance(data, torch.Tensor):
                print(f"{k}: {data.shape}")
            else:
                print(f"{k}: {data}")
        break


if __name__ == "__main__":
    main()
