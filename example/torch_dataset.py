import os
import sys

import torch
from torch.utils.data import DataLoader

# Add the root directory of the project to the path. Remove the following two lines
# if you have installed shift_dev as a package.
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend


def main():
    """Load the SHIFT dataset and print the tensor shape of the first batch."""

    dataset = SHIFTDataset(
        data_root="../../SHIFT_dataset/v2/public",
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

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
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
