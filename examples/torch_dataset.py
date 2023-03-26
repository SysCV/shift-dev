"""SHIFT dataset example using PyTorch."""

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
        data_root="./SHIFT_dataset/",
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
        shift_type="discrete",      # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=ZipBackend(),       # also supports HDF5Backend(), FileBackend()
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
    print('\n')
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        for k, data in batch["front"].items():
            if isinstance(data, torch.Tensor):
                print(f"{k}: {data.shape}")
            else:
                print(f"{k}: {data}")
        break

    # Print the sample indices within a video.
    # The video indices groups frames based on their video sequences. They are useful for training on videos.
    print('\n')
    video_to_indices = dataset.video_to_indices
    for video, indices in video_to_indices.items():
        print(f"Video name: {video}")
        print(f"Sample indices within a video: {indices}")
        break


if __name__ == "__main__":
    main()
