
import os
import sys

root_dir = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir)
)
sys.path.append(root_dir)

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend


def main():
    dataset = SHIFTDataset(
        data_root='../../SHIFT_dataset/v2/public',
        split="train",
        keys_to_load=[
            Keys.images,
            Keys.input_hw,
            Keys.intrinsics,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
            Keys.masks,
            Keys.segmentation_masks,
        ],
        views_to_load=["front"],
        backend=ZipBackend(),
        verbose=True,
    )

    print("Number of samples:", len(dataset))

    for k, data in dataset[0].items():
        if isinstance(data, tuple) or isinstance(data, str):
            print(f"{k}: {data}")
        else:
            print(f"{k}, {data.shape}")


if __name__ == "__main__":
    main()