"""SHIFT dataset for mmdet.

This is a reference code for mmdet style dataset of the SHIFT dataset. Note that
only single-view 2D detection, instance segmentation, and tracking are supported.
Please refer to the torch version of the dataloader for multi-view multi-task cases.

The codes are tested in mmdet-2.20.0.


Example
-------
Below is an example of how to use the SHIFTDataset class.

    >>> dataset = SHIFTDataset(
    >>>     data_root='./SHIFT_dataset/discrete/images'
    >>>     ann_file='train/front/det_2d.json',
    >>>     img_prefix='train/front/img.zip',
    >>>     backend_type='zip',
    >>>     pipeline=[
    >>>        ...
    >>>     ]
    >>> )


Notes
-----
1.  The `backend_type` must be one of ['file', 'zip', 'hdf5'] and the `img_prefix`
    must be consistent with the backend_type.
2.  Since the images are loaded before the pipeline, there is no need to add a 
    `LoadImageFromFile` module in the pipeline again.
3.  For instance segmentation please use the `det_insseg_2d.json` for the `ann_file`,
    and add a `LoadAnnotations(with_mask=True)` module in the pipeline.
"""

import json
import os
import sys

import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import LoadAnnotations

# Add the root directory of the project to the path. Remove the following two lines
# if you have installed shift_dev as a package.
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from shift_dev.utils.backend import HDF5Backend, ZipBackend


@DATASETS.register_module()
class SHIFTDataset(CustomDataset):
    CLASSES = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    WIDTH = 1280
    HEIGHT = 800

    def __init__(self, *args, backend_type: str = "file", **kwargs):
        """Initialize the SHIFT dataset.

        Args:
            backend_type (str, optional): The type of the backend. Must be one of
                ['file', 'zip', 'hdf5']. Defaults to "file".
        """
        super().__init__(*args, **kwargs)
        self.backend_type = backend_type
        if backend_type == "file":
            self.backend = None
        elif backend_type == "zip":
            self.backend = ZipBackend()
        elif backend_type == "hdf5":
            self.backend = HDF5Backend()
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}! "
                "Must be one of ['file', 'zip', 'hdf5']"
            )

    def load_annotations(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)

        data_infos = []
        for img_info in data["frames"]:
            img_filename = os.path.join(
                self.img_prefix, img_info["videoName"], img_info["name"]
            )

            bboxes = []
            labels = []
            track_ids = []
            masks = []
            for label in img_info["labels"]:
                bbox = label["box2d"]
                bboxes.append((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                labels.append(self.CLASSES.index(label["category"]))
                track_ids.append(label["id"])
                if "rle" in label and label["rle"] is not None:
                    masks.append(label["rle"])

            data_infos.append(
                dict(
                    filename=img_filename,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64),
                        track_ids=np.array(track_ids).astype(np.int64),
                        masks=masks if len(masks) > 0 else None,
                    ),
                )
            )
        return data_infos

    def get_img(self, idx):
        filename = self.data_infos[idx]["filename"]
        if self.backend_type == "zip":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        elif self.backend_type == "hdf5":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        else:
            return mmcv.imread(filename)

    def get_img_info(self, idx):
        return dict(
            filename=self.data_infos[idx]["filename"],
            width=self.WIDTH,
            height=self.HEIGHT,
        )

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]

    def prepare_train_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        results = dict(img=img, img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        results = dict(img=img, img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)


if __name__ == "__main__":
    """Example for loading the SHIFT dataset for instance segmentation."""

    dataset = SHIFTDataset(
        data_root="./SHIFT_dataset/discrete/images",
        ann_file="train/front/det_insseg_2d.json",
        img_prefix="train/front/img.zip",
        backend_type="zip",
        pipeline=[LoadAnnotations(with_mask=True)],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("img:", data["img"].shape)
        print("ann_info.bboxes:", data["ann_info"]["bboxes"].shape)
        print("ann_info.labels:", data["ann_info"]["labels"].shape)
        print("ann_info.track_ids:", data["ann_info"]["track_ids"].shape)
        if "gt_masks" in data:
            print("gt_masks:", data["gt_masks"])
        break
