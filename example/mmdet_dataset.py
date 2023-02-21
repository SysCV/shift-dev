"""
This is a reference code for mmdet style dataset of the SHIFT dataset. Note that
only single-view data is supported. Please refer to the torch version of the
dataset for multi-view data.

Example for mmdet config file:

    >>> dataset = SHIFTDataset(
    >>>     data_root='./shift_dataset/discrete/images'
    >>>     ann_file='train/front/det_2d.json',
    >>>     img_prefix='train/front/img.zip',
    >>>     backend_type='zip',
    >>>     pipeline=[
    >>>        ...
    >>>     ]
    >>> )

"""

import json
import os
import sys
import numpy as np

import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


root_dir = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir)
)
sys.path.append(root_dir)

from shift_dev.utils.backend import ZipBackend, HDF5Backend


@DATASETS.register_module()
class SHIFTDataset(CustomDataset):

    CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')

    WIDTH = 1280
    HEIGHT = 800

    def __init__(self, backend_type: str = "file", *args, **kwargs):
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
        with open(ann_file, 'r') as f:
            data = json.load(f)

        data_infos = []
        for img_info in data['frames']:
            img_filename = os.path.join(
                self.img_prefix, img_info['videoName'], img_info['name']
            )

            bboxes = []
            labels = []
            for label in img_info['labels']:
                bbox = label['box2d']
                bboxes.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                labels.append(self.CLASSES.index(label['category']))
            
            data_infos.append(
                dict(
                    filename=img_filename,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                )
            )
        return data_infos
        
    def get_img(self, idx):
        filename = self.data_infos[idx]['filename']
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
            filename=self.data_infos[idx]['filename'],
            width=self.WIDTH,
            height=self.HEIGHT,
        )

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def __getitem__(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        ann = self.get_ann_info(idx)
        return self.pipeline(dict(img=img, img_info=img_info, ann_info=ann))


def main():
    """Load the SHIFT dataset and print the tensor shape of the first batch."""

    dataset = SHIFTDataset(
        data_root='../../SHIFT_dataset/v2/public/discrete/images',
        ann_file='train/front/det_2d.json',
        img_prefix='train/front/img.zip',
        backend_type="zip",
        pipeline=[],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("img:", data["img"].shape)
        print("ann_info.bboxes:", data["ann_info"]["bboxes"].shape)
        print("ann_info.labels:", data["ann_info"]["labels"].shape)
        break


if __name__ == "__main__":
    main()