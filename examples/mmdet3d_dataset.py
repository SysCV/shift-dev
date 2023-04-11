"""SHIFT dataset for mmdet3d.

This is a reference code for mmdet3d style dataset of the SHIFT dataset. Note that
only monocular 3D detection and tracking are supported.
Please refer to the torch version of the dataloader for multi-view multi-task cases.

The codes are tested in mmdet3d-1.0.0.


Example
-------
Below is a snippet showing how to add the SHIFTDataset class in mmdet config files.

    >>> dict(
    >>>     type='SHIFTDataset',
    >>>     data_root='./SHIFT_dataset/discrete/images'
    >>>     ann_file='./SHIFT_dataset/discrete/images/train/front/det_3d.json',
    >>>     img_prefix='./SHIFT_dataset/discrete/images/train/front/img.zip',
    >>>     backend_type='zip',
    >>>     pipeline=[
    >>>        ...
    >>>     ]
    >>> )


Notes
-----
1.  Please copy this file to `mmdet/datasets/` and update the `mmdet/datasets/__init__.py`
    so that the `SHIFTDataset` class is imported. You can refer to their official tutorial at
    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html.
2.  The `backend_type` must be one of ['file', 'zip', 'hdf5'] and the `img_prefix`
    must be consistent with the backend_type.
3.  Since the images are loaded before the pipeline with the selected backend, there is no need
    to add a `LoadImageFromFileMono3D`/`LoadMultiViewImageFromFiles` module in the pipeline again.
"""

import json
import os
import sys

import mmcv
import numpy as np
from mmdet3d.core.bbox import CameraInstance3DBoxes
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import LoadAnnotations3D

# Add the root directory of the project to the path. Remove the following two lines
# if you have installed shift_dev as a package.
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from shift_dev.utils.backend import HDF5Backend, ZipBackend


@DATASETS.register_module()
class SHIFTDataset(Custom3DDataset):
    CLASSES = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    WIDTH = 1280
    HEIGHT = 800

    def __init__(self, *args, img_prefix: str = "", backend_type: str = "file", **kwargs):
        """Initialize the SHIFT dataset.

        Args:
            backend_type (str, optional): The type of the backend. Must be one of
                ['file', 'zip', 'hdf5']. Defaults to "file".
        """
        self.img_prefix = img_prefix
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
        self.cam_intrinsic = np.array(
            [
                [640, 0,    self.WIDTH / 2,    0],
                [0,   640,  self.HEIGHT / 2,   0],
                [0,   0,    1,                 0],
                [0,   0,    0,                 1],
            ],
            dtype=np.float32,
        )
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        print("Loading annotations...")
        data = mmcv.load(ann_file, file_format='json')
        data_infos = []
        for img_idx, img_info in enumerate(data["frames"]):
            img_filename = os.path.join(
                self.img_prefix, img_info["videoName"], img_info["name"]
            )

            boxes = []
            boxes_3d = []
            labels = []
            track_ids = []

            for label in img_info["labels"]:
                box2d = label["box2d"]
                box3d = label["box3d"]
                boxes.append((box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]))
                boxes_3d.append(
                    box3d["location"] + box3d["dimension"] + [box3d["orientation"][1] + np.pi / 2], # yaw
                )
                labels.append(self.CLASSES.index(label["category"]))
                track_ids.append(label["id"])

            data_infos.append(
                dict(
                    sample_idx=img_idx,
                    lidar_points=dict(lidar_path=""),   # dummy path for mmdet3d compatibility
                    image=dict(
                        image_idx=img_idx,
                        image_path=img_filename,
                        image_shape=np.array([self.HEIGHT, self.WIDTH], dtype=np.int32)
                    ),
                    annos=dict(
                        num=len(boxes),
                        boxes_2d=np.array(boxes).astype(np.float32),
                        boxes_3d=np.array(boxes_3d).astype(np.float32),
                        labels=np.array(labels).astype(np.int64),
                        names=[self.CLASSES[label] for label in labels],
                        track_ids=np.array(track_ids).astype(np.int64),
                    ),
                )
            )
        return data_infos

    def get_img(self, idx):
        filename = self.data_infos[idx]["image"]["image_path"]
        if self.backend_type == "zip":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        elif self.backend_type == "hdf5":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        else:
            return mmcv.imread(filename)

    def get_img_info(self, idx):
        return self.data_infos[idx]["image"]
    
    def get_ann_info(self, idx) -> dict:
        annos = self.data_infos[idx]["annos"]
        if annos["num"] != 0:
            gt_labels = annos["labels"]
            gt_bboxes_3d = annos["boxes_3d"]
            gt_bboxes = annos["boxes_2d"]
            gt_names = annos["names"]
            gt_track_ids = annos["track_ids"]
        else:
            gt_labels = np.zeros((0, ), dtype=np.int64)
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_names = []
            gt_track_ids = np.zeros((0, ), dtype=np.int64)

        gt_bboxes_3d = CameraInstance3DBoxes(
            gt_bboxes_3d, box_dim=7, with_yaw=True
        ).convert_to(self.box_mode_3d)
        
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            track_ids=gt_track_ids,
            gt_names=gt_names,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
        )
        return ann

    def prepare_train_data(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        # Filter out images without annotations during training
        if self.filter_empty_gt and len(ann_info["gt_bboxes_3d"]) == 0:
            return None
        results = dict(img=img, img_info=img_info, cam2img=self.cam_intrinsic, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        results = dict(img=img, img_info=img_info, cam2img=self.cam_intrinsic)
        self.pre_pipeline(results)
        return self.pipeline(results)


if __name__ == "__main__":
    """Example for loading the SHIFT dataset for monocular 3D detection."""

    dataset = SHIFTDataset(
        data_root="./SHIFT_dataset/discrete/images",
        ann_file="./SHIFT_dataset/discrete/images/val/front/det_3d.json",
        img_prefix="./SHIFT_dataset/discrete/images/val/front/img.zip",
        box_type_3d="Camera",
        backend_type="zip",
        pipeline=[LoadAnnotations3D(with_bbox=True)],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("img:", data["img"].shape)
        print("annos.gt_bboxes_3d:", data["ann_info"]["gt_bboxes_3d"])
        print("annos.gt_labels_3d:", data["ann_info"]["gt_labels_3d"])
        print("annos.gt_bboxes:", data["ann_info"]["bboxes"])
        print("annos.gt_labels:", data["ann_info"]["labels"])
        print("annos.track_ids:", data["ann_info"]["track_ids"])
        break
