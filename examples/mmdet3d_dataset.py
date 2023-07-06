"""SHIFT dataset for mmdet3d.

This is a reference code for mmdet3d style dataset of the SHIFT dataset. Note that
only monocular tasks are supported (i.e., 2D/3D detection, tracking, 2D instance segmentation, depth estimation).
Please refer to the torch version of the dataloader for multi-view multi-task cases.

The codes are tested in mmdet3d-1.0.0.


Example
-------
Below is a snippet showing how to add the SHIFTDataset class in mmdet config files.

    >>> dict(
    >>>     type='SHIFTDataset',
    >>>     data_root='./SHIFT_dataset/discrete/images'
    >>>     ann_file='train/front/det_3d.json',
    >>>     img_prefix='train/front/img.zip',
    >>>     backend_type='zip',
    >>>     pipeline=[
    >>>        dict(type='LoadAnnotations3D', with_bbox=True)
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

import os
import sys
import tempfile

import mmcv
import numpy as np
from mmdet3d.core.bbox import CameraInstance3DBoxes
from mmdet3d.core.bbox.structures import Box3DMode
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import LoadAnnotations3D

from scalabel.label.typing import Dataset, Frame, Label, Box3D, RLE
from scalabel.label.transforms import mask_to_rle

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
    DEPTH_FACTOR = 16777.216  #  256 ** 3 / 1000.0

    def __init__(
        self, 
        data_root: str, 
        ann_file: str, 
        img_prefix: str,
        insseg_ann_file: str = "",
        depth_prefix: str = "",
        backend_type: str = "file",
        img_to_float32: bool = False,
        yaw_offset: float = np.pi / 2,
        **kwargs
    ):
        """Initialize the SHIFT dataset.

        Args:
            data_root (str): The root path of the dataset.
            ann_file (str): The path to the annotation file for 3D detection.
            img_prefix (str): The base path to the image directory or archive.
            insseg_ann_file (str, optional): The path to the annotation file for 2D instance segmentation. If set, the
                instance masks will be loaded. Defaults to "".
            depth_prefix (str, optional): The base path to the depth directory or archive. If set, the depth maps will
                be loaded. Defaults to "".
            backend_type (str, optional): The type of the backend. Must be one of ['file', 'zip', 'hdf5'].
                Defaults to "file".
            img_to_float32 (bool, optional): Whether to convert the loaded image to float32. Defaults to False.
            yaw_offset (float, optional): The yaw offset of the 3D bounding boxes. Defaults to np.pi / 2. This is
                used to correct the heading of the 3D bounding boxes.
        """
        self.data_root = data_root
        self.ann_file = os.path.join(self.data_root, ann_file)
        self.img_prefix = os.path.join(self.data_root, img_prefix)
        self.img_to_float32 = img_to_float32
        self.yaw_offset = yaw_offset

        self.insseg_ann_file = os.path.join(self.data_root, insseg_ann_file) if insseg_ann_file != "" else ""
        self.depth_prefix = os.path.join(self.data_root, depth_prefix) if depth_prefix != "" else ""
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

        super().__init__(data_root, self.ann_file, **kwargs)

    def load_annotations(self, ann_file):
        print("Loading annotations...")
        data = mmcv.load(ann_file, file_format='json')
        if self.insseg_ann_file != "":
            data_insseg = mmcv.load(self.insseg_ann_file, file_format='json')
        else:
            data_insseg = None

        data_infos = []
        for img_idx, img_info in enumerate(data["frames"]):
            boxes = []
            boxes_3d = []
            labels = []
            track_ids = []
            masks = []
            for i, label in enumerate(img_info["labels"]):
                box2d = label["box2d"]
                box3d = label["box3d"]
                boxes.append((box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]))
                boxes_3d.append(
                    box3d["location"] + box3d["dimension"] + [box3d["orientation"][1] + self.yaw_offset], # yaw
                )
                labels.append(self.CLASSES.index(label["category"]))
                track_ids.append(label["id"])
                if data_insseg is not None:
                    masks.append(data_insseg["frames"][img_idx]["labels"][i]["rle"])

            data_infos.append(
                dict(
                    sample_idx=img_idx,
                    lidar_points=dict(lidar_path=""),   # dummy path for mmdet3d compatibility
                    image=dict(
                        image_idx=img_idx,
                        image_name=img_info["name"],
                        video_name=img_info["videoName"],
                        image_shape=np.array([self.HEIGHT, self.WIDTH], dtype=np.int32),
                        width=self.WIDTH,  # redundant but required by mmdet pipeline (LoadAnnotations(with_mask=True))
                        height=self.HEIGHT,
                    ),
                    annos=dict(
                        num=len(boxes),
                        boxes_2d=np.array(boxes).astype(np.float32),
                        boxes_3d=np.array(boxes_3d).astype(np.float32),
                        labels=np.array(labels).astype(np.int64),
                        names=[self.CLASSES[label] for label in labels],
                        track_ids=np.array(track_ids).astype(np.int64),
                        masks=masks if len(masks) > 0 else None,
                    ),
                )
            )
        return data_infos
    
    def read_image(self, filename):
        if self.backend_type == "zip":
            img_bytes = self.backend.get(filename)
            img = mmcv.imfrombytes(img_bytes)
        elif self.backend_type == "hdf5":
            img_bytes = self.backend.get(filename)
            img = mmcv.imfrombytes(img_bytes)
        else:
            img = mmcv.imread(filename)
        if self.img_to_float32:
            return img.astype(np.float32) / 255.0
        return img

    def get_img(self, idx):
        img_filename = os.path.join(
            self.img_prefix,
            self.data_infos[idx]["image"]["video_name"],
            self.data_infos[idx]["image"]["image_name"],
        )
        return self.read_image(img_filename)
        
    def get_depth(self, idx):
        depth_filename = os.path.join(
            self.depth_prefix,
            self.data_infos[idx]["image"]["video_name"],
            self.data_infos[idx]["image"]["image_name"].replace("jpg", "png").replace("img", "depth"),
        )
        depth_img = self.read_image(depth_filename)
        depth_img = depth_img.astype(np.float32)
        depth = depth_img[:, :, 0] * 256 * 256 + depth_img[:, :, 1] * 256 + depth_img[:, :, 2]
        depth /= self.DEPTH_FACTOR
        return depth

    def get_img_info(self, idx):
        return self.data_infos[idx]["image"]
    
    def get_ann_info(self, idx) -> dict:
        annos = self.data_infos[idx]["annos"]
        if annos["num"] != 0:
            gt_labels = annos["labels"]
            gt_bboxes_3d = annos["boxes_3d"]
            gt_bboxes = annos["boxes_2d"]
            gt_names = annos["names"]
            gt_masks = annos["masks"]
            gt_track_ids = annos["track_ids"]
        else:
            gt_labels = np.zeros((0, ), dtype=np.int64)
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_names = []
            gt_masks = []
            gt_track_ids = np.zeros((0, ), dtype=np.int64)

        gt_bboxes_3d = CameraInstance3DBoxes(
            gt_bboxes_3d, box_dim=7, with_yaw=True
        ).convert_to(self.box_mode_3d)
        
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            masks=gt_masks,
            track_ids=gt_track_ids,
            gt_names=gt_names,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
        )
        return ann

    def prepare_train_data(self, idx):
        img = [self.get_img(idx)]  # Note: mmdet3d expects a list of images
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        # Filter out images without annotations during training
        if self.filter_empty_gt and len(ann_info["gt_bboxes_3d"]) == 0:
            return None
        results = dict(img=img, img_info=img_info, cam2img=self.cam_intrinsic, ann_info=ann_info)
        if self.depth_prefix != "":
            results["gt_depth"] = self.get_depth(idx)
            
        # Add lidar2cam matrix for compatibility (e.g., PETR)
        if self.box_mode_3d == Box3DMode.LIDAR:
            results["lidar2cam"] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        elif self.box_mode_3d == Box3DMode.DEPTH:
            results["depth2cam"] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        # Set initial shape for mmdet3d pipeline compatibility
        img_shape = img[0][..., np.newaxis].shape
        results["img_shape"] = img_shape
        results["ori_shape"] = img_shape
        results["pad_shape"] = img_shape
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        img = [self.get_img(idx)]
        img_info = self.get_img_info(idx)
        results = dict(img=img, img_info=img_info, cam2img=self.cam_intrinsic)
        results["lidar2cam"] = np.eye(4)
        img_shape = img[0][..., np.newaxis].shape
        results["img_shape"] = img_shape
        results["ori_shape"] = img_shape
        results["pad_shape"] = img_shape
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def seg_results2scalabel(self, results, output_dir, seg_key="seg_results") -> str:
        """Convert instance segmentation results to Scalabel format (det_insseg_2d.json).

        Args:
            results (list[dict]): Testing results of the dataset.
            output_dir (str): Output directory of the results in Scalabel
                format.
            result_key (str): Key of semantic segmentation results in results.
                Default: "seg_results".

        Returns:
            str: Path of the converted results in Scalabel format.
        """
        frames = []
        for result in results:
            labels = []
            for seg_label in result[seg_key]:
                if "rle" in seg_label:
                    rle = seg_label["rle"]
                    label = Label(
                        id=seg_label["id"],
                        category=self.CLASSES[seg_label["category_id"]],
                        rle=RLE(
                            counts=rle["counts"].decode("utf-8"),
                            size=(rle["size"][0], rle["size"][1])
                        ),
                    )
                elif "mask" in seg_label:
                    label = Label(
                        id=seg_label["id"],
                        category=self.CLASSES[seg_label["category_id"]],
                        rle=mask_to_rle(seg_label["mask"]),
                    )
                labels.append(label)
            frame = Frame(
                name=result["image_name"],
                videoName=result["video_name"],
                frameIndex=result["image_name"].split("_")[0],
                labels=labels,
            )
            frames.append(frame)
        ds = Dataset(frames=frames, groups=None, config=None)

        jsonfile = os.path.join(output_dir, "det_insseg_2d.json")
        with open(jsonfile, "w") as f:
            f.write(ds.json(exclude_unset=True))
        return jsonfile
    
    def box_results2scalabel(self, results, output_dir, box_key="pts_bbox") -> str:
        """Convert 3D object detection results to Scalabel format (det_3d.json).

        Args:
            results (list[dict]): Testing results of the dataset.
            output_dir (str): Output directory of the results in Scalabel
                format.
            result_key (str): Key of semantic segmentation results in results.
                Default: "seg_results".

        Returns:
            str: Path of the output json file.
        """
        frames = []
        for result in results:
            labels = []
            for label_id, (box, label, score) in enumerate(zip(
                result[box_key]["boxes_3d"],
                result[box_key]["labels_3d"],
                result[box_key]["scores_3d"],
            )):
                box_cam = box.convert_to(Box3DMode.CAM)
                yaw = box_cam.tensor[0, 6]
                label = Label(
                    id=label_id,
                    category=self.CLASSES[label],
                    box3d=Box3D(
                        alpha=0,
                        location=tuple(box_cam[0, 0:3].tolist()),
                        dimension=tuple(box_cam[0, 3:6].tolist()),
                        orientation=(0, yaw - self.yaw_offset, 0)
                    ),
                    score=score,
                )
                labels.append(label)
            frame = Frame(
                name=result["image_name"],
                videoName=result["video_name"],
                frameIndex=result["image_name"].split("_")[0],
                labels=labels,
            )
            frames.append(frame)
        ds = Dataset(frames=frames, groups=None, config=None)

        jsonfile = os.path.join(output_dir, "det_3d.json")
        with open(jsonfile, "w") as f:
            f.write(ds.json(exclude_unset=True))
        return jsonfile

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for Scalabel evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            **kwargs: Other arguments are ignored.

        Note:
            The results are in the format of list[dict], where each dict contains:
                - "image_name" (str): Filename of the image.
                - "video_name" (str): Filename of the video.
                - "pts_bbox" (dict): 3D detection results, which contains:
                    - "boxes_3d" (BaseInstance3DBoxes): Predicted 3D boxes of shape (N, dim).
                    - "labels_3d" (torch.Tensor): Predicted labels of shape (N,).
                    - "scores_3d" (torch.Tensor): Scores of predicted boxes of shape (N,).
                - "seg_results" (list): Instance segmentation results, whose item contains either:
                    - "rle" (dict): RLE encoded segmentation mask, or
                    - "mask" (np.ndarray): Segmentation mask of shape (H, W).

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = {
            "det_insseg_2d": self.seg_results2scalabel(results, jsonfile_prefix, **kwargs),
            "det_3d": self.box_results2scalabel(results, jsonfile_prefix, **kwargs),
        }
        return result_files, tmp_dir


if __name__ == "__main__":
    """Example for loading the SHIFT dataset for monocular 3D detection."""

    dataset = SHIFTDataset(
        data_root="./SHIFT_dataset/discrete/images",
        ann_file="val/front/det_3d.json",
        insseg_ann_file="val/front/det_insseg_2d.json",
        img_prefix="val/front/img.zip",
        depth_prefix="val/front/depth.zip",
        box_type_3d="Camera",
        backend_type="zip",
        pipeline=[LoadAnnotations3D(with_bbox=True, with_mask=True)],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Check the tensor shapes.
    for i, data in enumerate(dataset):
        print(f"Sample {i}")

        # Image and camera intrinsic matrix
        print("img:", data["img"].shape)
        print("cam2img:", data["cam2img"].shape)

        # 3D bounding boxes
        print("gt_bboxes_3d:", data["gt_bboxes_3d"])
        print("gt_labels_3d:", data["gt_labels_3d"])

        # 2D bounding boxes and masks
        # Note that the mask labels are shared with 'gt_labels_3d'.
        print("gt_bboxes:", data["gt_bboxes"])
        print("gt_masks:", data["gt_masks"])

        # Depth map
        print("gt_depth:", data["gt_depth"].shape)

        # Track IDs
        print("track_ids:", data["ann_info"]["track_ids"])

        break
