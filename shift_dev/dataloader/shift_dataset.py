"""SHIFT dataset."""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import partial
import multiprocessing
from io import BytesIO

import numpy as np
import torch
from tqdm import tqdm
from scalabel.label.io import parse
from scalabel.label.typing import Config
from scalabel.label.typing import Dataset as ScalabelData
from torch import Tensor
from torch.utils.data import Dataset

from shift_dev.types import DataDict, Keys
from shift_dev.utils import setup_logger
from shift_dev.utils.backend import DataBackend, HDF5Backend, ZipBackend
from shift_dev.utils.load import im_decode, ply_decode

from .base import Scalabel

logger = setup_logger()


def _get_extension(backend: DataBackend):
    """Get the appropriate file extension for the given backend."""
    if isinstance(backend, HDF5Backend):
        return ".hdf5"
    if isinstance(backend, ZipBackend):
        return ".zip"
    return ""


class _SHIFTScalabelLabels(Scalabel):
    """Helper class for labels in SHIFT that are stored in Scalabel format."""

    VIEWS = [
        "front",
        "center",
        "left_45",
        "left_90",
        "right_45",
        "right_90",
        "left_stereo",
    ]

    def __init__(
        self,
        data_root: str,
        split: str,
        data_file: str = "",
        annotation_file: str = "",
        view: str = "front",
        framerate: str = "images",
        shift_type: str = "discrete",
        backend: DataBackend = HDF5Backend(),
        verbose: bool = False,
        num_workers: int = 1,
        **kwargs,
    ) -> None:
        """Initialize SHIFT dataset for one view.

        Args:
            data_root (str): Path to the root directory of the dataset.
            split (str): Which data split to load.
            data_file (str): Path to the data archive file. Default: "".
            annotation_file (str): Path to the annotation file. Default: "".
            view (str): Which view to load. Default: "front".
            backend (DataBackend): Backend to use for loading data. Default:
                HDF5Backend().
        """
        self.verbose = verbose
        self.num_workers = num_workers

        # Validate input
        assert split in set(("train", "val", "test")), f"Invalid split '{split}'"
        assert view in _SHIFTScalabelLabels.VIEWS, f"Invalid view '{view}'"

        # Set attributes
        ext = _get_extension(backend)
        if shift_type.startswith("continuous"):
            shift_speed = shift_type.split("/")[-1]
            annotation_path = os.path.join(
                data_root, "continuous", framerate, shift_speed, split, view, annotation_file
            )
            data_path = os.path.join(
                data_root, "continuous", framerate, shift_speed, split, view, f"{data_file}{ext}"
            )
        else:
            annotation_path = os.path.join(
                data_root, "discrete", framerate, split, view, annotation_file
            )
            data_path = os.path.join(
                data_root, "discrete", framerate, split, view, f"{data_file}{ext}"
            )
        super().__init__(data_path, annotation_path, data_backend=backend, **kwargs)

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        # NOTE: Skipping validation for much faster loading
        if self.verbose:
            logger.info(f"Loading annotation from '{self.annotation_path}' ...")
        return self._load(self.annotation_path)

    def _load(self, filepath: str) -> ScalabelData:
        """Load labels from a json file or a folder of json files."""
        raw_frames: List[DictStrAny] = []
        raw_groups: List[DictStrAny] = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")

        def process_file(filepath: str) -> Optional[DictStrAny]:
            raw_cfg = None
            with open(filepath, mode="r", encoding="utf-8") as fp:
                content = json.load(fp)
            if isinstance(content, dict):
                raw_frames.extend(content["frames"])
                if "groups" in content and content["groups"] is not None:
                    raw_groups.extend(content["groups"])
                if "config" in content and content["config"] is not None:
                    raw_cfg = content["config"]
            elif isinstance(content, list):
                raw_frames.extend(content)
            else:
                raise TypeError("The input file contains neither dict nor list.")
            if self.verbose:
                logger.info(f"Loading annotation from '{filepath}' Done.")
            return raw_cfg

        cfg = None
        if os.path.isfile(filepath) and filepath.endswith("json"):
            ret_cfg = process_file(filepath)
            if ret_cfg is not None:
                cfg = ret_cfg
        else:
            raise TypeError("Inputs must be a folder or a JSON file.")

        config = None
        if cfg is not None:
            config = Config(**cfg)

        parse_ = partial(parse, validate_frames=False)
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                frames = []
                with tqdm(total=len(raw_frames)) as pbar:
                    for result in pool.imap_unordered(parse_, raw_frames, chunksize=1000):
                        frames.append(result)
                        pbar.update()
        else:
            frames = list(map(parse_, raw_frames))
        return ScalabelData(frames=frames, config=config)


class SHIFTDataset(Dataset):
    """SHIFT dataset class, supporting multiple tasks and views."""

    DESCRIPTION = """SHIFT Dataset, a synthetic driving dataset for continuous
    multi-task domain adaptation"""
    HOMEPAGE = "https://www.vis.xyz/shift/"
    PAPER = "https://arxiv.org/abs/2206.08367"
    LICENSE = "CC BY-NC-SA 4.0"

    KEYS = [
        # Inputs
        Keys.images,
        Keys.original_hw,
        Keys.input_hw,
        Keys.points3d,
        # Scalabel formatted annotations
        Keys.intrinsics,
        Keys.extrinsics,
        Keys.timestamp,
        Keys.axis_mode,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
        Keys.masks,
        Keys.boxes3d,
        Keys.boxes3d_classes,
        Keys.boxes3d_track_ids,
        # Bit masks
        Keys.segmentation_masks,
        Keys.depth_maps,
        Keys.optical_flows,
    ]

    VIEWS = [
        "front",
        "center",
        "left_45",
        "left_90",
        "right_45",
        "right_90",
        "left_stereo",
    ]

    DATA_GROUPS = {
        "img": [
            Keys.images,
            Keys.original_hw,
            Keys.input_hw,
            Keys.intrinsics,
        ],
        "det_2d": [
            Keys.timestamp,
            Keys.axis_mode,
            Keys.extrinsics,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
        ],
        "det_3d": [
            Keys.boxes3d,
            Keys.boxes3d_classes,
            Keys.boxes3d_track_ids,
        ],
        "det_insseg_2d": [
            Keys.masks,
        ],
        "semseg": [
            Keys.segmentation_masks,
        ],
        "depth": [
            Keys.depth_maps,
        ],
        "flow": [
            Keys.optical_flows,
        ],
        "lidar": [
            Keys.points3d,
        ],
    }

    GROUPS_IN_SCALABEL = ["det_2d", "det_3d", "det_insseg_2d"]

    def __init__(
        self,
        data_root: str,
        split: str,
        keys_to_load: Sequence[str] = (Keys.images, Keys.boxes2d),
        views_to_load: Sequence[str] = ("front",),
        framerate: str = "images",
        shift_type: str = "discrete",
        backend: DataBackend = HDF5Backend(),
        num_workers: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize SHIFT dataset."""
        # Validate input
        assert split in {"train", "val", "test"}, f"Invalid split '{split}'."
        assert framerate in {"images", "videos"}, f"Invalid framerate '{framerate}'. Must be 'images' or 'videos'."
        assert shift_type in {"discrete", "continuous/1x", "continuous/10x", "continuous/100x"}, (
            f"Invalid shift_type '{shift_type}'. Must be one of 'discrete', 'continuous/1x', 'continuous/10x', "
            "or 'continuous/100x'."
        )
        self.validate_keys(keys_to_load)

        # Set attributes
        self.data_root = data_root
        self.split = split
        self.keys_to_load = keys_to_load
        self.views_to_load = views_to_load
        self.framerate = framerate
        self.shift_type = shift_type
        self.backend = backend
        self.verbose = verbose
        self.ext = _get_extension(backend)
        if self.shift_type.startswith("continuous"):
            shift_speed = self.shift_type.split("/")[-1]
            self.annotation_base = os.path.join(
                self.data_root, "continuous", self.framerate, shift_speed, self.split
            )
        else:
            self.annotation_base = os.path.join(
                self.data_root, self.shift_type, self.framerate, self.split
            )
        if self.verbose:
            logger.info(f"Base: {self.annotation_base}. Backend: {self.backend}")

        # Get the data groups' classes that need to be loaded
        self._data_groups_to_load = self._get_data_groups(keys_to_load)
        if "det_2d" not in self._data_groups_to_load:
            raise ValueError(
                "In current implementation, the 'det_2d' data group must be"
                "loaded to load any other data group."
            )

        self.scalabel_datasets = {}
        for view in self.views_to_load:
            if view == "center":
                # Load lidar data, only available for center view
                self.scalabel_datasets["center/lidar"] = _SHIFTScalabelLabels(
                    data_root=self.data_root,
                    split=self.split,
                    data_file="lidar",
                    annotation_file="det_3d.json",
                    view=view,
                    framerate=self.framerate,
                    shift_type=self.shift_type,
                    keys_to_load=(Keys.points3d, *self.DATA_GROUPS["det_3d"]),
                    backend=backend,
                    num_workers=num_workers,
                    verbose=verbose,
                )
            else:
                # Skip the lidar data group, which is loaded separately
                image_loaded = False
                for group in self._data_groups_to_load:
                    name = f"{view}/{group}"
                    keys_to_load = list(self.DATA_GROUPS[group])
                    # Load the image data group only once
                    if not image_loaded:
                        keys_to_load.extend(self.DATA_GROUPS["img"])
                        image_loaded = True
                    self.scalabel_datasets[name] = _SHIFTScalabelLabels(
                        data_root=self.data_root,
                        split=self.split,
                        data_file="img",
                        annotation_file=f"{group}.json",
                        view=view,
                        framerate=self.framerate,
                        shift_type=self.shift_type,
                        keys_to_load=keys_to_load,
                        backend=backend,
                        num_workers=num_workers,
                        verbose=verbose,
                    )

    def validate_keys(self, keys_to_load: Sequence[str]) -> None:
        """Validate that all keys to load are supported."""
        for k in keys_to_load:
            if k not in self.KEYS:
                raise ValueError(f"Key '{k}' is not supported!")

    def _get_data_groups(self, keys_to_load: Sequence[str]) -> list[str]:
        """Get the data groups that need to be loaded from Scalabel."""
        data_groups = []
        for data_group, group_keys in self.DATA_GROUPS.items():
            if data_group in self.GROUPS_IN_SCALABEL:
                # If the data group is loaded by Scalabel, add it to the list
                if any(key in group_keys for key in keys_to_load):
                    data_groups.append(data_group)
        return list(set(data_groups))

    def _load(
        self, view: str, data_group: str, file_ext: str, video: str, frame: str
    ) -> Tensor:
        """Load data from the given data group."""
        frame_number = frame.split("_")[0]
        filepath = os.path.join(
            self.annotation_base,
            view,
            f"{data_group}{self.ext}",
            video,
            f"{frame_number}_{data_group}_{view}.{file_ext}",
        )
        if data_group == "semseg":
            return self._load_semseg(filepath)
        if data_group == "depth":
            return self._load_depth(filepath)
        if data_group == "flow":
            return self._load_flow(filepath)
        raise ValueError(f"Invalid data group '{data_group}'")

    def _load_semseg(self, filepath: str) -> Tensor:
        """Load semantic segmentation data."""
        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)[..., 0]
        return torch.as_tensor(image, dtype=torch.int64).unsqueeze(0)

    def _load_depth(self, filepath: str, max_depth: float = 1000.0) -> Tensor:
        """Load depth data."""
        assert max_depth > 0, "Max depth value must be greater than 0."

        im_bytes = self.backend.get(filepath)
        image = im_decode(im_bytes)
        if image.shape[2] > 3:  # pragma: no cover
            image = image[:, :, :3]
        image = image.astype(np.float32)

        # Convert to depth
        depth = image[:, :, 2] * 256 * 256 + image[:, :, 1] * 256 + image[:, :, 0]
        depth /= 16777216.0  # 256 ** 3
        return torch.as_tensor(
            np.ascontiguousarray(depth * max_depth),
            dtype=torch.float32,
        ).unsqueeze(0)

    def _load_flow(self, filepath: str) -> Tensor:
        """Load optical flow data."""
        im_bytes = self.backend.get(filepath)
        flow = np.load(BytesIO(im_bytes))
        return (
            torch.as_tensor(flow["flow"], dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

    def _load_lidar(self, filepath: str) -> Tensor:
        """Load lidar data."""
        ply_bytes = self.backend.get(filepath)
        points = ply_decode(ply_bytes)
        return torch.as_tensor(points, dtype=torch.float32)

    def _get_frame_key(self, idx: int) -> tuple[str, str]:
        """Get the frame identifier (video name, frame name) by index."""
        if len(self.scalabel_datasets) > 0:
            frames = self.scalabel_datasets[
                list(self.scalabel_datasets.keys())[0]
            ].frames
            return frames[idx].videoName, frames[idx].name
        raise ValueError("No Scalabel file has been loaded.")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if len(self.scalabel_datasets) > 0:
            return len(self.scalabel_datasets[list(self.scalabel_datasets.keys())[0]])
        raise ValueError("No Scalabel file has been loaded.")

    def __getitem__(self, idx: int) -> DataDict:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        # load camera frames
        data_dict = {}
        for view in self.views_to_load:
            data_dict_view = {}
            video_name, frame_name = self._get_frame_key(idx)

            if view == "center":
                # Lidar is only available in the center view
                if Keys.points3d in self.keys_to_load:
                    data_dict_view.update(self.scalabel_datasets["center/lidar"][idx])
            else:
                # Load data from Scalabel
                for group in self._data_groups_to_load:
                    data_dict_view.update(
                        self.scalabel_datasets[f"{view}/{group}"][idx]
                    )

                # Load data from bit masks
                if Keys.segmentation_masks in self.keys_to_load:
                    data_dict_view[Keys.segmentation_masks] = self._load(
                        view, "semseg", "png", video_name, frame_name
                    )
                if Keys.depth_maps in self.keys_to_load:
                    data_dict_view[Keys.depth_maps] = self._load(
                        view, "depth", "png", video_name, frame_name
                    )
                if Keys.optical_flows in self.keys_to_load:
                    data_dict_view[Keys.optical_flows] = self._load(
                        view, "flow", "npz", video_name, frame_name
                    )
            
            data_dict[view] = data_dict_view

        return data_dict

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            dict[str, list[int]]: Mapping video to index.
        """
        if len(self.scalabel_datasets) > 0:
            return self.scalabel_datasets[list(self.scalabel_datasets.keys())[0]].video_to_indices
        raise ValueError("No Scalabel file has been loaded.")

    def get_video_indices(self, idx: int) -> list[int]:
        """Get all dataset indices in a video given a single dataset index."""
        for indices in self.video_to_indices.values():
            if idx in indices:
                return indices
        raise ValueError(f"Dataset index {idx} not found in video_to_indices!")
