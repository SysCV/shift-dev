"""Scalabel type dataset."""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Dict, Union

import numpy as np
import torch
from scalabel.label.io import load, load_label_config
from scalabel.label.transforms import (
    box2d_to_xyxy, poly2ds_to_mask, rle_to_mask
)
from scalabel.label.typing import Config
from scalabel.label.typing import Dataset as ScalabelData
from scalabel.label.typing import (
    Extrinsics, Frame, ImageSize, Intrinsics, Label
)
from scalabel.label.utils import (
    check_crowd, check_ignored, get_leaf_categories,
    get_matrix_from_extrinsics, get_matrix_from_intrinsics
)
from torch import Tensor
from torch.utils.data import Dataset

from shift_dev.types import DataDict, DictStrAny, Keys, NDArrayU8
from shift_dev.utils import Timer, setup_logger
from shift_dev.utils.backend import DataBackend, FileBackend
from shift_dev.utils.load import im_decode, ply_decode

from .cache import CacheMappingMixin, DatasetFromList

logger = setup_logger()


def load_intrinsics(intrinsics: Intrinsics) -> Tensor:
    """Transform intrinsic camera matrix according to augmentations."""
    intrinsic_matrix = torch.from_numpy(get_matrix_from_intrinsics(intrinsics)).to(
        torch.float32
    )
    return intrinsic_matrix


def load_extrinsics(extrinsics: Extrinsics) -> Tensor:
    """Transform extrinsics from Scalabel to Vis4D."""
    extrinsics_matrix = torch.from_numpy(get_matrix_from_extrinsics(extrinsics)).to(
        torch.float32
    )
    return extrinsics_matrix


def load_image(url: str, backend: DataBackend) -> Tensor:
    """Load image tensor from url."""
    im_bytes = backend.get(url)
    image = im_decode(im_bytes)
    return torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1)),
        dtype=torch.float32,
    ).unsqueeze(0)


def load_pointcloud(url: str, backend: DataBackend) -> Tensor:
    """Load pointcloud tensor from url."""
    assert url.endswith(".ply"), "Only PLY files are supported now."
    ply_bytes = backend.get(url)
    pointcloud = ply_decode(ply_bytes)
    return torch.as_tensor(pointcloud, dtype=torch.float32)


def instance_ids_to_global(
    frames: list[Frame], local_instance_ids: Dict[str, list[str]]
) -> None:
    """Use local (per video) instance ids to produce global ones."""
    video_names = list(local_instance_ids.keys())
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:  # pragma: no cover
            continue
        for label in ann.labels:
            assert label.attributes is not None
            if not check_crowd(label) and not check_ignored(label):
                video_name = (
                    ann.videoName
                    if ann.videoName is not None
                    else "no-video-" + str(frame_id)
                )
                sum_previous_vids = sum(
                    (
                        len(local_instance_ids[v])
                        for v in video_names[: video_names.index(video_name)]
                    )
                )
                label.attributes[
                    "instance_id"
                ] = sum_previous_vids + local_instance_ids[video_name].index(label.id)


def add_data_path(data_root: str, frames: list[Frame]) -> None:
    """Add filepath to frame using data_root."""
    for ann in frames:
        assert ann.name is not None
        if ann.url is None:
            if ann.videoName is not None:
                ann.url = os.path.join(data_root, ann.videoName, ann.name)
            else:
                ann.url = os.path.join(data_root, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.url)


def prepare_labels(frames: list[Frame], global_instance_ids: bool = False) -> None:
    """Add category id and instance id to labels, return class frequencies."""
    instance_ids: Dict[str, list[str]] = defaultdict(list)
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:
            continue

        for label in ann.labels:
            attr: Dict[str, bool | int | float | str] = {}
            if label.attributes is not None:
                attr = label.attributes

            if check_crowd(label) or check_ignored(label):
                continue

            assert label.category is not None
            video_name = (
                ann.videoName
                if ann.videoName is not None
                else "no-video-" + str(frame_id)
            )
            if label.id not in instance_ids[video_name]:
                instance_ids[video_name].append(label.id)
            attr["instance_id"] = instance_ids[video_name].index(label.id)
            label.attributes = attr

    if global_instance_ids:
        instance_ids_to_global(frames, instance_ids)


# Not using | operator because of a bug in Python 3.9
# https://bugs.python.org/issue42233
CategoryMap = Union[Dict[str, int], Dict[str, Dict[str, int]]]


class Scalabel(Dataset, CacheMappingMixin):
    """Scalabel type dataset.

    This class loads scalabel format data into Vis4D.
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        keys_to_load: Sequence[str] = (
            Keys.images,
            Keys.boxes2d,
        ),
        data_backend: None | DataBackend = None,
        category_map: None | CategoryMap = None,
        config_path: None | str = None,
        global_instance_ids: bool = False,
        bg_as_class: bool = False,
        use_cache: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of the data.
            annotation_path (str): Path to the annotation json(s).
            keys_to_load (Sequence[str, ...], optional): Keys to load from the
                dataset. Defaults to (Keys.images, Keys.boxes2d).
            data_backend (None | DataBackend, optional): Data backend, if None
                then classic file backend. Defaults to None.
            category_map (None | CategoryMap, optional): Mapping from a
                Scalabel category string to an integer index. If None, the
                standard mapping in the dataset config will be used. Defaults
                to None.
            config_path (None | str, optional): Path to the dataset config, can
                be added if it is not provided together with the labels or
                should be modified. Defaults to None.
            global_instance_ids (bool): Whether to convert tracking IDs of
                annotations into dataset global IDs or stay with local,
                per-video IDs. Defaults to false.
            bg_as_class (bool): Whether to include background pixels as an
                additional class for masks.
        """
        super().__init__()
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.keys_to_load = keys_to_load
        self.global_instance_ids = global_instance_ids
        self.bg_as_class = bg_as_class
        self.use_cache = use_cache
        self.data_backend = data_backend if data_backend is not None else FileBackend()
        self.config_path = config_path
        self.frames, self.cfg = self._load_mapping(self._generate_mapping, use_cache)

        assert self.cfg is not None, (
            "No dataset configuration found. Please provide a configuration "
            "via config_path."
        )

        self.cats_name2id: Dict[str, Dict[str, int]] = {}
        if category_map is None:
            class_list = list(c.name for c in get_leaf_categories(self.cfg.categories))
            assert len(set(class_list)) == len(
                class_list
            ), "Class names are not unique!"
            category_map = {c: i for i, c in enumerate(class_list)}
        self._setup_categories(category_map)

    def _setup_categories(self, category_map: CategoryMap) -> None:
        """Setup categories."""
        for target in self.keys_to_load:
            if isinstance(list(category_map.values())[0], int):
                self.cats_name2id[target] = category_map  # type: ignore
            else:
                assert (
                    target in category_map
                ), f"Target={target} not specified in category_mapping"
                target_map = category_map[target]
                assert isinstance(target_map, dict)
                self.cats_name2id[target] = target_map

    def _load_mapping(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> tuple[Dataset, Config]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        data = self._load_mapping_data(generate_map_func, use_cache)
        frames, cfg = data.frames, data.config  # type: ignore
        add_data_path(self.data_root, frames)
        prepare_labels(frames, global_instance_ids=self.global_instance_ids)
        frames = DatasetFromList(frames)
        logger.info(f"Loading annotation takes {timer.time():.2f} seconds.")
        return frames, cfg

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        data = load(self.annotation_path)
        if self.config_path is not None:
            data.config = load_label_config(self.config_path)
        return data

    def _load_inputs(self, frame: Frame) -> DictData:
        """Load inputs given a scalabel frame."""
        data: DictData = {}
        if frame.url is not None and Keys.images in self.keys_to_load:
            image = load_image(frame.url, self.data_backend)
            input_hw = (image.shape[2], image.shape[3])
            data[Keys.images] = image
            data[Keys.original_hw] = input_hw
            data[Keys.input_hw] = input_hw
            data[Keys.frame_ids] = frame.frameIndex
            # TODO how to properly integrate such metadata?
            data["name"] = frame.name
            data["videoName"] = frame.videoName

        if frame.url is not None and Keys.points3d in self.keys_to_load:
            data[Keys.points3d] = load_pointcloud(frame.url, self.data_backend)

        if frame.intrinsics is not None and Keys.intrinsics in self.keys_to_load:
            data[Keys.intrinsics] = load_intrinsics(frame.intrinsics)

        if frame.extrinsics is not None and Keys.extrinsics in self.keys_to_load:
            data[Keys.extrinsics] = load_extrinsics(frame.extrinsics)
        return data

    def _add_annotations(self, frame: Frame, data: DictData) -> None:
        """Add annotations given a scalabel frame and a data dictionary."""
        if frame.labels is None:
            return
        labels_used, instid_map = [], {}
        for label in frame.labels:
            assert label.attributes is not None and label.category is not None
            if not check_crowd(label) and not check_ignored(label):
                labels_used.append(label)
                if label.id not in instid_map:
                    instid_map[label.id] = int(label.attributes["instance_id"])
        # if not labels_used:
        #     return  # pragma: no cover

        image_size = (
            ImageSize(height=data[Keys.input_hw][0], width=data[Keys.input_hw][0])
            if Keys.input_hw in data
            else frame.size
        )

        if Keys.boxes2d in self.keys_to_load:
            cats_name2id = self.cats_name2id[Keys.boxes2d]
            boxes2d, classes, track_ids = boxes2d_from_scalabel(
                labels_used, cats_name2id, instid_map
            )
            data[Keys.boxes2d] = boxes2d
            data[Keys.boxes2d_classes] = classes
            data[Keys.boxes2d_track_ids] = track_ids

        if Keys.masks in self.keys_to_load:
            # NOTE: instance masks' mapping is consistent with boxes2d
            cats_name2id = self.cats_name2id[Keys.masks]
            instance_masks = instance_masks_from_scalabel(
                labels_used,
                cats_name2id,
                image_size=image_size,
                bg_as_class=self.bg_as_class,
            )
            data[Keys.masks] = instance_masks

        if Keys.boxes3d in self.keys_to_load:
            boxes3d, classes, track_ids = boxes3d_from_scalabel(
                labels_used, self.cats_name2id[Keys.boxes3d], instid_map
            )
            data[Keys.boxes3d] = boxes3d
            data[Keys.boxes3d_classes] = classes
            data[Keys.boxes3d_track_ids] = track_ids

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.frames)

    def __getitem__(self, index: int) -> DictData:
        """Get item from dataset at given index."""
        frame = self.frames[index]  # type: Frame
        data = self._load_inputs(frame)
        if len(self.keys_to_load) > 0:
            if len(self.cats_name2id) == 0:
                raise AttributeError(
                    "Category mapping is empty but keys_to_load is not. "
                    "Please specify a category mapping."
                )
            # load annotations to input sample
            self._add_annotations(frame, data)
        return data

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Group all dataset sample indices (int) by their video ID (str).

        Returns:
            dict[str, list[int]]: Mapping video to index.
        """
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frameidx: dict[str, list[int]] = defaultdict(list)
        for idx, frame in enumerate(self.frames):
            if frame.videoName is not None:
                assert (
                    frame.frameIndex is not None
                ), "found videoName but no frameIndex!"
                video_to_frameidx[frame.videoName].append(frame.frameIndex)
                video_to_indices[frame.videoName].append(idx)

        # sort dataset indices by frame indices
        for key, idcs in video_to_indices.items():
            zip_frame_idx = sorted(zip(video_to_frameidx[key], idcs))
            video_to_indices[key] = [idx for _, idx in zip_frame_idx]
        return video_to_indices

    def get_video_indices(self, idx: int) -> list[int]:
        """Get all dataset indices in a video given a single dataset index."""
        for indices in self.video_to_indices.values():
            if idx in indices:
                return indices
        raise ValueError(f"Dataset index {idx} not found in video_to_indices!")



def boxes3d_from_scalabel(
    labels: list[Label],
    class_to_idx: Dict[str, int],
    label_id_to_idx: Dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert 3D bounding boxes from scalabel format to Vis4D."""
    box_list, cls_list, idx_list = [], [], []
    for i, label in enumerate(labels):
        box, box_cls, l_id = label.box3d, label.category, label.id
        if box is None:
            continue
        if box_cls in class_to_idx:
            cls_list.append(class_to_idx[box_cls])
        else:
            continue

        box_list.append([*box.location, *box.dimension, *box.orientation])
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:
        return (
            torch.empty(0, 10),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    class_ids = torch.tensor(cls_list, dtype=torch.long)
    track_ids = torch.tensor(idx_list, dtype=torch.long)
    return box_tensor, class_ids, track_ids


def boxes2d_from_scalabel(
    labels: list[Label],
    class_to_idx: Dict[str, int],
    label_id_to_idx: Dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert from scalabel format to Vis4D.

    NOTE: The box definition in Scalabel includes x2y2 in the box area, whereas
    Vis4D and other software libraries like detectron2 and mmdet do not include
    this, which is why we convert via box2d_to_xyxy.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (Dict[str, int]): mapping from class name to index.
        label_id_to_idx (Dict[str, int] | None, optional): mapping from label
            id to index. Defaults to None.

    Returns:
        tuple[Tensor, Tensor, Tensor]: boxes, classes, track_ids
    """
    box_list, cls_list, idx_list = [], [], []
    for i, label in enumerate(labels):
        box, box_cls, l_id = (
            label.box2d,
            label.category,
            label.id,
        )
        if box is None:
            continue
        if box_cls in class_to_idx:
            cls_list.append(class_to_idx[box_cls])
        else:
            continue

        box_list.append(box2d_to_xyxy(box))
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:
        return (
            torch.empty(0, 4),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    class_ids = torch.tensor(cls_list, dtype=torch.long)
    track_ids = torch.tensor(idx_list, dtype=torch.long)
    return box_tensor, class_ids, track_ids


def instance_masks_from_scalabel(
    labels: list[Label],
    class_to_idx: Dict[str, int],
    image_size: ImageSize | None = None,
    bg_as_class: bool = False,
) -> Tensor:
    """Convert from scalabel format to Vis4D.

    Args:
        labels (list[Label]): list of scalabel labels.
        class_to_idx (Dict[str, int]): mapping from class name to index.
        image_size (ImageSize, optional): image size. Defaults to None.
        bg_as_class (bool, optional): whether to include background as a class.
            Defaults to False.

    Returns:
        Tensor: instance masks.
    """
    bitmask_list = []
    if bg_as_class:
        foreground = None
    for label in labels:
        if label.category not in class_to_idx:
            continue
        if label.poly2d is None and label.rle is None:
            continue
        if label.rle is not None:
            bitmask = rle_to_mask(label.rle)
        elif label.poly2d is not None:
            assert (
                image_size is not None
            ), "image size must be specified for masks with polygons!"
            bitmask_raw = poly2ds_to_mask(image_size, label.poly2d)
            bitmask: NDArrayU8 = (bitmask_raw > 0).astype(  # type: ignore
                bitmask_raw.dtype
            )
        bitmask_list.append(bitmask)
        if bg_as_class:
            foreground = (
                bitmask if foreground is None else np.logical_or(foreground, bitmask)
            )
    if bg_as_class:
        if foreground is None:  # pragma: no cover
            assert image_size is not None
            foreground = np.zeros((image_size.height, image_size.width), dtype=np.uint8)
        bitmask_list.append(np.logical_not(foreground))
    if len(bitmask_list) == 0:  # pragma: no cover
        return torch.empty(0, 0, 0, dtype=torch.uint8)
    mask_tensor = torch.tensor(np.array(bitmask_list), dtype=torch.uint8)
    return mask_tensor
