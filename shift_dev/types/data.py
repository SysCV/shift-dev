"""Defines data related constants.

While the datasets can hold arbitrary data types and formats, this file
provides some constants that are used to define a common data format which is
helpful to use for better data transformation.
"""
from dataclasses import dataclass
from enum import Enum


class AxisMode(Enum):
    """Enum for choosing among different coordinate frame conventions.

    ROS: The coordinate frame aligns with the right hand rule:
        x axis points forward
        y axis points left
        z axis points up
    See also: https://www.ros.org/reps/rep-0103.html#axis-orientation

    OpenCV: The coordinate frame aligns with a camera coordinate system:
        x axis points right
        y axis points down
        z axis points forward
    See also: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    """

    ROS = 0
    OPENCV = 1


@dataclass
class Keys:
    """Common supported keys for DictData.

    While DictData can hold arbitrary keys of data, we define a common set of
    keys where we expect a pre-defined format to enable the usage of common
    data pre-processing operations among different datasets.

    images (Tensor): Image of shape [1, C, H, W].
    original_hw (Tuple[int, int]): Original shape of image in (height, width).
    input_hw (Tuple[int, int]): Shape of image in (height, width) after
        transformations.
    frame_ids (int): If the dataset contains videos, this field indicates the
        temporal frame index of the current image / sample.

    boxes2d (Tensor): 2D bounding boxes of shape [N, 4] in xyxy format.
    boxes2d_classes (Tensor): Semantic classes of 2D bounding boxes, shape
        [N,].
    boxes2d_track_ids (Tensor): Tracking IDs of 2D bounding boxes, shape [N,].
    masks (Tensor): Instance segmentation masks of shape [N, H, W].
    segmentation_masks (Tensor):

    intrinsics (Tensor): Intrinsic sensor calibration. Shape [3, 3].
    extrinsics (Tensor): Extrinsic sensor calibration, transformation of sensor
        to world coordinate frame. Shape [4, 4].
    axis_mode (AxisMode): Coordinate convention of the current sensor.
    timestamp (int): Sensor timestamp in Unix format.

    points3d (Tensor): 3D pointcloud data, assumed to be [N, 3] and in sensor
        frame.

    boxes3d (Tensor): [N, 10], each row consists of center (XYZ), dimensions
        (WLH), and orientation quaternion (WXYZ).
    boxes3d_classes (Tensor): Associated semantic classes of 3D bounding boxes,
        [N,].
    boxes3d_track_ids (Tensor): Associated tracking IDs of 3D bounding boxes,
    """

    # image based inputs
    images = "images"
    original_hw = "original_hw"
    input_hw = "input_hw"
    frame_ids = "frame_ids"

    # 2D annotations
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    boxes2d_track_ids = "boxes2d_track_ids"
    masks = "masks"
    segmentation_masks = "segmentation_masks"
    depth_maps = "depth_maps"
    optical_flows = "optical_flows"

    # Image Classification
    categories = "categories"

    # sensor calibration
    intrinsics = "intrinsics"
    extrinsics = "extrinsics"
    axis_mode = "axis_mode"
    timestamp = "timestamp"

    # 3D data
    points3d = "points3d"

    # 3D annotation
    boxes3d = "boxes3d"
    boxes3d_classes = "boxes3d_classes"
    boxes3d_track_ids = "boxes3d_track_ids"
