"""Utility functions for data loading."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import numpy.typing as npt
import plyfile
from PIL import Image, ImageOps

NDArrayUI8 = npt.NDArray[np.uint8]
NDArrayF32 = npt.NDArray[np.float32]


def im_decode(im_bytes: bytes, mode: str = "RGB") -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in {
        "BGR",
        "RGB",
        "L",
    }, f"{mode} not supported for image decoding!"

    pil_img = Image.open(BytesIO(bytearray(im_bytes)))
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode == "L":  # pragma: no cover
        if mode == "L":
            img: NDArrayUI8 = np.array(pil_img)[..., None]
        else:
            # convert grayscale image to RGB
            pil_img = pil_img.convert("RGB")
    else:  # pragma: no cover
        if mode == "L":
            raise ValueError("Cannot convert colorful image to grayscale!")
    if mode == "BGR":  # pragma: no cover
        img = np.array(pil_img)[..., [2, 1, 0]]
    elif mode == "RGB":
        img = np.array(pil_img)
    return img


def ply_decode(ply_bytes: bytes, mode: str = "XYZI") -> NDArrayF32:
    """Decode to point clouds (numpy array) from bytes."""
    assert mode in {
        "XYZ",
        "XYZI",
    }, f"{mode} not supported for points decoding!"

    plydata = plyfile.PlyData.read(BytesIO(bytearray(ply_bytes)))
    num_points = plydata["vertex"].count
    num_channels = 3 if mode == "XYZ" else 4
    points = np.zeros((num_points, num_channels), dtype=np.float32)

    points[:, 0] = plydata["vertex"].data["x"]
    points[:, 1] = plydata["vertex"].data["y"]
    points[:, 2] = plydata["vertex"].data["z"]
    if mode == "XYZI":
        points[:, 3] = plydata["vertex"].data["intensity"]
    return points
