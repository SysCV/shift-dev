import numpy as np

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception as e:
    print("Error during importing pycocotools:", e, "\nconsider to reinstall the pycocotools manually.")

from ..types.scalabel import RLE


def mask_to_rle(mask) -> RLE:
    """Converting mask to RLE format."""
    assert 2 <= len(mask.shape) <= 3
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    rle = mask_utils.encode(np.array(mask, order="F", dtype="uint8"))[0]
    return RLE(counts=rle["counts"].decode("utf-8"), size=rle["size"])
