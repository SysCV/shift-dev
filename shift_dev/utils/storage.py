import hashlib
import io
import os
import tarfile
import zipfile

import numpy as np
from pycocotools import mask as mask_utils  # type: ignore

from ..types.scalabel import RLE


class ZipArchiveReader:
    def __init__(self, filename) -> None:
        self.file = zipfile.ZipFile(filename, 'r')
        # print(f"Loaded {filename}.")
        
    def get_file(self, name):
        data = self.file.read(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def get_list(self):
        return self.file.namelist()

    def close(self):
        self.file.close()


class TarArchiveReader:
    def __init__(self, filename) -> None:
        self.file = tarfile.TarFile(filename, 'r')
        # print(f"Loaded {filename}.")
        
    def get_file(self, name):
        data = self.file.extractfile(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def extract_file(self, name, output_dir):
        self.file.extract(name, output_dir)

    def get_list(self):
        return self.file.getnames()

    def close(self):
        self.file.close()


def string_hash(video):
    sha = hashlib.sha512(video.encode('utf-8'))
    return int(sha.hexdigest(), 16)


def mask_to_rle(mask) -> RLE:
    """Converting mask to RLE format."""
    assert 2 <= len(mask.shape) <= 3
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    rle = mask_utils.encode(np.array(mask, order="F", dtype="uint8"))[0]
    return RLE(counts=rle["counts"].decode("utf-8"), size=rle["size"])