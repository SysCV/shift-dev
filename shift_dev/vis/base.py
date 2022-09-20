import json
import os

import numpy as np
from PIL import Image

from ..types.scalabel import Frame
from ..utils.storage import ZipArchiveReader


class _BaseRender:
    def __init__(self, data_dir=None, label_path=None):
        self.data_dir = data_dir
        self.label_path = label_path
        if data_dir and data_dir.endswith(".zip"):
            self.zip = ZipArchiveReader(self.data_dir)
        else:
            self.zip = None

    def read_scalabel(self, video):
        assert self.label_path is not None, "label_path is not specified!"
        scalabel = json.load(open(self.label_path))
        results = []
        for frame in scalabel["frames"]:
            if frame["videoName"] == video:
                results.append(Frame.parse_obj(frame))
        return results

    def read_image(self, video: str, frame: int, group: str, view: str, ext: str):
        assert self.data_dir is not None, "data_dir is not specified!"
        filename = f"{frame:08d}_{group}_{view}.{ext}"
        filepath = os.path.join(self.data_dir, video, filename)
        try:
            if self.zip is None:
                img = Image.open(filepath)
            else:
                img = Image.open(self.zip.get_file(video + "/" + filename))
        except OSError as e:
            print(e)
            return None
        img = np.asarray(img)
        return img[:, :, :3]  # RGB(A) -> RGB
