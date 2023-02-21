"""

Config example:

    >>> dataset = ScalabelDataset(
    >>>     data_root='./shift_dataset/discrete/images'
    >>>     ann_file='train/front/det_2d.json',
    >>>     img_prefix='train/front/img',
    >>>     pipeline=[
    >>>        ...
    >>>     ]
    >>> )

"""

import json
import os
import numpy as np
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from shift_dev.utils.backend import ZipBackend, HDF5Backend


@DATASETS.register_module()
class SHIFTDataset(CustomDataset):

    CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')

    WIDTH = 1280
    HEIGHT = 800

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)

        data_infos = []
        for img_info in data['frames']:
            img_filename = os.path.join(self.img_prefix, img_info['name'])

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

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
        
    def __getitem__(self, idx):
        img = mmcv.imread(self.data_infos[idx]['filename'])
        ann = self.get_ann_info(idx)
        return self.pipeline(dict(img=img, img_info=img_info, ann_info=ann))


def main():
    """Load the SHIFT dataset and print the tensor shape of the first batch."""

    dataset = SHIFTDataset(
        data_root='../../SHIFT_dataset/v2/public/discrete/images',
        ann_file='train/front/det_2d.json',
        img_prefix='train/front/img',
        pipeline=[],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, batch in enumerate(dataset):
        print(f"Batch {i}:")
        for k, data in batch.items():
            if isinstance(data, torch.Tensor):
                print(f"{k}: {data.shape}")
            else:
                print(f"{k}: {data}")
        break    


if __name__ == "__main__":
    main()