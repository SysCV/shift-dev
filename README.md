<h1 align="center"> SHIFT Dataset DevKit </h1>

This repo contains tools and scripts for [SHIFT Dataset](https://www.vis.xyz/shift/)'s downloading, conversion, and more!

[**Homepage**](https://www.vis.xyz/shift) | [**Paper (CVPR 2022)**](https://arxiv.org/abs/2206.08367) | [**Poster**](https://github.com/SysCV/shift-dev/blob/main/assert/Poster%20SHIFT.pdf) | [**Talk**](https://www.youtube.com/watch?v=q39gJveIhRc) | [**Demo**](https://www.youtube.com/watch?v=BsqGrDd2Kzw)


<div align="center">
<div></div>

| **RGB**          |    **Optical Flow**    | **Depth**   | **LiDAR** |
|:----------------:|:----------------:|:----------------:|:---------:|
|  <img src="assert/figures/img.png">                |       <img src="assert/figures/flow.png">     |   <img src="assert/figures/depth.png">                       |   <img src="assert/figures/lidar.png" >         |
|   **Bounding box** | **Instance Segm.** | **Semantic Segm.**  | **Body Pose (soon)**  |
|   <img src="assert/figures/bbox2d.png">                 |     <img src="assert/figures/ins.png">            |         <img src="assert/figures/seg.png">           |       <img src="assert/figures/pose.png">      |

</div>



## News

- **[Sept 2020]** We released visualization scripts for annotation and sensor pose (issue https://github.com/SysCV/shift-dev/issues/6).
- **[June 2020]** We released the DevKit repo!


## Downloading
We recommend to download SHIFT using our Python download script. You can select the subset of views, data group, splits and framerates of the data to download. A usage example is shown below. You can find the abbreviation for views and data groups in the following tables.

```bash
python download.py --view  "[front, left_stereo]" \   # list of view abbreviation to download
                   --group "[img, semseg]" \          # list of data group abbreviation to download 
                   --split "[train, val, test]" \     # list of splits to download 
                   --framerate "[images, videos]" \   # chooses the desired frame rate (images=1fps, videos=10fps)
                   --shift "discrete" \               # type of domain shifts. Options: discrete, continuous/1x, continuous/10x, continuous/100x 
                   dataset_root                       # path where to store the downloaded data
```
Example

The command below downloads the entire RGB images and semantic segmentation from the discrete shift data.
```bash
python download.py --view "all" --group "[img, semseg]" --split "all" --framerate "[images]" ./data
```


## Tools
### Pack zip file into HDF5
Instead of unzipping the the downloaded zip files, you can also can convert them into corresponding [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files. HDF5 file is designed to store a large of dataset in a single file and, meanwhile, to support efficient I/O for training purpose. Converting to HDF5 is a good practice in an environment where the number of files that can be stored are limited. Example command:
```bash
# for zip files
python -m shift_dev.io.to_hdf5 "discrete/images/val/left_45/*.zip" --zip -j 1
# or unzipped folder
python -m shift_dev.io.to_hdf5 "discrete/images/val/left_45/img/" -j 1
```
Note: The converted HDF5 file will maintain the same file structure of the zip file / folder, i.e., `<seq>/<frame>_<group>_<view>.<ext>`.

Below is a code snippet for reading one image from a HDF5 file.
```python
import io
import h5py
import numpy as np
from PIL import Image

file_key = "0123-abcd/00000001_img_front.jpg"
with h5py.File("/path/to/file.hdf5", "r") as hdf5:      # load the HDF5 file
    data = np.array(hdf5[file_key])                     # select the file we want
    img = Image.open(io.BytesIO(data))                  # same as opening an ordinary png file.
```

### Decompress video files
For easier retrieval of frames during training, we recommend to decompress all video sequences into image frames before training. Make sure there is enough disk space to store the decompressed frames.

- To use your local FFmpeg libraries (4.x) is supported but not recommended. You can follow the command example below,
    ```bash
    python -m shift_dev.io.decompress_videos "discrete/videos/val/left_45/*.tar" -j 1
    ```

- To ensure reproducible decompression of videos, we recommend to use our Docker image. You could refer to the Docker engine's [installation doc](https://docs.docker.com/engine/install/).
    ```bash
    # build and install our Docker image
    docker build -t shift-devkit .
    # run the container
    docker run -v <path/to/data>:/data shift-devkit
    ```
    Here, `<path/to/data>` denotes the root path under which all tar files will be processed recursively.

### Visualization

We provide a visualization tool for object-level labels (e.g., bounding box, instance segmentation). The main rendering functions are provided in `shift_dev/vis/render.py` file. We believe you can reuse many of them for other kinds of visualization. 

We also provide a tool to make video with annotations:
```bash
python -m shift_dev.vis.video <seq_id> \    # specify the video sequence
    -d <path/to/img.zip> \                  # path to the img.zip or its unzipped folder
    -l <path/to/label.json> \               # path to the corresponding label ({det_2d/det_3d/det_insseg_2d}.json)
    -o <path/for/output> \                  # output path
    --view front                            # specify the view, needed to be corresponded with images and label file
```
This command will render an MP4 video with the bounding boxes or instance masks plotted over the background images. Checkout the example [here](https://www.youtube.com/watch?v=BsqGrDd2Kzw) (starting from 00:10)!



## Coordinate systems
<p align="center"> 
  <img src="assert/figures/coor_sys.png" alt="Coordinate systems" width="100%">
</p>


## Citation

The SHIFT Dataset is made freely available to academic and non-academic entities for research purposes such as academic research, teaching, scientific publications, or personal experimentation. If you use our dataset, we kindly ask you to cite our paper as:

```
@InProceedings{shift2022,
    author    = {Sun, Tao and Segu, Mattia and Postels, Janis and Wang, Yuxuan and Van Gool, Luc and Schiele, Bernt and Tombari, Federico and Yu, Fisher},
    title     = {{SHIFT:} A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21371-21382}
}
```


Copyright © 2022, [Tao Sun](https://suniique.com) ([@suniique](https://github.com/suniique)), [ETH VIS Group](https://cv.ethz.ch/).
