<h1 align="center"> SHIFT Dataset DevKit </h1>

This repo contains tools and scripts for [SHIFT Dataset](https://www.vis.xyz/shift/)'s downloading, conversion, and more!

<div align="center">
<div></div>

| **RGB**          |    **Optical Flow**    | **Depth**   | **LiDAR** |
|:----------------:|:----------------:|:----------------:|:---------:|
|  <img src="assert/figures/img.png">                |       <img src="assert/figures/flow.png">     |   <img src="assert/figures/depth.png">                       |   <img src="assert/figures/lidar.png" >         |
|   **Bounding box** | **Instance Segm.** | **Semantic Segm.**  | **Body Pose (soon)**  |
|   <img src="assert/figures/bbox2d.png">                 |     <img src="assert/figures/ins.png">            |         <img src="assert/figures/seg.png">           |       <img src="assert/figures/pose.png">      |

</div>

## News



## Data downloading
We recommend to download SHIFT using our Python download script. You can select the subset of views, data group, splits and framerates of the data to download. A usage example is shown below. You can find the abbreviation for views and data groups in the following tables.

```
python download.py --view  "[front, left_stereo]" \    # list of view abbreviation to download
                   --group "[img, semseg]" \          # list of data group abbreviation to download 
                   --split "[train, val, test]" \     # list of splits to download 
                   --framerate "[images, videos]" \   # chooses the desired frame rate (images=1fps, videos=10fps)
                   dataset_root                       # path where to store the downloaded data
```

## Tools

### Pack zip files into HDF5 file



### Coordinate systems
<p align="center"> 
  <img src="assert/figures/coor_sys.png" alt="Coordinate systems" width="100%">
</p>





Copyright © 2022, [Tao Sun](https://suniique.com) ([@suniique](https://github.com/suniique)), [ETH VIS Group](https://cv.ethz.ch/).