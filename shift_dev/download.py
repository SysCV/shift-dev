#!/usr/bin/env python

"""
Download script for SHIFT Dataset.

The data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.
Homepage: www.vis.xyz/shift/.
(C)2022, VIS Group, ETH Zurich.


Script usage example:
    python download.py --view  "[front, left_stereo]" \     # list of view abbreviation to download
                       --group "[img, semseg]" \            # list of data group abbreviation to download 
                       --split "[train, val, test]" \       # list of split to download 
                       --framerate "[images, videos]" \     # chooses the desired frame rate (images=1fps, videos=10fps)
                       --shift "discrete" \                 # type of domain shifts. Options: discrete, continuous/1x, continuous/10x, continuous/100x 
                       dataset_root                         # path where to store the downloaded data

You can set the option to "all" to download the entire data from this option. For example,
    python download.py --view "all" --group "[img]" --split "all" --framerate "[images]" .
downloads the entire RGB images from the dataset.  
"""

import argparse
import logging
import os
import sys
import tempfile

import tqdm

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib


BASE_URL = "https://dl.cv.ethz.ch/shift/"

FRAME_RATES = [("images", "images (1 fps)"), ("videos", "videos (10 fps)")]

SPLITS = [
    ("train", "training set"),
    ("val", "validation set"),
    ("minival", "mini validation set (for online evaluation)"),
    ("test", "testing set"),
    ("minitest", "mini testing set (for online evaluation)"),
]

VIEWS = [
    ("front", "Front"),
    ("left_45", "Left 45°"),
    ("left_90", "Left 90°"),
    ("right_45", "Right 45°"),
    ("right_90", "Right 90°"),
    ("left_stereo", "Front (Stereo)"),
    ("center", "Center (for LiDAR)"),
]

DATA_GROUPS = [
    ("img", "zip", "RGB Image"),
    ("det_2d", "json", "2D Detection and Tracking"),
    ("det_3d", "json", "3D Detection and Tracking"),
    ("semseg", "zip", "Semantic Segmentation"),
    ("det_insseg_2d", "json", "Instance Segmentation"),
    ("flow", "zip", "Optical Flow"),
    ("depth", "zip", "Depth Maps (24-bit)"),
    ("depth_8bit", "zip", "Depth Maps (8-bit)"),
    ("seq", "csv", "Sequence Info"),
    ("lidar", "zip", "LiDAR Point Cloud"),
]


class ProgressBar(tqdm.tqdm):
    def update_to(self, batch=1, batch_size=1, total=None):
        if total is not None:
            self.total = total
        self.update(batch * batch_size - self.n)


def setup_logger():
    log_formatter = logging.Formatter(
        "[%(asctime)s] SHIFT Downloader - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    return logger


def get_url_discrete(rate, split, view, group, ext):
    url = BASE_URL + "discrete/{rate}/{split}/{view}/{group}.{ext}".format(
        rate=rate, split=split, view=view, group=group, ext=ext
    )
    return url


def get_url_continuous(rate, shift_length, split, view, group, ext):
    url = BASE_URL + "continuous/{rate}/{shift_length}/{split}/{view}/{group}.{ext}".format(
        rate=rate, shift_length=shift_length, split=split, view=view, group=group, ext=ext
    )
    return url


def string_to_list(option_str):
    option_str = option_str.replace(" ", "").lstrip("[").rstrip("]")
    return option_str.split(",")


def parse_options(option_str, bounds, name):
    if option_str == "all":
        return bounds
    candidates = {}
    for item in bounds:
        candidates[item[0]] = item
    used = []
    try:
        option_list = string_to_list(option_str)
    except Exception as e:
        logger.error("Error in parsing options." + e)
    for option in option_list:
        if option not in candidates:
            logger.info(
                "Invalid option '{option}' for '{name}'. ".format(option=option, name=name)
                + "Please check the download document (https://www.vis.xyz/shift/download/)."
            )
        else:
            used.append(candidates[option])
    if len(used) == 0:
        logger.error(
            "No '{name}' is specified to download. ".format(name=name)
            + "If you want to download all {name}s, please use '--{name} all'.".format(name=name)
        )
        sys.exit(1)
    return used


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        logging.info("downloading " + url)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, "w")
        f.close()
        filename = url.split("/")[-1]
        with ProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
            urllib.urlretrieve(url, out_file_tmp, reporthook=t.update_to)
        os.rename(out_file_tmp, out_file)
    else:
        logger.warning("Skipping download of existing file " + out_file)


def main():
    parser = argparse.ArgumentParser(description="Downloads SHIFT Dataset public release.")
    parser.add_argument("out_dir", help="output directory in which to store the data.")
    parser.add_argument("--split", type=str, default="", help="specific splits to download.")
    parser.add_argument("--view", type=str, default="", help="specific views to download.")
    parser.add_argument("--group", type=str, default="", help="specific data groups to download.")
    parser.add_argument("--framerate", type=str, default="", help="specific frame rate to download.")
    parser.add_argument(
        "--shift",
        type=str,
        default="discrete",
        choices=["discrete", "continuous/1x", "continuous/10x", "continuous/100x"],
        help="specific shift type to download.",
    )
    args = parser.parse_args()

    print(
        "Welcome to use SHIFT Dataset download script! \n"
        "By continuing you confirm that you have agreed to the SHIFT's user license.\n"
    )

    frame_rates = parse_options(args.framerate, FRAME_RATES, "frame rate")
    splits = parse_options(args.split, SPLITS, "split")
    views = parse_options(args.view, VIEWS, "view")
    data_groups = parse_options(args.group, DATA_GROUPS, "data group")
    total_files = len(frame_rates) * len(splits) * len(views) * len(data_groups)
    logger.info("Number of files to download: " + str(total_files))

    if "lidar" in data_groups and views != ["center"]:
        logger.error("LiDAR data only available for Center view!")
        sys.exit(1)

    for rate, rate_name in frame_rates:
        for split, split_name in splits:
            for view, view_name in views:
                for group, ext, group_name in data_groups:
                    if rate == "videos" and group in ["img"]:
                        ext = "tar"
                    if args.shift == "discrete":
                        url = get_url_discrete(rate, split, view, group, ext)
                        out_file = os.path.join(args.out_dir, "discrete", rate, split, view, group + "." + ext)
                    else:
                        shift_length = args.shift.split("/")[-1]
                        url = get_url_continuous(rate, shift_length, split, view, group, ext)
                        out_file = os.path.join(
                            args.out_dir, "continuous", rate, shift_length, split, view, group + "." + ext
                        )
                    logger.info(
                        "Downloading - Shift: {shift}, Framerate: {rate}, Split: {split}, View: {view}, Data group: {group}.".format(
                            shift=args.shift,
                            rate=rate_name,
                            split=split_name,
                            view=view_name,
                            group=group_name,
                            url=url,
                        )
                    )
                    try:
                        download_file(url, out_file)
                    except Exception as e:
                        logger.error("Error in downloading " + str(e))

    logger.info("Done!")


if __name__ == "__main__":
    logger = setup_logger()
    main()
