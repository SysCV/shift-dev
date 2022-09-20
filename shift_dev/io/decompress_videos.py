"""Decompress video into image frames."""

import argparse
import glob
import multiprocessing as mp
import os
from functools import partial

import cv2

if cv2.__version__[0] != "4":
    print("Please upgrade your OpenCV package to 4.x.")
    exit(1)

import numpy as np
import tqdm

from ..utils.logs import setup_logger
from ..utils.storage import TarArchiveReader


def convert(tar_filepath, tmp_dir, show_progress_bar=False):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", ""), f.replace(".mp4", "")
            )
            os.makedirs(output_dir, exist_ok=True)
            convert_from_tar(tar_file, f, output_dir, tmp_dir)


def convert_from_tar(tar_file, video_name, output_dir, tmp_dir):
    tar_file.extract_file(video_name, tmp_dir)
    video = cv2.VideoCapture(os.path.join(tmp_dir, video_name))
    if not video.isOpened():
        logger.error("Error opening video stream or file!")
    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, "{:08d}.png".format(frame_id)), frame)
            frame_id += 1
        else:
            break
    video.release()
    os.remove(os.path.join(tmp_dir, video_name))


def main():
    parser = argparse.ArgumentParser(
        description="Decompress tar files of videos into image frames."
    )
    parser.add_argument("files", type=str, help="File pattern to match tar files.")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--tmp_dir",
        default="/tmp/shift-dataset/",
        help="Temporary folder for decompressed video files.",
    )
    args = parser.parse_args()

    if args.files[-4:] != ".tar":
        logger.error("File pattern must end with '.tar'!")
        exit()
    files = glob.glob(args.files)
    logger.info("Files to convert: " + str(len(files)))

    os.makedirs(args.tmp_dir, exist_ok=True)
    if args.jobs > 1:
        convert_fn = partial(convert, tmp_dir=args.tmp_dir)
        with mp.Pool(args.jobs) as pool:
            _ = list(tqdm.tqdm(pool.imap(convert_fn, files), total=len(files)))
    else:
        logger.info(
            "Note: You can also run this code using multi-processing by setting `-j` option."
        )
        for f in files:
            logger.info("Processing " + f)
            convert(f, args.tmp_dir, show_progress_bar=True)


if __name__ == "__main__":
    logger = setup_logger()
    main()
