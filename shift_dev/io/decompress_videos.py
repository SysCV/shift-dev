"""Decompress video into image frames."""

import argparse
import glob
import multiprocessing as mp
import os
import shutil
from functools import partial

import cv2
import h5py
import numpy as np
import tqdm

if cv2.__version__[0] != "4":
    print("Please upgrade your OpenCV package to 4.x.")
    exit(1)

from ..download import DATA_GROUPS, VIEWS
from ..utils.logs import setup_logger
from ..utils.storage import (TarArchiveReader, TarArchiveWriter,
                             ZipArchiveWriter)

DATA_GROUP_NAMES = [item[0] for item in DATA_GROUPS]
VIEW_NAMES = [item[0] for item in VIEWS]


def get_suffix(tar_file):
    filepath, filename = os.path.split(tar_file.filename)
    group_name = os.path.splitext(filename)[0]
    view_name = os.path.split(filepath)[1]
    assert (view_name in VIEW_NAMES) and (
        group_name in DATA_GROUP_NAMES
    ), f"It seems that {filename} doesn't follow the dataset structure."
    if group_name == "img":
        ext = "jpg"
    else:
        ext = "png"
    return view_name, group_name, ext


def extract_video(tar_file, video_name, output_dir, tmp_dir):
    view_name, group_name, ext = get_suffix(tar_file)
    tar_file.extract_file(video_name, tmp_dir)
    video = cv2.VideoCapture(os.path.join(tmp_dir, video_name))
    if not video.isOpened():
        logger.error("Error opening video stream or file!")
    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    "{:08d}_{}_{}.{}".format(frame_id, group_name, view_name, ext),
                ),
                frame,
            )
            frame_id += 1
        else:
            break
    video.release()
    os.remove(os.path.join(tmp_dir, video_name))


def convert_to_archive(
    tar_filepath, tmp_dir, show_progress_bar=False, writer=TarArchiveWriter
):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return
    try:
        out_filepath = tar_filepath.replace(
            ".tar", f"_decompressed.{writer.default_ext}"
        )
        archive_writer = writer(out_filepath)
    except Exception as e:
        logger.error("Cannot create {}. ".format(out_filepath) + e)
        return

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", "_tmp"),
                os.path.basename(f).split(".")[0],
            )
            os.makedirs(output_dir, exist_ok=True)
            extract_video(tar_file, f, output_dir, tmp_dir)
            archive_writer.add_file(
                output_dir,
                arcname=os.path.basename(f).split(".")[0],
            )
            shutil.rmtree(output_dir)
    tar_file.close()
    archive_writer.close()


def convert_to_hdf5(tar_filepath, tmp_dir, show_progress_bar=False):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return
    try:
        hdf5_filepath = tar_filepath.replace(".tar", "_decompressed.hdf5")
        hdf5_file = h5py.File(hdf5_filepath, mode="w")
    except Exception as e:
        logger.error("Cannot create {}. ".format(hdf5_filepath) + e)
        return

    def write_to_hdf5(seq, folder_path):
        for f in os.listdir(folder_path):
            if seq in hdf5_file:
                g = hdf5_file[seq]
            else:
                g = hdf5_file.create_group(seq)
            with open(os.path.join(folder_path, f), "rb") as fp:
                file_content = fp.read()
                g.create_dataset(f, data=np.frombuffer(file_content, dtype="uint8"))

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", "_tmp"),
                os.path.basename(f).split(".")[0],
            )
            os.makedirs(output_dir, exist_ok=True)
            extract_video(tar_file, f, output_dir, tmp_dir)
            write_to_hdf5(os.path.splitext(f)[0], output_dir)
            shutil.rmtree(output_dir)
    tar_file.close()
    hdf5_file.close()


def convert_to_folder(tar_filepath, tmp_dir, show_progress_bar=False):
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
            extract_video(tar_file, f, output_dir, tmp_dir)


CONVERT_MAP = dict(
    folder=convert_to_folder,
    tar=partial(convert_to_archive, writer=TarArchiveWriter),
    zip=partial(convert_to_archive, writer=ZipArchiveWriter),
    hdf5=convert_to_hdf5,
)


def main():
    parser = argparse.ArgumentParser(
        description="Decompress tar files of videos into image frames."
    )
    parser.add_argument("files", type=str, help="File pattern to match tar files.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="folder",
        choices=["folder", "tar", "zip", "hdf5"],
        help="Conversion mode. Defines the type of output.",
    )
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

    files = []
    for file in glob.glob(args.files, recursive=True):
        if file.endswith("_decompressed.tar"):
            logger.warning(f"Skip a decompressed tar file: {file}.")
        else:
            files.append(file)

    os.makedirs(args.tmp_dir, exist_ok=True)
    logger.info("Files to convert: " + str(len(files)))
    logger.info(f"Starting conversion to {args.mode}")
    convert = CONVERT_MAP[args.mode]

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
