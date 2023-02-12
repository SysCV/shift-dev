"""Pack a data group's zip file into an hdf5 file."""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
import tqdm

from ..utils.logs import setup_logger
from ..utils.storage import ZipArchiveReader


def convert_from_zip(zip_filepath, show_progress_bar=False):
    try:
        zip_file = ZipArchiveReader(zip_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(zip_filepath) + e)
        return
    try:
        hdf5_filepath = zip_filepath.replace(".zip", ".hdf5")
        hdf5_file = h5py.File(hdf5_filepath, mode="w")
    except Exception as e:
        logger.error("Cannot create {}. ".format(hdf5_filepath) + e)
        return

    file_list = zip_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        seq, frame = f.split("/")
        file_content = zip_file.get_file(f)
        bytes = np.frombuffer(file_content.read(), dtype="uint8")
        if seq in hdf5_file:
            g = hdf5_file[seq]
        else:
            g = hdf5_file.create_group(seq)
        g.create_dataset(frame, data=bytes)
    hdf5_file.close()


def convert_from_folder(path, show_progress_bar=False):
    try:
        hdf5_filepath = path.rstrip("/") + ".hdf5"
        hdf5_file = h5py.File(hdf5_filepath, mode="w")
        print(hdf5_filepath)
    except Exception as e:
        logger.error("Cannot create {}. ".format(hdf5_filepath) + e)
        return

    file_list = glob.glob(os.path.join(path, "*", "*"))
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        seq, frame = f.split("/")[-2:]
        with open(f, "rb") as fp:
            file_content = fp.read()
        bytes = np.frombuffer(file_content, dtype="uint8")
        if seq in hdf5_file:
            g = hdf5_file[seq]
        else:
            g = hdf5_file.create_group(seq)
        g.create_dataset(frame, data=bytes)
    hdf5_file.close()


def main():
    parser = argparse.ArgumentParser(description="Convert zip files to HDF5 files.")
    parser.add_argument(
        "path", type=str, help="Path pattern to match the zip files or folders."
    )
    parser.add_argument(
        "--zip", action="store_true", help="Whether process zip files, or folders."
    )
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="Number of jobs to run in parallel."
    )
    args = parser.parse_args()
    print(args.path)
    if args.zip:
        if args.path[-4:] != ".zip":
            logger.info("Path pattern must end with '.zip'!")
            sys.exit(1)
    else:
        if args.path[-1] != "/":
            logger.info("Path pattern must end with '/'!")
            sys.exit(1)
    files = glob.glob(args.path, recursive=True)
    logger.info("Files/folders to convert: " + str(len(files)))

    if args.zip:
        convert = convert_from_zip
    else:
        convert = convert_from_folder
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            _ = list(tqdm.tqdm(pool.imap(convert, files), total=len(files)))
    else:
        logger.info(
            "Note: You can also run this code using multi-processing by setting `-j` option."
        )
        for f in files:
            logger.info("Processing " + f)
            convert(f, show_progress_bar=True)


if __name__ == "__main__":
    logger = setup_logger()
    main()
