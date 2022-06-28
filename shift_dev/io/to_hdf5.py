"""Pack a data group's zip file into an hdf5 file."""

import argparse
import glob
import multiprocessing as mp

import h5py
import numpy as np
import tqdm

from ..utils.logs import setup_logger
from ..utils.storage import ZipArchiveReader


def convert(zip_filepath, show_progress_bar=False):
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
        print(seq, frame)
    hdf5_file.close()


def main():
    parser = argparse.ArgumentParser(description="Convert zip files to HDF5 files.")
    parser.add_argument(
        "files", type=str, help="File pattern to match zip files."
    )
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="Number of jobs to run in parallel."
    )
    args = parser.parse_args()

    if args.files[-4:] != ".zip":
        logger.error("File pattern must end with '.zip'!")
        exit()
    files = glob.glob(args.files)
    logger.info("Files to convert: " + str(len(files)))
	
    if args.jobs > 1:
	    with mp.Pool(args.jobs) as pool:
		    _ = list(tqdm.tqdm(pool.imap(convert, files), total=len(files)))
    else:
        logger.info("Note: You can also run this code using multi-processing by setting `-j` option.")
        for f in files:
            logger.info("Processing " + f)
            convert(f, show_progress_bar=True)


if __name__ == "__main__":
    logger = setup_logger()
    main()
