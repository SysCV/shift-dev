#!/bin/sh

# Install dependency
python3 -m pip install -r requirements.txt

# Example commands to downaload from all 1fps images and convert them into hdf5.
python3 download.py --view "all" --group "all" --split "all" --framerate "[images]" ./shift_dataset/
python3 -m shift_dev.io.to_hdf5 "./shift_dataset/**/*.zip" --zip -j 4
