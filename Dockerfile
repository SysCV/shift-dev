# Dockerfile

FROM python:3.8

WORKDIR /usr/src/app

# Update
RUN apt -y update
RUN apt -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
RUN apt -y install python3-h5py pkg-config libhdf5-dev 

# Install FFmpeg 4.3
RUN apt -y install ffmpeg=7:4.3.4-0+deb11u1

# Install Python libraries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start decompressing videos
CMD [ "python", "-m", "shift_dev.io.decompress_videos", "/data/**/*.tar" ]