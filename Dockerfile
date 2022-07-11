# Dockerfile

FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start decompressing videos
CMD [ "python", "-m", "shift_dev.io.decompress_videos", "/data/*.tar" ]