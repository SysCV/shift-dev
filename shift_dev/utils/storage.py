import hashlib
import io
import tarfile
import zipfile


class ZipArchiveReader:
    def __init__(self, filename) -> None:
        self.file = zipfile.ZipFile(filename, "r")
        # print(f"Loaded {filename}.")

    def get_file(self, name):
        data = self.file.read(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def get_list(self):
        return self.file.namelist()

    def close(self):
        self.file.close()


class TarArchiveReader:
    def __init__(self, filename) -> None:
        self.file = tarfile.TarFile(filename, "r")
        # print(f"Loaded {filename}.")

    def get_file(self, name):
        data = self.file.extractfile(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def extract_file(self, name, output_dir):
        self.file.extract(name, output_dir)

    def get_list(self):
        return self.file.getnames()

    def close(self):
        self.file.close()


class TarArchiveWriter:
    def __init__(self, filename) -> None:
        self.file = tarfile.TarFile(filename, "w")
        # print(f"Loaded {filename}.")

    def add_file(self, name, arcname):
        self.file.add(name, arcname=arcname)

    def get_list(self):
        return self.file.getnames()

    def close(self):
        self.file.close()


def string_hash(video):
    sha = hashlib.sha512(video.encode("utf-8"))
    return int(sha.hexdigest(), 16)
