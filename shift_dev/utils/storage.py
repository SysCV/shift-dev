import hashlib
import io
import os
import tarfile
import zipfile


class ArchiveReader:
    def __init__(self, filename) -> None:
        self.filename = filename

    def get_list(self):
        raise NotImplementedError

    def get_file(self, name):
        raise NotImplementedError

    def closs(self):
        raise NotImplementedError


class ArchiveWriter:
    def __init__(self, filename) -> None:
        self.filename = filename

    def get_list(self):
        raise NotImplementedError

    def add_file(self, name, arcname):
        raise NotImplementedError

    def closs(self):
        raise NotImplementedError


class ZipArchiveReader(ArchiveReader):
    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.file = zipfile.ZipFile(filename, "r")

    def get_file(self, name):
        data = self.file.read(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def get_list(self):
        return self.file.namelist()

    def close(self):
        self.file.close()


class ZipArchiveWriter(ArchiveWriter):
    default_ext = "zip"

    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.file = zipfile.ZipFile(filename, "w")

    def add_file(self, name, arcname="."):
        for root, dirs, files in os.walk(name):
            for file in files:
                filepath = os.path.join(root, file)
                self.file.write(filepath, os.path.join(arcname, file))

    def get_list(self):
        return self.file.namelist()

    def close(self):
        self.file.close()


class TarArchiveReader(ArchiveReader):
    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.file = tarfile.TarFile(filename, "r")

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


class TarArchiveWriter(ArchiveWriter):
    default_ext = "tar"

    def __init__(self, filename) -> None:
        self.filename = filename
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
