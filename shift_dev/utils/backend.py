"""Backends for the data types a dataset of interest is saved in.

Those can be used to load data from diverse storage backends, e.g. from HDF5
files which are more suitable for data centers. The naive backend is the
FileBackend, which loads from / saves to file naively.
"""
from __future__ import annotations

import os
from abc import abstractmethod
from zipfile import ZipFile

try:
    import h5py
    from h5py import File
except:
    raise ImportError("Please install h5py to enable HDF5Backend.")

import numpy as np


class DataBackend:
    """Abstract class of storage backends.

    All backends need to implement three functions: get(), set() and exists().
    get() reads the file as a byte stream and set() writes a byte stream to a
    file. exists() checks if a certain filepath exists.
    """

    @abstractmethod
    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content at the given filepath.

        Args:
            filepath (str): The filepath to store the data at.
            content (bytes): The content to store as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content at the given filepath as bytes.

        Args:
            filepath (str): The filepath to retrieve the data from."

        Returns:
            bytes: The content of the file as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): The filepath to check.

        Returns:
            bool: True if the filepath exists, False otherwise.
        """
        raise NotImplementedError


class FileBackend(DataBackend):
    """Raw file from hard disk data backend."""

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        return os.path.exists(filepath)

    def set(self, filepath: str, content: bytes) -> None:
        """Write the file content to disk.

        Args:
            filepath (str): Path to file.
            content (bytes): Content to write in bytes.
        """
        with open(filepath, "wb") as f:
            f.write(content)

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes.

        Args:
            filepath (str): Path to file.

        Raises:
            FileNotFoundError: If filepath does not exist.

        Returns:
            bytes: File content as bytes.
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"File not found:" f" {filepath}")
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf


class HDF5Backend(DataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at vis4d/data/datasets/to_hdf5.py to
    convert your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, File] = {}

    @staticmethod
    def _get_hdf5_path(filepath: str) -> tuple[str, list[str]]:
        """Get .hdf5 path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.hdf5/key1/key2'

        Returns:
            tuple[str, list[str]]: The .hdf5 path and the keys to retrieve.
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".hdf5" and not h5py.is_hdf5(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .hdf5 file
            if not filepath.endswith(".hdf5"):
                filepath = filepath + ".hdf5"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            return False
        value_buf = self._get_client(hdf5_path, "r")

        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                return False
        return True

    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content.

        Args:
            filepath: path/to/file.hdf5/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .hdf5 file
        """
        if ".hdf5" not in filepath:
            raise ValueError(f"{filepath} not a valid .hdf5 filepath!")
        hdf5_path, keys_str = filepath.split(".hdf5")
        key_list = keys_str.split("/")
        file = self._get_client(hdf5_path + ".hdf5", "a")
        if len(key_list) > 1:
            group_str = "/".join(key_list[:-1])
            if group_str == "":
                group_str = "/"

            group = file[group_str]
            key = key_list[-1]
            group.create_dataset(key, data=np.frombuffer(content, dtype="uint8"))

    def _get_client(self, hdf5_path: str, mode: str) -> File:
        """Get HDF5 client from path.

        Args:
            hdf5_path (str): Path to HDF5 file.
            mode (str): Mode to open the file in.

        Returns:
            File: the hdf5 file.
        """
        if hdf5_path not in self.db_cache:
            client = File(hdf5_path, mode)
            self.db_cache[hdf5_path] = [client, mode]
        else:
            client, current_mode = self.db_cache[hdf5_path]
            if current_mode != mode:
                client.close()
                client = File(hdf5_path, mode)
                self.db_cache[hdf5_path] = [client, mode]
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an HDF5 path
                together with the relative path inside it, e.g.: "/path/to/
                file.hdf5/key/subkey/data". If no .hdf5 given inside filepath,
                the function will search for the first .hdf5 file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.hdf5.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bytes: The file content in bytes
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {filepath}!")

        return bytes(value_buf[()])


class ZipBackend(DataBackend):
    """Backend for loading data from Zip files.

    This backend works with filepaths pointing to valid Zip files. We assume
    that the given Zip file contains the whole dataset associated to this
    backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, tuple[ZipFile, str]] = {}

    @staticmethod
    def _get_zip_path(filepath: str) -> tuple[str, list[str]]:
        """Get .zip path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.zip/key1/key2'

        Returns:
            tuple[str, list[str]]: The .zip path and the keys to retrieve.
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".zip" and not os.path.exists(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .zip file
            if not filepath.endswith(".zip"):
                filepath = filepath + ".zip"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        zip_path, keys = self._get_zip_path(filepath)
        if not os.path.exists(zip_path):
            return False
        file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        return url in file.namelist()

    def set(self, filepath: str, content: bytes) -> None:
        """Write the file content to the zip file.

        Args:
            filepath: path/to/file.zip/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .zip file
            NotImplementedError: If the method is not implemented.
        """
        if ".zip" not in filepath:
            raise ValueError(f"{filepath} not a valid .zip filepath!")

        zip_path, keys = self._get_zip_path(filepath)
        zip_file = self._get_client(zip_path, "a")
        url = "/".join(reversed(keys))
        zip_file.writestr(url, content)

    def _get_client(self, zip_path: str, mode: Literal["r", "w", "a", "x"]) -> ZipFile:
        """Get Zip client from path.

        Args:
            zip_path (str): Path to Zip file.
            mode (str): Mode to open the file in.

        Returns:
            ZipFile: the hdf5 file.
        """
        assert len(mode) == 1, "Mode must be a single character for zip file."
        if zip_path not in self.db_cache:
            client = ZipFile(zip_path, mode)
            self.db_cache[zip_path] = (client, mode)
        else:
            client, current_mode = self.db_cache[zip_path]
            if current_mode != mode:
                client.close()
                client = ZipFile(zip_path, mode)  # pylint:disable=consider-using-with
                self.db_cache[zip_path] = (client, mode)
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an Zip path
                together with the relative path inside it, e.g.: "/path/to/
                file.zip/key/subkey/data". If no .zip given inside filepath,
                the function will search for the first .zip file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.zip.

        Raises:
            ZipFileNotFoundError: If no suitable file exists.
            OSError: If the file cannot be opened.
            ValueError: If key not found inside zip file.

        Returns:
            bytes: The file content in bytes
        """
        zip_path, keys = self._get_zip_path(filepath)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Corresponding zip file not found:" f" {filepath}")
        zip_file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        try:
            with zip_file.open(url) as zf:
                content = zf.read()
        except KeyError as e:
            raise ValueError(f"Value '{url}' not found in {zip_path}!") from e
        return bytes(content)
