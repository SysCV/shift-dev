"""Utility functions for datasets."""
from __future__ import annotations

import copy
import hashlib
import os
import pickle
from collections.abc import Callable
from typing import Any

import appdirs
import numpy as np
from torch.utils.data import Dataset

from shift_dev.types import DictStrAny, NDArrayU8
from shift_dev.utils import Timer, setup_logger

logger = setup_logger()


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(  # type: ignore
        self, lst: list[Any], deepcopy: bool = False, serialize: bool = True
    ):
        """Creates an instance of the class.

        Args:
            lst: a list which contains elements to produce.
            deepcopy: whether to deepcopy the element when producing it, s.t.
            the result can be modified in place without affecting the source
            in the list.
            serialize: whether to hold memory using serialized objects. When
            enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
        """
        self._copy = deepcopy
        self._serialize = serialize

        def _serialize(data: Any) -> NDArrayU8:  # type: ignore
            """Serialize python object to numpy array."""
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            self._lst = [_serialize(x) for x in lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)  # type: ignore
        else:
            self._lst = lst  # pragma: no cover

    def __len__(self) -> int:
        """Return len of list."""
        if self._serialize:
            return len(self._addr)
        return len(self._lst)  # pragma: no cover

    def __getitem__(self, idx: int) -> Any:  # type: ignore
        """Return item of list at idx."""
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes_ = memoryview(self._lst[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes_)
        if self._copy:  # pragma: no cover
            return copy.deepcopy(self._lst[idx])

        return self._lst[idx]  # pragma: no cover


class CacheMappingMixin:
    """Caches a mapping for fast I/O and multi-processing.

    This class provides functionality for caching a mapping from dataset index
    requested by a call on __getitem__ to a dictionary that holds relevant
    information for loading the sample in question from the disk.
    Caching the mapping reduces startup time by loading the mapping instead of
    re-computing it at every startup.

    NOTE: The mapping will detect changes in the dataset by inspecting the
    string representation (__repr__) of your dataset. Make sure your __repr__
    implementation contains all parameters relevant to your mapping, so that
    the mapping will get updated once one of those parameters is changed.
    Conversely, make sure all non-relevant information is excluded from the
    string representation, so that the mapping can be loaded and re-used.
    """

    def _load_mapping_data(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> list[DictStrAny]:
        """Load possibly cached mapping via generate_map_func."""
        if use_cache:
            app_dir = os.getenv(
                "SHIFT_CACHE_DIR",
                os.getenv("TMPDIR", appdirs.user_cache_dir(appname="shift_dev")),
            )
            cache_dir = os.path.join(
                app_dir,
                "shfit_data_mapping",
                self.__class__.__name__,
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, self._get_hash() + ".pkl")
            if not os.path.exists(cache_path):
                logger.info(
                    f"Generating annotation cache and dumping to {cache_path} .."
                )
                data = generate_map_func()
                with open(cache_path, "wb") as file:
                    file.write(pickle.dumps(data))
            else:
                with open(cache_path, "rb") as file:
                    data = pickle.loads(file.read())
        else:
            data = generate_map_func()
        return data

    def _load_mapping(
        self,
        generate_map_func: Callable[[], list[DictStrAny]],
        use_cache: bool = True,
    ) -> Dataset[DictStrAny]:
        """Load cached mapping or generate if not exists."""
        timer = Timer()
        data = self._load_mapping_data(generate_map_func, use_cache)
        dataset = DatasetFromList(data)
        logger.info(f"Loading {str(self.__repr__)} takes {timer.time():.2f} seconds.")
        return dataset

    def _get_hash(self, length: int = 16) -> str:
        """Get hash of current dataset instance."""
        hasher = hashlib.sha256()
        hasher.update(str(self.__repr__).encode("utf8"))
        hash_value = hasher.hexdigest()[:length]
        return hash_value
