import multiprocessing as mp
from typing import Any
from typing import Union

import numpy as np
import tree  # noqa

NestedArray = tree.StructureKV[str, np.ndarray]


# todo maybe look into PortableQueue
# https://gist.github.com/FanchenBao/d8577599c46eab1238a81857bb7277c9
# https://github.com/portugueslab/arrayqueues/blob/master/arrayqueues/portable_queue.py


class ArrayView:
    def __init__(
            self,
            multiprocessing_array: mp.Array,
            numpy_array: np.ndarray,
            num_items: int,
    ):

        self._num_items = num_items
        self._item_shape = numpy_array.shape

        self.dtype = numpy_array.dtype
        self.shape = (num_items, *numpy_array.shape)

        self._view = np.frombuffer(
            buffer=multiprocessing_array,
            dtype=numpy_array.dtype,
            count=np.product(self.shape),
        ).reshape(self.shape)
        # todo maybe add a `set` to assert that the wrong items are not retrieved

    def put(self, element: np.ndarray, index: int) -> None:
        self._view[index, ...] = element

    def get(self, index: int) -> np.ndarray:
        return self._view[index, ...]


class ArrayQueue:
    def __init__(self, array: np.ndarray, maxsize: int):
        self.size = maxsize
        self.num_bytes: int = array.nbytes * maxsize
        self.array = mp.Array("c", self.num_bytes)

        self.queue = mp.Queue(maxsize=maxsize)
        self.next_index = mp.Value('i', 0)
        self.array_lock = mp.Lock()

        self.view = ArrayView(
            multiprocessing_array=self.array.get_obj(),
            numpy_array=array,
            num_items=maxsize,
        )

    def put(self, element: np.ndarray) -> None:
        # Avoid putting several elements at the same time
        with self.next_index.get_lock():
            self.queue.put(self.next_index.value)
            # Avoid array changes during putting and getting items
            with self.array_lock:
                self.view.put(element, self.next_index.value)
            self.next_index.value = (self.next_index.value + 1) % self.size

    def get(self, block=True, timeout=None) -> np.ndarray:
        index = self.queue.get(block=block, timeout=timeout)
        with self.array_lock:
            return self.view.get(index)

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()

    def qsize(self) -> int:
        return self.queue.qsize()


def get_queue(data: Any, maxsize: int) -> Union[mp.Queue, ArrayQueue]:
    if isinstance(data, np.ndarray):
        return ArrayQueue(data, maxsize=maxsize)
    else:
        return mp.Queue(maxsize=maxsize)


class TreeQueue:
    def __init__(
            self, nested_array: NestedArray, maxsize: int
    ):
        self.maxsize = maxsize
        self.nested_array = nested_array

        self._nested_queue = tree.map_structure(
            lambda data: get_queue(data=data, maxsize=maxsize), nested_array
        )

    def get(self, block=True, timeout=None) -> NestedArray:
        return tree.map_structure(
            lambda queue: queue.get(block=block, timeout=timeout),
            self._nested_queue
        )

    def put(self, nested_array: NestedArray, block=True, timeout=None) -> None:
        tree.map_structure(
            lambda queue, array: queue.put(array, block=block, timeout=timeout),
            self._nested_queue, nested_array
        )
