import multiprocessing as mp

import numpy as np
import tree  # noqa

NestedArray = tree.StructureKV[str, np.ndarray]


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
            self._put(element=element, index=self.next_index.value)
            self.next_index.value = (self.next_index.value + 1) % self.size

    def _put(self, element: np.ndarray, index: int):
        # Avoid array changes during putting and getting items
        with self.array_lock:
            self.view.put(element, index)

    def get(self, block=True, timeout=None) -> np.ndarray:
        index = self.queue.get(block=block, timeout=timeout)
        with self.array_lock:
            return self.view.get(index)

    def _get(self, index: int) -> np.ndarray:
        with self.array_lock:
            return self.view.get(index)

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()

    def qsize(self) -> int:
        return self.queue.qsize()


class TreeQueue:
    def __init__(
            self, nested_array: NestedArray, maxsize: int
    ):
        self.maxsize = maxsize
        self.nested_array = nested_array

        self.queue = mp.Queue(maxsize=maxsize)
        self.next_index = mp.Value('i', 0)
        self.lock = mp.Lock()

        self._nested_queue = tree.map_structure(
            lambda array: ArrayQueue(array=array, maxsize=maxsize), nested_array
        )

    def get(self, block=True, timeout=None) -> NestedArray:
        index = self.queue.get(block=block, timeout=timeout)
        with self.lock:
            return tree.map_structure(
                lambda queue: queue._get(index=index),  # noqa
                self._nested_queue
            )

    def put(self, nested_array: NestedArray, block=True, timeout=None) -> None:
        # Avoid putting several elements at the same time
        with self.next_index.get_lock():
            self.queue.put(self.next_index.value, block=block, timeout=timeout)
            tree.map_structure(
                lambda queue, array: queue._put(  # noqa
                    element=array,
                    index=self.next_index.value
                ),
                self._nested_queue, nested_array
            )
            self.next_index.value = (self.next_index.value + 1) % self.maxsize

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()

    def qsize(self) -> int:
        return self.queue.qsize()
