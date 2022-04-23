import multiprocessing as mp
import multiprocessing.managers
from abc import ABC, abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import TypeVar

import numpy as np
import tree  # noqa

T = TypeVar('T')
NestedArray = tree.StructureKV[str, np.ndarray]


class ArrayView:
    def __init__(
            self,
            multiprocessing_array: mp.Array,
            numpy_array: np.ndarray,
            num_items: int,
    ):
        self.num_items = num_items
        self.dtype = numpy_array.dtype
        self.shape = (num_items, *numpy_array.shape)
        self.nbytes: int = numpy_array.nbytes * num_items

        self._item_shape = numpy_array.shape
        self._multiprocessing_array = multiprocessing_array

        self._array_view = np.frombuffer(
            buffer=multiprocessing_array,
            dtype=numpy_array.dtype,
            count=np.product(self.shape),
        ).reshape(self.shape)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        del self.__dict__['_view']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._array_view = np.frombuffer(
            buffer=self._multiprocessing_array,
            dtype=self.dtype,
            count=np.product(self.shape),
        ).reshape(self.shape)

    def put(self, item: np.ndarray, index: int) -> None:
        assert item.shape == self._item_shape and item.dtype == self.dtype
        self._array_view[index, ...] = item

    def get(self, index: int) -> np.ndarray:
        return np.copy(self._array_view[index, ...])


class AbstractQueue(ABC):
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._queue = mp.Queue(maxsize=maxsize)

    @abstractmethod
    def put(self, item: T) -> None:
        # https://stackoverflow.com/a/42778801
        raise NotImplementedError

    @abstractmethod
    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        raise NotImplementedError

    @abstractmethod
    def empty(self) -> bool:
        return self._queue.empty()

    @abstractmethod
    def full(self) -> bool:
        return self._queue.full()

    @abstractmethod
    def qsize(self) -> int:
        return self._queue.qsize()


class ArrayQueue(AbstractQueue):
    def __init__(self, array: np.ndarray, maxsize: int):
        super(ArrayQueue, self).__init__(maxsize=maxsize)

        self._lock = mp.Lock()
        self._next_index = mp.Value('i', 0)
        self.nbytes: int = array.nbytes * maxsize

        self._array = mp.Array("c", self.nbytes)
        self._array_view = ArrayView(
            multiprocessing_array=self._array.get_obj(),
            numpy_array=array,
            num_items=maxsize,
        )

    def put(self, array: np.ndarray) -> None:
        # Avoid several simultaneous 'put' call
        with self._next_index.get_lock():
            self._queue.put(self._next_index.value)
            # Avoid ArrayQueue changes during a 'put' or 'get' call
            with self._lock:
                self._put(array=array, index=self._next_index.value)
            self._next_index.value = (self._next_index.value + 1) % self.maxsize

    def _put(self, array: np.ndarray, index: int) -> None:
        self._array_view.put(array, index)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> np.ndarray:
        index = self._queue.get(block=block, timeout=timeout)
        # Avoid ArrayQueue changes during a 'put' or 'get' call
        with self._lock:
            return self._get(index=index)

    def _get(self, index: int) -> np.ndarray:
        return self._array_view.get(index=index)


class TreeQueue(AbstractQueue): # rename legacy, simple, original
    """TreeQueue implemented with simple locking techniques."""
    def __init__(self, nested_array: NestedArray, maxsize: int):
        super().__init__(maxsize=maxsize)

        self._lock = mp.Lock()
        self._next_index = mp.Value('i', 0)

        self._nested_queue = tree.map_structure(
            lambda array: ArrayQueue(array=array, maxsize=maxsize), nested_array
        )

        self._nested_array = nested_array
        self.nbytes = sum([q.nbytes for q in tree.flatten(self._nested_queue)])

    def put(self, nested_array: NestedArray, block: bool = True, timeout: Optional[float] = None) -> None:
        # Avoid several simultaneous 'put' call
        with self._next_index.get_lock():
            self._queue.put(self._next_index.value, block=block, timeout=timeout)
            # Avoid ArrayQueue changes during a 'put' or 'get' call
            with self._lock:
                tree.map_structure(
                    lambda queue, array: queue._put(  # noqa
                        array=array,
                        index=self._next_index.value
                    ),
                    self._nested_queue, nested_array
                )
            self._next_index.value = (self._next_index.value + 1) % self.maxsize

    def get(self, block: bool = True, timeout: Optional[float] = None) -> NestedArray:
        index = self._queue.get(block=block, timeout=timeout)
        # Avoid ArrayQueue changes during a 'put' or 'get' call
        with self._lock:
            return tree.map_structure(
                lambda queue: queue._get(index=index),  # noqa
                self._nested_queue
            )
