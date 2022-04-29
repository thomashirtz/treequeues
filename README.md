# treequeues

This package contains queues for transferring arrays and nested arrays between processes using
pytree and `multiprocessing.Array`s. Compared to the vanilla `multiprocessing.Queue`, this implementation
can reach speeds up to 10 times higher depending on the tree shape and size as well as the number of processes involved.

By using numpy array buffered with multiprocessing array, the data can be send without the need for pickling.  
One of the drawback is that the total size (size of the nested array time the maxsize of the queue) needs to be 
preallocated.

This package contains `TreeQueue` and `ArrayQueue`, in both case, an instance of the data and the maximum size needs to be passed when creating the queue.

This repository was inspired by [ArrayQueues](https://github.com/portugueslab/arrayqueues) from [portugueslab](https://github.com/portugueslab).

### Raw performance

<p align="center">
    <img align="centre" width="1000"  src="benchmark_performance.png">
</p>

### Multiprocessing performance

<p align="center">
    <img align="center" width="600"  src="benchmark_multiprocessing.png">
</p>


## Installation
```
pip install git+https://github.com/thomashirtz/treequeues#egg=treequeues
```

## Usage example
```python
import numpy as np
from typing import NamedTuple
from multiprocessing import Process

from treequeues import TreeQueue


class ReadProcess(Process):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
      
    def run(self):
        print(self.source_queue.get())
        
        
class NestedArray(NamedTuple):
    item_1: np.ndarray
    item_2: np.ndarray
    item_3: np.ndarray

    
def generate_nested_array() -> NestedArray:
    return NestedArray(
        item_1=np.random.random((4, )),
        item_2=np.random.random((1, )),
        item_3=np.random.random((3, 2, )),
    )
    

if __name__ == "__main__":
    nested_array = generate_nested_array()
    # Initialize queue with the 'nested_array' structure, shape and dtype. 
    # This queue preallocate the space for 10 nested array with the same specification in a shared memory.
    queue = TreeQueue(nested_array=nested_array, maxsize=10)
    
    new_nested_array = generate_nested_array()
    queue.put(new_nested_array)
    
    process = ReadProcess(queue=queue)
    process.start()
    process.join()
    
```
