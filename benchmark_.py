import multiprocessing as mp
import time
from multiprocessing import Process
from typing import Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from treequeues import TreeQueue, NoLockTreeQueue

NUM_ITEMS = 200
QUEUE_MAXSIZE = 2


class CatchTime:
    def __enter__(self):
        self.initial_value = time.perf_counter()
        self.value: Optional[float] = None
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.value = time.perf_counter() - self.initial_value


class GetProcess(Process):
    def __init__(self, queue_data, queue_time: mp.Queue, barrier: mp.Barrier, num_iterations: int):
        super().__init__()
        self.queue_data = queue_data
        self.queue_time = queue_time
        self.barrier = barrier
        self.num_iterations = num_iterations

    def run(self):
        time_list = []
        self.barrier.wait()

        for i in range(self.num_iterations):
            start_time = time.perf_counter()
            self.queue_data.get()
            time_list.append(time.perf_counter() - start_time)
        for t in time_list:
            self.queue_time.put(t)
        self.queue_time.put(None)


class PutProcess(Process):
    def __init__(self, queue, barrier: mp.Barrier, num_iterations: int, data):
        super().__init__()
        self.queue = queue
        self.barrier = barrier
        self.num_iterations = num_iterations
        self.data = data

    def run(self):
        self.barrier.wait()
        for i in range(self.num_iterations):
            self.queue.put(self.data)


def get_tree(num_items: int, array_size: int):
    return {i: np.random.random(array_size) for i in range(num_items)}


def run_single_performance_test(
        num_processes: int,
        megabytes: int,
        num_tree_items: int,
        treequeue_type: str,
) -> Tuple[float, float]:
    barrier = mp.Barrier(num_processes * 2)
    array_size = int(megabytes * 1_000_000 / (num_tree_items * 8))
    data = get_tree(num_tree_items, array_size)
    queue_time = mp.Queue()

    if treequeue_type == 'TreeQueue':
        queue_data = TreeQueue(data, maxsize=4)
    elif treequeue_type == 'SoftLockTreeQueue':
        queue_data = NoLockTreeQueue(data, maxsize=4)
    else:
        queue_data = mp.Queue(maxsize=4)

    process_list = []
    for _ in range(num_processes):
        process = GetProcess(
            queue_data=queue_data,
            queue_time=queue_time,
            barrier=barrier,
            num_iterations=int(NUM_ITEMS/num_processes),
        )
        process.start()
        process_list.append(process)

    process_list = []
    for _ in range(num_processes):
        process = PutProcess(
            queue=queue_data,
            barrier=barrier,
            num_iterations=int(NUM_ITEMS/num_processes),
            data=data,
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    time_list = []
    none_counter = 0
    while none_counter < num_processes:
        data = queue_time.get()
        if data is None:
            none_counter += 1
        else:
            time_list.append(data)

    return float(np.mean(time_list)), float(np.std(time_list))


def run_benchmark_performance():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['figure.dpi'] = 300
    mpl.rcParams.update({'font.size': 14})

    num_processes = 1
    megabytes_list = [1., 5., 10., 20, 50, 100]

    fig, (axe_1, axe_2, axe_3) = plt.subplots(1, 3, figsize=(11, 4), sharey='all')

    experiences = {
        'Queue': {
            '1 item': {'num_tree_items': 1, 'color': 'r', 'linestyle': '-', 'axe': axe_1},
            '10 items': {'num_tree_items': 10, 'color': 'r', 'linestyle': '-', 'axe': axe_2},
            '100 items': {'num_tree_items': 100, 'color': 'r', 'linestyle': '-', 'axe': axe_3},
        },
        'TreeQueue': {
            '1 item': {'num_tree_items': 1, 'color': 'k', 'linestyle': '-', 'axe': axe_1},
            '10 items': {'num_tree_items': 10, 'color': 'k', 'linestyle': '-', 'axe': axe_2},
            '100 items': {'num_tree_items': 100, 'color': 'k', 'linestyle': '-', 'axe': axe_3},
        },
    }

    minimum = 1
    for queue_type, experience_dict in experiences.items():
        for experience_name, parameters in experience_dict.items():
            print(f'{queue_type}: {experience_name}')

            std_list = []
            mean_list = []
            for megabytes in megabytes_list:
                mean, std = run_single_performance_test(
                    num_processes=num_processes,
                    megabytes=megabytes,
                    num_tree_items=parameters['num_tree_items'],
                    use_treequeue=True if queue_type == 'TreeQueue' else False
                )
                minimum = min(minimum, 0.6*mean)
                mean_list.append(mean)
                std_list.append(std)

            parameters['axe'].errorbar(
                megabytes_list, mean_list, std_list, ecolor=parameters['color'], color=parameters['color'],
                linestyle=parameters['linestyle'], linewidth=1.5, label=f'{queue_type}',
                capsize=3, capthick=1.5)
            parameters['axe'].set_yscale('log')
            parameters['axe'].set_xscale('log')
            parameters['axe'].grid(linestyle='dotted', )
            parameters['axe'].grid(linestyle='dotted')
            parameters['axe'].set_title(f'Nested array composed of {experience_name}', fontsize=14)
            parameters['axe'].tick_params(axis='both', which='both', labelsize=13)
            parameters['axe'].yaxis.set_tick_params(labelbottom=True)
            parameters['axe'].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    axe_2.set_xlabel(r'Nested array size (megabytes)', fontsize=13)
    axe_1.set_ylabel(r'Time (s)', fontsize=13)
    axe_1.set_ylim(ymin=minimum)
    fig.suptitle('Queue performance (s) versus object size (megabytes)', y=0.96)
    handles, labels = plt.gca().get_legend_handles_labels()
    axe_1.legend(handles, labels, loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.06, top=0.9)
    plt.tight_layout()
    # plt.show()
    plt.savefig('benchmark_performance_.png', dpi=600)


def run_benchmark_multiprocessing():
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rcParams['figure.dpi'] = 300
    mpl.rcParams.update({'font.size': 14})

    num_tree_items = 10
    megabytes_list = [1., 5., 10., 20, 50, 100]

    fig, (axe_1, axe_2) = plt.subplots(1, 2, figsize=(8.5, 5), sharey='all')

    experiences = {
        'Queue': {
            '1 process': {'num_processes': 1, 'color': 'r', 'linestyle': '-', 'axe': axe_1},
            '8 processes': {'num_processes': 8, 'color': 'r', 'linestyle': '-', 'axe': axe_2},
        },
        'TreeQueue': {
            '1 process': {'num_processes': 1, 'color': 'k', 'linestyle': '-', 'axe': axe_1},
            '8 processes': {'num_processes': 8, 'color': 'k', 'linestyle': '-', 'axe': axe_2},
        },
        'SoftLockTreeQueue': {
            '1 process': {'num_processes': 1, 'color': 'b', 'linestyle': '-', 'axe': axe_1},
            '8 processes': {'num_processes': 8, 'color': 'b', 'linestyle': '-', 'axe': axe_2},
        },
    }

    minimum = 1
    for queue_type, experience_dict in experiences.items():
        for experience_name, parameters in experience_dict.items():
            print(f'{queue_type}: {experience_name}')

            std_list = []
            mean_list = []
            for megabytes in megabytes_list:
                with CatchTime() as ct:
                    mean, std = run_single_performance_test(
                        num_processes=parameters['num_processes'],
                        megabytes=megabytes,
                        num_tree_items=num_tree_items,
                        treequeue_type=queue_type,
                    )
                print(parameters['num_processes'], ct.value)

                minimum = min(minimum, 0.3*mean)
                mean_list.append(mean)
                std_list.append(std)

            parameters['axe'].errorbar(
                megabytes_list, mean_list, std_list, ecolor=parameters['color'], color=parameters['color'],
                linestyle=parameters['linestyle'], linewidth=1.5, label=f'{queue_type}',
                capsize=3, capthick=1.5)
            parameters['axe'].set_yscale('log')
            parameters['axe'].set_xscale('log')
            parameters['axe'].grid(linestyle='dotted')
            parameters['axe'].grid(linestyle='dotted')
            parameters['axe'].set_title(f'Experiment using {parameters["num_processes"]*2} processes', fontsize=15)
            parameters['axe'].tick_params(axis='both', which='both', labelsize=13)
            parameters['axe'].yaxis.set_tick_params(labelbottom=True)
            parameters['axe'].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    axe_1.set_ylabel(r'Time (s)', fontsize=13)
    fig.supxlabel(r'Nested array size (megabytes)', fontsize=13, y=0.066)
    axe_1.set_ylim(ymin=minimum)
    fig.suptitle('Queue performance (s) versus object size (megabytes)\nwith nested arrays composed of 10 items', y=0.96)
    axe_2.legend(loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0, top=0.9)
    plt.tight_layout()
    # plt.show()
    plt.savefig('benchmark_multiprocessing.png', dpi=600)


if __name__ == '__main__':
    # run_benchmark_multiprocessing()
    print(run_single_performance_test(
                    num_processes=8,
                    megabytes=1,
                    num_tree_items=3,
                    treequeue_type='SoftLockTreeQueue',
                ))
