""" Class for multiprocess parallel dataloader. Works with a dataloader class instance. """

# Kill remaining processes:
# pgrep -f /home/maccyz/Documents/LLM_Fluid/ | xargs kill
import atexit
import torch.multiprocessing as mp
import torch
import queue

from dataloader.MGN_dataloader import MGNDataloader
from cprint import c_print
from utils import set_seed


class ParallelDataGenerator:
    def __init__(self, dataloader: MGNDataloader, batch_size, num_procs=8, epoch_size=10):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.queue = mp.Queue(maxsize=batch_size*2)
        self.stop_signal = mp.Value('i', 0)
        self.num_procs = num_procs
        self.producers = []

        atexit.register(self.stop)
        c_print("Initialising Dataloader", color='green')

    def fetch_data(self):
        data = self.dataloader.ds_get()
        return data

    def data_producer(self, seed):
        set_seed(seed)
        while not self.stop_signal.value:
            data = self.fetch_data()
            try:
                self.queue.put(data, timeout=1)  # Timeout of 1 second
            except queue.Full:
                continue  # This allows checking the stop_signal again
        c_print("Producer stopped.", color="yellow")

    def get(self, timeout=1.):
        try:
            data = self.queue.get(timeout=timeout)
            return data
        except Exception as e:
            c_print(f"Error getting data from queue: {e}", color="magenta")
            return None

    def get_batch(self, timeout=1.):
        """ Combines several data samples into a single batch"""
        batch = []  # Initialize an empty list to store the batch

        while len(batch) < self.batch_size:
            try:
                data = self.queue.get(timeout=timeout)  # Try to get data from the queue
                batch.append(data)  # Add the data to the batch
            except Exception as e:
                c_print(f"Error getting data from queue: {e}", color="magenta")
                return None

        # Stack tensors
        batch = [torch.stack(tensors) for tensors in zip(*batch)]
        return batch  # Return the batch

    def stop(self):
        c_print("Stopping Dataloader", color='red')
        with self.stop_signal.get_lock():
            self.stop_signal.value = 1

        for p in self.producers:
            p.join()

    def run(self):
        init_seed = torch.random.initial_seed()
        for i in range(self.num_procs):
            p = mp.Process(target=self.data_producer, args=(init_seed + i,))
            p.start()
            self.producers.append(p)

    def __iter__(self):
        for _ in range(self.epoch_size):
            yield self.get_batch()


class SingleDataloader:
    """ Single threaded dataloader"""
    def __init__(self, dataloader: MGNDataloader, batch_size, epoch_size=10):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def get_batch(self):
        """ Combines several data samples into a single batch"""
        batch = []
        for _ in range(self.batch_size):
            data = self.dataloader.ds_get()
            batch.append(data)
        batch = [torch.stack(tensors) for tensors in zip(*batch)]
        return batch

    def get(self):
        data = self.dataloader.ds_get()
        return data

    def stop(self):
        pass

    def __iter__(self):
        for _ in range(self.epoch_size):
            yield self.get_batch()

