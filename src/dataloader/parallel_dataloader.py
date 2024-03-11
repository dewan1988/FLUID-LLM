""" Class for multiprocess parallel dataloader. Works with a dataloader class instance. """

# Kill remaining processes:
# pgrep -f /home/maccyz/Documents/LLM_Fluid/ | xargs kill
import torch.multiprocessing as mp
import torch

from dataloader.MGN_dataloader import MGNDataloader
from cprint import c_print
import queue


class ParallelDataGenerator:
    def __init__(self, dataloader: MGNDataloader, bs=1, num_producers=4, queue_maxsize=8):
        self.dataloader = dataloader
        self.bs = bs

        self.queue = mp.Queue(maxsize=queue_maxsize)
        self.stop_signal = mp.Value('i', 0)
        self.num_producers = num_producers
        self.producers = []

        c_print("Initialising Dataloader", color='green')

    def fetch_data(self):
        data = self.dataloader.ds_get()
        return data

    def data_producer(self):
        while not self.stop_signal.value:
            data = self.fetch_data()
            try:
                self.queue.put(data, timeout=3)  # Timeout of 1 second
            except queue.Full:
                continue  # This allows checking the stop_signal again
        c_print("Producer stopped.", color="yellow")

    def get(self, timeout=3.):
        # try:
        data = self.queue.get(timeout=timeout)
        return data
        # except Exception as e:
        #     c_print(f"Error getting data from queue: {e}", color="magenta")
        #     return None

    def get_batch(self, timeout=3.):
        """ Combines several data samples into a single batch"""
        batch = []  # Initialize an empty list to store the batch
        while len(batch) < self.bs:
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
        for _ in range(self.num_producers):
            p = mp.Process(target=self.data_producer, )
            p.start()
            self.producers.append(p)
