""" Class for multiprocess parallel dataloader. Works with a dataloader class instance. """
import torch.multiprocessing as mp
from dataloader.MGN_dataloader import MGNDataloader
from cprint import c_print
import queue


class ParallelDataGenerator:
    def __init__(self, dataloader: MGNDataloader, num_producers=4, queue_maxsize=10):
        self.dataloader = dataloader

        self.queue = mp.Queue(maxsize=queue_maxsize)
        self.stop_signal = mp.Value('i', 0)
        self.num_producers = num_producers
        self.producers = []

    def fetch_data(self):
        return self.dataloader.ds_get()

    def data_producer(self):
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
            print(f"Error getting data from queue: {e}")
            return None

    def stop(self):
        c_print("Stopping Dataloader", color='red')
        with self.stop_signal.get_lock():
            self.stop_signal.value = 1

        for p in self.producers:
            p.join()

    def run(self):
        for _ in range(self.num_producers):
            p = mp.Process(target=self.data_producer)
            p.start()
            self.producers.append(p)
