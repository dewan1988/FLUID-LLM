import multiprocessing as mp
import time
from dataloader import ds_get


class ParallelDataGenerator:
    def __init__(self, generator_fn, num_producers=4, queue_maxsize=10):
        self.generator_fn = generator_fn

        self.queue = mp.Queue(maxsize=queue_maxsize)
        self.stop_signal = mp.Value('i', 0)
        self.num_producers = num_producers
        self.producers = []

    def data_producer(self):
        while not self.stop_signal.value:
            data = self.generator_fn()
            self.queue.put(data)
        print("Producer stopped.")

    def get(self, timeout=None):
        try:
            data = self.queue.get(timeout=timeout)
            return data
        except Exception as e:
            print(f"Error getting data from queue: {e}")
            return None

    def stop(self):
        with self.stop_signal.get_lock():
            self.stop_signal.value = 1
        for p in self.producers:
            p.join()

    def run(self):
        for _ in range(self.num_producers):
            p = mp.Process(target=self.data_producer)
            p.start()
            self.producers.append(p)

if __name__ == "__main__":
    dataloader = ParallelDataGenerator(lambda: ds_get(0, 10),
                                       num_producers=4, queue_maxsize=10)
    dataloader.run()

    # Example of using the .get method outside the class in a training loop
    try:
        for _ in range(10):  # Simulate a training loop with 100 iterations
            data = dataloader.get()  # Adjust the timeout as needed
            if data is not None:
                print(f"Consumed data:")
            else:
                print("No data available.")
    except KeyboardInterrupt:
        print("jisodfj io sdfjgiofdjg")
        print("Stopping data generation.")
    finally:
        dataloader.stop()

