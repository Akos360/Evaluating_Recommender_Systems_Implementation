import time
import psutil
import GPUtil
import multiprocessing
import pandas as pd


class ResourceTracker:
    def __init__(self, pid, algorithm_name, rec_type, memory_file="memory_tracking.csv", cpu_file="cpu_tracking.csv", gpu_file="gpu_tracking.csv"):
        self.pid = pid
        self.algorithm_name = algorithm_name
        self.rec_type = rec_type
        self.memory_file = memory_file
        self.cpu_file = cpu_file
        self.gpu_file = gpu_file
        self.shared_memory_list = multiprocessing.Manager().list()
        self.shared_cpu_list = multiprocessing.Manager().list()
        self.shared_gpu_list = multiprocessing.Manager().list()
        self.stop_flag = multiprocessing.Manager().Event()
        self.status_flag = multiprocessing.Manager().dict()

    def reset_shared_lists(self):
        self.shared_memory_list = multiprocessing.Manager().list()
        self.shared_cpu_list = multiprocessing.Manager().list()
        self.shared_gpu_list = multiprocessing.Manager().list()

    def spawn_tracker(self, kind, run_index):
        if kind == "memory":
            proc = multiprocessing.Process(target=self.memory_tracker, args=(run_index,))
        elif kind == "cpu":
            proc = multiprocessing.Process(target=self.cpu_time_tracker, args=(run_index,))
        elif kind == "gpu":
            proc = multiprocessing.Process(target=self.gpu_tracker, args=(run_index,))
        else:
            raise ValueError(f"Unknown tracker type: {kind}")
        proc.start()
        return proc

    def wait_for_trackers(self, timeout=60):
        start = time.time()
        while not self.status_flag.get("finished", False):
            if time.time() - start > timeout:
                print("Timed out waiting for trackers to finish.")
                break
            time.sleep(0.5)

    # Sampling RAM Memory
    def memory_tracker(self, run_index):
        process = psutil.Process(self.pid)
        self.status_flag['started'] = True  

        while not self.stop_flag.is_set():
            mem_usage = process.memory_info().rss / (1024 * 1024)
            self.shared_memory_list.append((time.time(), mem_usage))
            time.sleep(1)

        memory_df = pd.DataFrame(list(self.shared_memory_list), columns=["timestamp", "memory_usage"])
        path = f"results/results_for_{self.rec_type}/memory_tracking/{self.algorithm_name}/run_{run_index}_memory_usage.csv"
        memory_df.to_csv(path, index=False)
        print(f"Memory tracking saved to {path}")
        self.status_flag['finished'] = True

    # Tracking CPU time
    def cpu_time_tracker(self, run_index):
        process = psutil.Process(self.pid)
        self.status_flag['started'] = True

        start_cpu_times = process.cpu_times()
        start_time = time.time()

        while not self.stop_flag.is_set():
            time.sleep(0.1)

        end_cpu_times = process.cpu_times()
        end_time = time.time()

        user_time = end_cpu_times.user - start_cpu_times.user
        system_time = end_cpu_times.system - start_cpu_times.system
        total_time = user_time + system_time
        elapsed_time = end_time - start_time

        cpu_df = pd.DataFrame([(elapsed_time, total_time, user_time, system_time)],
                              columns=["elapsed_time", "total_cpu_time", "user_time", "system_time"])
        path = f"results/results_for_{self.rec_type}/cpu_tracking/{self.algorithm_name}/run_{run_index}_cpu_time_usage.csv"
        cpu_df.to_csv(path, index=False)
        print(f"CPU tracking saved to {path}")
        self.status_flag['finished'] = True

    # Sampling GPU Memory
    def gpu_tracker(self, run_index):
        self.status_flag['started'] = True

        while not self.stop_flag.is_set():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.shared_gpu_list.append((time.time(), gpu.memoryUtil * 100, gpu.memoryUsed))
            time.sleep(1)

        gpu_df = pd.DataFrame(list(self.shared_gpu_list), columns=["timestamp", "gpu_usage", "gpu_memory"])
        path = f"results/results_for_{self.rec_type}/gpu_tracking/{self.algorithm_name}/run_{run_index}_gpu_usage.csv"
        gpu_df.to_csv(path, index=False)
        print(f"GPU tracking saved to {path}")
        self.status_flag['finished'] = True
