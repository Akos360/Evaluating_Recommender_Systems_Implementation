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

    def memory_tracker(self, run_index):
        process = psutil.Process(self.pid)
        
        # status as 'started' immediately when the process starts
        self.status_flag['started'] = True  
        
        while not self.stop_flag.is_set():  
            mem_usage = process.memory_info().rss / (1024 * 1024)       # Convert to MB
            self.shared_memory_list.append((time.time(), mem_usage))    # Append timestamp and memory usage
            time.sleep(1)                                               # Sample every second

        # After stop flag is set, save the collected data
        memory_data = list(self.shared_memory_list)
        memory_df = pd.DataFrame(memory_data, columns=["timestamp", "memory_usage"])
        if self.rec_type == "paragraph":
            memory_df.to_csv(f"memory_tracking/paragraph/{self.algorithm_name}/run_{run_index}_memory_usage.csv", index=False)
        elif self.rec_type == "description":
            memory_df.to_csv(f"memory_tracking/description/{self.algorithm_name}/run_{run_index}_memory_usage.csv", index=False)

        print(f"Memory usage data saved to 'memory_tracking/{self.algorithm_name}_run_{run_index}_memory_usage.csv'.")
        
        # status as 'finished' once the process ends
        self.status_flag['finished'] = True  


    def cpu_time_tracker(self, run_index):
        process = psutil.Process(self.pid)
        
        # status as 'started' immediately when the process starts
        self.status_flag['started'] = True  
        
        # Record the initial CPU time (user + system)
        start_cpu_times = process.cpu_times()
        start_time = time.time()  

        while not self.stop_flag.is_set():
            time.sleep(0.1)  

        # Record the final CPU time after the process ends
        end_cpu_times = process.cpu_times()
        end_time = time.time()  
        
        # Calculate the CPU time used (difference between start and end)
        user_time_used = end_cpu_times.user - start_cpu_times.user
        system_time_used = end_cpu_times.system - start_cpu_times.system
        total_cpu_time_used = user_time_used + system_time_used 
        
        # Elapsed real time
        elapsed_real_time = end_time - start_time
        
        # Save the CPU time data to the CSV
        cpu_time_data = [(elapsed_real_time, total_cpu_time_used, user_time_used, system_time_used)]
        cpu_time_df = pd.DataFrame(cpu_time_data, columns=["elapsed_time", "total_cpu_time", "user_time", "system_time"])
        if self.rec_type == "paragraph":
            cpu_time_df.to_csv(f"cpu_tracking/paragraph/{self.algorithm_name}/run_{run_index}_cpu_time_usage.csv", index=False)
        elif self.rec_type == "description":
            cpu_time_df.to_csv(f"cpu_tracking/description/{self.algorithm_name}/run_{run_index}_cpu_time_usage.csv", index=False)
        print(f"CPU time data saved to 'cpu_tracking/{self.algorithm_name}_run_{run_index}_cpu_time_usage.csv'.")
        
        # status as 'finished' once the process ends
        self.status_flag['finished'] = True

    def gpu_tracker(self, run_index):
        self.status_flag['started'] = True  
        
        while not self.stop_flag.is_set():  
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].memoryUtil * 100    # GPU memory usage percentage
                gpu_memory = gpus[0].memoryUsed         # Total memory in MB
                
                self.shared_gpu_list.append((time.time(), gpu_usage, gpu_memory))  # Append timestamp, usage, and total memory
            time.sleep(1)  

        # After stop flag is set, save the collected data
        gpu_data = list(self.shared_gpu_list)
        gpu_df = pd.DataFrame(gpu_data, columns=["timestamp", "gpu_usage", "gpu_memory"])
        if self.rec_type == "paragraph":
            gpu_df.to_csv(f"gpu_tracking/paragraph/{self.algorithm_name}/run_{run_index}_gpu_usage.csv", index=False)
        elif self.rec_type == "description":
            gpu_df.to_csv(f"gpu_tracking/description/{self.algorithm_name}/run_{run_index}_gpu_usage.csv", index=False)
        print(f"GPU usage data saved to 'gpu_tracking/{self.algorithm_name}_run_{run_index}_gpu_usage.csv'.")
        
        # status as 'finished' once the process ends
        self.status_flag['finished'] = True 