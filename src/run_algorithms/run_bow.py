from config.base_config import get_config, save_training_time_csv
from models.bow_model import BoWRecommender
from core.pipeline import run_recommendation_pipeline
from resource_tracking.resource_tracker import ResourceTracker
import pandas as pd
import os
import time

if __name__ == "__main__":
    config = get_config("bow")

    # Load data
    data = pd.read_csv(config["data_path"], encoding="ISO-8859-1")

    # Initialize resource tracker with correct paths
    tracker = ResourceTracker(
        pid=os.getpid(),
        algorithm_name=config["algorithm_name"],
        rec_type=config["rec_type"],
        memory_file=f"{config['tracking_base']}/memory_tracking/{config['algorithm_name']}_memory_usage.csv",
        cpu_file=f"{config['tracking_base']}/cpu_tracking/{config['algorithm_name']}_cpu_time_usage.csv",
        gpu_file=f"{config['tracking_base']}/gpu_tracking/{config['algorithm_name']}_gpu_usage.csv"
    )

    # Train the model
    start_time = time.time()
    model = BoWRecommender(config["rec_type"])
    model.train(data)
    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed} s")
    save_training_time_csv(
        algo_name=config["algorithm_name"],
        rec_type=config["rec_type"],
        train_time=elapsed,
        dataset_size=len(data)
    )

    # Save results per input pair
    def save_results(book_idx, para_idx, input_data, recommendations):
        file_path = f"{config['results_dir']}/{config['algorithm_name']}_results_{book_idx}_{para_idx}.csv"
        if recommendations:
            pd.DataFrame(recommendations).to_csv(file_path, index=False)
        else:
            pd.DataFrame(columns=model.format_recommendation(0, 0.0).keys()).to_csv(file_path, index=False)

    # Run the main pipeline
    results = run_recommendation_pipeline(
        model=model,
        data=data,
        input_pairs=config["input_pairs"],
        resource_tracker=tracker,
        stop_flag=tracker.stop_flag,
        top_n=config["top_n"],
        threshold=config["threshold"],
        test_coverage=config["test_coverage"],
        save_fn=save_results
    )

    timing_df = pd.DataFrame(results)
    timing_df.to_csv(f"{config['results_dir']}/{config['algorithm_name']}_timing_summary.csv", index=False)

    # Save performance summary
    # summary_path = f"{config['results_dir']}/{config['algorithm_name']}_summary.csv"
    # pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"✅ {config['algorithm_name']} completed! Results saved in {config['results_dir']}")
