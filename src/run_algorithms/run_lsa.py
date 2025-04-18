from config.base_config import get_config
from models.lsa_model import LSARecommender
from core.pipeline import run_recommendation_pipeline
from resource_tracking.resource_tracker import ResourceTracker

import pandas as pd
import os

if __name__ == "__main__":
    config = get_config("lsa")

    # Ensure results folder exists
    os.makedirs(config["results_dir"], exist_ok=True)

    # Load dataset
    data = pd.read_csv(config["data_path"], encoding="ISO-8859-1")

    # Setup resource tracker
    tracker = ResourceTracker(
        pid=os.getpid(),
        algorithm_name=config["algorithm_name"],
        rec_type=config["rec_type"],
        memory_file=f"{config['tracking_base']}/memory_tracking/{config['algorithm_name']}_memory_usage.csv",
        cpu_file=f"{config['tracking_base']}/cpu_tracking/{config['algorithm_name']}_cpu_time_usage.csv",
        gpu_file=f"{config['tracking_base']}/gpu_tracking/{config['algorithm_name']}_gpu_usage.csv"
    )

    # Initialize and train LSA model
    model = LSARecommender(config["rec_type"])
    model.train(data)

    # Save per-run results
    def save_results(book_idx, para_idx, input_data, recommendations):
        file_path = f"{config['results_dir']}/{config['algorithm_name']}_results_{book_idx}_{para_idx}.csv"
        pd.DataFrame(recommendations).to_csv(file_path, index=False)

    # Run the recommendation pipeline
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

    # Save summary file
    summary_path = f"{config['results_dir']}/{config['algorithm_name']}_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"✅ {config['algorithm_name']} completed! Results saved in {config['results_dir']}")
