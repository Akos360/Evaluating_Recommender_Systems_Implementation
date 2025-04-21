import pandas as pd
import os

def get_config(algo_name):
    """
    Returns a config dictionary for the given algorithm.
    """

    # --- Choose recommendation type ---
    rec_type = "paragraph"
    # rec_type = "description"

    # --- Set dataset path ---
    data_path = (
        "../datasets/paragraphs_limited_to_200_quarter.csv"
        if rec_type == "paragraph"
        else "../datasets/book_details_clean.csv"
    )

    # --- Set input test pairs ---
    input_pairs = (
        [
            (70, 1001),
            (231, 1560),
            (179, 141),
            (337, 24),
            (264, 64),
            (34, 1269),
            (85, 193),
            (67, 198),
            (36, 518),
            (297, 297),
        ]
        if rec_type == "paragraph"
        else [(i, None) for i in range(70, 80)]
    )

    # --- Set paths for results and tracking (inside src/results/) ---
    tracking_base = f"results/results_for_{rec_type}"
    results_dir = f"{tracking_base}/results/{algo_name}"

    thresholds = {
        "tf_idf": 0.5,
        "lsa": 0.5,
        "bow": 0.3,
        "glove": 0.99,
        "fasttext": 0.95,
        "bert": 0.5
    }
    threshold = thresholds.get(algo_name, 0.5)

    config = {
        # Core settings
        "algorithm_name": algo_name,
        "rec_type": rec_type,
        "data_path": data_path,
        "input_pairs": input_pairs,
 
        # Recommendation logic
        "top_n": 5,
        "threshold": threshold,
        "test_coverage": True,

        # Output paths
        "results_dir": results_dir,
        "tracking_base": tracking_base,
    }

    return config


def save_training_time_csv(algo_name, rec_type, train_time, dataset_size):
    save_path=f"results/results_for_{rec_type}/results/training_times.csv"

    new_row = pd.DataFrame([{
        "algorithm_name": algo_name,
        "rec_type": rec_type,
        "train_time": round(train_time, 4),
        "dataset_size": dataset_size
    }])

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        mask = (df["algorithm_name"] == algo_name) & (df["rec_type"] == rec_type)
        if mask.any():
            df.loc[mask, ["train_time", "dataset_size"]] = [round(train_time, 4), dataset_size]
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(save_path, index=False)
    print(f"Training time saved to {save_path}")
