def get_config(algo_name):
    """
    Returns a config dictionary for the given algorithm.
    """

    # --- Choose recommendation type ---
    rec_type = "paragraph"
    rec_type = "description"

    # --- Set dataset path ---
    data_path = (
        "datasets/paragraphs_limited_to_200_quarter.csv"
        if rec_type == "paragraph"
        else "datasets/book_details_clean.csv"
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
    tracking_base = f"src/results/results_for_{rec_type}"
    results_dir = f"{tracking_base}/results/{algo_name}"

    config = {
        # Core settings
        "algorithm_name": algo_name,
        "rec_type": rec_type,
        "data_path": data_path,
        "input_pairs": input_pairs,

        # Recommendation logic
        "top_n": 5,
        "threshold": 0.5,
        "test_coverage": True,

        # Output paths
        "results_dir": results_dir,
        "tracking_base": tracking_base,
    }

    return config
