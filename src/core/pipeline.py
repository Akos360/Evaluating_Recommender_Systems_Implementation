import joblib
from tqdm import tqdm

def run_recommendation_pipeline(model, data, input_pairs, resource_tracker, stop_flag,
                                top_n=5, threshold=0.5, test_coverage=False, save_fn=None):
    all_results = []

    for book_idx, para_idx in input_pairs:
        print(f"▶️ Running: {book_idx} - {para_idx}")
        resource_tracker.reset_shared_lists()
        resource_tracker.stop_flag.clear()
        resource_tracker.status_flag.clear()

        mem_proc = resource_tracker.spawn_tracker("memory", book_idx)
        cpu_proc = resource_tracker.spawn_tracker("cpu", book_idx)
        gpu_proc = resource_tracker.spawn_tracker("gpu", book_idx)

        try:
            model.prepare_input_and_filtered(data, book_idx, para_idx)
        except ValueError as e:
            print(f"⚠️ {e}")
            stop_flag.set()
            continue

        input_vector = model.get_input_vector()
        doc_vectors = model.get_doc_vectors()

        with tqdm(total=len(doc_vectors), desc="Computing Similarities", unit="vec") as pbar:
            similarities = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(model.compute_similarity)(input_vector, doc_vectors[idx])
                for idx in range(len(doc_vectors))
            )
            pbar.update(len(similarities))

        max_score = max(similarities)
        threshold_val = threshold * max_score

        filtered = [(i, sim) for i, sim in enumerate(similarities) if threshold_val <= sim < 1.0]
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
        if not test_coverage:
            filtered = filtered[:top_n]

        recs = [model.format_recommendation(i, score) for i, score in filtered]

        resource_tracker.wait_for_trackers()
        mem_proc.join(); cpu_proc.join(); gpu_proc.join()

        all_results.append({
            "input": model.input_data,
            "recommendations": recs
        })

        if save_fn:
            save_fn(book_idx, para_idx, model.input_data, recs)

    return all_results
