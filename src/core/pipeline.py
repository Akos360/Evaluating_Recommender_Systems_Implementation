import time
from tqdm import tqdm

def run_recommendation_pipeline(model, data, input_pairs, resource_tracker, stop_flag,
                                top_n=5, threshold=0.5, test_coverage=False, save_fn=None):
    
    timing_results = []

    # Loop to go through input pairs -> Runs
    for book_idx, para_idx in input_pairs:
        print(f"Running: {book_idx} - {para_idx}")
        start_time = time.time()

        # clean flags, lists
        resource_tracker.reset_shared_lists()
        resource_tracker.stop_flag.clear()
        resource_tracker.status_flag.clear()

        # start trackers
        mem_proc = resource_tracker.spawn_tracker("memory", book_idx)
        cpu_proc = resource_tracker.spawn_tracker("cpu", book_idx)
        gpu_proc = resource_tracker.spawn_tracker("gpu", book_idx)

        try:
            model.prepare_input_and_filtered(data, book_idx, para_idx)
        except ValueError as e:
            print(f"{e}")
            stop_flag.set()
            continue

        # Get input text -> vector
        start_time_input = time.time()
        input_vector = model.get_input_vector()
        elapsed_input = time.time() - start_time_input
        print(f"Input vector time:{elapsed_input} s")
        
        # get all other texts -> vectors
        start_time_vec = time.time()
        doc_vectors = model.get_doc_vectors()
        elapsed_vec = time.time() - start_time_vec
        print(f"All vectors time:{elapsed_vec} s")


        similarities = []

        # Compute similarities - Cosine Similarity
        if model.use_batch_similarity():
            with tqdm(total=doc_vectors.shape[0], desc="Computing Similarities (Batch)", unit="iteration") as pbar:
                similarities = model.compute_all_similarities(input_vector, doc_vectors)
                pbar.update(len(similarities))
        else:
            with tqdm(total=len(doc_vectors), desc="Computing Similarities (Loop)", unit="iteration") as pbar:
                for i, doc_vec in enumerate(doc_vectors):
                    sim = model.compute_similarity(input_vector, doc_vec)
                    similarities.append(sim)
                    pbar.update(1)

        # Get max Similarity Score - for threshold
        sorted_scores = sorted(similarities, reverse=True)
        max_score = sorted_scores[0]
        
        if max_score >= 0.999 and len(sorted_scores) > 1:
            max_score = sorted_scores[1]
            print("Top score = 1\n-> next score is set as max")
            
        threshold_val = threshold * max_score
        
        print(f"Max similarity score: {max_score}")
        print(f"Scores [:10]:\n{sorted_scores[:10]}")
        print(f"Threshold value ({threshold} × max): {threshold_val}")  

        # Filter based on threshold
        filtered = [(i, sim) for i, sim in enumerate(similarities) if threshold_val <= sim < 1.0]
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
        if not test_coverage:
            filtered = filtered[:top_n]

        # Format Output
        recs = [model.format_recommendation(i, score) for i, score in filtered]

        # Stop trackers
        resource_tracker.stop_flag.set()
        resource_tracker.wait_for_trackers()
        mem_proc.join(); 
        cpu_proc.join(); 
        gpu_proc.join()

        # save time
        elapsed = time.time() - start_time
        timing_results.append({
            "book_index": book_idx,
            "paragraph_index": para_idx,
            "elapsed_time_sec": round(elapsed, 2)
        })

        if save_fn:
            save_fn(book_idx, para_idx, model.input_data, recs)

    return timing_results
