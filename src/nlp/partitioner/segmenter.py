import numpy as np
from sklearn.cluster import AgglomerativeClustering


# Embedding clustering 

def cluster_embeddings(embeddings, n_clusters):
    """Cluster chunk embeddings using Agglomerative Clustering."""
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average"
    )

    labels = model.fit_predict(normed)
    return np.squeeze(labels), normed


# Similarity and smoothing

def compute_similarity(normed):
    """Cosine similarity between adjacent embeddings."""
    sims = [np.dot(normed[i], normed[i - 1]) for i in range(1, len(normed))]
    return np.array(sims)


def smooth(values, k=3):
    """Moving average smoothing."""
    if len(values) < k:
        return values
    return np.convolve(values, np.ones(k)/k, mode="same")


# Boundary extraction

def labels_to_boundaries(labels):
    """Find boundaries at cluster label transitions."""
    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    return boundaries


# Topic quality evaluator

def evaluate_topics(chunks, labels, normed):
    """Evaluate segmentation using cohesion + separation."""
    boundaries = labels_to_boundaries(labels)

    # Within topic similarity
    
    cohesion_scores = []
    for i, start in enumerate(boundaries):
        end = boundaries[i+1] if i+1 < len(boundaries) else len(labels)
        topic_vecs = normed[start:end]

        if len(topic_vecs) < 2:
            cohesion_scores.append(0.5)
            continue

        sims = np.dot(topic_vecs, topic_vecs.T)
        avg_sim = sims[np.triu_indices_from(sims, k=1)].mean()
        cohesion_scores.append(avg_sim)

    cohesion = float(np.mean(cohesion_scores))

    # Topic seperation
    
    separation_scores = []
    for i in range(1, len(boundaries)):
        prev = normed[boundaries[i] - 1]
        curr = normed[boundaries[i]]

        separation_scores.append(1 - np.dot(prev, curr))

    separation = float(np.mean(separation_scores)) if separation_scores else 0

    score = 0.6 * cohesion + 0.4 * separation
    return score


def auto_select_topic_count(chunks, embeddings, candidates=range(3, 11)):
    """Try multiple topic counts and select the best one safely."""
    n_chunks = len(chunks)
    if n_chunks < 2:
        return 1, [0] * n_chunks, None, 0    # trivial fallback

    # Clamp candidates so k <= number of chunks
    # INCREASED MAX CANDIDATES TO ALLOW MORE TOPICS FOR LONGER TRANSCRIPTS
    valid_candidates = [k for k in range(3, min(n_chunks, 15) + 1) if 2 <= k <= n_chunks] # Changed range(3, 11) to range(3, 15) + 1

    if not valid_candidates:
        valid_candidates = [min(2, n_chunks)]  # at least 2 if possible

    best_score = -1
    best_k = None
    best_labels = None
    best_normed = None

    for k in valid_candidates:
        try:
            labels, normed = cluster_embeddings(embeddings, n_clusters=k)
        except Exception as e:
            print(f"[auto_select_topic_count] Skipping k={k}: {e}")
            continue

        score = evaluate_topics(chunks, labels, normed)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_normed = normed

    # Final fallback if none selected
    if best_k is None:
        best_k = 2 if n_chunks >= 2 else 1
        best_labels = [0] * n_chunks
        best_normed = None

    return best_k, best_labels, best_normed, best_score

# Refinement and tuning

def validate_boundaries_with_similarity(boundaries, sims, high_sim_threshold=0.4):
    """
    Keep all cluster boundaries unless similarity is VERY high.
    This is intentionally permissive to avoid eliminating true topics.
    """
    # INCREASED THRESHOLD: Making the boundary validator more permissive (less likely to merge) 

    new = [boundaries[0]]
    for b in boundaries[1:]:
        if b == 0 or b >= len(sims):
            continue
        if sims[b - 1] > high_sim_threshold:
            continue
        new.append(b)
    return sorted(set(new))


def merge_small_topics(boundaries, min_chunks=1):
    """Optional: merge extremely tiny topics."""
    new = [boundaries[0]]
    for i in range(1, len(boundaries)):
        if boundaries[i] - new[-1] < min_chunks:
            continue
        new.append(boundaries[i])
    return new


def enforce_min_topic_length(boundaries, chunks, min_chunks=3):
    """
    Merge topics smaller than min_chunks into the previous topic.
    This is the simplest and highest-impact fix.
    """
    # DECREASED MIN_CHUNKS: Allowing for smaller, more precise topics (less aggressive merging)
    min_chunks = 2 
    
    new = [boundaries[0]]

    for i in range(1, len(boundaries)):
        prev = new[-1]
        curr = boundaries[i]

        # If this topic length is too small → merge
        if (curr - prev) < min_chunks:
            continue

        new.append(curr)

    return new



# Chunk grouping 

def group_chunks_into_topics(chunks, boundaries):
    topics = []

    for i, start in enumerate(boundaries):
        end = boundaries[i+1] if i+1 < len(boundaries) else len(chunks)
        group = chunks[start:end]

        topics.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(c["text"] for c in group),
            "chunk_count": len(group)
        })

    return topics


# Main topic segmentation

def topic_segmentation(chunks, embeddings):
    """Automatically selects topic count and segments."""
    
    # k-topic numbers auto-selection
    best_k, labels, normed, score = auto_select_topic_count(
        chunks, embeddings, candidates=range(3, 11)
    )

    raw_boundaries = labels_to_boundaries(labels)

    # Similarity smoothing
    sims = smooth(compute_similarity(normed), k=3)

    boundaries = validate_boundaries_with_similarity(raw_boundaries, sims)
    
    boundaries = enforce_min_topic_length(boundaries, chunks)
    
    topics = group_chunks_into_topics(chunks, boundaries)


    return {
        "topic_count": best_k,
        "score": score,
        "labels": labels.tolist(),
        "raw_boundaries": raw_boundaries,
        "boundaries": boundaries,
        "topics": topics,
        "similarity": sims.tolist()
    }