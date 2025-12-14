import numpy as np
from sklearn.cluster import AgglomerativeClustering


# CORE EMBEDDING CLUSTERING

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


# SIMILARITY AND SMOOTHING

def compute_similarity(normed):
    """Cosine similarity between adjacent embeddings."""
    sims = [np.dot(normed[i], normed[i - 1]) for i in range(1, len(normed))]
    return np.array(sims)


def compute_multimodal_similarity(
    embeddings,
    visual_features=None,
    audio_features=None,
    transcript_embeddings=None,
    weights=None
):
    """
    Combine multiple modality signals for better boundary detection.
    
    Args:
        embeddings: Semantic embeddings (normalized)
        visual_features: Frame-level visual features (optional)
        audio_features: Audio segment features (optional)
        transcript_embeddings: Transcript-based embeddings (optional)
        weights: Dict with keys 'semantic', 'visual', 'audio', 'transcript'
    
    Returns:
        Array of similarity scores between adjacent chunks
    """
    if weights is None:
        weights = {
            'semantic': 0.4,
            'visual': 0.3,
            'audio': 0.2,
            'transcript': 0.1
        }
    
    n = len(embeddings)
    sims = []
    
    for i in range(1, n):
        sim_score = 0
        weight_sum = 0
        
        # Semantic embedding similarity
        if embeddings is not None:
            sem_sim = np.dot(embeddings[i], embeddings[i-1])
            sim_score += weights.get('semantic', 0.4) * sem_sim
            weight_sum += weights.get('semantic', 0.4)
        
        # Visual scene similarity
        if visual_features is not None and len(visual_features) > i:
            vis_sim = np.dot(visual_features[i], visual_features[i-1])
            sim_score += weights.get('visual', 0.3) * vis_sim
            weight_sum += weights.get('visual', 0.3)
        
        # Audio pattern similarity
        if audio_features is not None and len(audio_features) > i:
            aud_sim = np.dot(audio_features[i], audio_features[i-1])
            sim_score += weights.get('audio', 0.2) * aud_sim
            weight_sum += weights.get('audio', 0.2)
        
        # Transcript coherence
        if transcript_embeddings is not None and len(transcript_embeddings) > i:
            trans_sim = np.dot(transcript_embeddings[i], transcript_embeddings[i-1])
            sim_score += weights.get('transcript', 0.1) * trans_sim
            weight_sum += weights.get('transcript', 0.1)
        
        # Normalize by actual weights used
        if weight_sum > 0:
            sim_score /= weight_sum
        
        sims.append(sim_score)
    
    return np.array(sims)


def smooth(values, k=3):
    """Moving average smoothing."""
    if len(values) < k:
        return values
    return np.convolve(values, np.ones(k)/k, mode="same")


# BOUNDARY EXTRACTION

def labels_to_boundaries(labels):
    """Find boundaries at cluster label transitions."""
    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    return boundaries


def detect_scene_changes(visual_features, threshold=0.3):
    """
    Detect visual scene changes based on feature discontinuity.
    
    Args:
        visual_features: Array of normalized visual features per chunk
        threshold: Similarity threshold below which a scene change is detected
    
    Returns:
        List of chunk indices where scene changes occur
    """
    if visual_features is None or len(visual_features) < 2:
        return []
    
    scene_boundaries = []
    
    for i in range(1, len(visual_features)):
        # Compute cosine similarity between consecutive frames
        similarity = np.dot(visual_features[i], visual_features[i-1])
        
        # Low similarity indicates a scene change
        if similarity < threshold:
            scene_boundaries.append(i)
    
    return scene_boundaries


def detect_audio_transitions(audio_features, pause_threshold=2.0):
    """
    Detect audio-based transitions (speaker changes, pauses, music shifts).
    
    Args:
        audio_features: List of dicts with keys like 'speaker', 'pause_before', 'type'
        pause_threshold: Minimum pause duration (seconds) to mark as boundary
    
    Returns:
        List of chunk indices where audio transitions occur
    """
    if audio_features is None or len(audio_features) < 2:
        return []
    
    boundaries = []
    
    for i in range(1, len(audio_features)):
        curr = audio_features[i]
        prev = audio_features[i-1]
        
        # Speaker change
        if curr.get('speaker') != prev.get('speaker'):
            boundaries.append(i)
        
        # Significant pause
        elif curr.get('pause_before', 0) > pause_threshold:
            boundaries.append(i)
        
        # Audio type transition (speech -> music, etc.)
        elif curr.get('type') != prev.get('type'):
            boundaries.append(i)
    
    return boundaries


def align_boundaries_with_scenes(semantic_boundaries, scene_boundaries, tolerance=5):
    """
    Align semantic boundaries with visual scene changes for better coherence.
    
    Args:
        semantic_boundaries: Boundaries from clustering
        scene_boundaries: Boundaries from scene detection
        tolerance: Max distance to snap semantic boundary to scene boundary
    
    Returns:
        Aligned boundary list
    """
    if not scene_boundaries:
        return semantic_boundaries
    
    aligned = []
    
    for sb in semantic_boundaries:
        # Find nearest scene boundary within tolerance
        distances = [abs(scb - sb) for scb in scene_boundaries]
        min_dist = min(distances)
        
        if min_dist <= tolerance:
            nearest_idx = distances.index(min_dist)
            aligned.append(scene_boundaries[nearest_idx])
        else:
            aligned.append(sb)
    
    return sorted(set(aligned))


# TOPIC QUALITY EVALUATION

def evaluate_topics(chunks, labels, normed):
    """Evaluate segmentation using cohesion + separation."""
    boundaries = labels_to_boundaries(labels)

    # Within topic similarity (cohesion)
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

    # Topic separation
    separation_scores = []
    for i in range(1, len(boundaries)):
        prev = normed[boundaries[i] - 1]
        curr = normed[boundaries[i]]
        separation_scores.append(1 - np.dot(prev, curr))

    separation = float(np.mean(separation_scores)) if separation_scores else 0

    score = 0.6 * cohesion + 0.4 * separation
    return score


def evaluate_video_segmentation(chunks, boundaries, embeddings, metadata=None):
    """
    Enhanced evaluation for video segmentation combining multiple factors.
    
    Args:
        chunks: List of chunk dicts
        boundaries: List of boundary indices
        embeddings: Normalized embeddings
        metadata: Optional dict with 'fps', 'duration_weights', etc.
    
    Returns:
        Combined quality score
    """
    if len(boundaries) < 2:
        return 0.5
    
    # Base semantic score
    labels = []
    for i, b in enumerate(boundaries[:-1]):
        next_b = boundaries[i+1] if i+1 < len(boundaries) else len(embeddings)
        labels.extend([i] * (next_b - b))
    
    # Pad if needed
    while len(labels) < len(embeddings):
        labels.append(labels[-1] if labels else 0)
    
    base_score = evaluate_topics(chunks, np.array(labels), embeddings)
    
    # Temporal regularity (penalize highly uneven segments)
    durations = []
    for i in range(len(boundaries)-1):
        start_chunk = chunks[boundaries[i]]
        end_chunk = chunks[boundaries[i+1]-1] if boundaries[i+1] < len(chunks) else chunks[-1]
        duration = end_chunk.get('end', 0) - start_chunk.get('start', 0)
        durations.append(duration)
    
    if durations:
        mean_dur = np.mean(durations)
        duration_variance = np.std(durations) / mean_dur if mean_dur > 0 else 1
        regularity_score = 1 / (1 + duration_variance)
    else:
        regularity_score = 0.5
    
    # Boundary sharpness (how distinct are the transitions)
    sharpness_scores = []
    for b in boundaries[1:-1]:
        if b >= 2 and b < len(embeddings) - 1:
            before = np.dot(embeddings[b-2], embeddings[b-1])
            after = np.dot(embeddings[b], embeddings[b+1])
            across = np.dot(embeddings[b-1], embeddings[b])
            sharpness = (before + after) / 2 - across
            sharpness_scores.append(max(0, sharpness))
    
    sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
    
    # Combined score with configurable weights
    weights = {
        'base': 0.5,
        'regularity': 0.2,
        'sharpness': 0.3
    }
    
    if metadata and 'score_weights' in metadata:
        weights.update(metadata['score_weights'])
    
    final_score = (
        weights['base'] * base_score +
        weights['regularity'] * regularity_score +
        weights['sharpness'] * sharpness
    )
    
    return final_score


# AUTO TOPIC COUNT SELECTION

def auto_select_topic_count(chunks, embeddings, candidates=None, metadata=None):
    """
    Try multiple topic counts and select the best one.
    
    Args:
        chunks: List of chunk dicts
        embeddings: Embedding array
        candidates: Range of k values to try (default: range(5, 20))
        metadata: Optional video metadata for enhanced evaluation
    
    Returns:
        Tuple of (best_k, best_labels, best_normed, best_score)
    """
    if candidates is None:
        # INCREASED: Try more granular segmentations (5-25 topics)
        candidates = range(5, 26)
    
    n_chunks = len(chunks)
    if n_chunks < 2:
        return 1, [0] * n_chunks, None, 0

    # Clamp candidates so k <= number of chunks, but prefer higher k values
    valid_candidates = [k for k in range(5, min(n_chunks, 30) + 1) if 2 <= k <= n_chunks]

    if not valid_candidates:
        valid_candidates = [min(2, n_chunks)]

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

        # Use video-aware evaluation if metadata provided
        if metadata and metadata.get('use_video_eval', False):
            boundaries = labels_to_boundaries(labels)
            score = evaluate_video_segmentation(chunks, boundaries, normed, metadata)
        else:
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



# REFINEMENT AND VALIDATION

def validate_boundaries_with_similarity(boundaries, sims, high_sim_threshold=0.65):
    """
    Keep all cluster boundaries unless similarity is VERY high.
    INCREASED threshold from 0.4 to 0.65 - more aggressive boundary detection.
    """
    new = [boundaries[0]]
    for b in boundaries[1:]:
        if b == 0 or b >= len(sims):
            continue
        # Only merge if similarity is REALLY high (>0.65)
        if sims[b - 1] > high_sim_threshold:
            continue
        new.append(b)
    return sorted(set(new))


def enforce_min_topic_length(boundaries, chunks, min_chunks=1):
    """
    Merge topics smaller than min_chunks into the previous topic.
    DECREASED from 2 to 1 - allows single-chunk topics for very granular segmentation.
    """
    new = [boundaries[0]]

    for i in range(1, len(boundaries)):
        prev = new[-1]
        curr = boundaries[i]

        # If this topic length is too small → merge
        if (curr - prev) < min_chunks:
            continue

        new.append(curr)

    return new


def enforce_temporal_constraints(boundaries, chunks, 
                                  min_duration_sec=15,  # DECREASED from 30
                                  max_duration_sec=180,  # DECREASED from 300
                                  fps=30):
    """
    Enforce realistic video segment durations based on time.
    UPDATED: More granular - min 15s, max 3min segments.
    
    Args:
        boundaries: List of boundary indices
        chunks: List of chunk dicts with 'start' and 'end' timestamps
        min_duration_sec: Minimum segment duration in seconds (default 15s)
        max_duration_sec: Maximum segment duration in seconds (default 180s)
        fps: Frames per second (for timestamp conversion if needed)
    
    Returns:
        Refined boundary list
    """
    if not chunks or len(boundaries) < 2:
        return boundaries
    
    new = [boundaries[0]]
    
    for i in range(1, len(boundaries)):
        prev_idx = new[-1]
        curr_idx = boundaries[i]
        
        # Get timestamps
        start_time = chunks[prev_idx].get('start', 0)
        end_time = chunks[curr_idx - 1].get('end', 0) if curr_idx > 0 else 0
        
        duration = end_time - start_time
        
        # Skip if segment is too short
        if duration < min_duration_sec:
            continue
        
        new.append(curr_idx)
    
    return new


def merge_small_topics(boundaries, min_chunks=1):
    """Optional: merge extremely tiny topics."""
    new = [boundaries[0]]
    for i in range(1, len(boundaries)):
        if boundaries[i] - new[-1] < min_chunks:
            continue
        new.append(boundaries[i])
    return new


# CHUNK GROUPING

def group_chunks_into_topics(chunks, boundaries):
    """Group chunks into topic segments based on boundaries."""
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


# MAIN SEGMENTATION FUNCTIONS

def topic_segmentation(chunks, embeddings, metadata=None, granularity='medium'):
    """
    Main segmentation function with video-aware enhancements.
    
    Args:
        chunks: List of chunk dicts with 'start', 'end', 'text'
        embeddings: Embedding array for chunks
        metadata: Optional dict with video-specific features:
            - 'visual_features': Visual embeddings per chunk
            - 'audio_features': Audio feature dicts per chunk
            - 'transcript_embeddings': Alternative text embeddings
            - 'scene_boundaries': Pre-detected scene changes
            - 'fps': Video frame rate
            - 'min_duration_sec': Minimum segment duration
            - 'max_duration_sec': Maximum segment duration
            - 'use_multimodal': Whether to use multimodal similarity
            - 'use_video_eval': Whether to use video-aware evaluation
        granularity: Control segmentation granularity - 'low', 'medium', 'high', 'very_high'
            - 'low': 3-8 topics (broad segments)
            - 'medium': 5-15 topics (balanced)
            - 'high': 8-25 topics (detailed)
            - 'very_high': 10-30 topics (very granular)
    
    Returns:
        Dict with segmentation results
    """
    if metadata is None:
        metadata = {}
    
    # Set granularity parameters
    granularity_configs = {
        'low': {
            'candidates': range(3, 9),
            'similarity_threshold': 0.5,
            'min_chunks': 3,
            'min_duration': 45,
            'max_duration': 300
        },
        'medium': {
            'candidates': range(5, 16),
            'similarity_threshold': 0.65,
            'min_chunks': 2,
            'min_duration': 30,
            'max_duration': 200
        },
        'high': {
            'candidates': range(8, 26),
            'similarity_threshold': 0.75,
            'min_chunks': 1,
            'min_duration': 20,
            'max_duration': 150
        },
        'very_high': {
            'candidates': range(10, 31),
            'similarity_threshold': 0.80,
            'min_chunks': 1,
            'min_duration': 15,
            'max_duration': 120
        }
    }
    
    config = granularity_configs.get(granularity, granularity_configs['high'])
    
    # Auto-select topic count with configured granularity
    best_k, labels, normed, score = auto_select_topic_count(
        chunks, embeddings, candidates=config['candidates'], metadata=metadata
    )

    raw_boundaries = labels_to_boundaries(labels)

    # Compute similarity with optional multimodal features
    if metadata.get('use_multimodal', False):
        sims = compute_multimodal_similarity(
            normed,
            visual_features=metadata.get('visual_features'),
            audio_features=metadata.get('audio_features'),
            transcript_embeddings=metadata.get('transcript_embeddings'),
            weights=metadata.get('modality_weights')
        )
    else:
        sims = compute_similarity(normed)
    
    # Smooth similarity scores
    sims = smooth(sims, k=3)

    # Validate boundaries against similarity with configured threshold
    boundaries = validate_boundaries_with_similarity(
        raw_boundaries, sims, high_sim_threshold=config['similarity_threshold']
    )
    
    # Align with scene changes if available
    if metadata.get('scene_boundaries'):
        boundaries = align_boundaries_with_scenes(
            boundaries, 
            metadata['scene_boundaries'],
            tolerance=metadata.get('scene_alignment_tolerance', 5)
        )
    
    # Enforce minimum topic length with configured minimum
    boundaries = enforce_min_topic_length(
        boundaries, chunks, min_chunks=config['min_chunks']
    )
    
    if metadata.get('fps') or metadata.get('min_duration_sec'):
        boundaries = enforce_temporal_constraints(
            boundaries, 
            chunks,
            min_duration_sec=metadata.get('min_duration_sec', config['min_duration']),
            max_duration_sec=metadata.get('max_duration_sec', config['max_duration']),
            fps=metadata.get('fps', 30)
        )
    
    # Group chunks into final topics
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