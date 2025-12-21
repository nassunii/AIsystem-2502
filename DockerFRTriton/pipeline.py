from typing import Any, Tuple

import numpy as np

from triton_service import run_inference


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    If embeddings are already normalized (like InsightFace normed_embedding),
    we can use dot product directly. Otherwise, normalize first.
    """
    # Normalize embeddings (InsightFace ArcFace outputs are typically normalized)
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    
    # Normalize if not already normalized (check if norm is close to 1.0)
    if abs(a_norm - 1.0) > 0.01:
        vec_a = vec_a / a_norm
    if abs(b_norm - 1.0) > 0.01:
        vec_b = vec_b / b_norm
    
    # Cosine similarity of normalized vectors is just dot product
    similarity = float(np.dot(vec_a, vec_b))
    # Clamp to [-1, 1] range
    return max(-1.0, min(1.0, similarity))


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """
    emb_a = run_inference(client, image_a)
    emb_b = run_inference(client, image_b)
    return emb_a.squeeze(0), emb_b.squeeze(0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    return _cosine_similarity(emb_a, emb_b)
