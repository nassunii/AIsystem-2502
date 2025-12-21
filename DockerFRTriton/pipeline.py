from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from triton_service import run_inference


def _unit_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    InsightFace ArcFace embeddings are typically unit-normalized.
    If the norm isn't close to 1.0, normalize it.
    """
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n if abs(n - 1.0) > 0.01 else x


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    If embeddings are already normalized (like InsightFace normed_embedding),
    we can use dot product directly. Otherwise, normalize first.
    """
    a = vec_a.reshape(-1)
    b = vec_b.reshape(-1)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0

    a = _unit_normalize(a)
    b = _unit_normalize(b)

    sim = float(np.dot(a, b))
    return float(np.clip(sim, -1.0, 1.0))


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """
    out_a = run_inference(client, image_a)
    out_b = run_inference(client, image_b)

    # keep the same behavior: return (emb_a.squeeze(0), emb_b.squeeze(0))
    return out_a.squeeze(axis=0), out_b.squeeze(axis=0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    ea, eb = get_embeddings(client, image_a, image_b)
    return _cosine_similarity(ea, eb)
