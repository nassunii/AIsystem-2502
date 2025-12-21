"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# lazy model setup
_FACE_ENGINE = None
def _engine():
    global _FACE_ENGINE
    if _FACE_ENGINE is None:
        model = FaceAnalysis(providers=["CPUExecutionProvider"])
        model.prepare(ctx_id=0, det_size=(640, 640))
        _FACE_ENGINE = model
    return _FACE_ENGINE


def _decode(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (bytes, bytearray)):
        arr = np.frombuffer(x, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img
    raise TypeError("Unsupported image format")


def detect_faces(image: Any) -> List[Any]:
    frame = _decode(image)
    faces = _engine().get(frame)

    result = []
    for f in faces:
        (x1, y1, x2, y2) = f.bbox.astype(int)
        face_crop = frame[y1:y2, x1:x2]
        result.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "crop": face_crop,
            "keypoints": f.kps,
            "embedding": f.normed_embedding,
            "confidence": float(f.det_score),
        })

    result.sort(key=lambda x: x["confidence"], reverse=True)
    return result


def compute_face_embedding(face_image: Any) -> Any:
    faces = detect_faces(face_image)
    if not faces:
        raise ValueError("No face found")
    return faces[0]["embedding"]


def detect_face_keypoints(face_image: Any) -> Any:
    faces = detect_faces(face_image)
    if not faces:
        raise ValueError("No face found")
    return faces[0]["keypoints"]


def warp_face(image: Any, homography_matrix: Any) -> Any:
    img = _decode(image)
    M = np.asarray(homography_matrix, dtype=np.float32)

    if M.shape == (2, 3):
        return cv2.warpAffine(img, M, (112, 112))
    elif M.shape == (3, 3):
        return cv2.warpPerspective(img, M, (112, 112))
    else:
        raise ValueError("Invalid homography matrix size")


def antispoof_check(face_image: Any) -> float:
    img = _decode(face_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_32F).var()
    score = (sharp / 280.0) ** 0.65
    return float(np.clip(score, 0.0, 1.0))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    fa = detect_faces(image_a)
    fb = detect_faces(image_b)

    if not fa or not fb:
        raise ValueError("Face missing")

    emb_a = fa[0]["embedding"]
    emb_b = fb[0]["embedding"]

    spoof_a = antispoof_check(fa[0]["crop"])
    spoof_b = antispoof_check(fb[0]["crop"])

    return _cos(emb_a, emb_b)
