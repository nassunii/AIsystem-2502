import cv2
import numpy as np
from numpy.linalg import norm
from typing import Any, List
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

RETINA = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _bytes_to_bgr(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode fail")
    return img


def _largest(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))


def _cos(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


def detect_faces(image: Any) -> List[Any]:
    return app.get(_bytes_to_bgr(image))


def detect_face_keypoints(face_image: Any) -> Any:
    return face_image.kps


def warp_face(image: Any, kps: Any, size=(112, 112)) -> Any:
    dst = RETINA.copy()
    dst[:, 0] *= size[0] / 112.0
    dst[:, 1] *= size[1] / 112.0
    M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    return cv2.warpAffine(image, M, size, borderValue=0)


def compute_face_embedding(face_image: Any) -> Any:
    r = app.get(face_image)
    if not r:
        raise ValueError("no face")
    return r[0].embedding.astype(np.float32)


def antispoof_check(face_image: Any) -> float:
    g = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    s = lap.var()
    return float(s / (s + 1000.0))


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    a = _bytes_to_bgr(image_a)
    b = _bytes_to_bgr(image_b)

    fa = _largest(app.get(a))
    fb = _largest(app.get(b))

    if fa is None or fb is None:
        raise ValueError("no face")

    al_a = warp_face(a, fa.kps)
    al_b = warp_face(b, fb.kps)

    if antispoof_check(al_a) < 0.3 or antispoof_check(al_b) < 0.3:
        return 0.0

    ea = compute_face_embedding(al_a)
    eb = compute_face_embedding(al_b)

    return _cos(ea, eb)
