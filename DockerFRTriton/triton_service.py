import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input.1"
MODEL_OUTPUT_NAME = "516"
MODEL_IMAGE_SIZE = (112, 112)

DETECTOR_NAME = "face_detector"
DETECTOR_INPUT_NAME = "input.1"
DETECTOR_IMAGE_SIZE = (640, 640)
DETECTOR_OUTPUT_NAMES = ["443", "468", "493", "446", "471", "496", "449", "474", "499"]

# InsightFace RetinaFace/SCRFD configuration (buffalo_sc style)
FMC = 3
FEAT_STRIDE_FPN = [8, 16, 32]
NUM_ANCHORS = 2
USE_KPS = True

# ArcFace standard alignment reference points for 112x112
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Estimate similarity transform matrix for face alignment (InsightFace style).
    """
    if lmk.shape != (5, 2):
        raise AssertionError("landmark must have shape (5, 2)")

    ref = ARCFACE_DST.copy()
    if image_size != 112:
        ref = ref * (image_size / 112.0)

    try:
        from skimage import transform as trans

        t = trans.SimilarityTransform()
        t.estimate(lmk, ref)
        mat = t.params[0:2, :]
    except ImportError:
        mat, _ = cv2.estimateAffinePartial2D(lmk, ref)
        if mat is None:
            mat = cv2.getAffineTransform(
                lmk[:3].astype(np.float32), ref[:3].astype(np.float32)
            )

    return mat


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Align + crop face using 5 landmarks (InsightFace standard).
    """
    mat = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, mat, (image_size, image_size), borderValue=0.0)


def _write_text(path: Path, content: str, created_msg: str, exists_msg: str) -> None:
    if not path.exists():
        path.write_text(content)
        print(created_msg)
    else:
        print(exists_msg)


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate Triton model repository with FR + Detector ONNX models and configs.
    (Keeps the same checks/behavior as the original code.)
    """
    # ---- FR ----
    fr_model_dir = model_repo / MODEL_NAME / MODEL_VERSION
    fr_model_path = fr_model_dir / "model.onnx"
    fr_config_path = fr_model_dir.parent / "config.pbtxt"

    # NOTE: Preserve original behavior: check existence BEFORE mkdir
    if not fr_model_path.exists():
        raise FileNotFoundError(
            f"Missing FR ONNX model at {fr_model_path}. "
            "Please place your exported model there."
        )

    fr_model_dir.mkdir(parents=True, exist_ok=True)

    fr_cfg = (
        textwrap.dedent(
            f"""
            name: "{MODEL_NAME}"
            platform: "onnxruntime_onnx"
            max_batch_size: 0
            default_model_filename: "model.onnx"
            input [
              {{
                name: "{MODEL_INPUT_NAME}"
                data_type: TYPE_FP32
                dims: [-1, 3, {MODEL_IMAGE_SIZE[0]}, {MODEL_IMAGE_SIZE[1]}]
              }}
            ]
            output [
              {{
                name: "{MODEL_OUTPUT_NAME}"
                data_type: TYPE_FP32
                dims: [1, 512]
              }}
            ]
            instance_group [
              {{ kind: KIND_CPU }}
            ]
            """
        ).strip()
        + "\n"
    )

    _write_text(
        fr_config_path,
        fr_cfg,
        f"[triton] Created FR model config at {fr_config_path}",
        f"[triton] Using existing FR model config at {fr_config_path}",
    )
    print(f"[triton] Prepared FR model repository at {fr_model_dir.parent}")

    # ---- Detector ----
    det_model_dir = model_repo / DETECTOR_NAME / MODEL_VERSION
    det_model_path = det_model_dir / "model.onnx"
    det_config_path = det_model_dir.parent / "config.pbtxt"

    # NOTE: Preserve original behavior: check existence BEFORE mkdir
    if not det_model_path.exists():
        raise FileNotFoundError(
            f"Missing Detector ONNX model at {det_model_path}. "
            "Please place your exported model there."
        )

    det_model_dir.mkdir(parents=True, exist_ok=True)

    det_out_lines = [
        '          { name: "443", data_type: TYPE_FP32, dims: [12800, 1] },',
        '          { name: "468", data_type: TYPE_FP32, dims: [3200, 1] },',
        '          { name: "493", data_type: TYPE_FP32, dims: [800, 1] },',
        '          { name: "446", data_type: TYPE_FP32, dims: [12800, 4] },',
        '          { name: "471", data_type: TYPE_FP32, dims: [3200, 4] },',
        '          { name: "496", data_type: TYPE_FP32, dims: [800, 4] },',
        '          { name: "449", data_type: TYPE_FP32, dims: [12800, 10] },',
        '          { name: "474", data_type: TYPE_FP32, dims: [3200, 10] },',
        '          { name: "499", data_type: TYPE_FP32, dims: [800, 10] }',
    ]

    det_cfg = (
        textwrap.dedent(
            f"""
            name: "{DETECTOR_NAME}"
            platform: "onnxruntime_onnx"
            max_batch_size: 0
            default_model_filename: "model.onnx"
            input [
              {{
                name: "{DETECTOR_INPUT_NAME}"
                data_type: TYPE_FP32
                dims: [1, 3, -1, -1]
              }}
            ]
            output [
{chr(10).join(det_out_lines)}
            ]
            instance_group [
              {{ kind: KIND_CPU }}
            ]
            """
        ).strip()
        + "\n"
    )

    _write_text(
        det_config_path,
        det_cfg,
        f"[triton] Created Detector model config at {det_config_path}",
        f"[triton] Using existing Detector model config at {det_config_path}",
    )
    print(f"[triton] Prepared Detector model repository at {det_model_dir.parent}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo.
    """
    triton_bin = subprocess.run(
        ["which", "tritonserver"], capture_output=True, text=True
    ).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    args = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(args)}")
    time.sleep(3)
    return proc


def stop_triton_server(server_handle: Any) -> None:
    """
    Stop Triton server started by start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Create Triton HTTP client.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    cli = httpclient.InferenceServerClient(url=url, verbose=False)
    if not cli.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return cli


def run_detector_inference(client: Any, image_bytes: bytes) -> dict:
    """
    Run detector model and return raw outputs + metadata.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("Pillow and tritonclient[http] are required.") from exc

    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        orig_wh = im.size  # (W, H)
        im = im.resize(DETECTOR_IMAGE_SIZE)
        arr = np.asarray(im, dtype=np.float32)

    # (img - 127.5) / 128.0
    arr = (arr - 127.5) / 128.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    inp = np.expand_dims(arr, axis=0)

    tr_in = httpclient.InferInput(DETECTOR_INPUT_NAME, inp.shape, "FP32")
    tr_in.set_data_from_numpy(inp)

    tr_outs = [httpclient.InferRequestedOutput(n) for n in DETECTOR_OUTPUT_NAMES]
    resp = client.infer(model_name=DETECTOR_NAME, inputs=[tr_in], outputs=tr_outs)

    out_map = {n: resp.as_numpy(n) for n in DETECTOR_OUTPUT_NAMES}
    return {"outputs": out_map, "original_size": orig_wh, "detector_size": DETECTOR_IMAGE_SIZE}


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distances to boxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack((x1, y1, x2, y2), axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distances to keypoints.
    """
    k = distance.shape[1] // 2
    out = np.zeros((points.shape[0], k, 2), dtype=np.float32)
    for i in range(k):
        out[:, i, 0] = points[:, 0] + distance[:, 2 * i]
        out[:, i, 1] = points[:, 1] + distance[:, 2 * i + 1]
    return out


def nms(dets: np.ndarray, thresh: float) -> list:
    """
    Classic NMS (same math/ordering).
    """
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    sc = dets[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (area[i] + area[order[1:]] - inter)
        idxs = np.where(ovr <= thresh)[0]
        order = order[idxs + 1]

    return keep


def parse_detector_outputs(
    detector_result: dict, conf_threshold: float = 0.5, nms_thresh: float = 0.4
) -> Optional[dict]:
    """
    Parse detector outputs using InsightFace anchor-free decoding.
    """
    outs = detector_result["outputs"]
    orig_wh = detector_result["original_size"]
    det_wh = detector_result["detector_size"]

    # (W,H) -> (H,W) for computing feature map sizes
    in_h, in_w = det_wh[1], det_wh[0]

    score_names = ["443", "468", "493"]
    bbox_names = ["446", "471", "496"]
    kps_names = ["449", "474", "499"]

    all_scores = []
    all_boxes = []
    all_kps = []

    center_cache = {}

    for level, stride in enumerate(FEAT_STRIDE_FPN):
        scr = outs[score_names[level]]
        bbp = outs[bbox_names[level]]
        kpp = outs[kps_names[level]] if (USE_KPS) else None

        fm_h = in_h // stride
        fm_w = in_w // stride

        cache_key = (fm_h, fm_w, stride)
        if cache_key in center_cache:
            centers = center_cache[cache_key]
        else:
            grid = np.stack(np.mgrid[:fm_h, :fm_w][::-1], axis=-1).astype(np.float32)
            grid = (grid + 0.5) * stride
            centers = grid.reshape((-1, 2))
            if NUM_ANCHORS > 1:
                centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
            center_cache[cache_key] = centers

        scr = scr.reshape(-1)
        bbp = bbp.reshape(-1, 4) * stride

        pos = np.where(scr >= conf_threshold)[0]
        if len(pos) == 0:
            continue

        ps = scr[pos]
        pc = centers[pos]
        pb = bbp[pos]

        boxes = distance2bbox(pc, pb)
        all_scores.append(ps)
        all_boxes.append(boxes)

        if USE_KPS and kpp is not None:
            kpp = kpp.reshape(-1, 10) * stride
            pk = kpp[pos]
            all_kps.append(distance2kps(pc, pk))

    if len(all_scores) == 0:
        # preserve recursive fallback behavior
        if conf_threshold > 0.3:
            return parse_detector_outputs(
                detector_result, conf_threshold=0.3, nms_thresh=nms_thresh
            )
        return None

    scores = np.concatenate(all_scores, axis=0)
    boxes = np.concatenate(all_boxes, axis=0)
    kpss = np.concatenate(all_kps, axis=0) if (USE_KPS and len(all_kps) > 0) else None

    pre = np.hstack((boxes, scores[:, None])).astype(np.float32)
    keep = nms(pre, nms_thresh)
    if len(keep) == 0:
        return None

    kept = pre[keep]
    best_local = int(np.argmax(kept[:, 4]))
    best = kept[best_local]

    best_box = best[:4]
    best_conf = float(best[4])

    sx = orig_wh[0] / det_wh[0]
    sy = orig_wh[1] / det_wh[1]

    scaled = best_box.copy()
    scaled[0] *= sx
    scaled[1] *= sy
    scaled[2] *= sx
    scaled[3] *= sy

    scaled[0] = max(0, scaled[0])
    scaled[1] = max(0, scaled[1])
    scaled[2] = min(orig_wh[0], scaled[2])
    scaled[3] = min(orig_wh[1], scaled[3])

    out = {"confidence": best_conf, "bbox": scaled, "original_size": orig_wh}

    if kpss is not None:
        # preserve original indexing logic: keep maps to pre indices, best_local is index within kept
        best_kps = kpss[keep[best_local]]
        kp = best_kps.copy()
        kp[:, 0] *= sx
        kp[:, 1] *= sy
        out["kps"] = kp

    return out


def run_inference(client: Any, image_bytes: bytes) -> np.ndarray:
    """
    Full pipeline: Detect -> (Align using landmarks or crop) -> FR embedding -> L2 normalize.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError(f"Import error: {exc}") from exc

    det_raw = run_detector_inference(client, image_bytes)
    det = parse_detector_outputs(det_raw)
    if det is None:
        raise ValueError("No face detected in the image")

    # support file-like objects exactly as before
    if hasattr(image_bytes, "read"):
        image_bytes.seek(0)
        blob = image_bytes.read()
    else:
        blob = image_bytes

    with Image.open(BytesIO(blob)) as im:
        im = im.convert("RGB")
        rgb = np.array(im, dtype=np.uint8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bbox = det["bbox"]
    if ("kps" in det) and (det["kps"] is not None):
        aligned = norm_crop(bgr, det["kps"], MODEL_IMAGE_SIZE[0])
    else:
        h, w = bgr.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        crop = bgr[y1:y2, x1:x2]
        aligned = cv2.resize(crop, MODEL_IMAGE_SIZE)

    arr = aligned.astype(np.float32)
    arr = (arr - 127.5) / 127.5
    arr = np.transpose(arr, (2, 0, 1))
    batch = np.expand_dims(arr, axis=0)

    tr_in = httpclient.InferInput(MODEL_INPUT_NAME, batch.shape, "FP32")
    tr_in.set_data_from_numpy(batch)
    tr_out = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)

    resp = client.infer(model_name=MODEL_NAME, inputs=[tr_in], outputs=[tr_out])
    emb = resp.as_numpy(MODEL_OUTPUT_NAME)

    # L2 normalize (same shape behavior)
    flat = emb.reshape(-1)
    n = np.linalg.norm(flat)
    if n > 0:
        flat = flat / n
    return flat.reshape(1, -1)
