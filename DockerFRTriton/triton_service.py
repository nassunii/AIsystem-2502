import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
import cv2


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

# InsightFace RetinaFace/SCRFD configuration for buffalo_sc style models
FMC = 3
FEAT_STRIDE_FPN = [8, 16, 32]
NUM_ANCHORS = 2
USE_KPS = True

# ArcFace standard alignment reference points for 112x112
# These are the target positions for: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041]    # right mouth corner
], dtype=np.float32)


def estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Estimate the similarity transformation matrix for face alignment.
    This is the same method used by InsightFace.
    
    Args:
        lmk: 5 facial landmarks, shape (5, 2)
        image_size: output image size (default 112 for ArcFace)
    
    Returns:
        M: 2x3 transformation matrix
    """
    assert lmk.shape == (5, 2)
    
    # Scale reference points if image size is different from 112
    dst = ARCFACE_DST.copy()
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    
    # Use skimage's SimilarityTransform if available, otherwise use cv2
    try:
        from skimage import transform as trans
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
    except ImportError:
        # Fallback: use cv2.estimateAffinePartial2D (similarity transform)
        M, _ = cv2.estimateAffinePartial2D(lmk, dst)
        if M is None:
            # If estimation fails, use simple affine
            M = cv2.getAffineTransform(lmk[:3].astype(np.float32), dst[:3].astype(np.float32))
    
    return M


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Align and crop face using 5 landmarks.
    This is the standard InsightFace alignment method.
    
    Args:
        img: input image (BGR)
        landmark: 5 facial landmarks, shape (5, 2)
        image_size: output size (default 112)
    
    Returns:
        aligned: aligned face image
    """
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the FR and Detector ONNX models and config.pbtxt.
    """
    # FR model config
    fr_model_dir = model_repo / MODEL_NAME / MODEL_VERSION
    fr_model_path = fr_model_dir / "model.onnx"
    fr_config_path = fr_model_dir.parent / "config.pbtxt"

    if not fr_model_path.exists():
        raise FileNotFoundError(
            f"Missing FR ONNX model at {fr_model_path}. "
            "Please place your exported model there."
        )

    fr_model_dir.mkdir(parents=True, exist_ok=True)
    
    if not fr_config_path.exists():
        fr_config_text = textwrap.dedent(
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
        ).strip() + "\n"

        fr_config_path.write_text(fr_config_text)
        print(f"[triton] Created FR model config at {fr_config_path}")
    else:
        print(f"[triton] Using existing FR model config at {fr_config_path}")
    
    print(f"[triton] Prepared FR model repository at {fr_model_dir.parent}")

    # Detector model config
    det_model_dir = model_repo / DETECTOR_NAME / MODEL_VERSION
    det_model_path = det_model_dir / "model.onnx"
    det_config_path = det_model_dir.parent / "config.pbtxt"

    if not det_model_path.exists():
        raise FileNotFoundError(
            f"Missing Detector ONNX model at {det_model_path}. "
            "Please place your exported model there."
        )

    det_model_dir.mkdir(parents=True, exist_ok=True)
    
    if not det_config_path.exists():
        det_output_configs = []
        det_output_configs.append('          { name: "443", data_type: TYPE_FP32, dims: [12800, 1] },')
        det_output_configs.append('          { name: "468", data_type: TYPE_FP32, dims: [3200, 1] },')
        det_output_configs.append('          { name: "493", data_type: TYPE_FP32, dims: [800, 1] },')
        det_output_configs.append('          { name: "446", data_type: TYPE_FP32, dims: [12800, 4] },')
        det_output_configs.append('          { name: "471", data_type: TYPE_FP32, dims: [3200, 4] },')
        det_output_configs.append('          { name: "496", data_type: TYPE_FP32, dims: [800, 4] },')
        det_output_configs.append('          { name: "449", data_type: TYPE_FP32, dims: [12800, 10] },')
        det_output_configs.append('          { name: "474", data_type: TYPE_FP32, dims: [3200, 10] },')
        det_output_configs.append('          { name: "499", data_type: TYPE_FP32, dims: [800, 10] }')
        
        det_config_text = textwrap.dedent(
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
{chr(10).join(det_output_configs)}
            ]
            instance_group [
              {{ kind: KIND_CPU }}
            ]
            """
        ).strip() + "\n"

        det_config_path.write_text(det_config_text)
        print(f"[triton] Created Detector model config at {det_config_path}")
    else:
        print(f"[triton] Using existing Detector model config at {det_config_path}")
    
    print(f"[triton] Prepared Detector model repository at {det_model_dir.parent}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
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
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def run_detector_inference(client: Any, image_bytes: bytes) -> dict:
    """
    Run face detector inference on an image.
    Returns detector outputs (confidence, bbox, landmarks).
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("Pillow and tritonclient[http] are required.") from exc

    # Load and preprocess image for detector
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        original_size = img.size  # (width, height)
        img_resized = img.resize(DETECTOR_IMAGE_SIZE)
        np_img = np.asarray(img_resized, dtype=np.float32)
    
    # InsightFace detector normalization: (img - 127.5) / 128.0
    np_img = (np_img - 127.5) / 128.0
    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(np_img, axis=0)

    # Prepare inputs
    infer_input = httpclient.InferInput(DETECTOR_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    # Prepare outputs
    infer_outputs = [httpclient.InferRequestedOutput(name) for name in DETECTOR_OUTPUT_NAMES]
    
    # Run inference
    response = client.infer(
        model_name=DETECTOR_NAME,
        inputs=[infer_input],
        outputs=infer_outputs
    )
    
    # Extract outputs
    outputs = {name: response.as_numpy(name) for name in DETECTOR_OUTPUT_NAMES}
    
    return {
        "outputs": outputs,
        "original_size": original_size,
        "detector_size": DETECTOR_IMAGE_SIZE
    }


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance predictions to bounding boxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance predictions to keypoints.
    """
    num_points = distance.shape[1] // 2
    kps = np.zeros((points.shape[0], num_points, 2), dtype=np.float32)
    for i in range(num_points):
        kps[:, i, 0] = points[:, 0] + distance[:, i * 2]
        kps[:, i, 1] = points[:, 1] + distance[:, i * 2 + 1]
    return kps


def nms(dets: np.ndarray, thresh: float) -> list:
    """
    Non-maximum suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def parse_detector_outputs(detector_result: dict, conf_threshold: float = 0.5, nms_thresh: float = 0.4) -> Optional[dict]:
    """
    Parse detector outputs using InsightFace's anchor-free decoding method.
    """
    outputs = detector_result["outputs"]
    original_size = detector_result["original_size"]
    detector_size = detector_result["detector_size"]
    
    input_height, input_width = detector_size[1], detector_size[0]
    
    score_names = ["443", "468", "493"]
    bbox_names = ["446", "471", "496"]
    kps_names = ["449", "474", "499"]
    
    scores_list = []
    bboxes_list = []
    kpss_list = []
    
    center_cache = {}
    
    for idx, stride in enumerate(FEAT_STRIDE_FPN):
        scores = outputs[score_names[idx]]
        bbox_preds = outputs[bbox_names[idx]]
        kps_preds = outputs[kps_names[idx]] if USE_KPS else None
        
        height = input_height // stride
        width = input_width // stride
        
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers + 0.5) * stride
            anchor_centers = anchor_centers.reshape((-1, 2))
            if NUM_ANCHORS > 1:
                anchor_centers = np.stack([anchor_centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
            center_cache[key] = anchor_centers
        
        scores = scores.reshape(-1)
        bbox_preds = bbox_preds.reshape(-1, 4) * stride
        
        pos_inds = np.where(scores >= conf_threshold)[0]
        
        if len(pos_inds) == 0:
            continue
        
        pos_scores = scores[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_anchor_centers = anchor_centers[pos_inds]
        
        bboxes = distance2bbox(pos_anchor_centers, pos_bbox_preds)
        
        scores_list.append(pos_scores)
        bboxes_list.append(bboxes)
        
        if USE_KPS and kps_preds is not None:
            kps_preds = kps_preds.reshape(-1, 10) * stride
            pos_kps_preds = kps_preds[pos_inds]
            kpss = distance2kps(pos_anchor_centers, pos_kps_preds)
            kpss_list.append(kpss)
    
    if len(scores_list) == 0:
        return parse_detector_outputs(detector_result, conf_threshold=0.3, nms_thresh=nms_thresh) if conf_threshold > 0.3 else None
    
    scores = np.concatenate(scores_list, axis=0)
    bboxes = np.concatenate(bboxes_list, axis=0)
    
    if USE_KPS and len(kpss_list) > 0:
        kpss = np.concatenate(kpss_list, axis=0)
    else:
        kpss = None
    
    pre_det = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)
    keep = nms(pre_det, nms_thresh)
    
    if len(keep) == 0:
        return None
    
    det = pre_det[keep]
    best_idx = np.argmax(det[:, 4])
    best_det = det[best_idx]
    
    best_bbox = best_det[:4]
    best_score = float(best_det[4])
    
    scale_x = original_size[0] / detector_size[0]
    scale_y = original_size[1] / detector_size[1]
    
    scaled_bbox = best_bbox.copy()
    scaled_bbox[0] *= scale_x
    scaled_bbox[1] *= scale_y
    scaled_bbox[2] *= scale_x
    scaled_bbox[3] *= scale_y
    
    scaled_bbox[0] = max(0, scaled_bbox[0])
    scaled_bbox[1] = max(0, scaled_bbox[1])
    scaled_bbox[2] = min(original_size[0], scaled_bbox[2])
    scaled_bbox[3] = min(original_size[1], scaled_bbox[3])
    
    result = {
        "confidence": best_score,
        "bbox": scaled_bbox,
        "original_size": original_size
    }
    
    if kpss is not None:
        best_kps = kpss[keep[best_idx]]
        scaled_kps = best_kps.copy()
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y
        result["kps"] = scaled_kps
    
    return result


def run_inference(client: Any, image_bytes: bytes) -> np.ndarray:
    """
    Full pipeline: Detect face → Align using landmarks → Extract embedding.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError(f"Import error: {exc}") from exc

    # Step 1: Run detector
    detector_result = run_detector_inference(client, image_bytes)
    
    # Step 2: Parse detector outputs
    detection = parse_detector_outputs(detector_result)
    
    if detection is None:
        raise ValueError("No face detected in the image")
    
    # Step 3: Load original image
    if hasattr(image_bytes, 'read'):
        image_bytes.seek(0)
        img_data = image_bytes.read()
    else:
        img_data = image_bytes
    
    with Image.open(BytesIO(img_data)) as img:
        img = img.convert("RGB")
        original_img = np.array(img, dtype=np.uint8)
    
    # Convert PIL RGB to OpenCV BGR
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # Step 4: Face alignment using landmarks
    bbox = detection["bbox"]
    
    if "kps" in detection and detection["kps"] is not None:
        landmarks = detection["kps"]
        aligned_face = norm_crop(original_img_bgr, landmarks, MODEL_IMAGE_SIZE[0])
    else:
        # Fallback: simple crop + resize
        img_h, img_w = original_img_bgr.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(img_w, int(bbox[2])), min(img_h, int(bbox[3]))
        face_crop = original_img_bgr[y1:y2, x1:x2]
        aligned_face = cv2.resize(face_crop, MODEL_IMAGE_SIZE)
    
    # Step 5: Preprocess for FR model
    np_img = aligned_face.astype(np.float32)
    np_img = (np_img - 127.5) / 127.5
    np_img = np.transpose(np_img, (2, 0, 1))
    batch = np.expand_dims(np_img, axis=0)

    # Step 6: Run FR model
    infer_input = httpclient.InferInput(MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)
    
    response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    embedding = response.as_numpy(MODEL_OUTPUT_NAME)
    
    # Normalize
    embedding = embedding.reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    embedding = embedding.reshape(1, -1)
    
    return embedding