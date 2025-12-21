"""
Download InsightFace ONNX models into the Triton model repository layout.

This script downloads InsightFace models (buffalo_l) and copies them to the Triton repository.
Models are downloaded to ~/.insightface/models/buffalo_l/ and then copied to the repository.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None


def download_insightface_models(model_name: str = "buffalo_l") -> Path:
    """
    Download InsightFace models using the InsightFace library.
    Returns the path to the downloaded models directory.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface is required. Install it with: pip install insightface onnxruntime"
        )

    print(f"[download] Downloading InsightFace models ({model_name})...")
    app = FaceAnalysis(name=model_name)
    app.prepare(ctx_id=-1)  # CPU mode

    # InsightFace models are stored in ~/.insightface/models/{model_name}/
    home = Path.home()
    model_dir = home / ".insightface" / "models" / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"InsightFace models not found at {model_dir}. "
            "Make sure InsightFace downloaded the models successfully."
        )

    print(f"[download] Models downloaded to: {model_dir}")
    return model_dir


def copy_model(src: Path, dest: Path, model_type: str) -> None:
    """Copy an ONNX model file to the Triton repository."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[copy] Exists, skipping: {dest}")
        return

    if not src.exists():
        raise FileNotFoundError(f"Source model not found: {src}")

    print(f"[copy] Copying {model_type}: {src} -> {dest}")
    shutil.copy2(src, dest)
    print(f"[copy] Saved: {dest}")


def check_model_io(model_path: Path) -> dict:
    """Check and return input/output names and shapes of an ONNX model."""
    if onnx is None:
        print("[check] onnx package not installed, skipping model inspection")
        return {"inputs": [], "outputs": []}

    try:
        model = onnx.load(str(model_path))
        inputs = [
            {"name": inp.name, "shape": [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]}
            for inp in model.graph.input
        ]
        outputs = [
            {"name": out.name, "shape": [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]}
            for out in model.graph.output
        ]
        return {"inputs": inputs, "outputs": outputs}
    except Exception as e:
        print(f"[check] Error inspecting model {model_path}: {e}")
        return {"inputs": [], "outputs": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download InsightFace ONNX models into a Triton repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models (FR + Detector)
  python download_models.py

  # Download only FR model
  python download_models.py --skip-detector

  # Use custom InsightFace model name
  python download_models.py --insightface-model buffalo_s

  # Check model I/O after copying
  python download_models.py --check-io
        """,
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path(__file__).parent / "model_repository",
        help="Root of the Triton model repository.",
    )
    parser.add_argument(
        "--insightface-model",
        type=str,
        default="buffalo_sc",
        help="InsightFace model name (e.g., buffalo_l, buffalo_sc). Default: buffalo_sc",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Only download the FR model, skip detector.",
    )
    parser.add_argument(
        "--check-io",
        action="store_true",
        help="Check and print input/output names and shapes of downloaded models.",
    )
    return parser.parse_args()


def find_model_file(model_dir: Path, patterns: list[str]) -> Path:
    """Find the first matching model file from a list of possible patterns."""
    for pattern in patterns:
        candidate = model_dir / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of the following models in {model_dir}: {patterns}"
    )


def main() -> int:
    args = parse_args()

    # Download InsightFace models
    insightface_dir = download_insightface_models(args.insightface_model)

    # Define possible FR model filenames (different model sets use different names)
    fr_patterns = ["w600k_r50.onnx", "w600k_mbf.onnx", "w600k_r100.onnx"]
    det_patterns = ["det_10g.onnx", "det_500m.onnx", "det_1g.onnx"]

    # Find actual model files
    fr_src = find_model_file(insightface_dir, fr_patterns)
    det_src = find_model_file(insightface_dir, det_patterns) if not args.skip_detector else None

    # Define destination paths
    fr_dest = args.model_repo / "fr_model" / "1" / "model.onnx"
    det_dest = args.model_repo / "face_detector" / "1" / "model.onnx"

    # Copy FR model
    print("\n[FR Model]")
    copy_model(fr_src, fr_dest, f"FR ({fr_src.name})")

    # Copy detector model
    if not args.skip_detector:
        print("\n[Detector Model]")
        copy_model(det_src, det_dest, f"Detector ({det_src.name})")

    # Check I/O if requested
    if args.check_io:
        print("\n" + "=" * 60)
        print("Model Input/Output Information")
        print("=" * 60)

        print("\n[FR Model]")
        fr_io = check_model_io(fr_dest)
        print("Inputs:")
        for inp in fr_io["inputs"]:
            print(f"  - Name: {inp['name']}, Shape: {inp['shape']}")
        print("Outputs:")
        for out in fr_io["outputs"]:
            print(f"  - Name: {out['name']}, Shape: {out['shape']}")

        if not args.skip_detector:
            print("\n[Detector Model]")
            det_io = check_model_io(det_dest)
            print("Inputs:")
            for inp in det_io["inputs"]:
                print(f"  - Name: {inp['name']}, Shape: {inp['shape']}")
            print("Outputs:")
            for out in det_io["outputs"]:
                print(f"  - Name: {out['name']}, Shape: {out['shape']}")

        print("\n" + "=" * 60)
        print("IMPORTANT: Update config.pbtxt files with the correct input/output names above!")
        print("=" * 60)

    print("\n[download] Done! Models are ready in the Triton repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
