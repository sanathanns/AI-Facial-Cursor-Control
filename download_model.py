"""Utility to ensure the MediaPipe Face Landmarker task model bundle is present."""
import os
import sys
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")


def ensure_model(model_path: str = MODEL_PATH) -> str:
    """Check if face_landmarker.task exists; if not, download it."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
        print(f"[Model] Face Landmarker model already exists: {model_path} ({os.path.getsize(model_path):,} bytes)")
        return model_path

    print(f"[Model] Downloading face_landmarker.task from {MODEL_URL}...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(count * block_size * 100 / total_size))
                sys.stdout.write(f"\rDownloading: {percent}% ({count * block_size}/{total_size} bytes)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=reporthook)
        print(f"\n[Model] Download complete: {model_path} ({os.path.getsize(model_path):,} bytes)")
    except Exception as e:
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download Face Landmarker model: {e}") from e

    return model_path


if __name__ == "__main__":
    path = ensure_model()
    print("Resolved model path:", path)
