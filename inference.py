"""Inference for the archived SigLIP 2 NaFlex Audited514 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts/audited514-error-logloss-a100-ajeng/model"
HEAD_PATH = MODEL_DIR / "balanced_lr.joblib"
INVENTORY_PATH = MODEL_DIR / "artifact_inventory.json"
DEFAULT_TEST_DIR = ROOT / "BDC2026/test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Use a checksum-verified encoder checkpoint for raw-image inference.",
    )
    parser.add_argument("--head", type=Path, default=HEAD_PATH)
    parser.add_argument(
        "--features-npz", type=Path, default=None,
        help="Use archived embeddings (default). Cannot be combined with --checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, inventory_path: Path) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected = next((item for item in inventory if item["filename"] == path.name), None)
    if expected is None:
        raise ValueError(f"{path.name} is not listed in {inventory_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}; use the archived run artifact.")


def load_head(head_path: Path, inventory_path: Path):
    verify_artifact(head_path, inventory_path)
    head = joblib.load(head_path)
    if not np.array_equal(np.asarray(head.classes_), np.array([0, 1, 2])):
        raise ValueError(f"Unexpected classifier class order: {head.classes_}")
    return head


def numbered_images(test_dir: Path) -> tuple[list[Path], np.ndarray]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [path for path in test_dir.iterdir()
             if path.is_file() and path.suffix.lower() in extensions]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {test_dir}")
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError("Test image filenames must have numeric stems, e.g. 1.jpg") from exc
    ids = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError("Test images must be numbered consecutively from 1")
    return paths, ids


def features_from_archive(features_path: Path, inventory_path: Path) -> tuple[np.ndarray, np.ndarray]:
    verify_artifact(features_path, inventory_path)
    with np.load(features_path, allow_pickle=False) as archive:
        if "inference" not in archive:
            raise ValueError(f"No 'inference' array in {features_path}")
        features = archive["inference"].astype(np.float16).astype(np.float32)
    return features, np.arange(1, len(features) + 1, dtype=np.int64)


def raw_image_features(
    test_dir: Path,
    model_dir: Path,
    checkpoint_path: Path,
    inventory_path: Path,
    batch_size: int,
    device_name: str,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    from torch import nn
    from transformers import AutoImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

    paths, ids = numbered_images(test_dir)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vision_values = config_data.get("vision_config")
    if not isinstance(vision_values, dict):
        raise ValueError("config.json does not contain vision_config")
    vision_config = Siglip2VisionConfig(
        **{key: value for key, value in vision_values.items() if key != "model_type"}
    )
    vision_config._attn_implementation = "sdpa"

    class ArchivedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Siglip2VisionModel(vision_config)
            self.classifier = nn.Linear(vision_config.hidden_size, 3)
            self.binary = nn.Linear(vision_config.hidden_size, 1)

    verify_artifact(checkpoint_path, inventory_path)
    model = ArchivedModel()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), use_fast=True, local_files_only=True
    )
    use_bf16 = (
        precision == "bf16" and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt", max_num_patches=256)
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 else nullcontext())
        with torch.inference_mode(), autocast:
            output = model.backbone(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )
        batch_features = output.pooler_output.float().cpu().numpy()
        # The archived Logistic Regression was fitted on float16-rounded features.
        features.append(batch_features.astype(np.float16).astype(np.float32))
        print(f"Encoded {min(start + len(batch_paths), len(paths))}/{len(paths)} images", flush=True)

    return np.concatenate(features), ids


def write_submission(output_path: Path, ids: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> None:
    if probabilities.shape != (len(ids), len(classes)):
        raise ValueError("Prediction dimensions do not match the image IDs")
    predictions = classes[probabilities.argmax(axis=1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "predicted"))
        writer.writerows(zip(ids.tolist(), predictions.astype(int).tolist()))


def main() -> None:
    args = parse_args()
    inventory_path = args.model_dir / "artifact_inventory.json"
    head = load_head(args.head, inventory_path)
    if args.checkpoint is not None:
        if args.features_npz is not None:
            raise ValueError("Choose either --features-npz or --checkpoint, not both")
        features, ids = raw_image_features(
            args.test_dir, args.model_dir, args.checkpoint, inventory_path,
            args.batch_size, args.device, args.precision,
        )
    else:
        features_path = args.features_npz or (args.model_dir / "features.npz")
        features, ids = features_from_archive(features_path, inventory_path)
    if features.shape[1] != head.n_features_in_:
        raise ValueError(
            f"Feature width {features.shape[1]} does not match head width {head.n_features_in_}"
        )
    probabilities = head.predict_proba(features)
    write_submission(args.output, ids, probabilities, np.asarray(head.classes_))
    print(f"Wrote {len(ids)} predictions to {args.output}")


if __name__ == "__main__":
    main()
