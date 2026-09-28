"""Bit-exact raw-image inference for the archived Audited514 run on Modal.

Mirrors pipeline_snapshot.py (image, GPU, model construction, bf16 autocast,
batch 32, float16 rounding) and pins the CPU kernel dispatch to AVX512, which
controls the uint8 antialiased resize inside the fast SigLIP 2 processor.

Expected volume layout (default volume name: bdc2026-verify):
  /run/naflex_audited.pt, features.npz, balanced_lr.joblib, probabilities.npz,
       submission.csv, artifact_inventory.json
  /BDC2026/test/1.jpg ... 1458.jpg
"""
import json
import os
from pathlib import Path

import modal

SEED = 2026
NAFLEX_ID = "google/siglip2-so400m-patch16-naflex"
NAFLEX_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"
RUN = Path("/cache/run")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "transformers==4.56.2",
        "huggingface-hub==0.34.4", "safetensors==0.6.2",
        "scikit-learn==1.7.2", "pandas==2.3.2", "numpy==2.2.6", "pillow==11.3.0",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": str(SEED), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "ATEN_CPU_CAPABILITY": "avx512"})
)
cache_volume = modal.Volume.from_name(os.environ.get("VERIFY_VOLUME", "bdc2026-verify"))
app = modal.App("satria-data-bdc-siglip2-naflex-exact-inference")


@app.function(
    image=image, gpu="A100-40GB", cpu=12, memory=49152, timeout=60 * 60,
    volumes={"/cache": cache_volume},
)
def verify_and_infer():
    import hashlib
    import platform
    import random

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, Siglip2VisionModel

    if torch.backends.cpu.get_cpu_capability() != "AVX512":
        raise RuntimeError("Bit-exact features need an AVX512 host; this worker "
                           f"reports {torch.backends.cpu.get_cpu_capability()}. Re-run.")

    # 1. Checksums of every file in the original run directory.
    inventory = json.loads((RUN / "artifact_inventory.json").read_text())
    checks = {}
    for item in (i for i in inventory if (RUN / i["filename"]).exists()):
        digest = hashlib.sha256()
        with (RUN / item["filename"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        checks[item["filename"]] = digest.hexdigest() == item["sha256"]

    # 2. Same seeding as seed_everything() in the snapshot.
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    processor = AutoImageProcessor.from_pretrained(NAFLEX_ID, revision=NAFLEX_REVISION, use_fast=True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Siglip2VisionModel.from_pretrained(
                NAFLEX_ID, revision=NAFLEX_REVISION, attn_implementation="sdpa")
            dimension = self.backbone.config.hidden_size
            self.classifier = nn.Linear(dimension, 3)
            self.binary = nn.Linear(dimension, 1)

    model = Model()
    model.load_state_dict(torch.load(RUN / "naflex_audited.pt", map_location="cpu", weights_only=True), strict=True)
    model.to("cuda:0").eval()

    test_dir = Path("/cache/BDC2026/test")
    paths = sorted((p for p in test_dir.iterdir() if p.is_file()), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in paths] == list(range(1, 1459))

    class Images(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            with Image.open(paths[index]) as source:
                return source.convert("RGB")

    def collate(images):
        return processor(images=list(images), return_tensors="pt", max_num_patches=256)

    loader = DataLoader(Images(), batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate)
    features = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to("cuda:0", non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.backbone(pixel_values=batch["pixel_values"],
                                     pixel_attention_mask=batch["pixel_attention_mask"],
                                     spatial_shapes=batch["spatial_shapes"])
            features.append(out.pooler_output.float().cpu().numpy())
    feats = np.concatenate(features).astype(np.float16)

    archived = np.load(RUN / "features.npz")["inference"]
    head = joblib.load(RUN / "balanced_lr.joblib")
    proba = head.predict_proba(feats.astype(np.float32))
    pred = head.classes_[proba.argmax(1)]
    archived_proba = np.load(RUN / "probabilities.npz")["inference"]
    sub = pd.read_csv(RUN / "submission.csv")
    csv_text = pd.DataFrame({"id": np.arange(1, 1459), "predicted": pred.astype(int)}).to_csv(index=False)
    return {
        "checksums_ok": checks,
        "cpu": {"capability": torch.backends.cpu.get_cpu_capability(), "model": next((l.split(":",1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"), "avx512": "avx512f" in open("/proc/cpuinfo").read()},
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()},
        "features_bit_identical_rows": int((feats == archived).all(1).sum()),
        "features_max_abs_diff": float(np.abs(feats.astype(np.float32) - archived.astype(np.float32)).max()),
        "proba_max_abs_diff": float(np.abs(proba - archived_proba).max()),
        "label_diff_vs_volume_submission": int((pred != sub.predicted.to_numpy()).sum()),
        "csv": csv_text,
    }


@app.local_entrypoint()
def main(out: str = "results/submission-exact-rerun.csv"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result = verify_and_infer.remote()
    Path(out).write_text(result.pop("csv"), encoding="utf-8")
    print(json.dumps(result, indent=2))
