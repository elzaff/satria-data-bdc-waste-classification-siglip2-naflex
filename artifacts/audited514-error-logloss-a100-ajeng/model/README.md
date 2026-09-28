# Inference bundle

This folder contains the compact files needed by `inference.py`:

- `config.json` and `preprocessor_config.json`: pinned SigLIP 2 NaFlex architecture and image preprocessing configuration.
- `balanced_lr.joblib`: the fitted three-class Logistic Regression head.
- `features.npz`: cached train and inference embeddings from the archived run.
- `probabilities.npz`: archived validation and inference probabilities.
- `manifest_audited_final.csv` and `manifest_changes.csv`: the exact training labels and the change list used by the run.
- `artifact_inventory.json` and `artifact_checksums.sha256`: original run sizes and SHA-256 digests.

`features.npz` contains the archived inference embeddings. Running `python inference.py` applies the saved head to those embeddings and reproduces the archived submission exactly, without a GPU.

The encoder checkpoints are not included in this bundle. Copies downloaded from Modal did not match the SHA-256 values recorded in `artifact_inventory.json`, so they are not presented as verified run weights. Raw-image inference requires a checkpoint that passes that checksum check.

The base model is `google/siglip2-so400m-patch16-naflex` at revision `cc24074f717b612951c2dead130904ab9b65a81e`. The upstream model is listed under Apache-2.0 on its [Hugging Face model page](https://huggingface.co/google/siglip2-so400m-patch16-naflex).
