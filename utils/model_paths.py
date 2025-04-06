import folder_paths
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import snapshot_download
from ..logger import niko_logger as logger


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lineart": {
        "repo_id": "Nikos7766/lineart-models",
        "subfolder": "controlnet/preprocessors/lineart"
    },
    "depthanythingv2": {
        "repo_id": "Nikos7766/DepthAnythingV2",
        "subfolder": "controlnet/preprocessors/depthanythingv2"
    },
}


def get_model_path(model_key: str, model_name: str) -> str:
    """Finds or downloads the model using pathlib for better path handling."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model key '{model_key}' not found in registry.")

    config = MODEL_REGISTRY[model_key]
    subfolder = Path(config["subfolder"])
    download_dir = Path(folder_paths.models_dir) / subfolder
    local_model_path = download_dir / model_name

    # Check in ComfyUI - configured paths
    registered_path = folder_paths.get_full_path(model_key, model_name)
    if registered_path and Path(registered_path).exists():
        logger.info(f"Found - {model_name} in registered path: {registered_path}")
        return str(registered_path)

    # Check existing paths
    if local_model_path.exists():
        logger.info(f"Found {model_name} at {local_model_path}")
        return str(local_model_path)

    # Download if missing
    logger.info(f"Model Not Found - Downloading {model_name}...")
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_file = Path(snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(download_dir),  # Convert Path to str for API compatibility
            allow_patterns=[model_name],
        )).resolve()

        model_path = downloaded_file / model_name
        logger.info(f"Downloaded {model_name} to {downloaded_file}")
        return str(model_path)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
