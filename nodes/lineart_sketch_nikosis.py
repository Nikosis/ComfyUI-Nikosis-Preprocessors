import gc
import torch
import numpy as np
import comfy.model_management as model_management
from ..processors.sketch.lineart_sketch import LineArtSketchDetector
from ..utils.model_paths import get_model_path

class LineArtSketchPreprocessorNikosis:
    @classmethod
    def INPUT_TYPES(cls):
        models_list = ["sk_model_fine.safetensors", "sk_model_coarse.safetensors"]
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (models_list, {"default": "sk_model_fine.safetensors"}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "keep_proportion": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "commence"
    CATEGORY = "Nikosis Nodes/preprocessors"

    def __init__(self):
        self.detector = None
        self.current_model = None
        self.device = model_management.get_torch_device()
        self.offload_device = "cpu"

    def _load_detector(self, model_name):
        """Loads or reuses the LineArtSketchDetector, ensuring the model is on GPU."""
        if self.detector is None or self.current_model != model_name:
            path = get_model_path("lineart", model_name)
            self.detector = LineArtSketchDetector(path, self.device)
            self.current_model = model_name
        else:
            # Move model back to GPU if it was offloaded
            self.detector.model.to(self.device)
            self.detector.device = self.device

    def _cleanup(self):
        """Moves the model to CPU and frees VRAM."""
        if self.detector and self.detector.model:
            self.detector.model.to(self.offload_device)
            self.detector.device = self.offload_device  # Update device state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def commence(self, images, model, resolution=1024, keep_proportion=True):
        # Load detector with model on GPU
        self._load_detector(model)

        # Process input
        image_np = (images.cpu().numpy() * 255).astype(np.uint8)
        result_np = self.detector.detect(image_np, resolution=resolution, keep_proportion=keep_proportion)

        # Offload model from VRAM
        self._cleanup()

        return (torch.from_numpy(result_np).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"LineArtSketchPreprocessorNikosis": LineArtSketchPreprocessorNikosis}
NODE_DISPLAY_NAME_MAPPINGS = {"LineArtSketchPreprocessorNikosis": "LineArt Sketch Preprocessor (nikosis)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'LineArtSketchPreprocessorNikosis']
