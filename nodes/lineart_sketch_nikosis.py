import torch
import numpy as np
import comfy.model_management as model_management
from ..processors.lineart.lineart_sketch import LineArtSketchDetector


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

    def _tensor_to_numpy(self, image_tensor):
        """Convert [B, H, W, C] tensor in [0, 1] to [B, H, W, C] NumPy array in [0, 255] uint8."""
        return (image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    def _process_batch(self, image_np, model_name, resolution, keep_proportion):
        """Process [B, H, W, C] NumPy array with LineArtSketchDetector."""
        # Use the device from model_management directly in from_pretrained
        detector = LineArtSketchDetector.from_pretrained(device=model_management.get_torch_device())
        result = detector.detect(image_np, model_name=model_name, resolution=resolution, keep_proportion=keep_proportion)
        del detector
        return result  # [B, H', W', 3] NumPy

    def _numpy_to_tensor(self, result_np):
        """Convert [B, H', W', C] NumPy to [B, H', W', C] tensor in [0, 1]."""
        return torch.from_numpy(result_np).float() / 255.0

    def commence(self, images, model, resolution=1024, keep_proportion=True,):
        """Process a batch of images and return a tensor."""
        image_np = self._tensor_to_numpy(images)  # [B, H, W, C] NumPy
        result_np = self._process_batch(image_np, model, resolution, keep_proportion)  # [B, H', W', 3] NumPy
        out_tensor = self._numpy_to_tensor(result_np)  # [B, H', W', 3] tensor
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {"LineArtSketchPreprocessorNikosis": LineArtSketchPreprocessorNikosis}
NODE_DISPLAY_NAME_MAPPINGS = {"LineArtSketchPreprocessorNikosis": "LineArt Sketch Preprocessor (nikosis)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'LineArtSketchPreprocessorNikosis']
