import torch
import numpy as np
import comfy.model_management as model_management
from ..utils.preprocessors.test_lineart import LineartDetector


class LineArtPreprocessorNikosis2:
    @classmethod
    def INPUT_TYPES(cls):
        models_list = ["sk_model_fine.safetensors", "sk_model_coarse.safetensors"]
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (models_list, {"default": "sk_model_fine.safetensors"}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "commence"
    CATEGORY = "Nikosis Nodes/Preprocessors/LineArt"

    def _tensor_to_numpy(self, image_tensor):
        """Convert [B, H, W, C] tensor in [0, 1] to [B, H, W, C] NumPy array in [0, 255] uint8."""
        # Convert float [0, 1] to uint8 [0, 255]
        image_np = (image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return image_np

    def _process_batch(self, image_np, model_name, resolution):
        """Process [B, H, W, C] NumPy array with LineartDetector."""
        detector = LineartDetector.from_pretrained().to(model_management.get_torch_device())
        result = detector(image_np, model_name=model_name, detect_resolution=resolution)
        del detector
        return result  # [B, H', W', 3] NumPy

    def _numpy_to_tensor(self, result_np):
        """Convert [B, H', W', C] NumPy to [B, H', W', C] tensor."""
        return torch.from_numpy(result_np).float() / 255.0  # [B, H', W', C], [0, 1]

    def commence(self, images, model, resolution=512):
        """Process [B, H, W, C] tensor input."""
        image_np = self._tensor_to_numpy(images)  # [B, H, W, C] NumPy
        result_np = self._process_batch(image_np, model, resolution)  # [B, H', W', 3] NumPy
        out_tensor = self._numpy_to_tensor(result_np)  # [B, H', W', 3] tensor
        return (out_tensor,)

NODE_CLASS_MAPPINGS = {"LineArtPreprocessorNikosis2": LineArtPreprocessorNikosis2}
NODE_DISPLAY_NAME_MAPPINGS = {"LineArtPreprocessorNikosis2": "🖌️ LineArt Preprocessor TEST (nikosis)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'LineArtPreprocessorNikosis2']
