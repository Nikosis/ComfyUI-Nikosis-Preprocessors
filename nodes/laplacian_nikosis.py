import torch
import numpy as np
from comfy.utils import ProgressBar
from ..processors.edge.edge import LaplacianDetector

class LaplacianPreprocessorNikosis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                "ksize": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),  # Odd numbers only
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "keep_proportion": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "commence"
    CATEGORY = "Nikosis Nodes/preprocessors"

    def commence(self, image, low_threshold, high_threshold, ksize, resolution, keep_proportion):
        detector = LaplacianDetector()
        detect_resolution = resolution if isinstance(resolution, int) and resolution >= 64 else 512
        batch_size = image.shape[0]
        pbar = ProgressBar(batch_size)
        out_tensor = None

        for i, img in enumerate(image):
            np_image = np.asarray(img.cpu() * 255., dtype=np.uint8)
            np_result = detector(
                input_image=np_image,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                detect_resolution=detect_resolution,
                ksize=ksize,
                upscale_method="INTER_LANCZOS4",
                keep_proportion=keep_proportion
            )
            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
            if out_tensor is None:
                out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
            out_tensor[i] = out
            pbar.update(1)

        return (out_tensor,)

NODE_CLASS_MAPPINGS = {"LaplacianPreprocessorNikosis": LaplacianPreprocessorNikosis}
NODE_DISPLAY_NAME_MAPPINGS = {"LaplacianPreprocessorNikosis": "Laplacian Preprocessor (nikosis)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'LaplacianPreprocessorNikosis']
