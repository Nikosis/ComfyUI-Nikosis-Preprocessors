import torch
import numpy as np

from comfy.utils import ProgressBar
from ..processors.edge.edge import CannyDetector, LaplacianDetector, PrewittDetector, PyraCannyDetector, SobelDetector


class EdgePreprocessorNikosis:
    def __init__(self):
        self.detector = {
            "canny": CannyDetector(), "laplacian": LaplacianDetector(), "prewitt": PrewittDetector(),
            "pyracanny": PyraCannyDetector(), "sobel": SobelDetector(),
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["canny", "laplacian", "prewitt", "pyracanny", "sobel"], {"default": "pyracanny"}),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "keep_proportion": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "commence"
    CATEGORY = "Nikosis Nodes/preprocessors"

    def commence(self, image, model, low_threshold, high_threshold, resolution, keep_proportion, upscale_method="INTER_LANCZOS4"):
        detector = self.detector[model]

        # Handle resolution
        detect_resolution = resolution if isinstance(resolution, int) and resolution >= 64 else 512

        # Convert tensor to numpy and process batch
        batch_size = image.shape[0]
        pbar = ProgressBar(batch_size)
        out_tensor = None

        for i, img in enumerate(image):
            # Convert from tensor (0-1) to numpy (0-255)
            np_image = np.asarray(img.cpu() * 255., dtype=np.uint8)

            # Process single image
            np_result = detector(
                input_image=np_image,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                detect_resolution=detect_resolution,
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
            )

            # Convert back to tensor format
            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)

            if out_tensor is None:
                out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
            out_tensor[i] = out

            pbar.update(1)

        return (out_tensor,)

NODE_CLASS_MAPPINGS = {"EdgePreprocessorNikosis": EdgePreprocessorNikosis}
NODE_DISPLAY_NAME_MAPPINGS = {"EdgePreprocessorNikosis": "Edge Preprocessor (nikosis)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'EdgePreprocessorNikosis']
