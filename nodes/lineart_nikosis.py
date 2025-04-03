import torch
import numpy as np
from comfy.utils import ProgressBar

from ..utils.processors.lineart_standard import LineArtStandardDetector, LineArtStandardCannyDetector, LineArtStandardSobelDetector, LineArtSobelDetector, LineArtCannyDetector


class LineArtStandardPreprocessorNikosis:
    def __init__(self):
        self.detector = {
            "standard": LineArtStandardDetector(), "sobel": LineArtStandardSobelDetector(), "canny": LineArtStandardCannyDetector(),
            "sobel-other": LineArtSobelDetector(), "canny-other": LineArtCannyDetector()
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["standard", "sobel", "canny", "sobel-other", "canny-other"], {"default": "standard"}),
                "sigma": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 100.0}),
                "intensity": ("INT", {"default": 8, "min": 0, "max": 16}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "commence"
    CATEGORY = "Nikosis Nodes/Preprocessors"

    def commence(self, image, model, sigma, intensity, resolution, **kwargs):
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
                guassian_sigma=sigma,
                intensity_threshold=intensity,
                detect_resolution=detect_resolution
            )

            # Convert back to tensor format
            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)

            if out_tensor is None:
                out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
            out_tensor[i] = out

            pbar.update(1)

        return (out_tensor,)
