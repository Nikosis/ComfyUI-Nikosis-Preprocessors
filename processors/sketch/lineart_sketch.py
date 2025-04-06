import torch
import numpy as np
from einops import rearrange
from ...utils.image_utils import img_to_hwc3, resize_and_crop_to_multiple_cv2
from ..sketch.lineart_sketch_core import load_lineart_model


class LineArtSketchDetector:
    """Pure image processing pipeline with model management."""

    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = load_lineart_model(model_path, device)

    def preprocess(self, images, resolution=512, upscale_method="INTER_CUBIC", keep_proportion=True):
        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise ValueError("Input must be a [B, H, W, C] NumPy array")
        processed = []
        for img in images:
            img = img_to_hwc3(img)
            img = resize_and_crop_to_multiple_cv2(
                input_image=img,
                target_res=resolution,
                multiple=16,
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
                crop_from="symmetric",
            )
            processed.append(img)
        return np.stack(processed)

    def to_tensor(self, images_np):
        tensor = torch.from_numpy(images_np).float().to(self.device) / 255.0
        return rearrange(tensor, 'b h w c -> b c h w')

    def run_model(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)
            line = output.squeeze(1)
            return (line.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    def postprocess(self, line_np):
        batch_size = line_np.shape[0]
        color_images = [img_to_hwc3(line_np[i]) for i in range(batch_size)]
        return 255 - np.stack(color_images)

    def detect(self, images, resolution=512, upscale_method="INTER_CUBIC", keep_proportion=True):
        images_np = self.preprocess(images, resolution, upscale_method, keep_proportion)
        tensor = self.to_tensor(images_np)
        line_np = self.run_model(tensor)
        return self.postprocess(line_np)
