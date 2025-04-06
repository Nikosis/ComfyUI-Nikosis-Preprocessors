import cv2
import numpy as np

from ...utils.image_utils import img_to_hwc3, resize_and_crop_to_multiple_cv2


class LineArtStandardDetector:
    def __call__(
            self,
            input_image=None,
            gaussian_sigma=6.0,
            intensity_threshold=8,
            detect_resolution=512,
            output_type=None,
            upscale_method="INTER_CUBIC",
            crop_from="symmetric",  # New parameter
            keep_proportion=True
    ):
        # Replace with new function, pass crop_from
        input_image = resize_and_crop_to_multiple_cv2(
            input_image,
            detect_resolution,
            multiple=16,
            upscale_method=upscale_method,
            crop_from=crop_from,
            keep_proportion=keep_proportion
        )

        x = input_image.astype(np.float32)
        gb = cv2.GaussianBlur(x, (0, 0), gaussian_sigma)
        intensity = np.min(gb - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = img_to_hwc3(detected_map)  # Ensure HWC format

        return detected_map
