import cv2
import numpy as np

#
### IMAGE TO (H,W,C3) #########################################################################################
#
def img_to_hwc3(img):
    """Converts an image to HWC format with 3 channels.

    Args:
        img (np.ndarray): Input image (grayscale, RGB, or RGBA).

    Returns:
        np.ndarray: Image in HWC format with 3 channels.

    Raises:
        AssertionError: If input dtype is not uint8 or channel count is invalid.
    """
    assert img.dtype == np.uint8

    # Convert grayscale (2D) to (H, W, 1)
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)  # Faster than concatenate

    # Ensure image is 3D
    assert img.ndim == 3
    if img.shape[0] in [1, 3, 4]:  # Likely CHW
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    height, width, channel = img.shape

    # Direct return if already 3 channels
    if channel == 3:
        return img

    # Convert grayscale (1 channel) to RGB
    if channel == 1:
        return np.repeat(img, 3, axis=2)  # Faster than np.concatenate

    # Convert RGBA to RGB using alpha blending
    if channel == 4:
        color = img[:, :, 0:3].astype(np.float32)  # Extract RGB
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0  # Normalize alpha
        return (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)  # Blend & return

    raise ValueError(f"Unsupported image shape: {img.shape}")  # Fail-fast if unexpected input
#
### RESIZE IMAGE AND CROP TO MULTIPLE OF - cv2 ##################################################################
#
def resize_and_crop_to_multiple_cv2(input_image, target_res, multiple=16, upscale_method="INTER_CUBIC",
                                    crop_from="symmetric", keep_proportion=True):
    H, W, C = input_image.shape

    is_multiple = (H % multiple == 0) and (W % multiple == 0)
    if is_multiple and target_res == min(H, W):
        return input_image.copy()

    aspect_ratio = W / H
    target_res = max(multiple, (target_res // multiple) * multiple)

    # Step 1: Calculate new dims
    if H < W:
        new_H = target_res
        new_W = int(new_H * aspect_ratio)
    else:
        new_W = target_res
        new_H = int(new_W / aspect_ratio)

    if keep_proportion:
        # Select interpolation method
        interpolation = getattr(cv2, upscale_method) if new_H > H or new_W > W else cv2.INTER_AREA
        resized_image = cv2.resize(input_image, (new_W, new_H), interpolation=interpolation)

        # Step 2: Crop to nearest multiples of 16
        H_resized, W_resized = resized_image.shape[:2]
        H_crop = (H_resized // multiple) * multiple  # Lower multiple
        W_crop = (W_resized // multiple) * multiple  # Lower multiple

        # Determine cropping based on crop_from
        if crop_from == "top/left":
            top_crop = H_resized - H_crop if H_resized != H_crop else 0
            left_crop = W_resized - W_crop if W_resized != W_crop else 0
        elif crop_from == "bottom/right":
            top_crop = 0
            left_crop = 0
        else:  # "symmetric" or invalid falls back to symmetric
            top_crop = (H_resized - H_crop) // 2 if H_resized != H_crop else 0
            left_crop = (W_resized - W_crop) // 2 if W_resized != W_crop else 0

        # Crop the image
        output_image = resized_image[
                       top_crop:top_crop + H_crop,
                       left_crop:left_crop + W_crop
                       ]
    else:

        if H < W:
            new_H = target_res  # Height is the smaller dimension, set to target_res
            # Find new_W as a multiple of 16 closest to target_res * aspect_ratio
            ideal_W = new_H * aspect_ratio
            new_W = round(ideal_W / multiple) * multiple
        else:
            new_W = target_res  # Width is the smaller dimension, set to target_res
            # Find new_H as a multiple of 16 closest to target_res / aspect_ratio
            ideal_H = new_W / aspect_ratio
            new_H = round(ideal_H / multiple) * multiple

            # Ensure minimum size
        new_H = max(new_H, multiple)
        new_W = max(new_W, multiple)

        # Resize directly to target_res x target_res (no cropping needed)
        interpolation = getattr(cv2, upscale_method) if new_H > H or new_W > W else cv2.INTER_AREA
        output_image = cv2.resize(input_image, (new_W, new_H), interpolation=interpolation)

    return output_image
