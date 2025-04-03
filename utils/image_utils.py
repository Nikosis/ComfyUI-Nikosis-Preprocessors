import warnings
import numpy as np
import torch
import cv2
from typing import Literal


def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
        warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")

    if "return_pil" in kwargs:
        warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"

    if type(output_type) is bool:
        warnings.warn(
            "Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"

    return input_image, output_type


# def resize_image_with_pad(input_image, resolution, upscale_method="", skip_hwc3=False,
#                           mode: Literal["constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"] = 'edge'):
#     if skip_hwc3:
#         img = input_image
#     else:
#         img = img_to_hwc3(input_image)
#     H_raw, W_raw, _ = img.shape
#     if resolution == 0:
#         return img, lambda x: x
#     k = float(resolution) / float(min(H_raw, W_raw))
#     H_target = int(np.round(float(H_raw) * k))
#     W_target = int(np.round(float(W_raw) * k))
#     img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
#     H_pad, W_pad = pad64(H_target), pad64(W_target)
#     # img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)
#     pad_width = ((0, H_pad), (0, W_pad), (0, 0))  # Use a tuple of tuples
#     img_padded = np.pad(img, pad_width, mode=mode)
#
#     def remove_pad(x):
#         return safer_memory(x[:H_target, :W_target, ...])
#
#     return safer_memory(img_padded), remove_pad

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]


def get_upscale_method(method_str):
    """Maps a string to an OpenCV interpolation method.

    Args:
        method_str (str): Name of the interpolation method (e.g., "INTER_LINEAR").

    Returns:
        int: Corresponding OpenCV interpolation constant.

    Raises:
        AssertionError: If method_str is not in UPSCALE_METHODS.
    """
    # Check if the provided method is valid
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    # Return the OpenCV interpolation constant for the given method
    return getattr(cv2, method_str)


# def resize_image_with_pad(input_image, resolution, upscale_method="", skip_hwc3=False,
#                           mode: Literal["constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"] = 'edge'):
#     """Resizes an image to a target resolution and pads it to a multiple of 64.
#
#     Args:
#         input_image (np.ndarray): Input image in HWC format.
#         resolution (int): Target resolution for the smaller dimension.
#         upscale_method (str, optional): Interpolation method for upscaling. Defaults to "".
#         skip_hwc3 (bool, optional): If True, skips HWC3 conversion. Defaults to False.
#         mode (str, optional): Padding mode for np.pad. Defaults to 'edge'.
#
#     Returns:
#         tuple: (padded image, function to remove padding)
#     """
#     # Skip HWC3 conversion if specified, otherwise convert to 3-channel HWC
#     if skip_hwc3:
#         img = input_image
#     else:
#         img = img_to_hwc3(input_image)
#
#     # Get original dimensions
#     height_raw, width_raw, _ = img.shape
#
#     # If resolution is 0, return the image as-is with a no-op remove_pad function
#     if resolution == 0:
#         return img, lambda x: x
#
#     # Calculate scaling factor based on the smaller dimension
#     k = float(resolution) / float(min(height_raw, width_raw))
#     # Compute target dimensions, rounding to nearest integer
#     height_target = int(np.round(float(height_raw) * k))
#     width_target = int(np.round(float(width_raw) * k))
#
#     # Resize the image using specified method for upscaling, INTER_AREA for downscaling
#     img = cv2.resize(img, (width_target, height_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
#
#     # Calculate padding needed to reach next multiple of 64
#     height_pad, width_pad = pad64(height_target), pad64(width_target)
#     # Define padding widths for height, width, and channels (no padding on channels)
#     pad_width = ((0, height_pad), (0, width_pad), (0, 0))  # Use a tuple of tuples
#     # Apply padding to the resized image
#     img_padded = np.pad(img, pad_width, mode=mode)
#
#     # Define a function to remove padding, returning only the original resized portion
#     def remove_pad(input_img):
#         return safer_memory(input_img[:height_target, :width_target, ...])
#
#     # Return padded image and remove_pad function, ensuring contiguous memory
#     return safer_memory(img_padded), remove_pad
def resize_image_with_pad(input_image, resolution, upscale_method="INTER_CUBIC", skip_hwc3=False,
                          mode: Literal["constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"] = 'edge'):
    """Resizes an image to a target resolution and pads it to a multiple of 64."""
    if skip_hwc3:
        img = input_image
    else:
        img = img_to_hwc3(input_image)

    height_raw, width_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x

    k = float(resolution) / float(min(height_raw, width_raw))
    height_target = int(np.round(float(height_raw) * k))
    width_target = int(np.round(float(width_raw) * k))
    img = cv2.resize(img, (width_target, height_target),
                     interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)

    height_pad, width_pad = pad64(height_target), pad64(width_target)
    pad_width = ((0, height_pad), (0, width_pad), (0, 0))
    img_padded = np.pad(img, pad_width, mode=mode)

    def remove_pad(input_img):
        return safer_memory(input_img[:height_target, :width_target, ...])

    return safer_memory(img_padded), remove_pad

def pad64(dim):
    """Calculates padding needed to reach the next multiple of 64.

    Args:
        dim (int): Dimension to pad (height or width).

    Returns:
        int: Number of pixels to add for padding.
    """
    # Compute the next multiple of 64 and subtract the original dimension
    return int(np.ceil(float(dim) / 64.0) * 64 - dim)


def safer_memory(x):
    """Ensures array has contiguous memory layout.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Contiguous array.
    """
    # Convert array to contiguous memory layout for better performance
    return np.ascontiguousarray(x)


def img_to_hwc3(img):
    """Converts an image to HWC format with 3 channels.

    Args:
        img (np.ndarray): Input image (grayscale, RGB, or RGBA).

    Returns:
        np.ndarray: Image in HWC format with 3 channels.

    Raises:
        AssertionError: If input dtype is not uint8 or channel count is invalid.
    """
    # Ensure input is uint8
    assert img.dtype == np.uint8
    # Add channel dimension if grayscale (2D)
    if img.ndim == 2:
        img = img[:, :, None]
    # Ensure image is 3D
    assert img.ndim == 3
    # Get dimensions
    height, width, channel = img.shape
    # Validate channel count
    assert channel == 1 or channel == 3 or channel == 4

    # Return as-is if already 3 channels
    if channel == 3:
        return img
    # Duplicate grayscale channel to RGB
    if channel == 1:
        return np.concatenate([img, img, img], axis=2)
    # Convert RGBA to RGB using alpha blending
    if channel == 4:
        color = img[:, :, 0:3].astype(np.float32)  # Extract RGB
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0  # Normalize alpha
        img_rgb = color * alpha + 255.0 * (1.0 - alpha)  # Blend with white background
        img_rgb = img_rgb.clip(0, 255).astype(np.uint8)  # Clip and convert to uint8
        return img_rgb


def torch_gc():
    """Performs garbage collection for PyTorch CUDA memory.

    Notes:
        Only runs if CUDA is available.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Clear CUDA memory cache
        torch.cuda.empty_cache()
        # Collect inter-process communication memory
        torch.cuda.ipc_collect()

# def safer_memory(x):
#     # Fix many MAC/AMD problems
#     return np.ascontiguousarray(x.copy()).copy()

# def safer_memory(x):
#     return np.ascontiguousarray(x)  # Ensures contiguous memory but avoids redundant copying
