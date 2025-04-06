import cv2
import numpy as np

from ...utils.image_utils import img_to_hwc3, resize_and_crop_to_multiple_cv2


def centered_canny(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.uint8

    y = cv2.Canny(x, int(canny_low_threshold), int(canny_high_threshold))
    y = y.astype(np.float32) / 255.0
    return y

def centered_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    result = [centered_canny(x[..., i], canny_low_threshold, canny_high_threshold) for i in range(3)]
    result = np.stack(result, axis=2)
    return result

def pyramid_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    H, W, C = x.shape
    acc_edge = None

    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hs, Ws = int(H * k), int(W * k)
        small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
        edge = centered_canny_color(small, canny_low_threshold, canny_high_threshold)
        if acc_edge is None:
            acc_edge = edge
        else:
            acc_edge = cv2.resize(acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc_edge = acc_edge * 0.75 + edge * 0.25

    return acc_edge

def norm255(x, low=4, high=96):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.float32

    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)

    x -= v_min
    x /= v_max - v_min

    return x * 255.0

def canny_pyramid(x, canny_low_threshold, canny_high_threshold):
    # For some reason, SAI's Control-lora Canny seems to be trained on canny maps with non-standard resolutions.
    # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

    color_canny = pyramid_canny_color(x, canny_low_threshold, canny_high_threshold)
    result = np.sum(color_canny, axis=2)

    return norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)

class PyraCannyDetector:
    def __call__(self,
                 input_image=None,
                 low_threshold=100,
                 high_threshold=200,
                 detect_resolution=512,
                 upscale_method="INTER_CUBIC",
                 keep_proportion=True): # output_type=None,

        detected_map = resize_and_crop_to_multiple_cv2(input_image=input_image, target_res=detect_resolution, upscale_method=upscale_method, keep_proportion=keep_proportion )
        detected_map = canny_pyramid(detected_map, low_threshold, high_threshold)
        detected_map = img_to_hwc3(detected_map)

        return detected_map

class SobelDetector:
    def __call__(self, input_image, low_threshold=100, high_threshold=200, detect_resolution=512,
                 upscale_method="INTER_CUBIC", keep_proportion=True):
        img = resize_and_crop_to_multiple_cv2(
            input_image, detect_resolution, upscale_method=upscale_method, keep_proportion=keep_proportion
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        _, strong_edges = cv2.threshold(magnitude, high_threshold, 255, cv2.THRESH_BINARY)
        weak_edges = cv2.inRange(magnitude, low_threshold, high_threshold)
        edges = strong_edges | weak_edges
        return img_to_hwc3(edges)

class PrewittDetector:
    def __call__(self, input_image, low_threshold=100, high_threshold=200, detect_resolution=512,
                 upscale_method="INTER_CUBIC", keep_proportion=True):
        img = resize_and_crop_to_multiple_cv2(
            input_image, detect_resolution, upscale_method=upscale_method, keep_proportion=keep_proportion
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
        prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
        magnitude = np.sqrt(prewittx**2 + prewitty**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        _, strong_edges = cv2.threshold(magnitude, high_threshold, 255, cv2.THRESH_BINARY)
        weak_edges = cv2.inRange(magnitude, low_threshold, high_threshold)
        edges = strong_edges | weak_edges
        return img_to_hwc3(edges)

class LaplacianDetector:
    def __call__(self, input_image, low_threshold=100, high_threshold=200, detect_resolution=512,
                 ksize=3, upscale_method="INTER_CUBIC", keep_proportion=True):
        img = resize_and_crop_to_multiple_cv2(
            input_image, detect_resolution, upscale_method=upscale_method, keep_proportion=keep_proportion
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        magnitude = np.abs(laplacian)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        _, strong_edges = cv2.threshold(magnitude, high_threshold, 255, cv2.THRESH_BINARY)
        weak_edges = cv2.inRange(magnitude, low_threshold, high_threshold)
        edges = strong_edges | weak_edges
        return img_to_hwc3(edges)

class CannyDetector:
    def __call__(self, input_image, low_threshold=100, high_threshold=200, detect_resolution=512,
                 upscale_method="INTER_CUBIC", keep_proportion=True):
        img = resize_and_crop_to_multiple_cv2(
            input_image, detect_resolution, upscale_method=upscale_method, keep_proportion=keep_proportion
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, low_threshold, high_threshold)
        return img_to_hwc3(canny)
