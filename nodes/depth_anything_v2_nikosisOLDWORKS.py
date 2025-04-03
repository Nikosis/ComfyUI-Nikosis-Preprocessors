import torch
import torch.nn.functional as F
from torchvision import transforms
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from ..utils.model_paths import get_model_path
from ..utils.preprocessors.depth_anything_v2.dpt import DepthAnythingV2

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    is_accelerate_available = True
except ImportError:
    is_accelerate_available = False


def calculate_dim(orig_height, orig_width, resolution):
    shorter_dim = orig_width if orig_width <= orig_height else orig_height
    scale_factor = resolution / shorter_dim
    if orig_width <= orig_height:
        new_width = resolution
        new_height = int(round(orig_height * scale_factor))
    else:
        new_height = resolution
        new_width = int(round(orig_width * scale_factor))
    return new_height, new_width


def closest_multiplier_16(dim, multiplier=16):
    return round(dim / multiplier) * multiplier


class DepthAnythingV2Node:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [
            'depth_anything_v2_vits_fp16.safetensors', 'depth_anything_v2_vits_fp32.safetensors',
            'depth_anything_v2_vitb_fp16.safetensors', 'depth_anything_v2_vitb_fp32.safetensors',
            'depth_anything_v2_vitl_fp16.safetensors', 'depth_anything_v2_vitl_fp32.safetensors',
            'depth_anything_v2_metric_hypersim_vitl_fp32.safetensors', 'depth_anything_v2_metric_vkitti_vitl_fp32.safetensors'
        ]
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (model_list, {"default": 'depth_anything_v2_vitl_fp32.safetensors'}),
                "manual_resolution": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "resolution": ("INT", {"default": 1024, "min": 112, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "commence"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = """
    https://depth-anything-v2.github.io
    """

    def __init__(self):
        self.model = None
        self.current_config = None

    def load_model(self, model_name):
        device = mm.get_torch_device()
        dtype = torch.float16 if "fp16" in model_name else torch.float32

        encoder = "vitl" if "vitl" in model_name else "vitb" if "vitb" in model_name else "vits"
        max_depth = 20.0 if "hypersim" in model_name else 80.0
        model_config = {
            "vits": {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if not self.model or self.current_config != model_name:
            self.current_config = model_name
            model_path = get_model_path("depthanythingv2", model_name)
            with (init_empty_weights() if is_accelerate_available else nullcontext()):
                self.model = DepthAnythingV2(**{**model_config[encoder], 'is_metric': 'metric' in model_name, 'max_depth': max_depth})
            state_dict = load_torch_file(model_path)
            if is_accelerate_available:
                for key in state_dict:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=state_dict[key])
            else:
                self.model.load_state_dict(state_dict)
                self.model.to(device)
            self.model.eval()

        return {"model": self.model, "dtype": dtype, "is_metric": self.model.is_metric}

    def resize_to_nearest_multiple(self, image, target_res, multiple=14):
        """
        Resizes an image so that the shorter dimension matches the nearest multiple of `multiple`
        close to `target_res`, keeping aspect ratio.
        """
        B, C, H, W = image.shape
        aspect_ratio = W / H

        # Round target resolution to nearest multiple of `multiple`
        target_res = max(multiple, (target_res // multiple) * multiple)

        # Determine which dimension is shorter and set it to `target_res`
        if H < W:
            new_H = target_res
            new_W = int(new_H * aspect_ratio)
        else:
            new_W = target_res
            new_H = int(new_W / aspect_ratio)

        # Ensure new width & height are multiples of `multiple`
        new_H = max(multiple, (new_H // multiple) * multiple)
        new_W = max(multiple, (new_W // multiple) * multiple)

        # Perform resizing
        resized_image = F.interpolate(image, size=(new_H, new_W), mode="bilinear", align_corners=False)
        # resized_image = F.interpolate(image, size=(new_H, new_W), mode="bicubic", align_corners=False)
        return resized_image, (H, W), (new_H, new_W)


    def commence(self, model, images, manual_resolution=False, resolution=1024):

        # Load the specified depth estimation model with its configuration
        da_model = self.load_model(model)

        # Get available processing devices
        device = mm.get_torch_device()  # Main computation device (typically GPU)
        offload_device = mm.unet_offload_device()  # Device for unloading the model

        # Extract model components from the loaded bundle
        model_instance = da_model['model']  # The actual PyTorch model
        dtype = da_model['dtype']  # Data type (e.g., float16/float32) for precision

        # Get input dimensions: Batch, Height, Width, Channels
        B, H, W, C = images.shape

        # Calculate target dimensions while maintaining aspect ratio (if manual resolution enabled)
        manual_H, manual_W = calculate_dim(images.shape[1], images.shape[2], resolution)

        # Store original dimensions for later resizing
        orig_H, orig_W = H, W

        shortest_orig_dim = min(orig_H, orig_W)

        shortest_orig_dim = closest_multiplier_16(shortest_orig_dim)

        images = images.to(device)

        # Convert image tensor from (B,H,W,C) to (B,C,H,W) format for PyTorch processing
        images = images.permute(0, 3, 1, 2)

        # calc_res = longest_orig_dim if longest_orig_dim > 1024 else 1024
        calc_res_auto = 1022 if shortest_orig_dim < 1022 else shortest_orig_dim
        calc_res_manual = resolution if calc_res_auto < resolution else calc_res_auto

        if manual_resolution:
            images, (orig_H, orig_W), (new_H, new_W) = self.resize_to_nearest_multiple(images, calc_res_manual)
        else:
            images, (orig_H, orig_W), (new_H, new_W) = self.resize_to_nearest_multiple(images, calc_res_auto)

        # Normalize images using ImageNet stats (standard for many pretrained models)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images)

        # Initialize progress tracking
        pbar = ProgressBar(B)
        out = []  # Storage for output depth maps

        # Move model to processing device
        model_instance.to(device)

        # Set up mixed precision if supported (faster computation with float16)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            # Process each image in batch
            for img in normalized_images:
                # Add batch dimension and move to device
                # depth = model_instance(img.unsqueeze(0).to(device))
                depth = model_instance(img.unsqueeze(0))

                # Normalize depth to [0,1] range per-image
                depth = (depth - depth.min()) / (depth.max() - depth.min())

                # Store result and update progress
                out.append(depth.cpu())
                pbar.update(1)

            # Offload model when done
            model_instance.to(offload_device)

        # Combine all batch results and format as RGB (repeating single channel)
        depth_out = torch.cat(out, dim=0).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()

        # Calculate final output dimensions (ensuring even numbers)
        final_H, final_W = (orig_H // 2) * 2, (orig_W // 2) * 2

        if manual_resolution and (depth_out.shape[1] != manual_H or depth_out.shape[2] != manual_W):
            depth_out = F.interpolate(depth_out.permute(0, 3, 1, 2), size=(manual_H, manual_W), mode="bilinear").permute(0, 2, 3, 1)
        else:
            depth_out = F.interpolate(depth_out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear").permute(0, 2, 3, 1)

        # Final normalization clamp for safety
        depth_out = torch.clamp((depth_out - depth_out.min()) / (depth_out.max() - depth_out.min()), 0, 1)

        # Invert depth if model uses metric scale (near=1, far=0)
        if da_model['is_metric']:
            depth_out = 1 - depth_out

        return (depth_out,)  # Return as tuple for consistency


NODE_CLASS_MAPPINGS = {"DepthAnythingV2Node": DepthAnythingV2Node}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthAnythingV2Node": "Depth Anything V2"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'DepthAnythingV2Node']
