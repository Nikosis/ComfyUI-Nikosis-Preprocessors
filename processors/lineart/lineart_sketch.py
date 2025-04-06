import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from safetensors.torch import load_file
from ...logger import niko_logger as logger
from ...utils.image_utils import img_to_hwc3, resize_and_crop_to_multiple_cv2
from ...utils.model_paths import get_model_path

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super().__init__()
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)
        model2 = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*model2)
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out


class LineArtSketchDetector:
    """Detects line art sketches from images using a pretrained Generator model."""

    def __init__(self, device="cpu"):
        self.model = None
        self.device = device

    @classmethod
    def from_pretrained(cls, device="cpu"):
        """Creates an instance with the specified device."""
        return cls(device)

    def load_model(self, model_name):
        """Loads the Generator model from a safetensors file."""
        if self.model is None:
            path = get_model_path("lineart", model_name)
            self.model = Generator(3, 1, 3)  # 3 input channels, 1 output, 3 residual blocks
            self.model.load_state_dict(load_file(path))
            self.model.to(self.device)
            logger.info(f"Loaded {model_name} model from: {path}")
        return self

    def preprocess(self, images, resolution=512, upscale_method="INTER_CUBIC", keep_proportion=True):
        """Prepares a batch of images by converting to 3 channels and resizing/cropping."""
        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise ValueError("Input must be a [B, H, W, C] NumPy array")

        processed = []
        for img in images:
            img = img_to_hwc3(img)  # Ensure 3 channels
            img = resize_and_crop_to_multiple_cv2(
                img,
                target_res=resolution,
                multiple=16,
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
                crop_from="symmetric",
            )
            processed.append(img)

        return np.stack(processed)  # [B, H', W', 3]

    def to_tensor(self, images_np):
        """Converts NumPy images to a PyTorch tensor for model input."""
        tensor = torch.from_numpy(images_np).float().to(self.device) / 255.0
        return rearrange(tensor, 'b h w c -> b c h w')  # [B, 3, H', W']

    def run_model(self, tensor):
        """Runs the model and converts output to NumPy."""
        with torch.no_grad():
            output = self.model(tensor)  # [B, 1, H', W']
            line = output.squeeze(1)  # [B, H', W']
            return (line.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # [B, H', W']

    def postprocess(self, line_np):
        """Converts single-channel output to 3-channel inverted image."""
        batch_size = line_np.shape[0]
        color_images = [img_to_hwc3(line_np[i]) for i in range(batch_size)]  # List of [H', W', 3]
        result = np.stack(color_images)  # [B, H', W', 3]
        return 255 - result  # Invert: black lines on white background

    def detect(self, images, model_name, resolution=512, upscale_method="INTER_CUBIC", keep_proportion=True):
        """Processes images to detect line art sketches."""
        # Preprocess images
        images_np = self.preprocess(images, resolution, upscale_method, keep_proportion)

        # Load model if not already loaded
        self.load_model(model_name)

        # Run inference
        tensor = self.to_tensor(images_np)
        line_np = self.run_model(tensor)

        # Postprocess and return
        return self.postprocess(line_np)  # [B, H', W', 3]
