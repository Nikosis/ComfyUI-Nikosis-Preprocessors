import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from safetensors.torch import load_file
from ....logger import niko_logger as logger
from ..image_utils import img_to_hwc3, resize_image_with_pad
from ...model_paths import get_model_path

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

class LineartDetector:
    def __init__(self):
        self.model = None
        # self.model_coarse = None
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls):
        return cls()

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self


    def _load_model(self, model_name):
        model_key = "lineArt"
        path = get_model_path(model_key, model_name)
        self.model = Generator(3, 1, 3)
        self.model.load_state_dict(load_file(path))
        self.model.to(self.device)
        logger.info(f"Loaded {model_name} model from: {path}")


    def _prepare_input(self, input_image, detect_resolution, upscale_method):
        assert isinstance(input_image, np.ndarray) and input_image.ndim == 4, "Input must be [B, H, W, C] NumPy array"
        batch_size = input_image.shape[0]
        resized_images = []
        remove_pad_funcs = []
        for i in range(batch_size):
            img, remove_pad = resize_image_with_pad(input_image[i], detect_resolution, upscale_method)
            resized_images.append(img)
            remove_pad_funcs.append(remove_pad)
        return np.stack(resized_images), lambda x: np.stack([remove_pad_funcs[i](x[i]) for i in range(batch_size)])

    def _to_model_tensor(self, image_np):
        image = torch.from_numpy(image_np).float().to(self.device) / 255.0
        return rearrange(image, 'b h w c -> b c h w')

    def _process_model_output(self, model_output):
        line = model_output.squeeze(1)
        return (line.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    def _post_process(self, line_np, remove_pad):
        """Convert [B, H, W] to [B, H', W', 3] NumPy array and remove padding."""
        batch_size = line_np.shape[0]
        detected_maps = [img_to_hwc3(line_np[i]) for i in range(batch_size)]  # List of [H, W, 3]
        detected_map = np.stack(detected_maps)  # [B, H, W, 3]
        return remove_pad(255 - detected_map)  # [B, H', W', 3], inverted

    def __call__(self, input_image, model_name, detect_resolution=512, upscale_method="INTER_CUBIC"):
        image_np, remove_pad = self._prepare_input(input_image, detect_resolution, upscale_method)
        self._load_model(model_name)
        model = self.model
        with torch.no_grad():
            image_tensor = self._to_model_tensor(image_np)
            model_output = model(image_tensor)
            line_np = self._process_model_output(model_output)
        detected_map = self._post_process(line_np, remove_pad)
        return detected_map
