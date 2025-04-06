import torch.nn as nn
from comfy.utils import load_torch_file as load_file

class ResidualBlock(nn.Module):
    """Residual block with conv_block substructure, matching saved weights."""
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    """Generator model for line art sketch processing."""
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
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

def load_lineart_model(model_path, device):
    """Loads a Generator model from the given path and moves it to the specified device."""
    model = Generator(3, 1, 3)
    model.load_state_dict(load_file(model_path))
    return model.to(device)
