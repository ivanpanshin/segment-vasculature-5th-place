import os
import random

import numpy as np
import torch
from segmentation_models_pytorch import Unet
from torch import nn


class UnetUpscale(nn.Module):
    def __init__(
        self, encoder_name, decoder_use_batchnorm, in_channels, classes, upscale_factor, encoder_weights="imagenet"
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.model = Unet(
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            decoder_use_batchnorm=decoder_use_batchnorm,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, (x.shape[-2] * self.upscale_factor, x.shape[-1] * self.upscale_factor), mode="bilinear"
        )
        x = self.model(x)
        x = torch.nn.functional.interpolate(
            x, (x.shape[-2] // self.upscale_factor, x.shape[-1] // self.upscale_factor), mode="bilinear"
        )
        return x


def seed_everything(seed=100500):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
